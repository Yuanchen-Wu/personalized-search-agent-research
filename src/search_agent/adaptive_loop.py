"""Adaptive retrieval loop (C3): retrieve -> assess -> continue/stop controller.

    ── STATUS: PARKED — not in current C3 scope (2026-07-18) ──────────────────
    Incremental deepening is DELIBERATELY set aside. Active C3 work is the
    "re-fanout until good" loop in ``adaptive_refanout.py``. This module,
    ``configs/adaptive_loop_v1.yaml`` and ``outputs/adaptive_loop_v1/`` are kept
    for a POSSIBLE LATER mechanism-A/B study (evidence-conditioned query
    *selection* / early-stop), but are NOT run or maintained right now.
    ───────────────────────────────────────────────────────────────────────────

Originally proposed as a C3 method. Instead of executing a fixed number of
fan-out branches (fixed_k1..k8), the controller:

  1. SEEDS with the top ``seed_size`` branches of the SAME ordered plan the
     fixed-k harness uses (so round-1 evidence == fixed_k{seed_size}, and the
     plan/search caches are shared -> the seed is free).
  2. After each retrieval round, asks an LLM assessor whether the evidence is
     SUFFICIENT; if not, the assessor also PROPOSES follow-up queries (depth /
     re-planning) targeting the gaps it found.
  3. Executes accepted follow-ups (genuinely new queries -> real Tavily calls),
     and repeats until sufficient / budget exhausted / no new queries / max rounds.

Cost is accounted honestly: the assessor call is a real LLM call and is counted
(``num_assessor_calls``), because whether adaptive Pareto-dominates fixed-k on
the *total*-LLM-call axis (not just retrieval calls) is the crux of the claim.

LEAK-FREE INVARIANT (load-bearing): the assessor self-estimates coverage from
AGENT-VISIBLE inputs only -- the query, ``persona.render_for_agent()`` (which
excludes the curated latent_profile), and its own retrieved evidence. This module
deliberately imports NOTHING from ``rubrics``; the frozen per-query rubric is
reserved for the judges. Peeking at it would leak the answer key and break
comparability with the leak-free judges on the shared frontier.

Termination is provable: every loop iteration either breaks or strictly increases
``realized`` toward the finite ``budget_cap``, and the number of assessor calls is
bounded by ``max_rounds - 1``; ``realized >= seed_size >= 1`` always, so synthesis
never sees empty evidence.

TWO MODES (set by ``fill_to_budget``):
  * variable-budget (``adaptive_bN``, ``fill_to_budget=False``, the default): the
    loop STOPS as soon as the assessor calls the evidence sufficient, so realized
    breadth varies per query -- this is the "spend less search for the same quality"
    (mechanism B) cost story. Stop reasons {sufficient, budget_exhausted,
    no_new_queries, max_rounds} are mutually exhaustive.
  * fixed-budget (``adaptive_kN``, ``fill_to_budget=True``): the loop NEVER stops on
    sufficiency; it spends EXACTLY ``budget_cap`` searches (iteratively selected,
    then backfilled from the shared plan if the assessor proposes too few), so the
    retrieval cost is matched to ``fixed_k{budget_cap}``. This isolates mechanism A
    -- the value of evidence-conditioned query *selection* -- holding breadth (and
    thus retrieval cost) constant, and stays directly comparable to the frozen
    fixed-k runs. Sufficiency is still recorded (``sufficient_before_budget``) as a
    side-signal for a later stopping study, but does not end the loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_RESULTS_PER_BRANCH,
    DEFAULT_SEARCH_DEPTH,
)
from .fanout import _extract_json, _persona_block
from .fixed_fanout import (
    PROMPT_VERSION_ORDERED_PLANNER,
    _is_near_duplicate,
    _parse_and_validate_branches,
    get_or_create_shared_plan,
    search_tavily_cached,
)
from .llm_gemini import call_gemini
from .meta_prompt import ADAPTIVE_ASSESS_PROPOSE_PROMPT_V1, STRICTNESS_INSTRUCTIONS
from .schemas import FanoutBranch, Persona, SearchResult

# NOTE: do NOT `import rubrics` here -- see the leak-free invariant in the module docstring.

# Candidate pool the seed plan is drawn from. MUST match the fixed-k harness
# (candidate_pool_size=8) so the shared plan cache is hit and the seed reuses
# exactly the branches fixed_k{seed_size} saw.
SEED_PLAN_POOL_SIZE = 8


@dataclass
class AdaptiveCost:
    """Honest per-run cost accounting for one adaptive run."""

    num_planner_calls: int = 0        # seed ordered-plan generation (0 on plan-cache hit)
    num_assessor_calls: int = 0       # assess+propose LLM calls
    num_tavily_calls: int = 0         # realized branches searched across all rounds
    realized_fanout_count: int = 0    # == num_tavily_calls (the variable "k" we spent)
    num_rounds: int = 0               # retrieval rounds executed (seed counts as round 1)
    num_backfilled: int = 0           # fixed-budget mode: branches taken from the plan to hit the cap
    num_cache_hits: int = 0
    num_cache_misses: int = 0
    planner_latency: float = 0.0
    assessor_latency: float = 0.0     # summed across assessor calls
    search_latency: float = 0.0       # summed real per-round search wall-clock
    sufficient_before_budget: bool = False  # fixed-budget mode: assessor judged evidence enough before the cap
    stop_reason: str = ""             # sufficient | budget_exhausted | no_new_queries | max_rounds | filled_to_budget


@dataclass
class AssessDecision:
    """Parsed output of one assess+propose call."""

    sufficient: bool
    coverage_gaps: List[str] = field(default_factory=list)
    proposed_branches: List[FanoutBranch] = field(default_factory=list)
    rationale: str = ""
    latency: float = 0.0
    parse_ok: bool = True
    attempts: int = 1                 # assessor LLM calls made (>1 => a retry was needed)
    assessor_context_size: int = 0    # chars of the prompt we showed the assessor (a logged cost)


@dataclass
class AdaptiveResult:
    """Everything the runner needs to synthesize + log an adaptive run."""

    branches: List[FanoutBranch]          # executed branches, priority_rank reassigned 1..N
    raw_results: List[SearchResult]       # all rounds concatenated
    events: List[Dict[str, Any]]          # plan events + per-round decisions -> RunLog.events
    cost: AdaptiveCost


def _domain(url: str) -> str:
    try:
        net = urlparse(url or "").netloc.lower()
        return net[4:] if net.startswith("www.") else net
    except Exception:
        return ""


def _render_evidence_digest(
    results: List[SearchResult],
    snippet_chars: int = 180,
    max_items: int = 40,
) -> Tuple[str, int]:
    """Render a bounded, citable digest of evidence-so-far for the assessor prompt.

    Returns (digest_text, num_distinct_branch_queries).
    """
    num_branches = len({r.branch_query for r in results})
    if not results:
        return "(no evidence retrieved yet)", 0
    lines: List[str] = []
    for i, r in enumerate(results[:max_items], start=1):
        snippet = " ".join((r.content or "").split())
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars] + "…"
        dom = _domain(r.url)
        dom_str = f" ({dom})" if dom else ""
        lines.append(f"[{i}] ({r.branch_type}) {r.title} — {snippet}{dom_str}")
    if len(results) > max_items:
        lines.append(f"... (+{len(results) - max_items} more results not shown)")
    return "\n".join(lines), num_branches


def _backfill_from_plan(
    plan: List[FanoutBranch],
    executed_queries: List[str],
    needed: int,
) -> List[FanoutBranch]:
    """Pick up to ``needed`` not-yet-searched branches from the shared ordered plan.

    Fixed-budget (adaptive_kN) mode only: when the assessor proposes too few genuinely
    new queries, we top up from the SAME plan the fixed-k arm uses -- in rank order,
    skipping near-duplicates of anything already searched -- so realized breadth still
    hits ``budget_cap`` exactly (retrieval cost matched to fixed_k). In the degenerate
    case (assessor adds nothing) this collapses adaptive_kN back to fixed_k{N}.
    """
    picked: List[FanoutBranch] = []
    seen = list(executed_queries)
    for branch in plan:
        if len(picked) >= needed:
            break
        if not _is_near_duplicate(branch.query, seen):
            picked.append(branch)
            seen.append(branch.query)
    return picked


def assess_and_propose(
    *,
    user_query: str,
    persona: Optional[Persona],
    evidence: List[SearchResult],
    executed_queries: List[str],
    max_new_queries: int,
    strictness: str = "balanced",
    model: str = DEFAULT_GEMINI_MODEL,
    seed: int = 42,
    max_llm_retries: int = 2,
) -> AssessDecision:
    """One combined assess+propose LLM call (leak-free; agent-visible inputs only).

    json_mode occasionally returns empty/blocked text under free-tier load; we retry
    with a varied seed (and tolerate a bare-array response) so a transient flake does
    not silently collapse the loop into a premature stop.
    """
    digest, num_branches = _render_evidence_digest(evidence)
    prompt = ADAPTIVE_ASSESS_PROPOSE_PROMPT_V1.format(
        strictness_instruction=STRICTNESS_INSTRUCTIONS.get(
            strictness, STRICTNESS_INSTRUCTIONS["balanced"]
        ),
        user_query=user_query,
        persona_block=_persona_block(persona) or "(no user context provided)",
        num_evidence=len(evidence),
        num_branches=num_branches,
        evidence_digest=digest,
        executed_queries_block="\n".join(f"  - {q}" for q in executed_queries) or "  (none)",
        max_new_queries=max_new_queries,
    )

    parsed: Any = None
    raw = ""
    latency = 0.0
    attempts = 0
    for attempt in range(max_llm_retries + 1):
        attempts += 1
        t0 = time.time()
        raw = call_gemini(
            prompt,
            model=model,
            response_mime_type="application/json",
            temperature=0.2,
            seed=seed + attempt,  # vary sampling so a retry can break a transient empty/blocked reply
        )
        latency += time.time() - t0
        parsed = _extract_json(raw)
        if isinstance(parsed, list):
            # Tolerate a bare array of proposed-query objects (still "not sufficient").
            parsed = {"sufficient": False, "coverage_gaps": [], "proposed_queries": parsed, "rationale": ""}
        if isinstance(parsed, dict):
            break

    if not isinstance(parsed, dict):
        # Defensive: after retries, a still-unparseable response cleanly stops the loop
        # rather than risking a runaway. Rare; logged via parse_ok=False for filtering.
        return AssessDecision(
            sufficient=True,
            rationale=f"(assessor response unparseable after {attempts} attempts; stopping)",
            latency=latency,
            parse_ok=False,
            attempts=attempts,
            assessor_context_size=len(prompt),
        )

    proposed_branches = _parse_and_validate_branches(parsed.get("proposed_queries") or [])
    # If `sufficient` is absent, infer it: no proposals => sufficient, else not.
    sufficient = bool(parsed.get("sufficient", len(proposed_branches) == 0))
    gaps = parsed.get("coverage_gaps") or []
    if not isinstance(gaps, list):
        gaps = [str(gaps)]
    return AssessDecision(
        sufficient=sufficient,
        coverage_gaps=[str(g) for g in gaps],
        proposed_branches=proposed_branches,
        rationale=str(parsed.get("rationale", "")),
        latency=latency,
        parse_ok=True,
        attempts=attempts,
        assessor_context_size=len(prompt),
    )


def run_adaptive_retrieval(
    *,
    user_query: str,
    persona: Optional[Persona],
    query_id: str,
    budget_cap: int,
    seed_size: int = 2,
    max_rounds: int = 5,
    per_round_cap: int = 4,
    strictness: str = "balanced",
    planner_model: str = DEFAULT_GEMINI_MODEL,
    assessor_model: str = DEFAULT_GEMINI_MODEL,
    seed: int = 42,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    max_results_per_branch: int = DEFAULT_MAX_RESULTS_PER_BRANCH,
    plans_cache_path: Optional[str] = None,
    search_cache_path: Optional[str] = None,
    use_cache: bool = True,
    fill_to_budget: bool = False,
) -> AdaptiveResult:
    """Run the adaptive retrieve->assess->continue/stop loop for one (query, persona).

    ``budget_cap`` (B) is the ceiling on total branches searched. The controller does
    retrieval + control only; the caller synthesizes/dedups exactly like the fixed
    path (comparability).

    ``fill_to_budget`` selects the mode (see the module docstring):
      * False (adaptive_bN): stop early on sufficiency; realized breadth varies (B is a
        cap). Traces the cost-savings frontier.
      * True (adaptive_kN): never stop on sufficiency; spend EXACTLY ``budget_cap``
        searches (iterative selection + plan backfill), holding retrieval cost equal to
        fixed_k{budget_cap} so the quality delta isolates iterative query selection.
    """
    cost = AdaptiveCost()
    events: List[Dict[str, Any]] = []
    executed_branches: List[FanoutBranch] = []
    raw_results: List[SearchResult] = []
    executed_queries: List[str] = []

    eff_seed_size = max(1, min(seed_size, budget_cap))

    def _do_search(branch: FanoutBranch) -> None:
        """Search one branch, append results, update counters/latency."""
        t0 = time.time()
        results, hit = search_tavily_cached(
            query=branch.query,
            branch_type=branch.branch_type,
            max_results=max_results_per_branch,
            search_depth=search_depth,
            cache_path=search_cache_path,
            use_cache=use_cache,
        )
        cost.search_latency += time.time() - t0
        raw_results.extend(results)
        executed_branches.append(branch)
        executed_queries.append(branch.query)
        cost.num_tavily_calls += 1
        cost.num_cache_hits += 1 if hit else 0
        cost.num_cache_misses += 0 if hit else 1

    # --- Round 1: seed from the shared ordered plan (reuses C2's plan + search caches).
    t0 = time.time()
    plan_id, full_plan, plan_events, plan_cache_hit = get_or_create_shared_plan(
        query_id=query_id,
        user_query=user_query,
        persona=persona,
        candidate_pool_size=SEED_PLAN_POOL_SIZE,
        planner_model=planner_model,
        prompt_version=PROMPT_VERSION_ORDERED_PLANNER,
        seed=seed,
        cache_path=plans_cache_path,
        use_cache=use_cache,
    )
    cost.planner_latency = time.time() - t0
    cost.num_planner_calls = 0 if plan_cache_hit else 1
    events.extend(plan_events)

    seed_branches = full_plan[:eff_seed_size]
    for branch in seed_branches:
        _do_search(branch)
    cost.num_rounds = 1
    realized = len(executed_branches)
    events.append({
        "event_type": "adaptive_seed",
        "plan_id": plan_id,
        "plan_cache_hit": plan_cache_hit,
        "seed_size": realized,
        "budget_cap": budget_cap,
    })

    # --- Assess/continue rounds.
    assess_rounds = 0
    stop_reason = ""
    while True:
        remaining = budget_cap - realized
        if remaining <= 0:
            stop_reason = "budget_exhausted"
            break
        if assess_rounds >= max_rounds - 1:
            stop_reason = "max_rounds"
            break

        allow = min(remaining, per_round_cap)
        decision = assess_and_propose(
            user_query=user_query,
            persona=persona,
            evidence=raw_results,
            executed_queries=executed_queries,
            max_new_queries=allow,
            strictness=strictness,
            model=assessor_model,
            seed=seed,
        )
        assess_rounds += 1
        cost.num_assessor_calls += 1
        cost.assessor_latency += decision.latency

        if decision.sufficient:
            cost.sufficient_before_budget = True
            # Variable-budget (adaptive_bN) stops here. Fixed-budget (adaptive_kN) records
            # the signal but keeps going -- it must spend exactly budget_cap searches.
            if not fill_to_budget:
                stop_reason = "sufficient"
                events.append({
                    "event_type": "assess_round", "round": assess_rounds,
                    "sufficient": True, "num_proposed": len(decision.proposed_branches),
                    "num_accepted": 0, "coverage_gaps": decision.coverage_gaps,
                    "rationale": decision.rationale, "parse_ok": decision.parse_ok,
                    "attempts": decision.attempts,
                    "assessor_latency": decision.latency, "realized_after": realized,
                })
                break

        # Accept up to `allow` genuinely-new proposed queries.
        accepted: List[FanoutBranch] = []
        for b in decision.proposed_branches:
            if len(accepted) >= allow:
                break
            if not _is_near_duplicate(b.query, executed_queries):
                accepted.append(b)
                executed_queries.append(b.query)  # reserve so within-round dups are caught

        events.append({
            "event_type": "assess_round", "round": assess_rounds,
            "sufficient": decision.sufficient, "num_proposed": len(decision.proposed_branches),
            "num_accepted": len(accepted), "coverage_gaps": decision.coverage_gaps,
            "rationale": decision.rationale, "parse_ok": decision.parse_ok,
            "attempts": decision.attempts,
            "assessor_latency": decision.latency, "realized_after": realized + len(accepted),
        })

        if not accepted:
            # No usable new queries. Variable mode stops; fixed-budget mode breaks out
            # and backfills from the plan below so realized still reaches budget_cap.
            stop_reason = "assessor_exhausted" if fill_to_budget else "no_new_queries"
            break

        # `_do_search` re-appends the query; drop the reservation to avoid duplicates.
        for b in accepted:
            executed_queries.pop(executed_queries.index(b.query))
            _do_search(b)
        cost.num_rounds += 1
        realized = len(executed_branches)

        if realized >= budget_cap:
            stop_reason = "budget_exhausted"
            break

    # --- Fixed-budget backfill (adaptive_kN): guarantee realized == budget_cap.
    # The loop may exit early (assessor called it sufficient / proposed too few / hit
    # max_rounds) with realized < cap. Top up from the shared ordered plan so retrieval
    # cost is matched to fixed_k{budget_cap}; the worst case collapses to fixed_k.
    if fill_to_budget and realized < budget_cap:
        backfill = _backfill_from_plan(full_plan, executed_queries, budget_cap - realized)
        for branch in backfill:
            _do_search(branch)
        cost.num_backfilled = len(backfill)
        realized = len(executed_branches)
        events.append({
            "event_type": "adaptive_backfill",
            "num_backfilled": len(backfill),
            "prior_stop_reason": stop_reason,
            "realized_after": realized,
        })
        stop_reason = "filled_to_budget"
        if realized < budget_cap:
            # Only reachable if the shared plan pool (SEED_PLAN_POOL_SIZE) < budget_cap.
            events.append({
                "event_type": "adaptive_underfilled",
                "realized_fanout_count": realized,
                "budget_cap": budget_cap,
            })

    # Reassign sequential priority ranks over the execution order (well-formed for judges).
    for idx, b in enumerate(executed_branches, start=1):
        b.priority_rank = idx

    cost.realized_fanout_count = realized
    cost.stop_reason = stop_reason
    events.append({
        "event_type": "adaptive_stop",
        "stop_reason": stop_reason,
        "realized_fanout_count": realized,
        "num_rounds": cost.num_rounds,
        "num_assessor_calls": cost.num_assessor_calls,
    })

    return AdaptiveResult(
        branches=executed_branches,
        raw_results=raw_results,
        events=events,
        cost=cost,
    )
