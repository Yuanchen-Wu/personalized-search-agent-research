"""Adaptive re-fan-out retrieval loop (C3, "re-fanout until good").

The C3 method the paper actually wants. Each round is a FULL fan-out (not an
incremental add):

  1. FAN-OUT k search queries (persona-conditioned; round 1 from scratch, later
     rounds regenerated from the judge's feedback).
  2. SEARCH all k, collecting this round's evidence.
  3. JUDGE the RETRIEVED EVIDENCE (not the final answer): good enough to write a
     high-quality, grounded answer?
       - good     -> synthesize from THIS round's docs (caller does synthesis).
       - not good -> DISCARD this round's docs, carry the judge's feedback into a
                     revised fan-out, and retry.
  4. Loop until a round is approved, capped at ``max_rounds``.

Synthesis uses ONLY the approved round's evidence -- rejected rounds are thrown
away (they never reach the final answer). If no round is approved within the cap,
we fall back to the BEST round's evidence (highest coverage) so synthesis always
has something -- re-fanout can drift, so the last round is not necessarily the best.
Each round's evidence is logged (per-round), so smaller ``max_rounds`` are derivable
post-hoc from a single high-cap run (nested, like the fixed-k prefixes).

COST is variable and honestly counted: total = rounds x (fan-out gen + k searches
+ K retrieval-judge votes) + 1 synthesis. So this arm sits to the RIGHT of ``fixed_k`` on
the cost axis -- it trades extra retrieval + re-planning for a vetted fan-out.
That is the "adaptive Pareto-dominates the fixed-k curve" question.

LEAK-FREE INVARIANT (load-bearing): the fan-out generator and the retrieval judge
see only AGENT-VISIBLE inputs -- the query, ``persona.render_for_agent()`` (which
excludes the curated latent_profile), prior queries, and the agent's own retrieved
evidence. This module imports NOTHING from ``rubrics``; the frozen per-query rubric
is reserved for the evaluation judges, so peeking here would leak the answer key.

Termination is provable: the loop runs at most ``max_rounds`` iterations, each
doing at most ``fanout_size`` searches, so total searches <= max_rounds * k.
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
    _is_near_duplicate,
    _parse_and_validate_branches,
    search_tavily_cached,
)
from .llm_gemini import call_gemini
from .meta_prompt import (
    REFANOUT_INITIAL_FANOUT_PROMPT_V1,
    REFANOUT_RETRIEVAL_JUDGE_PROMPT_V1,
    REFANOUT_REVISED_FANOUT_PROMPT_V1,
)
from .schemas import FanoutBranch, Persona, SearchResult

# NOTE: do NOT `import rubrics` here -- see the leak-free invariant in the docstring.


@dataclass
class RefanoutCost:
    """Honest per-run cost accounting for one re-fanout run (counts ALL rounds)."""

    num_fanout_gen_calls: int = 0     # one fan-out-generation LLM call per round
    num_judge_calls: int = 0          # one retrieval-judge LLM call per round
    num_tavily_calls: int = 0         # TOTAL searches across ALL rounds (incl. discarded)
    num_rounds: int = 0               # rounds executed
    approved_round: Optional[int] = None   # 1-indexed round that was approved (None => none)
    fallback_round: Optional[int] = None   # round whose evidence was used on exhaustion (best; None if approved)
    approved_fanout_size: int = 0     # k of the round whose evidence feeds synthesis
    approval_threshold: float = 0.0   # coverage-score bar this run used (logged for the sweep)
    approved_score: Optional[float] = None  # mean coverage_score of the approved (or best-fallback) round
    num_cache_hits: int = 0
    num_cache_misses: int = 0
    fanout_gen_latency: float = 0.0   # summed across rounds
    judge_latency: float = 0.0        # summed across rounds
    search_latency: float = 0.0       # summed across rounds
    stop_reason: str = ""             # approved | max_rounds_exhausted


@dataclass
class RetrievalJudgeDecision:
    """Aggregated output of the retrieval-coverage judge for one round.

    With ``judge_samples`` (K) > 1 the judge is polled K times on the SAME evidence
    and ``coverage_score`` is the MEAN of the per-sample votes (a fractional 1-5).
    Averaging denoises the judge's run-to-run swing and gives the approval threshold
    a finer sweep than integer scores; ``sample_scores`` keeps the raw K votes so the
    per-call noise floor stays measurable from the logs.
    """

    coverage_score: float             # mean of sample_scores (1-5); approve when >= threshold
    coverage_gaps: List[str] = field(default_factory=list)
    feedback: str = ""                # guidance for regenerating the fan-out
    rationale: str = ""
    latency: float = 0.0              # summed across all K samples x their retries
    parse_ok: bool = True
    attempts: int = 1                 # summed judge LLM calls (K samples x retries)
    sample_scores: List[int] = field(default_factory=list)  # the raw K per-sample votes
    num_samples: int = 1              # number of parseable votes that fed the mean


@dataclass
class RefanoutResult:
    """Everything the runner needs to synthesize + log a re-fanout run."""

    approved_branches: List[FanoutBranch]   # the approved (or fallback last) round's fan-out
    approved_results: List[SearchResult]    # ONLY the approved round's docs -> synthesis
    events: List[Dict[str, Any]]            # per-round records + stop -> RunLog.events
    cost: RefanoutCost


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
    """Bounded, citable digest of this round's evidence for the judge prompt.

    Returns (digest_text, num_distinct_branch_queries).
    """
    num_branches = len({r.branch_query for r in results})
    if not results:
        return "(no evidence retrieved this round)", 0
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


def _dedupe_to_size(branches: List[FanoutBranch], fanout_size: int) -> List[FanoutBranch]:
    """Drop near-duplicate queries and cap at ``fanout_size``, reassigning ranks."""
    unique: List[FanoutBranch] = []
    seen: List[str] = []
    for b in branches:
        if len(unique) >= fanout_size:
            break
        if b.query and not _is_near_duplicate(b.query, seen):
            unique.append(b)
            seen.append(b.query)
    for i, b in enumerate(unique, start=1):
        b.priority_rank = i
    return unique


def generate_fanout(
    *,
    user_query: str,
    persona: Optional[Persona],
    fanout_size: int,
    round_idx: int,
    prior_queries: List[str],
    coverage_gaps: List[str],
    feedback: str,
    model: str = DEFAULT_GEMINI_MODEL,
    seed: int = 42,
    max_llm_retries: int = 2,
) -> Tuple[List[FanoutBranch], float, int]:
    """Generate one round's fan-out of ``fanout_size`` queries (leak-free).

    Round 1 uses the initial prompt; later rounds use the revised prompt seeded
    with the previous (rejected) queries + the judge's gaps/feedback so the new
    fan-out targets what was missing. Returns (branches, latency, attempts).
    """
    if round_idx <= 1:
        prompt = REFANOUT_INITIAL_FANOUT_PROMPT_V1.format(
            user_query=user_query,
            persona_block=_persona_block(persona) or "(no user context provided)",
            fanout_size=fanout_size,
        )
    else:
        prompt = REFANOUT_REVISED_FANOUT_PROMPT_V1.format(
            user_query=user_query,
            persona_block=_persona_block(persona) or "(no user context provided)",
            fanout_size=fanout_size,
            prior_queries_block="\n".join(f"  - {q}" for q in prior_queries) or "  (none)",
            coverage_gaps_block="\n".join(f"  - {g}" for g in coverage_gaps) or "  (none identified)",
            judge_feedback=feedback or "(no specific feedback)",
        )

    branches: List[FanoutBranch] = []
    latency = 0.0
    attempts = 0
    for attempt in range(max_llm_retries + 1):
        attempts += 1
        t0 = time.time()
        raw = call_gemini(
            prompt,
            model=model,
            response_mime_type="application/json",
            temperature=0.4,
            seed=seed + attempt,  # vary sampling so a retry can break a transient empty reply
        )
        latency += time.time() - t0
        branches = _parse_and_validate_branches(_extract_json(raw))
        if branches:
            break

    return _dedupe_to_size(branches, fanout_size), latency, attempts


def _call_judge_once(
    prompt: str,
    *,
    model: str,
    seed: int,
    temperature: float,
    max_llm_retries: int,
) -> Tuple[Any, float, int]:
    """One judge LLM call with unparseable-retry. Returns (parsed_json_or_None, latency, attempts).

    json_mode occasionally returns empty/blocked text under free-tier load; retry
    with a varied seed so a transient flake does not poison a sample.
    """
    parsed: Any = None
    latency = 0.0
    attempts = 0
    for attempt in range(max_llm_retries + 1):
        attempts += 1
        t0 = time.time()
        raw = call_gemini(
            prompt,
            model=model,
            response_mime_type="application/json",
            temperature=temperature,
            seed=seed + attempt,
        )
        latency += time.time() - t0
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            break
    return parsed, latency, attempts


def _interpret_judge_json(parsed: Dict[str, Any]) -> Tuple[int, List[str], str, str]:
    """Pull (clamped 1-5 score, gaps, feedback, rationale) out of one judge dict."""
    gaps = parsed.get("coverage_gaps") or []
    if not isinstance(gaps, list):
        gaps = [str(gaps)]
    try:
        score = int(round(float(parsed.get("coverage_score", 3))))
    except (ValueError, TypeError):
        score = 3
    score = max(1, min(5, score))  # clamp to the 1-5 scale
    return score, [str(g) for g in gaps], str(parsed.get("feedback", "")), str(parsed.get("rationale", ""))


def judge_retrieval(
    *,
    user_query: str,
    persona: Optional[Persona],
    evidence: List[SearchResult],
    fanout_branches: List[FanoutBranch],
    model: str = DEFAULT_GEMINI_MODEL,
    seed: int = 42,
    max_llm_retries: int = 2,
    judge_samples: int = 1,
    judge_temperature: float = 0.2,
) -> RetrievalJudgeDecision:
    """Score this round's retrieval, averaging ``judge_samples`` (K) judge votes.

    Leak-free (agent-visible inputs only). The judge is polled K times on the SAME
    evidence at ``judge_temperature`` (which must be > 0 so the votes actually vary);
    the returned ``coverage_score`` is their MEAN -- a fractional 1-5 that both
    denoises the judge's run-to-run swing and gives the CONTROLLER's approval
    threshold a finer sweep than integers. The judge never sees the threshold (so its
    score is not biased toward the bar) and it does not decide stop/continue. If EVERY
    sample is unparseable after retries we default to a top score so the loop stops
    cleanly rather than burning every round. ``sample_scores`` keeps the raw K votes
    so the per-call noise floor is measurable from the logs.
    """
    digest, num_branches = _render_evidence_digest(evidence)
    prompt = REFANOUT_RETRIEVAL_JUDGE_PROMPT_V1.format(
        user_query=user_query,
        persona_block=_persona_block(persona) or "(no user context provided)",
        fanout_queries_block="\n".join(f"  - {b.query}" for b in fanout_branches) or "  (none)",
        num_evidence=len(evidence),
        num_branches=num_branches,
        evidence_digest=digest,
    )

    k = max(1, judge_samples)
    sample_scores: List[int] = []
    sample_records: List[Tuple[int, List[str], str, str]] = []  # (score, gaps, feedback, rationale)
    total_latency = 0.0
    total_attempts = 0
    for s in range(k):
        # Distinct seed offset per sample -> K independent draws of the SAME prompt.
        parsed, latency, attempts = _call_judge_once(
            prompt, model=model, seed=seed + 1000 * s,
            temperature=judge_temperature, max_llm_retries=max_llm_retries,
        )
        total_latency += latency
        total_attempts += attempts
        if isinstance(parsed, dict):
            rec = _interpret_judge_json(parsed)
            sample_scores.append(rec[0])
            sample_records.append(rec)

    if not sample_scores:
        # Every sample unparseable after retries -> clean stop, not a runaway.
        return RetrievalJudgeDecision(
            coverage_score=5.0,
            rationale=f"(judge unparseable across all {k} sample(s); accepting round)",
            latency=total_latency,
            parse_ok=False,
            attempts=total_attempts,
            sample_scores=[],
            num_samples=0,
        )

    mean_score = round(sum(sample_scores) / len(sample_scores), 3)
    # Gaps: deduped union across samples (surface every identified miss). Feedback +
    # rationale: from the LOWEST-scoring sample -- the most critical read gives the
    # most actionable guidance for the next fan-out.
    seen: set = set()
    gaps_union: List[str] = []
    for _score, gaps, _fb, _rat in sample_records:
        for g in gaps:
            if g not in seen:
                seen.add(g)
                gaps_union.append(g)
    worst = min(sample_records, key=lambda r: r[0])
    return RetrievalJudgeDecision(
        coverage_score=mean_score,
        coverage_gaps=gaps_union,
        feedback=worst[2],
        rationale=worst[3],
        latency=total_latency,
        parse_ok=True,
        attempts=total_attempts,
        sample_scores=sample_scores,
        num_samples=len(sample_scores),
    )


def run_refanout_retrieval(
    *,
    user_query: str,
    persona: Optional[Persona],
    query_id: str,
    fanout_size: int = 4,
    max_rounds: int = 3,
    approval_threshold: float = 4.0,
    planner_model: str = DEFAULT_GEMINI_MODEL,
    judge_model: str = DEFAULT_GEMINI_MODEL,
    judge_samples: int = 1,
    judge_temperature: float = 0.2,
    seed: int = 42,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    max_results_per_branch: int = DEFAULT_MAX_RESULTS_PER_BRANCH,
    search_cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> RefanoutResult:
    """Run the re-fanout retrieve->judge->retry loop for one (query, persona).

    ``fanout_size`` (k) is the per-round fan-out width (matched to fixed_k{k} for
    comparability); ``max_rounds`` caps the retries. The controller does retrieval
    + control only; the caller synthesizes from ``approved_results`` exactly like
    the fixed path, so the only differences vs. fixed_k are (a) the fan-out is
    persona-conditioned per round and (b) it may re-fan-out until the retrieval
    passes the judge (at extra, honestly-counted cost).
    """
    cost = RefanoutCost()
    cost.approval_threshold = approval_threshold
    events: List[Dict[str, Any]] = []

    prior_queries: List[str] = []
    coverage_gaps: List[str] = []
    feedback: str = ""

    approved_branches: List[FanoutBranch] = []
    approved_results: List[SearchResult] = []
    last_branches: List[FanoutBranch] = []
    last_results: List[SearchResult] = []
    last_score: Optional[float] = None
    # Track the BEST round (highest coverage) so an exhausted loop falls back to it
    # rather than the last round -- re-fanout can drift, so a later round may be worse.
    best_score: Optional[float] = None
    best_branches: List[FanoutBranch] = []
    best_results: List[SearchResult] = []
    best_round_idx: Optional[int] = None

    for round_idx in range(1, max_rounds + 1):
        # --- 1. Fan-out (round 1 from scratch; later rounds from judge feedback).
        branches, gen_lat, gen_attempts = generate_fanout(
            user_query=user_query,
            persona=persona,
            fanout_size=fanout_size,
            round_idx=round_idx,
            prior_queries=prior_queries,
            coverage_gaps=coverage_gaps,
            feedback=feedback,
            model=planner_model,
            seed=seed,
        )
        cost.num_fanout_gen_calls += 1
        cost.fanout_gen_latency += gen_lat
        cost.num_rounds = round_idx

        # --- 2. Search all k -> THIS round's fresh evidence.
        round_results: List[SearchResult] = []
        t0 = time.time()
        for b in branches:
            results, hit = search_tavily_cached(
                query=b.query,
                branch_type=b.branch_type,
                max_results=max_results_per_branch,
                search_depth=search_depth,
                cache_path=search_cache_path,
                use_cache=use_cache,
            )
            round_results.extend(results)
            cost.num_tavily_calls += 1
            cost.num_cache_hits += 1 if hit else 0
            cost.num_cache_misses += 0 if hit else 1
        cost.search_latency += time.time() - t0

        last_branches, last_results = branches, round_results

        # --- 3. Judge THIS round's retrieval.
        decision = judge_retrieval(
            user_query=user_query,
            persona=persona,
            evidence=round_results,
            fanout_branches=branches,
            model=judge_model,
            seed=seed,
            judge_samples=judge_samples,
            judge_temperature=judge_temperature,
        )
        cost.num_judge_calls += 1
        cost.judge_latency += decision.latency
        last_score = decision.coverage_score
        # Best-round tracking (strict > so the earliest round wins ties -> fewer rounds).
        if best_score is None or decision.coverage_score > best_score:
            best_score = decision.coverage_score
            best_branches, best_results = branches, round_results
            best_round_idx = round_idx

        approved = decision.coverage_score >= approval_threshold
        events.append({
            "event_type": "refanout_round",
            "round": round_idx,
            "fanout_size": len(branches),
            "queries": [b.query for b in branches],
            "num_results": len(round_results),
            "coverage_score": decision.coverage_score,
            "sample_scores": decision.sample_scores,
            "num_judge_samples": decision.num_samples,
            "approval_threshold": approval_threshold,
            "approved": approved,
            "coverage_gaps": decision.coverage_gaps,
            "feedback": decision.feedback,
            "rationale": decision.rationale,
            "parse_ok": decision.parse_ok,
            "judge_attempts": decision.attempts,
            "gen_attempts": gen_attempts,
            # Per-round retrieved evidence -> lets max_rounds be derived post-hoc
            # (nested, like fixed-k prefixes) and rounds re-synthesized/re-scored later.
            "results": [r.as_dict() for r in round_results],
        })

        if approved:
            approved_branches, approved_results = branches, round_results
            cost.approved_round = round_idx
            cost.approved_score = decision.coverage_score
            cost.stop_reason = "approved"
            break

        # Not good enough: DISCARD this round's docs, carry feedback into the next
        # fan-out. (round_results is simply not retained beyond `last_results`.)
        prior_queries = [b.query for b in branches]
        coverage_gaps = decision.coverage_gaps
        feedback = decision.feedback

    # No round approved within the cap -> fall back to the BEST round's evidence
    # (highest coverage), NOT the last: re-fanout can drift, so a later round may be
    # worse. We must synthesize something; flag it distinctly from an approval.
    if cost.approved_round is None:
        approved_branches, approved_results = best_branches, best_results
        cost.approved_score = best_score
        cost.fallback_round = best_round_idx
        cost.stop_reason = "max_rounds_exhausted"
        events.append({
            "event_type": "refanout_fallback",
            "reason": "max_rounds_exhausted",
            "fallback_policy": "best_round",
            "using_round": best_round_idx,
            "best_coverage_score": best_score,
            "last_coverage_score": last_score,
        })

    # Well-formed sequential ranks on the branches that feed synthesis/judges.
    for i, b in enumerate(approved_branches, start=1):
        b.priority_rank = i
    cost.approved_fanout_size = len(approved_branches)

    events.append({
        "event_type": "refanout_stop",
        "stop_reason": cost.stop_reason,
        "num_rounds": cost.num_rounds,
        "approved_round": cost.approved_round,
        "fallback_round": cost.fallback_round,
        "approval_threshold": approval_threshold,
        "approved_score": cost.approved_score,
        "total_tavily_calls": cost.num_tavily_calls,
        "approved_fanout_size": cost.approved_fanout_size,
    })

    return RefanoutResult(
        approved_branches=approved_branches,
        approved_results=approved_results,
        events=events,
        cost=cost,
    )
