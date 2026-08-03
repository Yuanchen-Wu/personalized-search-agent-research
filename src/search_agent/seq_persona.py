"""Sequential persona-gap retrieval (seqpersona_k*): one query at a time,
each next query proposed by a LEAK-FREE persona-gap critic.

Cost-matched to fixed_k{budget}: exactly ``budget`` searches. The critic sees
only agent-visible inputs (query, persona.render_for_agent() via the shared
persona block, retrieved evidence) — never the rubric; this module imports
nothing from ``rubrics``. Synthesis happens ONCE at the end, by the CALLER,
with the same base synthesizer as the fixed arms — so the fixed-vs-seqpersona
comparison isolates the retrieval policy alone.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import DEFAULT_GEMINI_MODEL, DEFAULT_MAX_RESULTS_PER_BRANCH, DEFAULT_SEARCH_DEPTH
from .adaptive_refanout import _render_evidence_digest, generate_fanout
from .fanout import _extract_json, _persona_block
from .fixed_fanout import _generate_deterministic_fallback_branch, search_tavily_cached
from .llm_gemini import call_gemini
from .meta_prompt import SEQPERSONA_ASSESS_PROPOSE_V1
from .schemas import FanoutBranch, Persona, SearchResult

# NOTE: do NOT import rubrics here — leak-free invariant (see docstring).


@dataclass
class SeqPersonaResult:
    branches: List[FanoutBranch]          # the budget executed queries, in order
    raw_results: List[SearchResult]       # concatenated evidence (pre-dedup)
    num_tavily_calls: int
    num_planner_calls: int                # seed call + per-step propose calls
    planner_latency: float
    search_latency: float
    events: List[Dict[str, Any]] = field(default_factory=list)


def _propose_next(user_query: str, persona: Optional[Persona], evidence: List[SearchResult],
                  prior_queries: List[str], remaining: int, model: str, seed: int):
    """One critic call: unmet persona needs + the next query. Returns (branch, event)."""
    digest, _ = _render_evidence_digest(evidence)
    prompt = SEQPERSONA_ASSESS_PROPOSE_V1.format(
        user_query=user_query,
        persona_block=_persona_block(persona) or "(no user context provided)",
        prior_queries_block="\n".join(f"{i}. {q}" for i, q in enumerate(prior_queries, 1)),
        num_evidence=len(evidence),
        evidence_digest=digest,
        remaining=remaining,
    )
    parsed, err = None, None
    try:
        raw = call_gemini(prompt, model=model, response_mime_type="application/json", seed=seed)
        parsed = _extract_json(raw)
    except Exception as e:  # noqa: BLE001 - fallback handles it
        err = f"{type(e).__name__}: {e}"

    next_query, needs, rationale = "", [], ""
    if isinstance(parsed, dict):
        next_query = str(parsed.get("next_query") or "").strip()
        needs = parsed.get("unmet_user_needs") or []
        if not isinstance(needs, list):
            needs = [str(needs)]
        rationale = str(parsed.get("rationale") or "").strip()

    step = len(prior_queries) + 1
    if next_query and next_query.lower() not in {q.lower() for q in prior_queries}:
        branch = FanoutBranch(branch_type="personalized", query=next_query,
                              rationale=rationale or "seqpersona proposal",
                              information_need="; ".join(str(n) for n in needs[:3]),
                              priority_rank=step)
        fallback = False
    else:
        branch = _generate_deterministic_fallback_branch(user_query, step, prior_queries)
        branch.priority_rank = step
        fallback = True

    event = {"event_type": "seqpersona_step", "step": step, "query": branch.query,
             "unmet_user_needs": [str(n) for n in needs], "rationale": rationale,
             "fallback": fallback, "parse_error": err}
    return branch, event


def run_seqpersona_retrieval(*, user_query: str, persona: Optional[Persona], query_id: str,
                             budget: int = 8,
                             planner_model: str = DEFAULT_GEMINI_MODEL,
                             seed: int = 42,
                             search_depth: str = DEFAULT_SEARCH_DEPTH,
                             max_results_per_branch: int = DEFAULT_MAX_RESULTS_PER_BRANCH,
                             search_cache_path: Optional[str] = None,
                             use_cache: bool = True) -> SeqPersonaResult:
    """Spend exactly ``budget`` searches one at a time, persona-gap-guided."""
    branches: List[FanoutBranch] = []
    raw: List[SearchResult] = []
    events: List[Dict[str, Any]] = []
    n_planner = 0
    t_plan = t_search = 0.0

    def _search(branch: FanoutBranch) -> None:
        nonlocal t_search
        t0 = time.time()
        res, hit = search_tavily_cached(
            query=branch.query, branch_type=branch.branch_type,
            max_results=max_results_per_branch, search_depth=search_depth,
            cache_path=search_cache_path, use_cache=use_cache)
        t_search += time.time() - t0
        raw.extend(res)
        events.append({"event_type": "seqpersona_search", "step": branch.priority_rank,
                       "query": branch.query, "num_results": len(res), "cache_hit": hit})

    # Step 1: persona-conditioned seed query (same generator as the re-fanout loop).
    t0 = time.time()
    seed_branches, _lat, _att = generate_fanout(
        user_query=user_query, persona=persona, fanout_size=1, round_idx=1,
        prior_queries=[], coverage_gaps=[], feedback="", model=planner_model, seed=seed)
    t_plan += time.time() - t0
    n_planner += 1
    seed_branch = seed_branches[0] if seed_branches else _generate_deterministic_fallback_branch(user_query, 1, [])
    seed_branch.priority_rank = 1
    branches.append(seed_branch)
    _search(seed_branch)

    # Steps 2..budget: critique the pool against the persona, fetch the gap.
    while len(branches) < budget:
        t0 = time.time()
        branch, ev = _propose_next(user_query, persona, raw,
                                   [b.query for b in branches],
                                   budget - len(branches), planner_model,
                                   seed + len(branches))
        t_plan += time.time() - t0
        n_planner += 1
        events.append(ev)
        branches.append(branch)
        _search(branch)

    return SeqPersonaResult(branches=branches, raw_results=raw,
                            num_tavily_calls=len(branches), num_planner_calls=n_planner,
                            planner_latency=t_plan, search_latency=t_search, events=events)
