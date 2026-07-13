"""JSONL logging utilities for run records.

One run == one JSON object on one line in ``outputs/runs.jsonl``. We append so
that batch experiments accumulate cleanly and can be analyzed later with simple
line-by-line tooling.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from typing import List, Optional

from .config import DEFAULT_RUNS_LOG, OUTPUTS_DIR
from .schemas import (
    CostProxy,
    FanoutBranch,
    Persona,
    RunLog,
    SearchResult,
    QueryRecord,
)


def new_run_id() -> str:
    """Generate a short, unique run id."""
    return uuid.uuid4().hex[:12]


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def build_run_log(
    *,
    variant: str,
    query_record: QueryRecord,
    persona: Optional[Persona],
    fanout_branches: List[FanoutBranch],
    raw_search_results: List[SearchResult],
    final_answer: str,
    cost_proxy: CostProxy,
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    experiment_name: str = "placement_ablation_v1",
    method: str = "",
    seed: Optional[int] = None,
    planner_model: Optional[str] = None,
    synthesis_model: Optional[str] = None,
    requested_fanout_count: Optional[int] = None,
    realized_fanout_count: Optional[int] = None,
    full_candidate_plan_id: Optional[str] = None,
    executed_fanout_prefix: Optional[List[Dict[str, Any]]] = None,
    branch_types_executed: Optional[List[str]] = None,
    information_needs_executed: Optional[List[str]] = None,
    priority_ranks_executed: Optional[List[int]] = None,
    deduplicated_search_results: Optional[List[Dict[str, Any]]] = None,
    exact_synthesis_evidence: Optional[List[Dict[str, Any]]] = None,
    num_planner_calls: int = 0,
    num_synthesis_calls: int = 0,
    num_tavily_calls: int = 0,
    num_cache_hits: int = 0,
    num_cache_misses: int = 0,
    num_raw_results: int = 0,
    num_unique_results: int = 0,
    total_retrieved_context_size: int = 0,
    total_synthesis_context_size: int = 0,
    planner_latency: float = 0.0,
    search_latency: float = 0.0,
    synthesis_latency: float = 0.0,
    total_latency: float = 0.0,
    events: Optional[List[Dict[str, Any]]] = None,
) -> RunLog:
    """Assemble a :class:`RunLog` from pipeline outputs."""
    return RunLog(
        run_id=run_id or new_run_id(),
        experiment_name=experiment_name,
        timestamp=timestamp or utc_timestamp(),
        variant=variant,
        user_query=query_record.query,
        query_id=query_record.query_id,
        task_type=query_record.task_type,
        task_category=query_record.task_category,
        macro_domain=query_record.macro_domain,
        persona_relevant_dimensions=query_record.persona_relevant_dimensions,
        search_required=query_record.search_required,
        expected_personalization_stage=query_record.expected_personalization_stage,
        persona_id=persona.persona_id if persona else None,
        persona=persona.as_dict() if persona else None,
        fanout_branches=[b.as_dict() for b in fanout_branches],
        raw_search_results=[r.as_dict() for r in raw_search_results],
        final_answer=final_answer,
        cost_proxy=cost_proxy.as_dict(),
        method=method or variant,
        seed=seed,
        planner_model=planner_model,
        synthesis_model=synthesis_model,
        requested_fanout_count=requested_fanout_count,
        realized_fanout_count=realized_fanout_count,
        full_candidate_plan_id=full_candidate_plan_id,
        executed_fanout_prefix=executed_fanout_prefix or [],
        branch_types_executed=branch_types_executed or [],
        information_needs_executed=information_needs_executed or [],
        priority_ranks_executed=priority_ranks_executed or [],
        deduplicated_search_results=deduplicated_search_results or [],
        exact_synthesis_evidence=exact_synthesis_evidence or [],
        num_planner_calls=num_planner_calls,
        num_synthesis_calls=num_synthesis_calls,
        num_tavily_calls=num_tavily_calls,
        num_cache_hits=num_cache_hits,
        num_cache_misses=num_cache_misses,
        num_raw_results=num_raw_results,
        num_unique_results=num_unique_results,
        total_retrieved_context_size=total_retrieved_context_size,
        total_synthesis_context_size=total_synthesis_context_size,
        planner_latency=planner_latency,
        search_latency=search_latency,
        synthesis_latency=synthesis_latency,
        total_latency=total_latency,
        events=events or [],
    )



def append_run_log(run_log: RunLog, path: str = DEFAULT_RUNS_LOG) -> str:
    """Append a run log as one JSON line. Creates the outputs dir if needed.

    Returns the path written to.
    """
    os.makedirs(os.path.dirname(path) or OUTPUTS_DIR, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_log.as_dict(), ensure_ascii=False) + "\n")
    return path
