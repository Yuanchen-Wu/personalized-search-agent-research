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
        persona_relevant_dimensions=query_record.persona_relevant_dimensions,
        persona_id=persona.persona_id if persona else None,
        persona=persona.as_dict() if persona else None,
        fanout_branches=[b.as_dict() for b in fanout_branches],
        raw_search_results=[r.as_dict() for r in raw_search_results],
        final_answer=final_answer,
        cost_proxy=cost_proxy.as_dict(),
    )


def append_run_log(run_log: RunLog, path: str = DEFAULT_RUNS_LOG) -> str:
    """Append a run log as one JSON line. Creates the outputs dir if needed.

    Returns the path written to.
    """
    os.makedirs(os.path.dirname(path) or OUTPUTS_DIR, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_log.as_dict(), ensure_ascii=False) + "\n")
    return path
