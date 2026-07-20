"""Frozen per-query rubric loading for the leak-free judges.

The data generator authors a complete grading rubric for every query (inside the
query record's ``metadata``), but it is dropped from ``runs.jsonl`` at logging
time (``RunLog`` does not carry query metadata). The evaluators therefore
re-join the rubric here, keyed by ``query_id``, so they grade against the
*frozen* rubric authored before any answer existed — instead of the
agent-invisible persona answer key (``latent_profile`` / ``description``).

See ``reports/local/benchmark_data_spec.md`` §7 for the judge contract.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# Which rubric fields each judge grades against. The fan-out judge only sees the
# search queries, so it gets the retrieval-side gold signals; the final-response
# judge sees the answer, so it gets the full synthesis-side rubric.
FANOUT_RUBRIC_FIELDS: List[str] = [
    "gold_retrieval_intent",
    "desired_fanout_keywords",
    "must_use",
    "positive_persona_signals",
    "distractor_signals_to_ignore",
]
RETRIEVAL_RUBRIC_FIELDS: List[str] = [
    "gold_retrieval_intent",
    "must_use",
    "positive_persona_signals",
    "distractor_signals_to_ignore",
    "safety_expectations",
    "risk_level",
]
FINAL_RUBRIC_FIELDS: List[str] = [
    "gold_synthesis_intent",
    "gold_retrieval_intent",
    "must_use",
    "should_not_use",
    "desired_synthesis_behavior",
    "positive_persona_signals",
    "distractor_signals_to_ignore",
    "safety_expectations",
    "risk_level",
]

_EMPTY = (None, "", [], {})


def load_rubrics(queries_path: str) -> Dict[str, Dict[str, Any]]:
    """Map ``query_id -> rubric dict`` (the query record's ``metadata``).

    Returns an empty dict if the file is missing so callers can degrade
    gracefully rather than crash. Accepts ``id`` / ``example_id`` as fallback
    keys for legacy query files.
    """
    rubrics: Dict[str, Dict[str, Any]] = {}
    if not queries_path or not os.path.exists(queries_path):
        return rubrics
    with open(queries_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("query_id") or rec.get("id") or rec.get("example_id")
            if not qid:
                continue
            md = rec.get("metadata")
            rubrics[qid] = md if isinstance(md, dict) else {}
    return rubrics


def _fmt_value(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "    (none)"
        return "\n".join(f"    - {item}" for item in value)
    return f"    {value}"


def format_rubric(rubric: Dict[str, Any], fields: List[str]) -> str:
    """Render the selected rubric fields into a prompt block.

    Only non-empty fields are shown. Returns a clear sentinel when no rubric is
    available so the judge is told explicitly rather than seeing a blank.
    """
    if not rubric:
        return "(no frozen rubric found for this query_id)"
    lines: List[str] = []
    for field in fields:
        value = rubric.get(field)
        if value in _EMPTY:
            continue
        lines.append(f"  {field}:")
        lines.append(_fmt_value(value))
    return "\n".join(lines) if lines else "(no frozen rubric found for this query_id)"


def format_latent_profile(persona: Dict[str, Any]) -> str:
    """Render the agent-invisible answer key (``description`` + ``latent_profile``).

    This is ONLY for the optional ``--include-latent-profile`` A/B mode that
    deliberately re-introduces the leak so we can measure how much it shifts
    scores. It is never included in the default (leak-free) judge prompt.
    """
    persona = persona or {}
    latent = (persona.get("attributes") or {}).get("latent_profile") or {}
    description = persona.get("description") or ""
    lines: List[str] = []
    if description:
        lines.append(f"  description: {description}")
    if latent:
        lines.append("  latent_profile:")
        for key, value in latent.items():
            lines.append(f"    - {key}: {value}")
    return "\n".join(lines) if lines else "  (none)"
