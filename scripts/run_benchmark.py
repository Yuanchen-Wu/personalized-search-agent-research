"""Batch experiment runner for the personalization-placement ablation.

Reads sample queries and personas, then runs the variants and appends one
JSONL record per run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import yaml
from typing import Dict, List, Optional, Tuple

# Make the src/ modules importable when running this script directly.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from search_agent.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_RESULTS_PER_BRANCH,
    DEFAULT_RUNS_LOG,
)
from search_agent.logging_utils import append_run_log
from search_agent.run_agent import (
    PERSONALIZED_SYNTHESIS_VARIANTS,
    load_personas,
    run_agent,
)
from search_agent.schemas import Persona, VARIANTS, QueryRecord

# Variants that ignore the persona entirely (fan-out + synthesis both generic).
NON_PERSONALIZED_VARIANTS = [
    v for v in VARIANTS if v not in PERSONALIZED_SYNTHESIS_VARIANTS
]

def load_queries(path: str) -> List[QueryRecord]:
    """Load queries from a JSONL file into QueryRecord objects."""
    queries: List[QueryRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            queries.append(QueryRecord.from_dict(json.loads(line)))
    return queries

def build_plan(
    queries: List[QueryRecord],
    personas: Dict[str, Persona],
    variants_to_run: List[str],
) -> List[tuple[QueryRecord, Optional[Persona], str]]:
    """Build the list of (query, persona, variant) jobs to run."""
    persona_list = list(personas.values())
    plan: List[tuple[QueryRecord, Optional[Persona], str]] = []

    for q in queries:
        # For this ablation, we don't deduplicate V0/V1 if we want strict
        # output grouping by (query, persona, variant).
        # But keeping deduplication for V0/V1 is okay if we log persona=None.
        # Actually to keep it simple and fulfill the "every logged run must include persona_id"
        # we will run it once per persona even for V0/V1 so that metrics can easily join by persona.
        for persona in persona_list:
            for variant in variants_to_run:
                plan.append((q, persona, variant))
    return plan

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the ablation batch.")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--queries", default=None)
    parser.add_argument("--personas", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Resolve paths
    q_path = args.queries or config.get("data", {}).get("queries_path", os.path.join(_PROJECT_ROOT, "data", "queries", "queries_v1.jsonl"))
    p_path = args.personas or config.get("data", {}).get("personas_path", os.path.join(_PROJECT_ROOT, "data", "personas", "personas_v1.jsonl"))
    out_path = args.output or config.get("outputs", {}).get("runs_path", DEFAULT_RUNS_LOG)
    variants = args.variants or config.get("variants", list(VARIANTS))
    experiment_name = config.get("experiment_name", "placement_ablation_v1")

    # If relative path from config, prepend _PROJECT_ROOT
    if not os.path.isabs(q_path): q_path = os.path.join(_PROJECT_ROOT, q_path)
    if not os.path.isabs(p_path): p_path = os.path.join(_PROJECT_ROOT, p_path)
    if not os.path.isabs(out_path): out_path = os.path.join(_PROJECT_ROOT, out_path)

    queries = load_queries(q_path)
    personas = load_personas(p_path)
    plan = build_plan(queries, personas, variants)
    if args.limit is not None:
        plan = plan[: args.limit]

    total = len(plan)
    print(f"[run_batch] {total} runs planned.")

    if args.dry_run:
        print("[run_batch] Dry run mode. Printing plan preview:")
        for i, (q, p, v) in enumerate(plan[:5]):
            print(f"  {i+1}: variant={v} persona={p.persona_id} query_id={q.query_id} task_type={q.task_type}")
        if total > 5: print("  ...")
        return

    failures = 0
    for i, (q, persona, variant) in enumerate(plan, start=1):
        pid = persona.persona_id if persona else None
        print(f"[{i}/{total}] variant={variant} persona={pid} query_id={q.query_id}")
        try:
            run_log = run_agent(
                query_record=q,
                persona=persona,
                variant=variant,
                model=args.model,
                experiment_name=experiment_name,
            )
            append_run_log(run_log, path=out_path)
        except Exception as err:
            failures += 1
            print(f"    ERROR: {err}")

    print(f"[run_batch] done. {total - failures}/{total} succeeded. Logs appended to {out_path}")

if __name__ == "__main__":
    main()
