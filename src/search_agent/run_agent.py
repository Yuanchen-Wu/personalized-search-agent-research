"""End-to-end agent orchestration and CLI entrypoint.

Pipeline:
    user query (+ optional persona)
      -> fan-out generation (variant-dependent)
      -> Tavily search per branch
      -> collect/normalize results
      -> final synthesis with Gemini (persona-dependent)
      -> structured JSONL log

Run a single query:
    python src/run_agent.py \
        --query "What laptop should I buy for ML research?" \
        --persona_id ml_phd_budget \
        --variant V4_mixed_fanout
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from .config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_RESULTS_PER_BRANCH,
    DEFAULT_RUNS_LOG,
    DEFAULT_SEARCH_DEPTH,
    PROJECT_ROOT,
)
from .fanout import generate_fanout_queries
from .logging_utils import append_run_log, build_run_log, new_run_id, utc_timestamp
from .schemas import CostProxy, FanoutBranch, Persona, RunLog, SearchResult, VARIANTS, QueryRecord
from .search_tavily import collect_search_results
from .synthesize import synthesize_answer

# Variants that pass persona context into final synthesis.
PERSONALIZED_SYNTHESIS_VARIANTS = {
    "V2_synthesis_only_personalization",
    "V3_personalized_fanout",
    "V4_mixed_fanout",
}

DEFAULT_PERSONAS_PATH = os.path.join(
    PROJECT_ROOT, "experiments", "sample_personas.jsonl"
)


def load_personas(path: str = DEFAULT_PERSONAS_PATH) -> Dict[str, Persona]:
    """Load personas from a JSONL file, keyed by persona_id."""
    personas: Dict[str, Persona] = {}
    if not os.path.exists(path):
        return personas
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            persona = Persona.from_dict(data)
            personas[persona.persona_id] = persona
    return personas


def run_agent(
    query_record: QueryRecord,
    persona: Optional[Persona],
    variant: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
    max_results_per_branch: int = DEFAULT_MAX_RESULTS_PER_BRANCH,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    experiment_name: str = "placement_ablation_v1",
) -> RunLog:
    """Run the full pipeline for one (query, persona, variant) combination.

    The persona is always available to fan-out generation only for the variants
    that call for it (handled inside ``generate_fanout_queries``). Persona is
    passed to synthesis only for PERSONALIZED_SYNTHESIS_VARIANTS.
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Choose from: {', '.join(VARIANTS)}"
        )

    run_id = new_run_id()
    timestamp = utc_timestamp()

    # 1) Fan-out generation.
    fanout_branches: List[FanoutBranch] = generate_fanout_queries(
        query_record.query, persona, variant, model=model
    )
    # Count Gemini calls used for fan-out: V0 makes none, others make one.
    num_fanout_gemini_calls = 0 if variant == "V0_generic_single" else 1

    # 2 + 3) Search each branch and collect normalized results.
    raw_results: List[SearchResult] = collect_search_results(
        fanout_branches,
        max_results_per_branch=max_results_per_branch,
        search_depth=search_depth,
    )

    # 4) Final synthesis. Persona passed only for personalized-synthesis variants.
    synth_persona = persona if variant in PERSONALIZED_SYNTHESIS_VARIANTS else None
    final_answer = synthesize_answer(
        query_record.query, synth_persona, raw_results, variant, model=model
    )

    # 5) Cost accounting (transparent proxy, not real billing).
    cost_proxy = CostProxy(
        num_gemini_calls=num_fanout_gemini_calls + 1,  # +1 for synthesis
        num_tavily_calls=len(fanout_branches),
        num_fanout_branches=len(fanout_branches),
        num_raw_results=len(raw_results),
    )

    return build_run_log(
        variant=variant,
        query_record=query_record,
        persona=persona,
        fanout_branches=fanout_branches,
        raw_search_results=raw_results,
        final_answer=final_answer,
        cost_proxy=cost_proxy,
        run_id=run_id,
        timestamp=timestamp,
        experiment_name=experiment_name,
    )


def _print_run(run_log: RunLog) -> None:
    """Pretty-print a run for interactive inspection."""
    sep = "=" * 70
    print(sep)
    print(f"RUN {run_log.run_id} | variant={run_log.variant}")
    print(f"query: {run_log.user_query}")
    print(f"persona_id: {run_log.persona_id}")
    print(sep)

    print("\nFAN-OUT BRANCHES")
    print("-" * 70)
    for i, b in enumerate(run_log.fanout_branches, start=1):
        print(f"  [{i}] ({b['branch_type']}) {b['query']}")
        if b.get("rationale"):
            print(f"      rationale: {b['rationale']}")
        if b.get("used_persona_fields"):
            print(f"      persona_fields: {b['used_persona_fields']}")

    print("\nTOP TAVILY RESULTS PER BRANCH")
    print("-" * 70)
    current_branch = None
    for r in run_log.raw_search_results:
        if r["branch_query"] != current_branch:
            current_branch = r["branch_query"]
            print(f"\n  branch ({r['branch_type']}): {current_branch}")
        dup = " [dup]" if r.get("is_duplicate_url") else ""
        score = r.get("score")
        score_str = f"{score:.3f}" if isinstance(score, float) else "n/a"
        print(f"    #{r['rank']} (score={score_str}){dup} {r['title']}")
        print(f"        {r['url']}")

    print("\nFINAL ANSWER")
    print("-" * 70)
    print(run_log.final_answer)

    print("\nCOST PROXY")
    print("-" * 70)
    print(f"  {run_log.cost_proxy}")
    print(sep)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the personalization search agent on one query."
    )
    parser.add_argument("--query", required=True, help="User query string.")
    parser.add_argument(
        "--persona_id",
        default=None,
        help="Persona id from experiments/sample_personas.jsonl (optional).",
    )
    parser.add_argument(
        "--variant",
        default="V4_mixed_fanout",
        choices=list(VARIANTS),
        help="Experimental variant to run.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model name.",
    )
    parser.add_argument(
        "--max_results_per_branch",
        type=int,
        default=DEFAULT_MAX_RESULTS_PER_BRANCH,
        help="Max Tavily results per branch.",
    )
    parser.add_argument(
        "--personas_path",
        default=DEFAULT_PERSONAS_PATH,
        help="Path to personas JSONL.",
    )
    parser.add_argument(
        "--log_path",
        default=DEFAULT_RUNS_LOG,
        help="Where to append the JSONL run log.",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Do not write the run to the JSONL log.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    persona: Optional[Persona] = None
    if args.persona_id:
        personas = load_personas(args.personas_path)
        persona = personas.get(args.persona_id)
        if persona is None:
            raise SystemExit(
                f"persona_id {args.persona_id!r} not found in "
                f"{args.personas_path}. Available: {list(personas)}"
            )

    run_log = run_agent(
        user_query=args.query,
        persona=persona,
        variant=args.variant,
        model=args.model,
        max_results_per_branch=args.max_results_per_branch,
    )

    _print_run(run_log)

    if not args.no_log:
        path = append_run_log(run_log, path=args.log_path)
        print(f"\n[saved] run appended to {path}")


if __name__ == "__main__":
    main()
