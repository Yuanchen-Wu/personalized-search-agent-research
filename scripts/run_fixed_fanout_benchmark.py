"""Benchmark runner for fixed_fanout_scaling_v1 experiment.

Usage:
    python scripts/run_fixed_fanout_benchmark.py --config configs/fixed_fanout_scaling_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import yaml
from typing import Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from search_agent.config import DEFAULT_GEMINI_MODEL, DEFAULT_SEARCH_DEPTH
from search_agent.evidence import (
    compute_context_character_count,
    deduplicate_search_results,
    filter_unique_documents,
    select_evidence_for_synthesis,
)
from search_agent.fixed_fanout import (
    PROMPT_VERSION_ORDERED_PLANNER,
    get_or_create_shared_plan,
    search_tavily_cached,
)
from search_agent.logging_utils import append_run_log, build_run_log
from search_agent.run_agent import load_personas
from scripts.run_benchmark import load_queries
from search_agent.schemas import (
    CostProxy,
    FanoutBranch,
    Persona,
    QueryRecord,
    SearchResult,
)
from search_agent.synthesize import synthesize_answer


def load_completed_run_keys(runs_path: str) -> Set[Tuple[str, str, str, int]]:
    """Return set of (query_id, persona_id, method, seed) keys already written to runs_path."""
    keys: Set[Tuple[str, str, str, int]] = set()
    if not os.path.exists(runs_path):
        return keys
    with open(runs_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                q_id = str(data.get("query_id", ""))
                p_id = str(data.get("persona_id", ""))
                method = str(data.get("method") or data.get("variant") or "")
                seed = int(data.get("seed", 42))
                keys.add((q_id, p_id, method, seed))
            except Exception:
                continue
    return keys


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run fixed fanout scaling benchmark.")
    parser.add_argument("--config", default="configs/fixed_fanout_scaling_v1.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query_ids", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--no_cache", action="store_true", default=False)
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_name = config.get("experiment_name", "fixed_fanout_scaling_v1")
    methods = args.methods or config.get("methods", ["fixed_k1", "fixed_k2", "fixed_k4", "fixed_k8"])
    seed = args.seed if args.seed is not None else config.get("reproducibility", {}).get("seed", 42)
    use_cache = not args.no_cache

    # Paths resolution
    data_cfg = config.get("data", {})
    q_path = data_cfg.get("queries_path", "data/queries/queries_v1.jsonl")
    p_path = data_cfg.get("personas_path", "data/personas/personas_v1.jsonl")
    if not os.path.isabs(q_path): q_path = os.path.join(_PROJECT_ROOT, q_path)
    if not os.path.isabs(p_path): p_path = os.path.join(_PROJECT_ROOT, p_path)

    out_cfg = config.get("outputs", {})
    runs_path = out_cfg.get("runs_path", "outputs/fixed_fanout_scaling_v1/runs.jsonl")
    plans_cache_path = out_cfg.get("fanout_plans_path", "outputs/fixed_fanout_scaling_v1/fanout_plans.jsonl")
    search_cache_path = out_cfg.get("search_cache_path", "outputs/fixed_fanout_scaling_v1/search_cache.jsonl")

    if not os.path.isabs(runs_path): runs_path = os.path.join(_PROJECT_ROOT, runs_path)
    if not os.path.isabs(plans_cache_path): plans_cache_path = os.path.join(_PROJECT_ROOT, plans_cache_path)
    if not os.path.isabs(search_cache_path): search_cache_path = os.path.join(_PROJECT_ROOT, search_cache_path)

    # Models & params
    planner_model = config.get("models", {}).get("planner") or DEFAULT_GEMINI_MODEL
    synthesis_model = config.get("models", {}).get("synthesizer") or DEFAULT_GEMINI_MODEL
    search_depth = config.get("search", {}).get("search_depth", DEFAULT_SEARCH_DEPTH)
    max_results_per_branch = config.get("search", {}).get("max_results_per_branch", 5)

    synth_cfg = config.get("synthesis", {})
    evidence_budget_mode = synth_cfg.get("evidence_budget_mode", "all")
    max_documents = synth_cfg.get("max_documents")
    max_context_chars = synth_cfg.get("max_context_chars")

    candidate_pool_size = config.get("fixed_fanout", {}).get("candidate_pool_size", 8)

    queries = load_queries(q_path)
    personas = load_personas(p_path)

    if args.query_ids:
        queries = [q for q in queries if q.query_id in args.query_ids]

    if args.limit is not None:
        queries = queries[:args.limit]

    # Build unique query/persona pairs
    qp_pairs: List[Tuple[QueryRecord, Optional[Persona]]] = []
    persona_map = personas
    for q in queries:
        target_pid = q.metadata.get("persona_id")
        if target_pid and target_pid in persona_map:
            qp_pairs.append((q, persona_map[target_pid]))
        else:
            for p in persona_map.values():
                qp_pairs.append((q, p))

    total_planned_runs = len(qp_pairs) * len(methods)

    if args.dry_run:
        print("=== DRY RUN MODE ===")
        print(f"Experiment Name: {exp_name}")
        print(f"Query/Persona Pairs: {len(qp_pairs)}")
        print(f"Methods to Run: {methods}")
        print(f"Total Planned Runs: {total_planned_runs}")
        print(f"Max Shared Planner Calls: {len(qp_pairs)}")
        print(f"Max Tavily Search Calls: {len(qp_pairs) * candidate_pool_size}")
        print(f"Max Synthesis Calls: {total_planned_runs}")
        print("Preview of first 3 pairs:")
        for idx, (q, p) in enumerate(qp_pairs[:3], start=1):
            pid = p.persona_id if p else "none"
            print(f"  [{idx}] Query ID: {q.query_id} | Persona ID: {pid}")
            print(f"      Methods: {methods}")
        print("=== DRY RUN COMPLETE ===")
        return

    # Handle overwrite mode
    if args.overwrite and os.path.exists(runs_path):
        os.remove(runs_path)

    completed_keys = load_completed_run_keys(runs_path) if args.resume else set()

    print(f"Starting benchmark: {len(qp_pairs)} pairs x {len(methods)} methods = {total_planned_runs} total runs.")
    if completed_keys:
        print(f"Found {len(completed_keys)} previously completed runs in {runs_path}. Resuming...")

    method_k_map = {
        "fixed_k1": 1,
        "fixed_k2": 2,
        "fixed_k4": 4,
        "fixed_k8": 8,
    }

    run_counter = 0
    start_time = time.time()

    for pair_idx, (q, persona) in enumerate(qp_pairs, start=1):
        pid = persona.persona_id if persona else "none"
        print(f"\n--- Processing Pair [{pair_idx}/{len(qp_pairs)}] Query ID: {q.query_id} | Persona ID: {pid} ---")

        # 1. Generate or load shared candidate plan (1 planner call per pair)
        t_plan_start = time.time()
        plan_id, full_plan, plan_events, plan_cache_hit = get_or_create_shared_plan(
            query_id=q.query_id,
            user_query=q.query,
            persona=persona,
            candidate_pool_size=candidate_pool_size,
            planner_model=planner_model,
            prompt_version=PROMPT_VERSION_ORDERED_PLANNER,
            seed=seed,
            cache_path=plans_cache_path,
            use_cache=use_cache,
        )
        t_planner_lat = time.time() - t_plan_start

        # 2. Execute searches for all 8 candidate branches once
        search_results_by_branch: Dict[int, List[SearchResult]] = {}
        pair_cache_hits = 0
        pair_cache_misses = 0
        t_search_start = time.time()

        for branch in full_plan:
            rank = branch.priority_rank or 1
            results, s_hit = search_tavily_cached(
                query=branch.query,
                branch_type=branch.branch_type,
                max_results=max_results_per_branch,
                search_depth=search_depth,
                cache_path=search_cache_path,
                use_cache=use_cache,
            )
            search_results_by_branch[rank] = results
            if s_hit:
                pair_cache_hits += 1
            else:
                pair_cache_misses += 1

        t_search_lat = time.time() - t_search_start

        # 3. Process each requested fixed-k method using exact nested prefixes
        for method in methods:
            if method not in method_k_map:
                print(f"Skipping unknown method: {method}")
                continue

            k = method_k_map[method]
            key = (q.query_id, pid, method, seed)
            if args.resume and key in completed_keys:
                print(f"  [Skipped - Resumed] method={method} for query_id={q.query_id} persona_id={pid}")
                continue

            # Slicing nested prefix of candidate plan: Q_1 \subset Q_2 \subset Q_4 \subset Q_8
            executed_branches = full_plan[:k]
            realized_fanout_count = len(executed_branches)

            if realized_fanout_count != k:
                print(f"  [ERROR] Realized fanout count ({realized_fanout_count}) does not match requested k ({k}). Failing run.")
                continue

            # Gather raw search results strictly from the executed prefix branches
            prefix_raw_results: List[SearchResult] = []
            for b in executed_branches:
                rank = b.priority_rank or 1
                prefix_raw_results.extend(search_results_by_branch.get(rank, []))

            # Evidence processing
            deduped_results = deduplicate_search_results(prefix_raw_results)
            unique_results = filter_unique_documents(prefix_raw_results)

            synthesis_evidence = select_evidence_for_synthesis(
                search_results=deduped_results,
                evidence_budget_mode=evidence_budget_mode,
                max_documents=max_documents,
                max_context_chars=max_context_chars,
            )

            # 4. Synthesize answer for this k
            t_synth_start = time.time()
            final_answer = synthesize_answer(
                user_query=q.query,
                persona=persona,
                search_results=synthesis_evidence,
                variant=method,
                model=synthesis_model,
                select_results=False,  # Pre-selected evidence passed
                seed=seed,
            )
            t_synth_lat = time.time() - t_synth_start

            total_run_latency = t_planner_lat + t_search_lat + t_synth_lat

            # 5. Build and write run log record
            cost_proxy = CostProxy(
                num_gemini_calls=1 + 1,  # 1 planner + 1 synth (amortized planner call)
                num_tavily_calls=k,
                num_fanout_branches=k,
                num_raw_results=len(prefix_raw_results),
            )

            run_log = build_run_log(
                variant=method,
                method=method,
                query_record=q,
                persona=persona,
                fanout_branches=executed_branches,
                raw_search_results=prefix_raw_results,
                final_answer=final_answer,
                cost_proxy=cost_proxy,
                experiment_name=exp_name,
                seed=seed,
                planner_model=planner_model,
                synthesis_model=synthesis_model,
                requested_fanout_count=k,
                realized_fanout_count=realized_fanout_count,
                full_candidate_plan_id=plan_id,
                executed_fanout_prefix=[b.as_dict() for b in executed_branches],
                branch_types_executed=[b.branch_type for b in executed_branches],
                information_needs_executed=[b.information_need for b in executed_branches],
                priority_ranks_executed=[b.priority_rank for b in executed_branches if b.priority_rank],
                deduplicated_search_results=[r.as_dict() for r in deduped_results],
                exact_synthesis_evidence=[r.as_dict() for r in synthesis_evidence],
                num_planner_calls=0 if plan_cache_hit else 1,
                num_synthesis_calls=1,
                num_tavily_calls=k,
                num_cache_hits=pair_cache_hits,
                num_cache_misses=pair_cache_misses,
                num_raw_results=len(prefix_raw_results),
                num_unique_results=len(unique_results),
                total_retrieved_context_size=compute_context_character_count(prefix_raw_results),
                total_synthesis_context_size=compute_context_character_count(synthesis_evidence),
                planner_latency=t_planner_lat,
                search_latency=t_search_lat,
                synthesis_latency=t_synth_lat,
                total_latency=total_run_latency,
                events=plan_events,
            )

            append_run_log(run_log, path=runs_path)
            run_counter += 1
            print(f"  [SUCCESS] Written run {run_log.run_id} | method={method} (k={k}) | latency={total_run_latency:.2f}s")

    elapsed = time.time() - start_time
    print(f"\nBenchmark completed {run_counter} runs in {elapsed:.2f}s. Results saved to {runs_path}")


if __name__ == "__main__":
    main()
