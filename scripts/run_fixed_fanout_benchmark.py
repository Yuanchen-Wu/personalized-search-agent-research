"""Benchmark runner for fixed_fanout_scaling_v1 experiment.

Usage:
    python scripts/run_fixed_fanout_benchmark.py --config configs/fixed_fanout_scaling_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import yaml
from typing import Any, Dict, List, Optional, Set, Tuple

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
from search_agent.adaptive_loop import run_adaptive_retrieval
from search_agent.adaptive_refanout import run_refanout_retrieval
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


_ADAPTIVE_METHOD_RE = re.compile(r"^adaptive_([bk])(\d+)(?:_([a-z]+))?$")


def parse_adaptive_method(method: str) -> Optional[Dict[str, Any]]:
    """Parse an adaptive method name.

    The prefix letter selects the mode:
      'b' = variable budget (early-stop; mechanism B, the cost-savings story).
      'k' = fixed budget (spend exactly k; mechanism A, cost-matched to fixed_k{k}).

    'adaptive_b8'        -> {'budget_cap': 8, 'strictness': None, 'fill_to_budget': False}
    'adaptive_b8_strict' -> {'budget_cap': 8, 'strictness': 'strict', 'fill_to_budget': False}
    'adaptive_k4'        -> {'budget_cap': 4, 'strictness': None, 'fill_to_budget': True}
    Returns None for non-adaptive (e.g. fixed_k*) methods.
    """
    m = _ADAPTIVE_METHOD_RE.match(method)
    if not m:
        return None
    return {
        "budget_cap": int(m.group(2)),
        "strictness": m.group(3),
        "fill_to_budget": m.group(1) == "k",
    }


_REFANOUT_METHOD_RE = re.compile(r"^refanout_k(\d+)(?:_t(\d+(?:p\d+)?))?$")


def parse_refanout_method(method: str) -> Optional[Dict[str, Any]]:
    """Parse a re-fanout method name.

    'refanout_k4'      -> {'fanout_size': 4, 'threshold': None}   (uses config default)
    'refanout_k4_t3'   -> {'fanout_size': 4, 'threshold': 3.0}    (approval threshold 3)
    'refanout_k4_t4p5' -> {'fanout_size': 4, 'threshold': 4.5}    ('p' = decimal point)
    Re-fanout (C3): each round is a full k-query fan-out; the loop re-fans-out with
    the judge's feedback until a round's coverage_score >= the approval threshold,
    then synthesizes from that approved round. The threshold suffix _tN is the knob
    swept to trace the frontier. Returns None for non-refanout methods.
    """
    m = _REFANOUT_METHOD_RE.match(method)
    if not m:
        return None
    threshold = float(m.group(2).replace("p", ".")) if m.group(2) else None
    return {"fanout_size": int(m.group(1)), "threshold": threshold}


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
    parser.add_argument("--all_persona_pairings", action="store_true", default=False, help="Cross-pair queries with all personas (up to 1387 pairs) instead of targeted 1-to-1 personas (72 pairs)")
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

    # Adaptive loop (C3) params
    assessor_model = config.get("models", {}).get("assessor") or DEFAULT_GEMINI_MODEL
    adaptive_cfg = config.get("adaptive", {})
    adaptive_seed_size = adaptive_cfg.get("seed_size", 2)
    adaptive_max_rounds = adaptive_cfg.get("max_rounds", 5)
    adaptive_per_round_cap = adaptive_cfg.get("per_round_cap", 4)
    # Fixed-budget (adaptive_kN) proposes ONE query per round by default so each new
    # search is conditioned on all prior evidence (max adaptivity); variable (bN) batches.
    adaptive_fixed_per_round_cap = adaptive_cfg.get("fixed_budget_per_round_cap", 1)
    default_strictness = adaptive_cfg.get("default_strictness", "balanced")

    # Re-fanout loop (C3) params. Judge falls back to the assessor model, then the default.
    judge_model = config.get("models", {}).get("judge") or assessor_model
    refanout_cfg = config.get("refanout", {})
    refanout_max_rounds = refanout_cfg.get("max_rounds", 3)
    refanout_default_threshold = refanout_cfg.get("approval_threshold", 4)
    refanout_judge_samples = refanout_cfg.get("judge_samples", 1)
    refanout_judge_temperature = refanout_cfg.get("judge_temperature", 0.2)

    queries = load_queries(q_path)
    personas = load_personas(p_path)

    if args.query_ids:
        queries = [q for q in queries if q.query_id in args.query_ids]

    # Build unique query/persona pairs
    qp_pairs: List[Tuple[QueryRecord, Optional[Persona]]] = []
    persona_map = personas
    for q in queries:
        target_pid = q.metadata.get("persona_id")
        if not args.all_persona_pairings and target_pid and target_pid in persona_map:
            qp_pairs.append((q, persona_map[target_pid]))
        else:
            for p in persona_map.values():
                qp_pairs.append((q, p))

    if args.limit is not None:
        qp_pairs = qp_pairs[:args.limit]

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

    print("\n======================================================================")
    print(" [STAGE 1/3] BENCHMARK EXECUTION: Generating Plans, Searching & Synthesizing")
    print("======================================================================")
    print(f"Total pairs to process: {len(qp_pairs)} | Methods per pair: {methods}")
    print(f"Total planned runs: {total_planned_runs} | Config file: {args.config}")
    if completed_keys:
        print(f"Found {len(completed_keys)} previously completed runs in {runs_path}. Resuming...")

    method_k_map = {
        "fixed_k1": 1,
        "fixed_k2": 2,
        "fixed_k4": 4,
        "fixed_k8": 8,
    }
    # Fixed methods share one pre-generated 8-branch plan + one all-8 search pass per
    # pair. Adaptive methods seed/search inside their own controller, so this pre-work
    # is skipped when the run set is purely adaptive.
    has_fixed = any(m in method_k_map for m in methods)

    run_counter = 0
    start_time = time.time()

    for pair_idx, (q, persona) in enumerate(qp_pairs, start=1):
        pid = persona.persona_id if persona else "none"
        print(f"\n--- Processing Pair [{pair_idx}/{len(qp_pairs)}] Query ID: {q.query_id} | Persona ID: {pid} ---")

        # 1+2. Shared ordered plan + one all-8 search pass -- only the fixed-k methods
        # (nested prefixes) need this. Per-branch search latency is recorded so each
        # fixed method can report the real latency of *its* prefix (metering fix B),
        # instead of the whole-8 latency being written to every k.
        plan_id: Optional[str] = None
        full_plan: List[FanoutBranch] = []
        plan_events: List[Dict[str, Any]] = []
        plan_cache_hit = False
        t_planner_lat = 0.0
        search_results_by_branch: Dict[int, List[SearchResult]] = {}
        search_lat_by_rank: Dict[int, float] = {}
        pair_cache_hits = 0
        pair_cache_misses = 0

        if has_fixed:
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

            for branch in full_plan:
                rank = branch.priority_rank or 1
                t_b = time.time()
                results, s_hit = search_tavily_cached(
                    query=branch.query,
                    branch_type=branch.branch_type,
                    max_results=max_results_per_branch,
                    search_depth=search_depth,
                    cache_path=search_cache_path,
                    use_cache=use_cache,
                )
                search_lat_by_rank[rank] = time.time() - t_b
                search_results_by_branch[rank] = results
                if s_hit:
                    pair_cache_hits += 1
                else:
                    pair_cache_misses += 1

        # 3. Process each requested method (fixed nested-prefix, or adaptive loop).
        for method in methods:
            adaptive_spec = parse_adaptive_method(method)
            refanout_spec = parse_refanout_method(method)
            if method not in method_k_map and adaptive_spec is None and refanout_spec is None:
                print(f"Skipping unknown method: {method}")
                continue

            key = (q.query_id, pid, method, seed)
            if args.resume and key in completed_keys:
                print(f"  [Skipped - Resumed] method={method} for query_id={q.query_id} persona_id={pid}")
                continue

            # --- Re-fanout loop (C3): fan-out -> search -> judge retrieval -> retry ---
            if refanout_spec is not None:
                threshold = (refanout_spec["threshold"]
                             if refanout_spec["threshold"] is not None
                             else refanout_default_threshold)
                result = run_refanout_retrieval(
                    user_query=q.query,
                    persona=persona,
                    query_id=q.query_id,
                    fanout_size=refanout_spec["fanout_size"],
                    max_rounds=refanout_max_rounds,
                    approval_threshold=threshold,
                    planner_model=planner_model,
                    judge_model=judge_model,
                    judge_samples=refanout_judge_samples,
                    judge_temperature=refanout_judge_temperature,
                    seed=seed,
                    search_depth=search_depth,
                    max_results_per_branch=max_results_per_branch,
                    search_cache_path=search_cache_path,
                    use_cache=use_cache,
                )
                rc = result.cost
                rf_branches = result.approved_branches
                rf_raw = result.approved_results
                rf_dedup = deduplicate_search_results(rf_raw)
                rf_unique = filter_unique_documents(rf_raw)
                rf_evidence = select_evidence_for_synthesis(
                    search_results=rf_dedup,
                    evidence_budget_mode=evidence_budget_mode,
                    max_documents=max_documents,
                    max_context_chars=max_context_chars,
                )
                t_synth_start = time.time()
                rf_answer = synthesize_answer(
                    user_query=q.query,
                    persona=persona,
                    search_results=rf_evidence,
                    variant=method,
                    model=synthesis_model,
                    select_results=False,
                    seed=seed,
                )
                rf_synth_lat = time.time() - t_synth_start
                rf_total_lat = (
                    rc.fanout_gen_latency + rc.judge_latency + rc.search_latency + rf_synth_lat
                )

                rf_cost_proxy = CostProxy(
                    num_gemini_calls=rc.num_fanout_gen_calls + rc.num_judge_calls + 1,
                    num_tavily_calls=rc.num_tavily_calls,
                    num_fanout_branches=rc.approved_fanout_size,
                    num_raw_results=len(rf_raw),
                )
                rf_run_log = build_run_log(
                    variant=method,
                    method=method,
                    query_record=q,
                    persona=persona,
                    fanout_branches=rf_branches,
                    raw_search_results=rf_raw,
                    final_answer=rf_answer,
                    cost_proxy=rf_cost_proxy,
                    experiment_name=exp_name,
                    seed=seed,
                    planner_model=planner_model,
                    synthesis_model=synthesis_model,
                    requested_fanout_count=refanout_spec["fanout_size"],
                    realized_fanout_count=rc.approved_fanout_size,
                    full_candidate_plan_id=None,
                    executed_fanout_prefix=[b.as_dict() for b in rf_branches],
                    branch_types_executed=[b.branch_type for b in rf_branches],
                    information_needs_executed=[b.information_need for b in rf_branches],
                    priority_ranks_executed=[b.priority_rank for b in rf_branches if b.priority_rank],
                    deduplicated_search_results=[r.as_dict() for r in rf_dedup],
                    exact_synthesis_evidence=[r.as_dict() for r in rf_evidence],
                    num_planner_calls=rc.num_fanout_gen_calls,
                    num_synthesis_calls=1,
                    num_assessor_calls=rc.num_judge_calls,
                    num_refanout_rounds=rc.num_rounds,
                    approved_round=rc.approved_round,
                    num_tavily_calls=rc.num_tavily_calls,
                    num_cache_hits=rc.num_cache_hits,
                    num_cache_misses=rc.num_cache_misses,
                    num_raw_results=len(rf_raw),
                    num_unique_results=len(rf_unique),
                    total_retrieved_context_size=compute_context_character_count(rf_raw),
                    total_synthesis_context_size=compute_context_character_count(rf_evidence),
                    planner_latency=rc.fanout_gen_latency,
                    search_latency=rc.search_latency,
                    synthesis_latency=rf_synth_lat,
                    assessor_latency=rc.judge_latency,
                    total_latency=rf_total_lat,
                    events=result.events,
                )
                append_run_log(rf_run_log, path=runs_path)
                run_counter += 1
                print(f"  [SUCCESS-REFANOUT] {rf_run_log.run_id} | method={method} "
                      f"| rounds={rc.num_rounds} approved_round={rc.approved_round} "
                      f"| score={rc.approved_score}>=tau{rc.approval_threshold} "
                      f"| fanout={rc.approved_fanout_size} total_searches={rc.num_tavily_calls} "
                      f"| stop={rc.stop_reason} | lat={rf_total_lat:.2f}s")
                continue

            # --- Adaptive loop (C3): retrieve -> assess -> continue/stop ---
            if adaptive_spec is not None:
                strictness = adaptive_spec["strictness"] or default_strictness
                fill_to_budget = adaptive_spec["fill_to_budget"]
                eff_per_round_cap = (
                    adaptive_fixed_per_round_cap if fill_to_budget else adaptive_per_round_cap
                )
                result = run_adaptive_retrieval(
                    user_query=q.query,
                    persona=persona,
                    query_id=q.query_id,
                    budget_cap=adaptive_spec["budget_cap"],
                    seed_size=adaptive_seed_size,
                    max_rounds=adaptive_max_rounds,
                    per_round_cap=eff_per_round_cap,
                    strictness=strictness,
                    planner_model=planner_model,
                    assessor_model=assessor_model,
                    seed=seed,
                    search_depth=search_depth,
                    max_results_per_branch=max_results_per_branch,
                    plans_cache_path=plans_cache_path,
                    search_cache_path=search_cache_path,
                    use_cache=use_cache,
                    fill_to_budget=fill_to_budget,
                )
                rc = result.cost
                a_branches = result.branches
                a_raw = result.raw_results
                a_dedup = deduplicate_search_results(a_raw)
                a_unique = filter_unique_documents(a_raw)
                a_evidence = select_evidence_for_synthesis(
                    search_results=a_dedup,
                    evidence_budget_mode=evidence_budget_mode,
                    max_documents=max_documents,
                    max_context_chars=max_context_chars,
                )
                t_synth_start = time.time()
                a_answer = synthesize_answer(
                    user_query=q.query,
                    persona=persona,
                    search_results=a_evidence,
                    variant=method,
                    model=synthesis_model,
                    select_results=False,
                    seed=seed,
                )
                a_synth_lat = time.time() - t_synth_start
                a_total_lat = rc.planner_latency + rc.assessor_latency + rc.search_latency + a_synth_lat

                a_cost_proxy = CostProxy(
                    num_gemini_calls=rc.num_planner_calls + rc.num_assessor_calls + 1,
                    num_tavily_calls=rc.num_tavily_calls,
                    num_fanout_branches=rc.realized_fanout_count,
                    num_raw_results=len(a_raw),
                )
                a_run_log = build_run_log(
                    variant=method,
                    method=method,
                    query_record=q,
                    persona=persona,
                    fanout_branches=a_branches,
                    raw_search_results=a_raw,
                    final_answer=a_answer,
                    cost_proxy=a_cost_proxy,
                    experiment_name=exp_name,
                    seed=seed,
                    planner_model=planner_model,
                    synthesis_model=synthesis_model,
                    requested_fanout_count=adaptive_spec["budget_cap"],
                    realized_fanout_count=rc.realized_fanout_count,
                    full_candidate_plan_id=None,
                    executed_fanout_prefix=[b.as_dict() for b in a_branches],
                    branch_types_executed=[b.branch_type for b in a_branches],
                    information_needs_executed=[b.information_need for b in a_branches],
                    priority_ranks_executed=[b.priority_rank for b in a_branches if b.priority_rank],
                    deduplicated_search_results=[r.as_dict() for r in a_dedup],
                    exact_synthesis_evidence=[r.as_dict() for r in a_evidence],
                    num_planner_calls=rc.num_planner_calls,
                    num_synthesis_calls=1,
                    num_assessor_calls=rc.num_assessor_calls,
                    num_tavily_calls=rc.num_tavily_calls,
                    num_cache_hits=rc.num_cache_hits,
                    num_cache_misses=rc.num_cache_misses,
                    num_raw_results=len(a_raw),
                    num_unique_results=len(a_unique),
                    total_retrieved_context_size=compute_context_character_count(a_raw),
                    total_synthesis_context_size=compute_context_character_count(a_evidence),
                    planner_latency=rc.planner_latency,
                    search_latency=rc.search_latency,
                    synthesis_latency=a_synth_lat,
                    assessor_latency=rc.assessor_latency,
                    total_latency=a_total_lat,
                    events=result.events,
                )
                append_run_log(a_run_log, path=runs_path)
                run_counter += 1
                print(f"  [SUCCESS-ADAPTIVE] {a_run_log.run_id} | method={method} "
                      f"| realized_k={rc.realized_fanout_count}/{adaptive_spec['budget_cap']} "
                      f"| rounds={rc.num_rounds} assessor={rc.num_assessor_calls} "
                      f"backfill={rc.num_backfilled} "
                      f"| stop={rc.stop_reason} | lat={a_total_lat:.2f}s")
                continue

            # --- Fixed nested-prefix method ---
            k = method_k_map[method]
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

            # Metering fix B: this method's search latency is the sum over ITS prefix
            # branches, not the whole-8 latency previously written to every k.
            t_search_lat = sum(
                search_lat_by_rank.get(b.priority_rank or 1, 0.0) for b in executed_branches
            )
            total_run_latency = t_planner_lat + t_search_lat + t_synth_lat

            # 5. Build and write run log record
            cost_proxy = CostProxy(
                # Metering fix C: real LLM-call count (planner is 0 on a plan-cache hit) + 1 synth.
                num_gemini_calls=(0 if plan_cache_hit else 1) + 1,
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
    print(f"\n[STAGE 1/3 COMPLETE] Benchmark finished {run_counter} runs in {elapsed/60.0:.2f} minutes. Runs written to {runs_path}")


if __name__ == "__main__":
    main()
