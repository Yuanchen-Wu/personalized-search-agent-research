"""Statistical summarization and paired marginal-gain analysis for fixed_fanout_scaling_v1.

Outputs:
  - summary_by_method.csv
  - summary_by_method_task_type.csv
  - summary_by_method_task_category.csv
  - summary_by_method_domain.csv
  - quality_cost_frontier.csv
  - marginal_gains.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import numpy as np
import yaml
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))


def compute_bootstrap_ci(
    diffs: List[float],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute deterministic bootstrap confidence interval for paired differences."""
    if not diffs:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = []
    arr = np.array(diffs, dtype=float)
    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = 1.0 - ci_level
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, sem, median, sample count for a list of numbers."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "median": 0.0, "count": 0}
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    sem = std / math.sqrt(n) if n > 0 else 0.0
    sorted_v = sorted(values)
    if n % 2 == 1:
        median = sorted_v[n // 2]
    else:
        median = (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2.0

    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "median": median,
        "count": float(n),
    }


def extract_domain(url: str) -> str:
    """Extract domain host from URL."""
    if not url:
        return ""
    url = url.lower()
    if "://" in url:
        url = url.split("://", 1)[1]
    url = url.split("/", 1)[0]
    return url.split(":", 1)[0]


_FIXED_K = {"fixed_k1": 1, "fixed_k2": 2, "fixed_k4": 4, "fixed_k8": 8}
_ADAPTIVE_RE = re.compile(r"^adaptive_[bk](\d+)")


def _requested_k(method: str) -> int:
    """Requested breadth: fixed_kK -> K; adaptive_[bk]N -> N (budget cap / fixed k); else 0."""
    if method in _FIXED_K:
        return _FIXED_K[method]
    m = _ADAPTIVE_RE.match(method or "")
    return int(m.group(1)) if m else 0


def _merge_runs(path, runs_by_id, runs_by_pair):
    """Load a runs.jsonl into runs_by_id / runs_by_pair (run_ids are globally unique)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rid = d["run_id"]
            runs_by_id[rid] = d
            q_id = d["query_id"]
            p_id = d.get("persona_id") or "none"
            method = d.get("method") or d.get("variant")
            runs_by_pair[(q_id, p_id)][method] = d


def _merge_scores(path, prefix, scores_by_run):
    """Load one score file, prefixing metric names (final_/retrieval_/fanout_)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                for k, v in d.get("scores", {}).items():
                    if isinstance(v, (int, float)):
                        scores_by_run[d["run_id"]][f"{prefix}{k}"] = float(v)


def main():
    parser = argparse.ArgumentParser(description="Summarize fixed fanout scaling experiment results.")
    parser.add_argument("--config", default="configs/fixed_fanout_scaling_v1.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_cfg = config.get("outputs", {})
    runs_path = out_cfg.get("runs_path", "outputs/fixed_fanout_scaling_v1/runs.jsonl")
    fanout_scores_path = out_cfg.get("fanout_scores_path", "outputs/fixed_fanout_scaling_v1/fanout_scores.jsonl")
    retrieval_scores_path = out_cfg.get("retrieval_scores_path", "outputs/fixed_fanout_scaling_v1/retrieval_scores.jsonl")
    final_scores_path = out_cfg.get("final_response_scores_path", "outputs/fixed_fanout_scaling_v1/final_response_scores.jsonl")

    if not os.path.isabs(runs_path): runs_path = os.path.join(_PROJECT_ROOT, runs_path)
    if not os.path.isabs(fanout_scores_path): fanout_scores_path = os.path.join(_PROJECT_ROOT, fanout_scores_path)
    if not os.path.isabs(retrieval_scores_path): retrieval_scores_path = os.path.join(_PROJECT_ROOT, retrieval_scores_path)
    if not os.path.isabs(final_scores_path): final_scores_path = os.path.join(_PROJECT_ROOT, final_scores_path)

    # Output CSV paths
    summary_method_path = out_cfg.get("summary_by_method_path", "outputs/fixed_fanout_scaling_v1/summary_by_method.csv")
    summary_tt_path = out_cfg.get("summary_by_method_task_type_path", "outputs/fixed_fanout_scaling_v1/summary_by_method_task_type.csv")
    summary_tc_path = out_cfg.get("summary_by_method_task_category_path", "outputs/fixed_fanout_scaling_v1/summary_by_method_task_category.csv")
    summary_domain_path = out_cfg.get("summary_by_method_domain_path", "outputs/fixed_fanout_scaling_v1/summary_by_method_domain.csv")
    frontier_path = out_cfg.get("quality_cost_frontier_path", "outputs/fixed_fanout_scaling_v1/quality_cost_frontier.csv")
    marginal_gains_path = out_cfg.get("marginal_gains_path", "outputs/fixed_fanout_scaling_v1/marginal_gains.csv")

    if not os.path.isabs(summary_method_path): summary_method_path = os.path.join(_PROJECT_ROOT, summary_method_path)
    if not os.path.isabs(summary_tt_path): summary_tt_path = os.path.join(_PROJECT_ROOT, summary_tt_path)
    if not os.path.isabs(summary_tc_path): summary_tc_path = os.path.join(_PROJECT_ROOT, summary_tc_path)
    if not os.path.isabs(summary_domain_path): summary_domain_path = os.path.join(_PROJECT_ROOT, summary_domain_path)
    if not os.path.isabs(frontier_path): frontier_path = os.path.join(_PROJECT_ROOT, frontier_path)
    if not os.path.isabs(marginal_gains_path): marginal_gains_path = os.path.join(_PROJECT_ROOT, marginal_gains_path)

    print("\n======================================================================")
    print(" [STAGE 3/3] SUMMARIZATION: Quality-Cost Frontier & Paired Marginal Gains")
    print("======================================================================")

    # 1+2. Load run records + scores. Merge this experiment's dir with any
    # `additional_result_dirs` so multiple method families (e.g. fixed_k* + adaptive_*)
    # land on ONE frontier without re-scoring already-scored runs. run_ids are globally
    # unique, so the merge is collision-free.
    runs_by_id: Dict[str, Dict[str, Any]] = {}
    runs_by_pair: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    scores_by_run: Dict[str, Dict[str, float]] = defaultdict(dict)

    _merge_runs(runs_path, runs_by_id, runs_by_pair)
    _merge_scores(final_scores_path, "final_", scores_by_run)
    _merge_scores(retrieval_scores_path, "retrieval_", scores_by_run)
    _merge_scores(fanout_scores_path, "fanout_", scores_by_run)

    for extra_dir in config.get("additional_result_dirs", []):
        if not os.path.isabs(extra_dir):
            extra_dir = os.path.join(_PROJECT_ROOT, extra_dir)
        _merge_runs(os.path.join(extra_dir, "runs.jsonl"), runs_by_id, runs_by_pair)
        _merge_scores(os.path.join(extra_dir, "final_response_scores.jsonl"), "final_", scores_by_run)
        _merge_scores(os.path.join(extra_dir, "retrieval_scores.jsonl"), "retrieval_", scores_by_run)
        _merge_scores(os.path.join(extra_dir, "fanout_scores.jsonl"), "fanout_", scores_by_run)
        print(f"  merged additional result dir: {extra_dir}")

    # Enrich run records with direct non-LLM metrics
    for rid, run in runs_by_id.items():
        raw_results = run.get("raw_search_results", [])
        urls = [r.get("url") for r in raw_results if r.get("url")]
        unique_urls = set(urls)
        domains = set(extract_domain(u) for u in unique_urls)
        # Bug D fix: is_duplicate_url is never set on raw_search_results (dedup writes
        # it onto deduplicated_search_results instead), so the old dup_count was always 0.
        # Compute the real cross-branch URL redundancy from unique/raw URL counts.
        n_urls = len(urls)
        dup_rate = 1.0 - (len(unique_urls) / n_urls) if n_urls else 0.0

        scores_by_run[rid]["realized_k"] = float(run.get("realized_fanout_count", len(run.get("fanout_branches", []))))
        scores_by_run[rid]["tavily_calls"] = float(run.get("num_tavily_calls", run.get("cost_proxy", {}).get("num_tavily_calls", 0)))
        scores_by_run[rid]["unique_urls"] = float(len(unique_urls))
        scores_by_run[rid]["unique_domains"] = float(len(domains))
        scores_by_run[rid]["duplicate_url_rate"] = dup_rate
        scores_by_run[rid]["total_retrieved_context_size"] = float(run.get("total_retrieved_context_size", 0))
        scores_by_run[rid]["total_synthesis_context_size"] = float(run.get("total_synthesis_context_size", 0))
        scores_by_run[rid]["planner_calls"] = float(run.get("num_planner_calls", 1))
        scores_by_run[rid]["synthesis_calls"] = float(run.get("num_synthesis_calls", 1))
        scores_by_run[rid]["assessor_calls"] = float(run.get("num_assessor_calls", 0))
        # Honest total-LLM-call cost axis (the assessor is a real cost for adaptive).
        scores_by_run[rid]["total_llm_calls"] = (
            float(run.get("num_planner_calls", 1))
            + float(run.get("num_synthesis_calls", 1))
            + float(run.get("num_assessor_calls", 0))
        )
        scores_by_run[rid]["assessor_latency"] = float(run.get("assessor_latency", 0.0))
        scores_by_run[rid]["search_latency"] = float(run.get("search_latency", 0.0))
        scores_by_run[rid]["synthesis_latency"] = float(run.get("synthesis_latency", 0.0))
        scores_by_run[rid]["total_latency"] = float(run.get("total_latency", 0.0))

    # 3. Groupings for standard summaries
    by_method = defaultdict(lambda: defaultdict(list))
    by_method_tt = defaultdict(lambda: defaultdict(list))
    by_method_tc = defaultdict(lambda: defaultdict(list))
    by_method_domain = defaultdict(lambda: defaultdict(list))

    for rid, run in runs_by_id.items():
        method = run.get("method") or run.get("variant")
        tt = run.get("task_type", "unknown")
        tc = run.get("task_category", "unknown")
        domain = run.get("macro_domain", "unknown")

        run_scores = scores_by_run.get(rid, {})
        for metric, val in run_scores.items():
            by_method[method][metric].append(val)
            by_method_tt[(method, tt)][metric].append(val)
            by_method_tc[(method, tc)][metric].append(val)
            by_method_domain[(method, domain)][metric].append(val)

    def write_summary_csv(path: str, group_dict: Dict[Any, Dict[str, List[float]]], key_names: List[str]):
        if not group_dict:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        all_metrics = sorted(list(set(m for v in group_dict.values() for m in v.keys())))
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = key_names + ["sample_count"] + [f"{m}_mean" for m in all_metrics] + [f"{m}_sem" for m in all_metrics]
            writer.writerow(headers)

            for keys, metrics_data in sorted(group_dict.items()):
                row_keys = list(keys) if isinstance(keys, tuple) else [keys]
                sample_count = max(len(v) for v in metrics_data.values()) if metrics_data else 0
                row = row_keys + [sample_count]
                
                # Means
                for m in all_metrics:
                    stats = compute_stats(metrics_data.get(m, []))
                    row.append(f"{stats['mean']:.4f}")
                # SEMs
                for m in all_metrics:
                    stats = compute_stats(metrics_data.get(m, []))
                    row.append(f"{stats['sem']:.4f}")

                writer.writerow(row)

    write_summary_csv(summary_method_path, by_method, ["method"])
    write_summary_csv(summary_tt_path, by_method_tt, ["method", "task_type"])
    write_summary_csv(summary_tc_path, by_method_tc, ["method", "task_category"])
    write_summary_csv(summary_domain_path, by_method_domain, ["method", "macro_domain"])

    # 4. Quality-Cost Frontier
    frontier_groups: Dict[Tuple[str, str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for rid, run in runs_by_id.items():
        method = run.get("method") or run.get("variant")
        tt = run.get("task_type", "unknown")
        tc = run.get("task_category", "unknown")
        domain = run.get("macro_domain", "unknown")
        run_scores = scores_by_run.get(rid, {})

        for grp in [("overall", "overall", method), (tt, "task_type", method), (tc, "task_category", method), (domain, "macro_domain", method)]:
            for metric, val in run_scores.items():
                frontier_groups[grp][metric].append(val)

    os.makedirs(os.path.dirname(frontier_path), exist_ok=True)
    with open(frontier_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = [
            "analysis_group",
            "group_type",
            "method",
            "sample_count",
            "requested_k",
            "realized_mean_k",
            "mean_tavily_calls",
            "mean_unique_urls",
            "mean_unique_domains",
            "duplicate_url_rate",
            "mean_synthesis_context_size",
            "mean_planner_calls",
            "mean_synthesis_calls",
            "mean_assessor_calls",
            "mean_total_llm_calls",
            "mean_search_latency",
            "mean_total_latency",
        ]
        # Append score metric headers
        score_metrics = sorted(list(set(m for v in scores_by_run.values() for m in v.keys() if m.startswith(("final_", "retrieval_", "fanout_")))))
        headers.extend([f"{m}_mean" for m in score_metrics])
        writer.writerow(headers)

        for (grp_name, grp_type, method), m_data in sorted(frontier_groups.items()):
            sc = max(len(v) for v in m_data.values()) if m_data else 0
            req_k = _requested_k(method)
            realized_k = compute_stats(m_data.get("realized_k", []))["mean"]
            tavily_calls = compute_stats(m_data.get("tavily_calls", []))["mean"]
            u_urls = compute_stats(m_data.get("unique_urls", []))["mean"]
            u_doms = compute_stats(m_data.get("unique_domains", []))["mean"]
            dup_rate = compute_stats(m_data.get("duplicate_url_rate", []))["mean"]
            synth_ctx = compute_stats(m_data.get("total_synthesis_context_size", []))["mean"]
            p_calls = compute_stats(m_data.get("planner_calls", []))["mean"]
            s_calls = compute_stats(m_data.get("synthesis_calls", []))["mean"]
            a_calls = compute_stats(m_data.get("assessor_calls", []))["mean"]
            tot_llm = compute_stats(m_data.get("total_llm_calls", []))["mean"]
            s_lat = compute_stats(m_data.get("search_latency", []))["mean"]
            t_lat = compute_stats(m_data.get("total_latency", []))["mean"]

            row = [
                grp_name,
                grp_type,
                method,
                sc,
                req_k,
                f"{realized_k:.2f}",
                f"{tavily_calls:.2f}",
                f"{u_urls:.2f}",
                f"{u_doms:.2f}",
                f"{dup_rate:.4f}",
                f"{synth_ctx:.1f}",
                f"{p_calls:.2f}",
                f"{s_calls:.2f}",
                f"{a_calls:.2f}",
                f"{tot_llm:.2f}",
                f"{s_lat:.2f}",
                f"{t_lat:.2f}",
            ]
            for sm in score_metrics:
                row.append(f"{compute_stats(m_data.get(sm, []))['mean']:.4f}")
            writer.writerow(row)

    # 5. Paired Marginal-Gain Analysis
    comparisons = [
        ("fixed_k2", "fixed_k1", "k1_to_k2"),
        ("fixed_k4", "fixed_k2", "k2_to_k4"),
        ("fixed_k8", "fixed_k4", "k4_to_k8"),
        ("fixed_k4", "fixed_k1", "k1_to_k4"),
        ("fixed_k8", "fixed_k1", "k1_to_k8"),
        # Mechanism A (fixed budget): adaptive_kN vs fixed_kN at the SAME breadth ->
        # identical retrieval cost, so the paired diff isolates iterative query selection.
        # This is the clean, headline comparison.
        ("adaptive_k4", "fixed_k4", "adaptiveK4_vs_k4"),
        # Mechanism B (variable budget): instance-paired (same query,persona), NOT
        # evidence-nested. "does adaptive match/beat fixed-k quality at lower per-query cost?"
        ("adaptive_b4", "fixed_k2", "adaptiveB4_vs_k2"),
        ("adaptive_b6", "fixed_k4", "adaptiveB6_vs_k4"),
        ("adaptive_b8", "fixed_k2", "adaptiveB8_vs_k2"),
        ("adaptive_b8", "fixed_k8", "adaptiveB8_vs_k8"),
        ("adaptive_b12", "fixed_k8", "adaptiveB12_vs_k8"),
    ]

    paired_diffs: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for (q_id, p_id), method_runs in runs_by_pair.items():
        for method_high, method_low, comp_name in comparisons:
            if method_high in method_runs and method_low in method_runs:
                rid_high = method_runs[method_high]["run_id"]
                rid_low = method_runs[method_low]["run_id"]

                scores_high = scores_by_run.get(rid_high, {})
                scores_low = scores_by_run.get(rid_low, {})

                all_keys = set(scores_high.keys()) | set(scores_low.keys())
                for metric in all_keys:
                    val_h = scores_high.get(metric, 0.0)
                    val_l = scores_low.get(metric, 0.0)
                    paired_diffs[comp_name][metric].append(val_h - val_l)

    os.makedirs(os.path.dirname(marginal_gains_path), exist_ok=True)
    with open(marginal_gains_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "comparison",
            "metric",
            "sample_count",
            "mean_paired_diff",
            "sem",
            "median_paired_diff",
            "ci95_low",
            "ci95_high",
            "pct_improved",
            "pct_unchanged",
            "pct_worsened",
        ])

        for comp_name in [c[2] for c in comparisons]:
            metrics_dict = paired_diffs.get(comp_name, {})
            for metric in sorted(metrics_dict.keys()):
                diffs = metrics_dict[metric]
                stats = compute_stats(diffs)
                ci_low, ci_high = compute_bootstrap_ci(diffs, seed=42)
                n = len(diffs)
                n_imp = sum(1 for d in diffs if d > 1e-6)
                n_unc = sum(1 for d in diffs if abs(d) <= 1e-6)
                n_wor = sum(1 for d in diffs if d < -1e-6)

                pct_imp = (n_imp / float(n)) * 100.0 if n > 0 else 0.0
                pct_unc = (n_unc / float(n)) * 100.0 if n > 0 else 0.0
                pct_wor = (n_wor / float(n)) * 100.0 if n > 0 else 0.0

                writer.writerow([
                    comp_name,
                    metric,
                    n,
                    f"{stats['mean']:.4f}",
                    f"{stats['sem']:.4f}",
                    f"{stats['median']:.4f}",
                    f"{ci_low:.4f}",
                    f"{ci_high:.4f}",
                    f"{pct_imp:.2f}",
                    f"{pct_unc:.2f}",
                    f"{pct_wor:.2f}",
                ])

    print(f"\n[STAGE 3/3 COMPLETE] Statistical summarization finished successfully! Written files:\n"
          f"  - {summary_method_path}\n"
          f"  - {summary_tt_path}\n"
          f"  - {summary_tc_path}\n"
          f"  - {summary_domain_path}\n"
          f"  - {frontier_path}\n"
          f"  - {marginal_gains_path}")


if __name__ == "__main__":
    main()
