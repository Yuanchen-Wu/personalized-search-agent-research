"""Phase-0 oracle-adaptive pre-check (read-only analysis; no new runs).

Quantifies the HEADROOM for an adaptive (per-query) fan-out policy from the
already-scored fixed-k runs of fixed_fanout_scaling_v1.

For each (query_id, persona_id) pair we have fixed-k quality q(k) for k in {1,2,4,8}.
Define, per pair and per metric, the oracle stopping point
    k*_eps = min{ k : q(k) >= best_q - eps },  best_q = max_k q(k)   (higher-is-better metrics)
i.e. the smallest breadth that reaches within eps of that pair's own best.

If k*_eps is HETEROGENEOUS across pairs (some saturate at k=1, others need k=8),
then a per-query policy can beat every single fixed k on average -> C3 has headroom.
The prefix-oracle frontier point (mean k*, mean q(k*)) is a MOTIVATING LOWER BOUND
on what the real (leak-free, depth-capable) adaptive loop can achieve.

Usage:
    python scripts/oracle_adaptive_precheck.py
    python scripts/oracle_adaptive_precheck.py --results-dir outputs/fixed_fanout_scaling_v1 --eps 0.25
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from statistics import mean
from typing import Dict, List, Optional, Tuple

# Higher-is-better quality metrics we care about for the frontier. (Risk / lower-is-better
# metrics are intentionally excluded here -- the oracle "best = max" logic assumes higher=better.)
PRIMARY_METRICS = [
    "intent_satisfaction",       # headline answer-quality (may be saturated)
    "constraint_coverage",       # retrieval-side, steeper k-gradient
    "groundedness",              # faithfulness
    "citation_support",          # faithfulness
    "specificity",               # answer-quality
    "non_genericness",           # answer-quality
    "evidence_relevance",        # retrieval
    "result_persona_fit",        # retrieval (personalization)
    "disconfirming_coverage",    # retrieval (C4 signal)
    "missing_constraint_awareness",
]

FIXED_METHODS = ["fixed_k1", "fixed_k2", "fixed_k4", "fixed_k8"]
METHOD_K = {"fixed_k1": 1, "fixed_k2": 2, "fixed_k4": 4, "fixed_k8": 8}


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_merged(results_dir: str) -> Tuple[Dict[str, dict], Dict[str, Dict[str, float]]]:
    """Return run_meta[run_id] and merged scores[run_id][metric]."""
    runs = _read_jsonl(os.path.join(results_dir, "runs.jsonl"))
    run_meta: Dict[str, dict] = {}
    for r in runs:
        rid = r["run_id"]
        run_meta[rid] = {
            "query_id": r.get("query_id"),
            "persona_id": r.get("persona_id"),
            "method": r.get("method") or r.get("variant"),
            "realized_k": r.get("realized_fanout_count"),
            "task_type": r.get("task_type", "unknown"),
            "macro_domain": r.get("macro_domain", "unknown"),
        }

    scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    for fn in ("final_response_scores.jsonl", "retrieval_scores.jsonl", "fanout_scores.jsonl"):
        for row in _read_jsonl(os.path.join(results_dir, fn)):
            rid = row["run_id"]
            for metric, val in row.get("scores", {}).items():
                if isinstance(val, (int, float)):
                    scores[rid][metric] = float(val)
    return run_meta, scores


def build_pairs(run_meta, scores):
    """pair[(qid,pid)] = {method: {metric: val, ...}, _meta: {...}} for complete pairs only."""
    pair: Dict[Tuple[str, str], dict] = defaultdict(dict)
    meta_by_pair: Dict[Tuple[str, str], dict] = {}
    for rid, m in run_meta.items():
        key = (m["query_id"], m["persona_id"])
        method = m["method"]
        pair[key][method] = scores.get(rid, {})
        meta_by_pair[key] = {"task_type": m["task_type"], "macro_domain": m["macro_domain"]}
    complete = {k: v for k, v in pair.items() if all(fm in v for fm in FIXED_METHODS)}
    return complete, meta_by_pair


def q_at_k(pair_scores: dict, metric: str) -> Optional[List[Tuple[int, float]]]:
    """Return [(k, q(k)), ...] for the 4 fixed methods if metric present in all, else None."""
    pts = []
    for fm in FIXED_METHODS:
        s = pair_scores.get(fm, {})
        if metric not in s:
            return None
        pts.append((METHOD_K[fm], s[metric]))
    return pts


def k_star(pts: List[Tuple[int, float]], eps: float) -> Tuple[int, float, float]:
    """Smallest k within eps of best. Return (k*, q(k*), best_q)."""
    best_q = max(q for _, q in pts)
    for k, q in sorted(pts):  # ascending k
        if q >= best_q - eps:
            return k, q, best_q
    k, q = max(pts)  # fallback (unreachable)
    return k, q, best_q


def fixed_curve(pairs, metric) -> Dict[int, float]:
    """Mean q(k) across pairs for each fixed k."""
    acc = defaultdict(list)
    for _, ps in pairs.items():
        pts = q_at_k(ps, metric)
        if pts is None:
            continue
        for k, q in pts:
            acc[k].append(q)
    return {k: mean(v) for k, v in acc.items()}


def analyze_metric(pairs, meta_by_pair, metric, eps):
    kstars, kq, bestqs = [], [], []
    by_tt = defaultdict(list)
    by_dom = defaultdict(list)
    n = 0
    for key, ps in pairs.items():
        pts = q_at_k(ps, metric)
        if pts is None:
            continue
        ks, qks, bq = k_star(pts, eps)
        kstars.append(ks); kq.append(qks); bestqs.append(bq); n += 1
        by_tt[meta_by_pair[key]["task_type"]].append(ks)
        by_dom[meta_by_pair[key]["macro_domain"]].append(ks)
    if n == 0:
        return None
    curve = fixed_curve(pairs, metric)
    hist = Counter(kstars)
    return {
        "metric": metric, "n": n, "eps": eps,
        "curve": curve,
        "grad_k1_k8": curve.get(8, 0) - curve.get(1, 0),
        "hist": {k: hist.get(k, 0) for k in (1, 2, 4, 8)},
        "mean_kstar": mean(kstars),
        "oracle_quality": mean(kq),
        "mean_best": mean(bestqs),
        "frac_kstar_le2": sum(1 for k in kstars if k <= 2) / n,
        "frac_kstar_ge4": sum(1 for k in kstars if k >= 4) / n,
        "by_task_type": {t: (len(v), mean(v)) for t, v in sorted(by_tt.items())},
        "by_domain": {d: (len(v), mean(v)) for d, v in sorted(by_dom.items())},
    }


def fmt_curve(c: Dict[int, float]) -> str:
    return " -> ".join(f"k{k}:{c.get(k, float('nan')):.2f}" for k in (1, 2, 4, 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="outputs/fixed_fanout_scaling_v1")
    ap.add_argument("--eps", type=float, default=0.25, help="tolerance band on the 1-5 scale")
    args = ap.parse_args()

    run_meta, scores = load_merged(args.results_dir)
    pairs, meta_by_pair = build_pairs(run_meta, scores)
    print(f"Loaded {len(run_meta)} runs; {len(pairs)} complete (query,persona) pairs "
          f"(all 4 fixed-k present).  eps={args.eps}\n")

    results = []
    for metric in PRIMARY_METRICS:
        r = analyze_metric(pairs, meta_by_pair, metric, args.eps)
        if r:
            results.append(r)

    # Rank by k-gradient (headroom is only meaningful where quality actually moves with k).
    results.sort(key=lambda r: r["grad_k1_k8"], reverse=True)

    print("=" * 100)
    print(f"{'metric':<30}{'fixed curve (mean q)':<34}{'grad':>6}   k* hist [1/2/4/8]   mean_k*  oracleQ")
    print("=" * 100)
    for r in results:
        h = r["hist"]
        print(f"{r['metric']:<30}{fmt_curve(r['curve']):<34}{r['grad_k1_k8']:>+6.2f}   "
              f"{h[1]:>2}/{h[2]:>2}/{h[4]:>2}/{h[8]:>2}          {r['mean_kstar']:>5.2f}   {r['oracle_quality']:.2f}")

    print("\n" + "=" * 100)
    print("HEADROOM SUMMARY (per metric): oracle point vs fixed curve")
    print("=" * 100)
    for r in results:
        c = r["curve"]
        # savings vs k8 at ~equal quality; gain vs k2 at ~equal cost
        print(f"\n[{r['metric']}]  n={r['n']}  grad(k1->k8)={r['grad_k1_k8']:+.2f}")
        print(f"   fixed:  {fmt_curve(c)}")
        print(f"   oracle: cost(mean k*)={r['mean_kstar']:.2f}  quality={r['oracle_quality']:.2f}  "
              f"(mean per-pair best={r['mean_best']:.2f})")
        print(f"   heterogeneity: {r['frac_kstar_le2']*100:.0f}% of pairs saturate by k<=2, "
              f"{r['frac_kstar_ge4']*100:.0f}% still need k>=4")
        print(f"   k* by task_type: " + ", ".join(f"{t}={mk:.2f}(n{nn})" for t, (nn, mk) in r["by_task_type"].items()))
        print(f"   k* by domain:    " + ", ".join(f"{d}={mk:.2f}(n{nn})" for d, (nn, mk) in r["by_domain"].items()))

    # Verdict heuristic
    print("\n" + "=" * 100)
    print("GATE VERDICT")
    print("=" * 100)
    steep = [r for r in results if r["grad_k1_k8"] >= 0.30]
    heterogeneous = [r for r in steep if 0.15 <= r["frac_kstar_ge4"] <= 0.85]
    print(f"Metrics with a real k-gradient (>=0.30): {[r['metric'] for r in steep]}")
    print(f"...of those, with genuine k* heterogeneity (15-85% need k>=4): "
          f"{[r['metric'] for r in heterogeneous]}")
    if heterogeneous:
        best = max(heterogeneous, key=lambda r: r["grad_k1_k8"])
        print(f"\n=> HEADROOM EXISTS. Suggested primary metric for F5: '{best['metric']}' "
              f"(grad {best['grad_k1_k8']:+.2f}, mean k*={best['mean_kstar']:.2f} vs fixed k8).")
        print("   Proceed with the adaptive build.")
    else:
        print("\n=> LOW HEADROOM (quality saturates fast / homogeneous k*). "
              "Reframe C3 as equal-quality-at-lower-cost/latency before building.")


if __name__ == "__main__":
    main()
