"""F5: the quality-cost frontier -- adaptive curve vs the fixed-k curve.

Reads the merged quality_cost_frontier.csv (fixed_k* + adaptive_*) plus
marginal_gains.csv, overlays the Phase-0 prefix-oracle point, and reports:
  - a text summary table (cost + headline quality per method),
  - the adaptive-vs-fixed paired deltas (with bootstrap CIs),
  - F5 figures (PDF+PNG): quality vs cost on TWO honest axes
      (mean_tavily_calls = realized retrieval breadth, and
       mean_total_llm_calls = planner+assessor+synthesis -- the axis that keeps
       the Pareto claim fair, since the assessor is a real LLM cost).

Usage:
    python scripts/plot_adaptive_frontier.py \
        --frontier outputs/adaptive_loop_v1/quality_cost_frontier.csv \
        --marginal outputs/adaptive_loop_v1/marginal_gains.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # to import the Phase-0 oracle helpers

# (frontier column, raw score-metric name for the oracle, human label)
PRIMARY = [
    ("retrieval_result_persona_fit_mean", "result_persona_fit", "Result persona-fit"),
    ("retrieval_constraint_coverage_mean", "constraint_coverage", "Constraint coverage"),
]
# Extra metrics shown in the text table (steep + saturated + C4 signal).
TABLE_METRICS = [
    ("retrieval_result_persona_fit_mean", "persona_fit"),
    ("retrieval_constraint_coverage_mean", "constr_cov"),
    ("final_groundedness_mean", "grounded"),
    ("retrieval_disconfirming_coverage_mean", "disconf_cov"),
    ("final_intent_satisfaction_mean", "intent"),
]
FIXED = ["fixed_k1", "fixed_k2", "fixed_k4", "fixed_k8"]
COST_AXES = [
    ("mean_tavily_calls", "Realized retrieval calls (mean k)"),
    ("mean_total_llm_calls", "Total LLM calls (planner+assessor+synth)"),
]


def load_overall(path):
    """method -> row dict, for the 'overall' analysis group only."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["method"]: r for r in csv.DictReader(f) if r["group_type"] == "overall"}


def adaptive_methods(fr):
    ms = [m for m in fr if m.startswith("adaptive_b")]
    return sorted(ms, key=lambda m: int(m.split("_b")[1].split("_")[0]))


def gv(row, col):
    try:
        return float(row.get(col, "") or "nan")
    except ValueError:
        return float("nan")


def print_table(fr, adaptive):
    hdr = f"{'method':14}{'realK':>7}{'totLLM':>8}" + "".join(f"{lbl:>12}" for _, lbl in TABLE_METRICS)
    print("\n" + "=" * len(hdr))
    print("QUALITY-COST SUMMARY (overall)")
    print("=" * len(hdr))
    print(hdr)
    for m in FIXED + adaptive:
        if m not in fr:
            continue
        r = fr[m]
        line = f"{m:14}{gv(r,'realized_mean_k'):>7.2f}{gv(r,'mean_total_llm_calls'):>8.2f}"
        line += "".join(f"{gv(r, col):>12.3f}" for col, _ in TABLE_METRICS)
        print(line)


def print_marginal(path):
    if not os.path.exists(path):
        print("\n(no marginal_gains.csv yet)")
        return
    key_metrics = {"retrieval_result_persona_fit", "retrieval_constraint_coverage",
                   "final_groundedness", "retrieval_disconfirming_coverage",
                   "final_intent_satisfaction"}
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
            if r["comparison"].startswith("adaptive") and r["metric"] in key_metrics]
    if not rows:
        print("\n(no adaptive-vs-fixed comparisons in marginal_gains.csv yet)")
        return
    print("\n" + "=" * 92)
    print("ADAPTIVE vs FIXED -- instance-paired deltas (mean [95% CI], % of pairs improved)")
    print("=" * 92)
    print(f"{'comparison':22}{'metric':34}{'Δmean':>9}{'ci95':>18}{'%improved':>10}")
    for r in sorted(rows, key=lambda x: (x["comparison"], x["metric"])):
        ci = f"[{float(r['ci95_low']):+.2f},{float(r['ci95_high']):+.2f}]"
        print(f"{r['comparison']:22}{r['metric']:34}{float(r['mean_paired_diff']):>+9.3f}"
              f"{ci:>18}{float(r['pct_improved']):>9.0f}%")


def oracle_points(c2_dir, eps):
    """metric -> (mean_kstar, oracle_quality) from the Phase-0 prefix oracle."""
    from oracle_adaptive_precheck import load_merged, build_pairs, analyze_metric
    rm, sc = load_merged(c2_dir)
    pairs, meta = build_pairs(rm, sc)
    out = {}
    for _, metric, _ in PRIMARY:
        r = analyze_metric(pairs, meta, metric, eps)
        if r:
            out[metric] = (r["mean_kstar"], r["oracle_quality"])
    return out


def make_figure(fr, adaptive, oracles, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"\n[warn] matplotlib unavailable ({e}); skipping figure.")
        return
    nrows, ncols = len(PRIMARY), len(COST_AXES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.0 * nrows), squeeze=False)
    for i, (mcol, mname, mlabel) in enumerate(PRIMARY):
        for j, (xcol, xlabel) in enumerate(COST_AXES):
            ax = axes[i][j]
            fx = [gv(fr[m], xcol) for m in FIXED if m in fr]
            fy = [gv(fr[m], mcol) for m in FIXED if m in fr]
            ax.plot(fx, fy, "-o", color="#444444", label="fixed-k", zorder=3)
            for m in FIXED:
                if m in fr:
                    ax.annotate(m.replace("fixed_", ""), (gv(fr[m], xcol), gv(fr[m], mcol)),
                                textcoords="offset points", xytext=(4, -10), fontsize=8, color="#444")
            ax_x = [gv(fr[m], xcol) for m in adaptive]
            ax_y = [gv(fr[m], mcol) for m in adaptive]
            ax.plot(ax_x, ax_y, "D", color="#1f77b4", label="adaptive", zorder=4, ms=8)
            for m in adaptive:
                ax.annotate(m.replace("adaptive_", ""), (gv(fr[m], xcol), gv(fr[m], mcol)),
                            textcoords="offset points", xytext=(4, 5), fontsize=8, color="#1f77b4")
            if xcol == "mean_tavily_calls" and mname in oracles:
                ox, oy = oracles[mname]
                ax.plot([ox], [oy], "*", color="#d62728", ms=18, label="prefix-oracle (LB)", zorder=5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(mlabel)
            ax.grid(True, alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("F5 — Adaptive retrieval vs fixed-k: quality–cost frontier", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"F5_adaptive_frontier.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", default="outputs/adaptive_loop_v1/quality_cost_frontier.csv")
    ap.add_argument("--marginal", default="outputs/adaptive_loop_v1/marginal_gains.csv")
    ap.add_argument("--c2-dir", default="outputs/fixed_fanout_scaling_v1")
    ap.add_argument("--out", default="outputs/adaptive_loop_v1/figures")
    ap.add_argument("--eps", type=float, default=0.25)
    args = ap.parse_args()

    fr = load_overall(args.frontier)
    adaptive = adaptive_methods(fr)
    print(f"methods on frontier: fixed={[m for m in FIXED if m in fr]} adaptive={adaptive}")
    print_table(fr, adaptive)
    print_marginal(args.marginal)
    try:
        oracles = oracle_points(args.c2_dir, args.eps)
    except Exception as e:
        print(f"[warn] oracle overlay unavailable ({e})")
        oracles = {}
    make_figure(fr, adaptive, oracles, args.out)


if __name__ == "__main__":
    main()
