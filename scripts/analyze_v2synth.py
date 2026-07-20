"""Analysis for the v2-synth 'fixed synthesis' grid, with v2 as the DEFAULT final query.

Produces 3 CSVs for the rebuilt C3 notebook:
  synth_tau_frontier.csv    : {baseline, v2} x {tau 3,4,5} per-metric means (answer + retrieval).
  v2synth_tau_gains.csv     : paired tau diffs UNDER v2 synthesis (t3->t4, t4->t5, t3->t5) w/ 95% CI
                              -- "does re-fanout's better evidence now reach the good answer?"
  synth_improve_by_tau.csv  : paired (v2 - baseline) at each tau w/ 95% CI -- the final-query
                              improvement at each evidence level (the end comparison).

Run: PYTHONPATH=src conda run -n eacl-search python scripts/analyze_v2synth.py
"""
import csv
import os
import json
import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "outputs/adaptive_refanout_v1")

FINAL_HI = ["final_intent_satisfaction", "final_personalization_target_use", "final_specificity",
            "final_non_genericness", "final_missing_constraint_awareness",
            "final_actionability_without_overclaiming", "final_uncertainty_calibration",
            "final_domain_safety", "final_safety", "final_groundedness", "final_citation_support",
            "final_evidence_usage_quality"]
RET_HI = ["retrieval_evidence_relevance", "retrieval_constraint_coverage",
          "retrieval_result_persona_fit", "retrieval_source_quality"]
RISK_LO = ["final_overpersonalization", "final_unsupported_claim_risk"]
ALL = FINAL_HI + RET_HI + RISK_LO


def load(p):
    fp = os.path.join(D, p)
    return [json.loads(l) for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def scores_of(final, retr, fan):
    s = {}
    for fn, pre in [(final, "final_"), (retr, "retrieval_"), (fan, "fanout_")]:
        for r in load(fn):
            s.setdefault(r["run_id"], {}).update(
                {pre + k: float(v) for k, v in r.get("scores", {}).items() if isinstance(v, (int, float))})
    return s


def tau_of(method):
    return int(method.replace("_v2", "").split("_t")[-1])


def main():
    base_runs = [r for r in load("runs_tau_grid.jsonl") if r.get("method") in ("refanout_k4_t3", "refanout_k4_t4", "refanout_k4_t5")]
    v2_runs = load("runs_v2synth_grid.jsonl")
    base_sc = scores_of("final_response_scores.jsonl", "retrieval_scores.jsonl", "fanout_scores.jsonl")
    v2_sc = scores_of("final_response_scores_v2synth.jsonl", "retrieval_scores_v2synth.jsonl", "fanout_scores_v2synth.jsonl")

    base_id = {(tau_of(r["method"]), r["query_id"]): r["run_id"] for r in base_runs}
    v2_id = {(tau_of(r["method"]), r["query_id"]): r["run_id"] for r in v2_runs}
    qids = sorted({r["query_id"] for r in v2_runs})
    print(f"pairs={len(qids)} | baseline scored={len(base_sc)} | v2 scored={len(v2_sc)}")

    def col(sc, idmap, tau, metric):
        return np.array([sc.get(idmap.get((tau, q), ""), {}).get(metric, np.nan) for q in qids], float)

    def paired(a, b):
        d = b - a; d = d[~np.isnan(d)]
        rng = np.random.default_rng(42)
        boot = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(2000)]
        return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # 1. frontier: {baseline, v2} x {3,4,5}
    with open(os.path.join(D, "synth_tau_frontier.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["synth", "tau", "n"] + [m + "_mean" for m in ALL])
        for label, sc, idmap in [("baseline", base_sc, base_id), ("v2", v2_sc, v2_id)]:
            for tau in (3, 4, 5):
                row = [label, tau, len(qids)]
                for m in ALL:
                    v = col(sc, idmap, tau, m)
                    row.append(f"{np.nanmean(v):.4f}")
                w.writerow(row)

    # 2. v2-synth paired tau gains
    with open(os.path.join(D, "v2synth_tau_gains.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "metric", "higher_is_better", "mean_paired_diff", "ci95_low", "ci95_high"])
        for hi, lo, name in [(4, 3, "t3_to_t4"), (5, 4, "t4_to_t5"), (5, 3, "t3_to_t5")]:
            for m in ALL:
                mean, l, h = paired(col(v2_sc, v2_id, lo, m), col(v2_sc, v2_id, hi, m))
                w.writerow([name, m, 0 if m in RISK_LO else 1, f"{mean:.4f}", f"{l:.4f}", f"{h:.4f}"])

    # 3. old-vs-new final query improvement, per tau
    with open(os.path.join(D, "synth_improve_by_tau.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau", "metric", "higher_is_better", "base_mean", "v2_mean",
                    "mean_paired_diff", "ci95_low", "ci95_high"])
        for tau in (3, 4, 5):
            for m in ALL:
                b = col(base_sc, base_id, tau, m); v = col(v2_sc, v2_id, tau, m)
                mean, l, h = paired(b, v)
                w.writerow([tau, m, 0 if m in RISK_LO else 1,
                            f"{np.nanmean(b):.4f}", f"{np.nanmean(v):.4f}",
                            f"{mean:.4f}", f"{l:.4f}", f"{h:.4f}"])

    print("wrote synth_tau_frontier.csv, v2synth_tau_gains.csv, synth_improve_by_tau.csv")
    # quick console read: does the v2 ANSWER climb with tau?
    print("\nv2-synth answer intent by tau:", [round(float(np.nanmean(col(v2_sc, v2_id, t, "final_intent_satisfaction"))), 3) for t in (3, 4, 5)])
    print("v2-synth retrieval cons_cov by tau:", [round(float(np.nanmean(col(v2_sc, v2_id, t, "retrieval_constraint_coverage"))), 3) for t in (3, 4, 5)])


if __name__ == "__main__":
    main()
