"""Paired tau-gain analysis for the refanout_k4 tau in {3,4,5} grid.

All three tau are derived from ONE t5 trajectory per pair, so they are perfectly
paired by query_id (72 pairs). This reports per-method means and paired diffs
(t3->t4, t4->t5, t3->t5) with bootstrap 95% CIs for the headline answer-quality,
retrieval, and faithfulness metrics -- the "does the re-fanout loop help" read.
(The shared summarize_fixed_fanout.py hardcodes its marginal-gains list to
fixed_k/adaptive, so this fills the refanout gap.)

Run: PYTHONPATH=src conda run -n eacl-search python scripts/analyze_tau_gains.py
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "outputs/adaptive_refanout_v1")
METHODS = ["refanout_k4_t3", "refanout_k4_t4", "refanout_k4_t5"]

# Higher-is-better headline metrics + (flagged) lower-is-better risk metrics.
HIGHER = [
    "final_intent_satisfaction", "final_specificity", "final_non_genericness",
    "final_personalization_target_use", "final_missing_constraint_awareness",
    "final_actionability_without_overclaiming", "final_uncertainty_calibration",
    "final_groundedness", "final_citation_support", "final_evidence_usage_quality",
    "retrieval_evidence_relevance", "retrieval_constraint_coverage",
    "retrieval_result_persona_fit", "retrieval_source_quality",
]
LOWER = ["final_overpersonalization", "final_unsupported_claim_risk",
         "retrieval_unsafe_or_overpersonalized_retrieval_risk"]
ALL = HIGHER + LOWER


def load(p):
    fp = os.path.join(D, p)
    return [json.loads(l) for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def main():
    runs = {r["run_id"]: r for r in load("runs_tau_grid.jsonl")}
    scores = {}
    for fname, prefix in [("final_response_scores.jsonl", "final_"),
                          ("retrieval_scores.jsonl", "retrieval_"),
                          ("fanout_scores.jsonl", "fanout_")]:
        for r in load(fname):
            d = scores.setdefault(r["run_id"], {})
            for k, v in r.get("scores", {}).items():
                if isinstance(v, (int, float)):
                    d[prefix + k] = float(v)

    by_qm = {(run["query_id"], run["method"]): rid for rid, run in runs.items()}
    qids = sorted({run["query_id"] for run in runs.values()})
    print(f"{len(qids)} pairs; scored run_ids: {len(scores)}")

    def col(method, metric):
        vals = [scores.get(by_qm.get((q, method), ""), {}).get(metric) for q in qids]
        return [v for v in vals if v is not None]

    print("\n=== PER-METHOD MEANS ===  (risk metrics ↓ marked *lower-better*)")
    print(f"{'metric':46s} {'t3':>7s} {'t4':>7s} {'t5':>7s}")
    for m in ALL:
        mm = [np.mean(col(meth, m)) if col(meth, m) else float('nan') for meth in METHODS]
        tag = " *↓" if m in LOWER else ""
        print(f"{m:46s} {mm[0]:7.3f} {mm[1]:7.3f} {mm[2]:7.3f}{tag}")

    def paired(metric, mhi, mlo):
        diffs = []
        for q in qids:
            hi = scores.get(by_qm.get((q, mhi), ""), {}).get(metric)
            lo = scores.get(by_qm.get((q, mlo), ""), {}).get(metric)
            if hi is not None and lo is not None:
                diffs.append(hi - lo)
        if not diffs:
            return None
        arr = np.array(diffs)
        rng = np.random.default_rng(42)
        boot = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(2000)]
        lo95, hi95 = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return dict(mean=float(arr.mean()), sem=sem, median=float(np.median(arr)),
                    lo=lo95, hi=hi95, n=len(arr),
                    imp=int((arr > 1e-6).sum()), unc=int((np.abs(arr) <= 1e-6).sum()),
                    wor=int((arr < -1e-6).sum()))

    for mhi, mlo, label in [("refanout_k4_t5", "refanout_k4_t3", "t3 -> t5  (loop OFF -> full persistence)"),
                            ("refanout_k4_t4", "refanout_k4_t3", "t3 -> t4"),
                            ("refanout_k4_t5", "refanout_k4_t4", "t4 -> t5")]:
        print(f"\n=== PAIRED {label} ===  mean Δ [95% CI]  (↑improved/↓worsened of n); * = CI excludes 0")
        for m in ALL:
            r = paired(m, mhi, mlo)
            if not r:
                continue
            sig = "*" if (r['lo'] > 0 or r['hi'] < 0) else " "
            note = " (↓lower-better: +Δ = WORSE)" if m in LOWER else ""
            print(f"{m:46s} {r['mean']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]{sig}  {r['imp']:2d}↑/{r['wor']:2d}↓ of {r['n']}{note}")

    # Write a marginal-gains CSV for the refanout tau comparisons (the shared
    # summarize_fixed_fanout.py hardcodes its list to fixed_k/adaptive, so this fills
    # the refanout gap and the C3 notebook reads it like C2 reads marginal_gains.csv).
    import csv
    out_csv = os.path.join(D, "tau_marginal_gains.csv")
    comparisons = [("refanout_k4_t4", "refanout_k4_t3", "t3_to_t4"),
                   ("refanout_k4_t5", "refanout_k4_t4", "t4_to_t5"),
                   ("refanout_k4_t5", "refanout_k4_t3", "t3_to_t5")]
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["comparison", "metric", "sample_count", "mean_paired_diff", "sem",
                    "median_paired_diff", "ci95_low", "ci95_high", "pct_improved",
                    "pct_unchanged", "pct_worsened", "higher_is_better"])
        for mhi, mlo, name in comparisons:
            for m in ALL:
                r = paired(m, mhi, mlo)
                if not r:
                    continue
                n = r["n"]
                w.writerow([name, m, n, f"{r['mean']:.4f}", f"{r['sem']:.4f}",
                            f"{r['median']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}",
                            f"{100*r['imp']/n:.2f}", f"{100*r['unc']/n:.2f}",
                            f"{100*r['wor']/n:.2f}", 0 if m in LOWER else 1])
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
