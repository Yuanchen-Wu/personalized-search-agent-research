"""Synthesis-ablation comparison CSV: baseline t5 synth vs hardened v1 vs hardened v2,
all on the SAME tau=5 evidence (paired by query_id). Feeds the C3 notebook.

Writes outputs/adaptive_refanout_v1/synth_ablation_summary.csv with per-metric means for
base / v1 / v2 and paired diffs (v2-base, v2-v1) with bootstrap 95% CIs.
"""
import csv
import json
import os
import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "outputs/adaptive_refanout_v1")


def load(p):
    fp = os.path.join(D, p)
    return [json.loads(l) for l in open(fp) if l.strip()] if os.path.exists(fp) else []


HIGHER = ["intent_satisfaction", "personalization_target_use", "specificity", "non_genericness",
          "missing_constraint_awareness", "actionability_without_overclaiming",
          "uncertainty_calibration", "domain_safety", "safety",
          "groundedness", "citation_support", "evidence_usage_quality"]
LOWER = ["overpersonalization", "unsupported_claim_risk", "contradiction_with_evidence"]


def main():
    base = {r["query_id"]: r["scores"] for r in load("final_response_scores.jsonl") if r["method"] == "refanout_k4_t5"}
    v1 = {r["query_id"]: r["scores"] for r in load("final_response_scores_synthhard.jsonl")}
    v2 = {r["query_id"]: r["scores"] for r in load("final_response_scores_synthhard_v2.jsonl")}
    q = sorted(set(base) & set(v1) & set(v2))

    def col(d, m):
        return np.array([d[k].get(m, np.nan) for k in q], float)

    def paired(a, b):
        dd = b - a
        dd = dd[~np.isnan(dd)]
        rng = np.random.default_rng(42)
        boot = [float(np.mean(rng.choice(dd, len(dd), replace=True))) for _ in range(2000)]
        return float(dd.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    out = os.path.join(D, "synth_ablation_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "higher_is_better", "n", "base_mean", "v1_mean", "v2_mean",
                    "d_v2_base", "d_v2_base_lo", "d_v2_base_hi",
                    "d_v2_v1", "d_v2_v1_lo", "d_v2_v1_hi",
                    "d_v1_base", "d_v1_base_lo", "d_v1_base_hi"])
        for m in HIGHER + LOWER:
            b, x, y = col(base, m), col(v1, m), col(v2, m)
            d2b = paired(b, y); d2v1 = paired(x, y); d1b = paired(b, x)
            w.writerow([m, 0 if m in LOWER else 1, len(q),
                        f"{np.nanmean(b):.4f}", f"{np.nanmean(x):.4f}", f"{np.nanmean(y):.4f}",
                        f"{d2b[0]:.4f}", f"{d2b[1]:.4f}", f"{d2b[2]:.4f}",
                        f"{d2v1[0]:.4f}", f"{d2v1[1]:.4f}", f"{d2v1[2]:.4f}",
                        f"{d1b[0]:.4f}", f"{d1b[1]:.4f}", f"{d1b[2]:.4f}"])
    print(f"wrote {out} ({len(q)} pairs, {len(HIGHER)+len(LOWER)} metrics)")


if __name__ == "__main__":
    main()
