"""Restrict the tau analysis to the ENGAGED subset: pairs whose tau=5 loop ran >1 round
(the first fan-out was NOT immediately approved). On these pairs the loop actually
re-fanned, so this isolates whether re-fanout's retrieval gain reaches the answer WHERE
it engages -- vs being diluted by the pairs that approve on round 1.

Emits (for the v2 = default synthesizer):
  engaged_frontier.csv     : {baseline,v2} x tau per-metric means, ENGAGED subset only
  engaged_v2_tau_gains.csv : v2 paired tau diffs (t3->t4,t4->t5,t3->t5) w/ 95% CI, ENGAGED
  engaged_full_compare.csv : v2 intent/retrieval-by-tau, FULL vs ENGAGED (for the overlay)
"""
import csv
import json
import os
from collections import Counter
import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "outputs/adaptive_refanout_v1")

FINAL_HI = ["final_intent_satisfaction", "final_personalization_target_use", "final_specificity",
            "final_non_genericness", "final_missing_constraint_awareness", "final_groundedness"]
RET_HI = ["retrieval_evidence_relevance", "retrieval_constraint_coverage",
          "retrieval_result_persona_fit", "retrieval_source_quality"]
ALL = FINAL_HI + RET_HI


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


def tau_of(m):
    return int(m.replace("_v2", "").split("_t")[-1])


def main():
    t5 = {r["query_id"]: r for r in load("runs_tau_grid.jsonl") if r["method"] == "refanout_k4_t5"}
    rounds = {q: (r.get("num_refanout_rounds") or 1) for q, r in t5.items()}
    engaged = sorted([q for q, n in rounds.items() if n > 1])
    full = sorted(t5)
    print(f"ENGAGED (t5 rounds>1): {len(engaged)}/{len(full)} pairs | t5 rounds dist: {dict(sorted(Counter(rounds.values()).items()))}")

    base_sc = scores_of("final_response_scores.jsonl", "retrieval_scores.jsonl", "fanout_scores.jsonl")
    v2_sc = scores_of("final_response_scores_v2synth.jsonl", "retrieval_scores_v2synth.jsonl", "fanout_scores_v2synth.jsonl")
    base_id = {(tau_of(r["method"]), r["query_id"]): r["run_id"]
               for r in load("runs_tau_grid.jsonl") if r["method"] in ("refanout_k4_t3", "refanout_k4_t4", "refanout_k4_t5")}
    v2_id = {(tau_of(r["method"]), r["query_id"]): r["run_id"] for r in load("runs_v2synth_grid.jsonl")}

    def col(sc, idmap, tau, metric, qids):
        return np.array([sc.get(idmap.get((tau, q), ""), {}).get(metric, np.nan) for q in qids], float)

    def paired(a, b):
        d = b - a; d = d[~np.isnan(d)]
        rng = np.random.default_rng(42)
        boot = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(2000)]
        return float(np.nanmean(b - a)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # 1. engaged frontier
    with open(os.path.join(D, "engaged_frontier.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["synth", "tau", "n"] + [m + "_mean" for m in ALL])
        for lab, sc, idm in [("baseline", base_sc, base_id), ("v2", v2_sc, v2_id)]:
            for tau in (3, 4, 5):
                w.writerow([lab, tau, len(engaged)] + [f"{np.nanmean(col(sc, idm, tau, m, engaged)):.4f}" for m in ALL])

    # 2. engaged v2 paired tau gains
    with open(os.path.join(D, "engaged_v2_tau_gains.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["comparison", "metric", "mean_paired_diff", "ci95_low", "ci95_high"])
        for hi, lo, name in [(4, 3, "t3_to_t4"), (5, 4, "t4_to_t5"), (5, 3, "t3_to_t5")]:
            for m in ALL:
                mean, l, h = paired(col(v2_sc, v2_id, lo, m, engaged), col(v2_sc, v2_id, hi, m, engaged))
                w.writerow([name, m, f"{mean:.4f}", f"{l:.4f}", f"{h:.4f}"])

    # 3. full vs engaged, v2 intent + retrieval by tau (for overlay)
    with open(os.path.join(D, "engaged_full_compare.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["subset", "tau", "n", "v2_intent", "v2_constraint_coverage", "v2_source_quality"])
        for lab, qids in [("full", full), ("engaged", engaged)]:
            for tau in (3, 4, 5):
                w.writerow([lab, tau, len(qids),
                            f"{np.nanmean(col(v2_sc, v2_id, tau, 'final_intent_satisfaction', qids)):.4f}",
                            f"{np.nanmean(col(v2_sc, v2_id, tau, 'retrieval_constraint_coverage', qids)):.4f}",
                            f"{np.nanmean(col(v2_sc, v2_id, tau, 'retrieval_source_quality', qids)):.4f}"])

    # console headline
    def by_tau(sc, idm, metric, qids): return [round(float(np.nanmean(col(sc, idm, t, metric, qids))), 3) for t in (3, 4, 5)]
    print("\nv2 intent by tau      FULL:", by_tau(v2_sc, v2_id, "final_intent_satisfaction", full),
          " ENGAGED:", by_tau(v2_sc, v2_id, "final_intent_satisfaction", engaged))
    print("v2 constraint_cov tau FULL:", by_tau(v2_sc, v2_id, "retrieval_constraint_coverage", full),
          " ENGAGED:", by_tau(v2_sc, v2_id, "retrieval_constraint_coverage", engaged))
    ig = paired(col(v2_sc, v2_id, 3, "final_intent_satisfaction", engaged), col(v2_sc, v2_id, 5, "final_intent_satisfaction", engaged))
    cg = paired(col(v2_sc, v2_id, 3, "retrieval_constraint_coverage", engaged), col(v2_sc, v2_id, 5, "retrieval_constraint_coverage", engaged))
    sg = paired(col(v2_sc, v2_id, 3, "retrieval_source_quality", engaged), col(v2_sc, v2_id, 5, "retrieval_source_quality", engaged))
    print(f"\nENGAGED paired t3->t5:  intent Δ={ig[0]:+.3f} [{ig[1]:+.3f},{ig[2]:+.3f}] | "
          f"constraint_cov Δ={cg[0]:+.3f} [{cg[1]:+.3f},{cg[2]:+.3f}] | source_qual Δ={sg[0]:+.3f} [{sg[1]:+.3f},{sg[2]:+.3f}]")
    print("wrote engaged_frontier.csv, engaged_v2_tau_gains.csv, engaged_full_compare.csv")


if __name__ == "__main__":
    main()
