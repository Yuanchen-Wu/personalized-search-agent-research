"""Pre-specified low-coverage cut (no selection bias).

Stratify the 72 pairs by ROUND-1 retrieval coverage -- a PREDICTOR judged before the
loop re-fans, not the realized outcome -- and measure the tau3->tau5 answer gain per
stratum (v2 = default synthesizer). Tests "does the loop help the answer more where the
first fan-out was weak?" without conditioning on realized coverage improvement.

Bands follow the in-loop judge's rubric:
  weak  (round-1 <= 3): generic / a materially relevant constraint missing -> most headroom
  mid   (3 < round-1 < 5): strong but imperfect
  saturated (round-1 = 5): first fan-out already excellent -> tau=5 approves round 1, so
                           tau3 and tau5 share evidence and the gain is ~0 by construction.
"""
import csv
import json
import os
from collections import Counter
import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "outputs/adaptive_refanout_v1")
METRICS = ["final_intent_satisfaction", "final_personalization_target_use",
           "retrieval_constraint_coverage", "retrieval_source_quality"]
BANDS = ["weak (r1<=3)", "mid (3<r1<5)", "saturated (r1=5)"]


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


def round1_coverage(run):
    ev = sorted([e for e in run.get("events", []) if e.get("event_type") == "refanout_round"],
                key=lambda e: e.get("round", 0))
    return ev[0].get("coverage_score") if ev else None


def band_of(c):
    if c is None:
        return None
    if c <= 3:
        return "weak (r1<=3)"
    if c >= 5:
        return "saturated (r1=5)"
    return "mid (3<r1<5)"


def main():
    t5 = [r for r in load("runs_tau_grid.jsonl") if r["method"] == "refanout_k4_t5"]
    band = {r["query_id"]: band_of(round1_coverage(r)) for r in t5}
    print("band sizes (by round-1 coverage):", dict(Counter(band.values())))

    v2_sc = scores_of("final_response_scores_v2synth.jsonl", "retrieval_scores_v2synth.jsonl", "fanout_scores_v2synth.jsonl")
    v2_id = {(int(r["method"].replace("_v2", "").split("_t")[-1]), r["query_id"]): r["run_id"]
             for r in load("runs_v2synth_grid.jsonl")}

    def col(tau, metric, qids):
        return np.array([v2_sc.get(v2_id.get((tau, q), ""), {}).get(metric, np.nan) for q in qids], float)

    def paired(a, b):
        d = b - a; d = d[~np.isnan(d)]
        if len(d) == 0:
            return 0.0, 0.0, 0.0, 0
        rng = np.random.default_rng(42)
        boot = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(2000)]
        return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(d)

    out = []
    for b in BANDS:
        qids = sorted([q for q, bb in band.items() if bb == b])
        for m in METRICS:
            mean, lo, hi, n = paired(col(3, m, qids), col(5, m, qids))
            out.append([b, len(qids), m, f"{mean:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    with open(os.path.join(D, "lowcov_cut.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["band", "n", "metric", "mean_paired_diff", "ci95_low", "ci95_high"])
        w.writerows(out)

    print("\ntau3 -> tau5 paired gain by round-1 coverage band (v2 synthesis):")
    print(f"{'band':20s}{'n':>4s}  {'intent':>22s}  {'personalization':>22s}  {'src_quality':>22s}")
    for b in BANDS:
        qids = sorted([q for q, bb in band.items() if bb == b])
        cells = []
        for m in ["final_intent_satisfaction", "final_personalization_target_use", "retrieval_source_quality"]:
            mean, lo, hi, n = paired(col(3, m, qids), col(5, m, qids))
            sig = "*" if (lo > 0 or hi < 0) else " "
            cells.append(f"{mean:+.2f}[{lo:+.2f},{hi:+.2f}]{sig}")
        print(f"{b:20s}{len(qids):>4d}  {cells[0]:>22s}  {cells[1]:>22s}  {cells[2]:>22s}")
    print("\nwrote lowcov_cut.csv")


if __name__ == "__main__":
    main()
