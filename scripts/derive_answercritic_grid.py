"""Derive the answer-critic answer-score-threshold grid {3,4,5} from the completed
threshold=5 runs (the "t=5 trick").

Because the answer-critic loop synthesizes a DRAFT every round, this needs NO re-synthesis:
for each threshold tau, pick the stop round (first round with answer_score >= tau, else the
best-scoring round) and read off that round's already-computed draft + evidence. Output is
schema-compatible with evaluate_fixed_fanout.py. Isolated dir; nothing else is touched.

Run: PYTHONPATH=src python scripts/derive_answercritic_grid.py
"""
import copy
import json
import os
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(_ROOT, "outputs/answercritic_v1")
IN = os.path.join(D, "runs_answercritic.jsonl")
OUT = os.path.join(D, "runs_answercritic_grid.jsonl")
TAUS = [3.0, 4.0, 5.0]


def stop_round(scores, tau):
    for i, s in enumerate(scores, 1):
        if s is not None and s >= tau:
            return i, True
    return (scores.index(max(scores)) + 1 if scores else 1), False   # best-round fallback (earliest max)


def main():
    rows = [json.loads(l) for l in open(IN) if l.strip()]
    out = []
    for r in rows:
        ev = sorted([e for e in r.get("events", []) if e.get("event_type") == "answercritic_round"],
                    key=lambda e: e.get("round", 0))
        if not ev:
            continue
        scores = [e.get("answer_score") for e in ev]
        for tau in TAUS:
            s, approved = stop_round(scores, tau)
            rev = ev[s - 1]
            rec = copy.deepcopy(r)
            rec["run_id"] = f"{r['run_id']}_t{int(tau)}"
            rec["method"] = rec["variant"] = f"answercritic_t{int(tau)}"
            rec["final_answer"] = rev.get("draft", "")
            rec["exact_synthesis_evidence"] = rev.get("evidence", [])
            rec["raw_search_results"] = rev.get("evidence", [])
            rec["fanout_branches"] = rev.get("branches", [])
            rec["approved_round"] = s if approved else None
            rec["num_refanout_rounds"] = s
            rec["events"] = [{
                "event_type": "derived_from_answercritic_t5", "tau": tau, "stop_round": s,
                "approved": approved, "answer_score": scores[s - 1],
                "trajectory_scores": scores, "source_run_id": r["run_id"],
            }]
            out.append(rec)

    with open(OUT, "w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"derived {len(out)} records from {len(rows)} runs -> {OUT}")
    print("  by method:", dict(Counter(r["method"] for r in out)))


if __name__ == "__main__":
    main()
