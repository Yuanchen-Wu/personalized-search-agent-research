"""Smoke the leak-free draft-answer-critic loop on a few pairs; print the per-round trajectory.

Run:
  GEMINI_MAX_RPM=120 PYTHONPATH=src conda run -n eacl-search --no-capture-output \
      python -u scripts/run_answercritic_smoke.py --query_ids q_1 q_56
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from search_agent.schemas import Persona
from search_agent.adaptive_answercritic import run_answercritic_loop

D = os.path.join(_ROOT, "outputs/adaptive_refanout_v1")
CACHE = os.path.join(_ROOT, "outputs/fixed_fanout_scaling_v1/search_cache.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_ids", nargs="+", default=["q_1", "q_56"])
    ap.add_argument("--max_rounds", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=4.0)
    args = ap.parse_args()

    t5 = {r["query_id"]: r for r in (json.loads(l) for l in open(f"{D}/runs_tau_grid.jsonl") if l.strip())
          if r["method"] == "refanout_k4_t5"}

    for qid in args.query_ids:
        r = t5[qid]
        persona = Persona.from_dict(r["persona"]) if r.get("persona") else None
        print("=" * 84)
        print(f"{qid} | {r.get('task_type')} | {r.get('macro_domain')}")
        print("Q:", r["user_query"])
        res = run_answercritic_loop(
            user_query=r["user_query"], persona=persona, query_id=qid,
            max_rounds=args.max_rounds, approval_threshold=args.threshold,
            judge_samples=3, seed=42, search_cache_path=CACHE, use_cache=True)
        for e in res.events:
            if e["event_type"] == "answercritic_round":
                print(f"  round {e['round']}: draft_score={e['answer_score']} samples={e['sample_scores']} "
                      f"approved={e['approved']} needs_more_evidence={e['needs_more_evidence']} n_ev={e['num_results']}")
                print(f"     queries : {e['queries']}")
                print(f"     gaps    : {e['answer_gaps'][:3]}")
                if e['feedback']:
                    print(f"     feedback: {e['feedback'][:170]}")
        print(f"  -> stop={res.stop_reason} approved_round={res.approved_round} best_round={res.best_round} "
              f"final_score={res.approved_score}")
        print(f"     cost: tavily={res.num_tavily_calls} synth(drafts)={res.num_synthesis_calls} critic={res.num_critic_calls}")
        print(f"     FINAL ANSWER[:400]: {res.final_answer[:400]}")


if __name__ == "__main__":
    main()
