"""Resumable, incrementally-saved runner for the leak-free draft-answer-critic loop.

Runs the loop at the max-persistence threshold (so answer-score thresholds 3/4/5 are
derivable post-hoc from the per-round drafts) over the shared 72 (query, persona) pairs.

Money-safe by construction:
  * Each pair's result is APPENDED to the output JSONL and flushed+fsync'd immediately,
    so an interruption never loses completed work.
  * On resume, pairs already present (for this seed) are SKIPPED -- so you can run in
    parts (`--limit`, `--query_ids`) and just re-invoke to continue.
  * A single pair that errors is logged and skipped (retried on the next resume), never
    crashing the whole run.
  * Output goes to an ISOLATED dir (outputs/answercritic_v1/) -- phase-1 data is untouched.

Usage:
  GEMINI_MAX_RPM=120 PYTHONPATH=src conda run -n eacl-search --no-capture-output \
      python -u scripts/run_answercritic_benchmark.py --config configs/answercritic_v1.yaml --limit 20
  # later, continue the rest (skips the first 20):
      python -u scripts/run_answercritic_benchmark.py --config configs/answercritic_v1.yaml
"""
import argparse
import datetime
import json
import os
import sys
import traceback
import uuid

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from search_agent.schemas import Persona
from search_agent.adaptive_answercritic import run_answercritic_loop


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


def load_pairs(cfg):
    src = _abs(cfg["data"]["pairs_source"])
    method = cfg["data"].get("pairs_method", "refanout_k4_t5")
    return [r for r in (json.loads(l) for l in open(src) if l.strip()) if r.get("method") == method]


def completed_query_ids(out_path, seed):
    """query_ids already saved for this seed (resume key)."""
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path) as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                if d.get("seed") == seed:
                    done.add(d["query_id"])
    return done


def build_record(src, res, cfg, seed, models):
    ac = cfg["answercritic"]
    return {
        "run_id": uuid.uuid4().hex[:12],
        "experiment_name": cfg["experiment_name"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "answercritic_t5", "variant": "answercritic_t5", "seed": seed,
        "query_id": src["query_id"], "user_query": src["user_query"],
        "task_type": src.get("task_type"), "task_category": src.get("task_category"),
        "macro_domain": src.get("macro_domain"), "search_required": src.get("search_required", True),
        "expected_personalization_stage": src.get("expected_personalization_stage"),
        "persona_relevant_dimensions": src.get("persona_relevant_dimensions"),
        "persona_id": src.get("persona_id"), "persona": src.get("persona"),
        "final_answer": res.final_answer,
        "approved_round": res.approved_round, "best_round": res.best_round,
        "approved_score": res.approved_score, "stop_reason": res.stop_reason,
        "num_refanout_rounds": res.num_rounds,
        "fanout_branches": [b.as_dict() for b in res.approved_branches],
        "exact_synthesis_evidence": [r.as_dict() for r in res.approved_evidence],
        "raw_search_results": [r.as_dict() for r in res.approved_evidence],
        "num_tavily_calls": res.num_tavily_calls,
        "num_synthesis_calls": res.num_synthesis_calls,
        "num_critic_calls": res.num_critic_calls,
        "approval_threshold": ac["approval_threshold"], "max_rounds": ac["max_rounds"],
        "judge_samples": ac["judge_samples"],
        "planner_model": models["planner"], "synthesis_model": models["synthesizer"],
        "critic_model": models["critic"],
        "events": res.events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/answercritic_v1.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--query_ids", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="start the output file fresh (danger)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(_abs(args.config)))
    ac, models, se = cfg["answercritic"], cfg["models"], cfg["search"]
    seed = args.seed if args.seed is not None else cfg["reproducibility"]["seed"]
    out_path = _abs(cfg["outputs"]["runs_path"])
    cache = _abs(cfg["outputs"]["search_cache_path"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pairs = load_pairs(cfg)
    if args.query_ids:
        pairs = [p for p in pairs if p["query_id"] in args.query_ids]
    done = set() if args.overwrite else completed_query_ids(out_path, seed)
    todo = [p for p in pairs if p["query_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"[answercritic] pairs={len(pairs)} already_done(seed={seed})={len(done)} "
          f"to_run={len(todo)} | threshold={ac['approval_threshold']} max_rounds={ac['max_rounds']} "
          f"K={ac['judge_samples']} -> {out_path}")
    if args.dry_run:
        print("DRY RUN (no API). first to run:", [p["query_id"] for p in todo[:8]])
        return

    if args.overwrite and os.path.exists(out_path):
        open(out_path, "w").close()

    ok = fail = 0
    for i, src in enumerate(todo, 1):
        qid = src["query_id"]
        try:
            persona = Persona.from_dict(src["persona"]) if src.get("persona") else None
            res = run_answercritic_loop(
                user_query=src["user_query"], persona=persona, query_id=qid,
                fanout_size=4, max_rounds=ac["max_rounds"], approval_threshold=ac["approval_threshold"],
                planner_model=models["planner"], synthesizer_model=models["synthesizer"],
                critic_model=models["critic"], judge_samples=ac["judge_samples"],
                judge_temperature=ac["judge_temperature"], seed=seed,
                search_depth=se["search_depth"], max_results_per_branch=se["max_results_per_branch"],
                search_cache_path=cache, use_cache=True)
            rec = build_record(src, res, cfg, seed, models)
            with open(out_path, "a", encoding="utf-8") as fh:      # <-- append + flush + fsync per pair
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            ok += 1
            print(f"  [{i}/{len(todo)}] {qid} SAVED  rounds={res.num_rounds} "
                  f"approved_round={res.approved_round} score={res.approved_score} stop={res.stop_reason}", flush=True)
        except Exception as e:
            fail += 1
            traceback.print_exc()
            print(f"  [{i}/{len(todo)}] {qid} FAILED: {type(e).__name__}: {str(e)[:200]} "
                  f"(skipped; retried on next resume)", flush=True)
            continue
    print(f"[answercritic] done this invocation: saved={ok} failed={fail} | total in file="
          f"{len(completed_query_ids(out_path, seed))}/{len(pairs)}")


if __name__ == "__main__":
    main()
