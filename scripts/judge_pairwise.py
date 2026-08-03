"""Pairwise preference judging between two arms' final answers.

A sensitivity instrument, not a replacement for the absolute 1-5 judges: for
each shared item, the rubric-aware judge compares the two answers head-to-head,
in BOTH presentation orders (position-bias control), K votes per order. An item
counts as a win only by majority across all votes; answer-length deltas are
reported so verbosity confounds are visible.

Calibrate before trusting (known contrasts from the C1 flash arm):
  .venv/bin/python scripts/judge_pairwise.py \
      --runs-a outputs/placement_ablation_v2_gemini-3.6-flash/runs.jsonl --variant-a V1_generic_fanout \
      --runs-b outputs/placement_ablation_v2_gemini-3.6-flash/runs.jsonl --variant-b V4_personalized_fanout \
      --label-a V1 --label-b V4 --out outputs/pairwise/calibration_v1_vs_v4.jsonl
  # expect a lopsided B win-rate (~85%+); then V2 vs V3 (expect mild B lean).

Resume-safe: rows append+flush per item; already-judged query_ids are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from search_agent.llm_gemini import call_gemini
from search_agent.meta_prompt import PAIRWISE_FINAL_JUDGE_PROMPT_V1
from search_agent.rubrics import FINAL_RUBRIC_FIELDS, format_rubric, load_rubrics


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(_PROJECT_ROOT, p)


def load_answers(path, variant=None, method=None):
    """query_id -> (final_answer, meta) for one arm, optionally filtered."""
    out = {}
    for line in open(_abs(path), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if variant and r.get("variant") != variant:
            continue
        if method and r.get("method") != method:
            continue
        out[r["query_id"]] = (r.get("final_answer", ""), r)
    return out


def _clean(text):
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    return t[:-3].strip() if t.endswith("```") else t.strip()


def judge_item(qid, ans_a, ans_b, meta, rubric, model, samples, seed):
    votes = []  # normalized to "a" / "b" / "tie" in ARM terms (not position terms)
    for order in ("ab", "ba"):
        first, second = (ans_a, ans_b) if order == "ab" else (ans_b, ans_a)
        prompt = PAIRWISE_FINAL_JUDGE_PROMPT_V1.format(
            user_query=meta.get("user_query", ""),
            task_type=meta.get("task_type", "unknown"),
            task_category=meta.get("task_category", "unknown"),
            macro_domain=meta.get("macro_domain", "unknown"),
            rubric_block=format_rubric(rubric, FINAL_RUBRIC_FIELDS),
            answer_a=first,
            answer_b=second,
        )
        for s in range(samples):
            try:
                raw = call_gemini(prompt, model=model, temperature=0.2,
                                  seed=seed + 100 * s, throttle=False)
                w = json.loads(_clean(raw)).get("winner", "tie")
                w = w.strip().lower() if isinstance(w, str) else "tie"
            except Exception:
                w = "parse_error"
            if w not in ("a", "b", "tie"):
                votes.append("parse_error")
                continue
            if order == "ab":
                votes.append("a" if w == "a" else "b" if w == "b" else "tie")
            else:  # positions swapped: displayed A is arm B
                votes.append("b" if w == "a" else "a" if w == "b" else "tie")
    wa, wb = votes.count("a"), votes.count("b")
    majority = "a" if wa > wb else "b" if wb > wa else "tie"
    return {
        "query_id": qid, "votes": votes, "wins_a": wa, "wins_b": wb,
        "ties": votes.count("tie"), "parse_errors": votes.count("parse_error"),
        "majority": majority, "len_a": len(ans_a), "len_b": len(ans_b),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-a", required=True)
    ap.add_argument("--runs-b", required=True)
    ap.add_argument("--variant-a"); ap.add_argument("--variant-b")
    ap.add_argument("--method-a"); ap.add_argument("--method-b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--queries", default="data/v2/synthetic_queries_v2.jsonl",
                    help="rubric source (use the benchmark version the runs used)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--samples", type=int, default=2, help="votes per presentation order")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    A = load_answers(args.runs_a, args.variant_a, args.method_a)
    B = load_answers(args.runs_b, args.variant_b, args.method_b)
    rubrics = load_rubrics(_abs(args.queries))
    shared = sorted(set(A) & set(B))
    if args.limit:
        shared = shared[: args.limit]

    out_path = _abs(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            if line.strip():
                try:
                    done.add(json.loads(line)["query_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [q for q in shared if q not in done]
    print(f"[pairwise] {args.label_a} vs {args.label_b}: shared={len(shared)} "
          f"done={len(done)} to_judge={len(todo)} | {2 * args.samples} votes/item "
          f"| judge={args.model}")

    with open(out_path, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(judge_item, q, A[q][0], B[q][0], A[q][1],
                                rubrics.get(q, {}), args.model, args.samples,
                                args.seed): q for q in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  judged {i}/{len(todo)}")

    rows = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["query_id"] in set(shared)]
    na = sum(1 for r in rows if r["majority"] == "a")
    nb = sum(1 for r in rows if r["majority"] == "b")
    nt = len(rows) - na - nb
    dec = na + nb
    import random
    rng = random.Random(0)
    outcomes = [1] * nb + [0] * na
    if dec:
        boots = sorted(sum(rng.choices(outcomes, k=dec)) / dec for _ in range(4000))
        lo, hi = boots[100], boots[3899]
    else:
        lo = hi = float("nan")
    mean_len = lambda k: sum(r[k] for r in rows) / max(1, len(rows))
    print(f"\n[pairwise summary] items={len(rows)}  {args.label_a} wins={na}  "
          f"{args.label_b} wins={nb}  ties={nt}")
    print(f"  {args.label_b} win-rate among decided: "
          f"{(nb / dec if dec else float('nan')):.1%}  [95% CI {lo:.1%}, {hi:.1%}]"
          f"  {'(CI clears 50%)' if dec and (lo > 0.5 or hi < 0.5) else '(CI includes 50%)'}")
    print(f"  mean answer length: {args.label_a}={mean_len('len_a'):.0f} chars, "
          f"{args.label_b}={mean_len('len_b'):.0f} chars  (check for verbosity confound)")


if __name__ == "__main__":
    main()
