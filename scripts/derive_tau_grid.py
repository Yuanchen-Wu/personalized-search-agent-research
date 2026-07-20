"""Derive the tau in {3,4,5} answer grid from completed t5 (max_rounds=6) trajectories.

WHY THIS IS VALID (the "t5 is a superset" argument):
  The in-loop retrieval judge is tau-BLIND -- it scores coverage; a separate
  controller applies the approval threshold. The revised fan-out depends only on
  the judge's tau-blind gaps/feedback. So a run's per-round trajectory
  (fan-out -> evidence -> coverage) for rounds 1..N is INDEPENDENT of tau; tau only
  decides where the loop STOPS reading it. Because ``refanout_k4_t5`` runs the loop
  the longest (it rarely approves early) and every round's evidence is logged, a t5
  run CONTAINS the stop point of every lower tau:

      stop_round(tau) = first round whose mean coverage_score >= tau,
                        else best-round fallback (earliest max)   [matches the loop]

  We re-synthesize the final answer from that round's evidence, using the exact same
  dedup -> select_evidence_for_synthesis -> synthesize_answer path as the live runner,
  so the derived record is schema- and semantics-identical to a real ``refanout_k4_t{tau}``
  run. tau=5 reuses the existing run answer as-is; a lower tau also reuses it whenever
  its stop round == the t5 synthesis round (same evidence => identical answer, and we
  skip the redundant LLM call).

OUTPUT: outputs/adaptive_refanout_v1/runs_tau_grid.jsonl -- 72 pairs x 3 tau = 216
records, consumable by evaluate_fixed_fanout.py without modification.

Run in the eacl-search env with an RPM override, e.g.:
  GEMINI_MAX_RPM=120 PYTHONPATH=src conda run -n eacl-search --no-capture-output \
      python -u scripts/derive_tau_grid.py
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from search_agent.evidence import (
    compute_context_character_count,
    deduplicate_search_results,
    filter_unique_documents,
    select_evidence_for_synthesis,
)
from search_agent.schemas import FanoutBranch, Persona, SearchResult
from search_agent.synthesize import synthesize_answer

SYN_MODEL = "gemini-3.5-flash"   # pinned, matches the generation grid
SEED = 42
TAUS = [3.0, 4.0, 5.0]


def _method_name(tau: float) -> str:
    # tau=3 -> refanout_k4_t3 ; tau=4.5 -> refanout_k4_t4p5 (parser convention)
    s = f"{tau:g}".replace(".", "p")
    return f"refanout_k4_t{s}"


def _sr_from_dict(d: Dict[str, Any], idx: int) -> SearchResult:
    return SearchResult(
        title=d.get("title", ""),
        url=d.get("url", ""),
        content=d.get("content", ""),
        score=d.get("score"),
        rank=d.get("rank", idx + 1),
        branch_type=d.get("branch_type", "generic"),
        branch_query=d.get("branch_query", ""),
        is_duplicate_url=d.get("is_duplicate_url", False),
    )


def _round_events(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    ev = [e for e in run.get("events", []) if e.get("event_type") == "refanout_round"]
    return sorted(ev, key=lambda e: e.get("round", 0))


def _synthesis_round(run: Dict[str, Any]) -> int:
    """The round whose evidence produced this t5 run's existing final_answer."""
    if run.get("approved_round"):
        return run["approved_round"]
    for e in run.get("events", []):
        if e.get("event_type") == "refanout_fallback":
            return e.get("using_round") or 1
    return run.get("num_refanout_rounds", 1)  # defensive


def _stop_round(scores: List[float], tau: float) -> (int, bool):
    """(1-based stop round, approved?) — first round >= tau, else best-round (earliest max)."""
    for i, s in enumerate(scores, start=1):
        if s is not None and s >= tau:
            return i, True
    best = scores.index(max(scores)) + 1 if scores else 1  # earliest max == loop's strict-> fallback
    return best, False


def _reconstruct_branches(event: Dict[str, Any], raw: List[SearchResult]) -> List[FanoutBranch]:
    """Rebuild fan-out branches for a round from its logged queries + evidence branch_type.

    The per-round event logs only query strings; branch_type is recovered from the
    round's evidence (each SearchResult carries its branch_query + branch_type).
    information_need/rationale were not logged per-round -> left blank (used only by
    fan-out/retrieval judges, not the headline answer-quality judge).
    """
    type_by_q: Dict[str, str] = {}
    for r in raw:
        type_by_q.setdefault(r.branch_query, r.branch_type)
    branches: List[FanoutBranch] = []
    for i, q in enumerate(event.get("queries", []), start=1):
        branches.append(FanoutBranch(
            branch_type=type_by_q.get(q, "generic"),
            query=q,
            information_need="",
            priority_rank=i,
        ))
    return branches


def _derive_record(base: Dict[str, Any], tau: float, stop: int, approved: bool,
                   scores: List[float]) -> Dict[str, Any]:
    """Build one derived tau record from a t5 run, re-synthesizing from the stop round.

    Reuses base's answer/evidence (no LLM call) when the stop round IS the t5
    synthesis round -- same evidence => identical answer.
    """
    method = _method_name(tau)
    rec = copy.deepcopy(base)
    rec["run_id"] = f"{base['run_id']}_t{tau:g}".replace(".", "p")
    rec["method"] = method
    rec["variant"] = method

    if stop == _synthesis_round(base):
        # Identical round -> reuse base's synthesized answer + evidence verbatim.
        reused = True
    else:
        reused = False
        event = _round_events(base)[stop - 1]
        raw = [_sr_from_dict(d, i) for i, d in enumerate(event.get("results", []))]
        dedup = deduplicate_search_results(raw)
        unique = filter_unique_documents(raw)
        evidence = select_evidence_for_synthesis(dedup, evidence_budget_mode="all",
                                                 max_documents=None, max_context_chars=None)
        persona = Persona.from_dict(base["persona"]) if base.get("persona") else None
        answer = synthesize_answer(
            user_query=base["user_query"], persona=persona, search_results=evidence,
            variant=method, model=SYN_MODEL, select_results=False, seed=SEED,
        )
        branches = _reconstruct_branches(event, raw)
        rec["final_answer"] = answer
        rec["fanout_branches"] = [b.as_dict() for b in branches]
        rec["executed_fanout_prefix"] = [b.as_dict() for b in branches]
        rec["branch_types_executed"] = [b.branch_type for b in branches]
        rec["information_needs_executed"] = [b.information_need for b in branches]
        rec["priority_ranks_executed"] = [b.priority_rank for b in branches]
        rec["raw_search_results"] = [r.as_dict() for r in raw]
        rec["deduplicated_search_results"] = [r.as_dict() for r in dedup]
        rec["exact_synthesis_evidence"] = [r.as_dict() for r in evidence]
        rec["num_raw_results"] = len(raw)
        rec["num_unique_results"] = len(unique)
        rec["realized_fanout_count"] = len(branches)
        rec["total_retrieved_context_size"] = compute_context_character_count(raw)
        rec["total_synthesis_context_size"] = compute_context_character_count(evidence)

    rec["approved_round"] = stop if approved else None
    rec["num_refanout_rounds"] = stop
    rec["events"] = [{
        "event_type": "derived_from_t5",
        "tau": tau,
        "stop_round": stop,
        "approved": approved,
        "reused_t5_answer": reused,
        "coverage_score": scores[stop - 1] if scores else None,
        "trajectory_scores": scores,
        "source_run_id": base["run_id"],
    }]
    return rec, reused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_runs", default="outputs/adaptive_refanout_v1/runs_grid.jsonl")
    ap.add_argument("--out_runs", default="outputs/adaptive_refanout_v1/runs_tau_grid.jsonl")
    args = ap.parse_args()

    in_path = os.path.join(_ROOT, args.in_runs) if not os.path.isabs(args.in_runs) else args.in_runs
    out_path = os.path.join(_ROOT, args.out_runs) if not os.path.isabs(args.out_runs) else args.out_runs

    rows = [json.loads(l) for l in open(in_path) if l.strip()]
    t5 = [r for r in rows if r.get("method") == "refanout_k4_t5"]
    print(f"Loaded {len(t5)} t5 runs from {os.path.basename(in_path)}")

    out: List[Dict[str, Any]] = []
    n_synth = 0
    n_reused = 0
    for run in t5:
        scores = [e.get("coverage_score") for e in _round_events(run)]
        for tau in TAUS:
            if tau == 5.0:
                rec = copy.deepcopy(run)
                rec["method"] = rec["variant"] = "refanout_k4_t5"
                out.append(rec)
                continue
            stop, approved = _stop_round(scores, tau)
            rec, reused = _derive_record(run, tau, stop, approved, scores)
            n_reused += 1 if reused else 0
            n_synth += 0 if reused else 1
            out.append(rec)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"Wrote {len(out)} records -> {out_path}")
    print("  by method:", dict(Counter(r["method"] for r in out)))
    print(f"  re-synthesized: {n_synth}   reused-t5-answer: {n_reused}")

    # --- Validation vs the real t3 runs already in the input (q_1..q_6) ---
    real_t3 = {r["query_id"]: r for r in rows if r.get("method") == "refanout_k4_t3"}
    if real_t3:
        derived_t3 = {r["query_id"]: r for r in out if r.get("method") == "refanout_k4_t3"}
        print(f"\nVALIDATION vs {len(real_t3)} real t3 runs (derived t3 uses t5's round-1 evidence):")
        for qid, real in sorted(real_t3.items()):
            d = derived_t3.get(qid)
            if not d:
                continue
            from urllib.parse import urlparse
            rdoms = {n for r in real.get("exact_synthesis_evidence", []) if (n := urlparse(r.get("url", "")).netloc)}
            ddoms = {n for r in d.get("exact_synthesis_evidence", []) if (n := urlparse(r.get("url", "")).netloc)}
            overlap = len(rdoms & ddoms)
            print(f"  {qid}: real rounds={real.get('num_refanout_rounds')} derived stop={d.get('num_refanout_rounds')} "
                  f"| domain overlap {overlap}/{len(rdoms)} | ans_len real={len(real.get('final_answer',''))} der={len(d.get('final_answer',''))}")


if __name__ == "__main__":
    main()
