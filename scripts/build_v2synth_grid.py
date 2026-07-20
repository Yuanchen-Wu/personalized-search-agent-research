"""Build the 'fixed synthesis' grid: hardened-v2 synthesis over the SAME per-tau evidence
for tau in {3,4,5}. Treats v2 (SYNTHESIS_PROMPT_HARDENED_V2) as the DEFAULT synthesizer and
re-asks the whole tau analysis with the good final query, so we can see whether re-fanout's
retrieval gains now reach the ANSWER (they didn't under the old conservative synthesis).

Evidence is held fixed per tau (reused from runs_tau_grid.jsonl); ONLY the synthesizer
changes. Methods: refanout_k4_t{3,4,5}_v2 -> runs_v2synth_grid.jsonl (216 records).
Isolated output; nothing existing is overwritten.

Run:
  GEMINI_MAX_RPM=120 PYTHONPATH=src conda run -n eacl-search --no-capture-output \
      python -u scripts/build_v2synth_grid.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from search_agent.schemas import Persona, SearchResult
from search_agent.synthesize import _format_evidence, _persona_block
from search_agent.llm_gemini import call_gemini
from search_agent.meta_prompt import SYNTHESIS_PROMPT_HARDENED_V2

SYN_MODEL = "gemini-3.5-flash"
SEED = 42
IN = os.path.join(_ROOT, "outputs/adaptive_refanout_v1/runs_tau_grid.jsonl")
OUT = os.path.join(_ROOT, "outputs/adaptive_refanout_v1/runs_v2synth_grid.jsonl")
EVIDENCE_METHODS = ["refanout_k4_t3", "refanout_k4_t4", "refanout_k4_t5"]


def _sr(d, i):
    return SearchResult(
        title=d.get("title", ""), url=d.get("url", ""), content=d.get("content", ""),
        score=d.get("score"), rank=d.get("rank", i + 1),
        branch_type=d.get("branch_type", "generic"), branch_query=d.get("branch_query", ""),
        is_duplicate_url=d.get("is_duplicate_url", False),
    )


def synth_v2(run):
    persona = Persona.from_dict(run["persona"]) if run.get("persona") else None
    evidence = [_sr(d, i) for i, d in enumerate(run.get("exact_synthesis_evidence", []))]
    prompt = SYNTHESIS_PROMPT_HARDENED_V2.format(
        user_query=run["user_query"],
        persona_block=_persona_block(persona),
        evidence_block=_format_evidence(evidence),
    )
    return call_gemini(prompt, model=SYN_MODEL, temperature=0.4, seed=SEED)


def build(run):
    n = run["method"].split("_t")[-1]           # "3" | "4" | "5"
    rec = copy.deepcopy(run)                     # inherit query/persona/evidence/rubric-link
    rec["run_id"] = run["run_id"] + "_v2"
    rec["method"] = rec["variant"] = f"refanout_k4_t{n}_v2"
    rec["final_answer"] = synth_v2(run)          # v2 synthesizer, same evidence
    rec["synthesis_prompt_version"] = "SYNTHESIS_PROMPT_HARDENED_V2"
    rec["events"] = [{"event_type": "v2synth_grid", "base_run_id": run["run_id"],
                      "evidence_tau": n, "synthesis_prompt": "hardened_v2"}]
    return rec


def main():
    rows = [json.loads(l) for l in open(IN) if l.strip()]
    grid = [r for r in rows if r.get("method") in EVIDENCE_METHODS]
    print(f"{len(grid)} records (v2 synthesis over tau in {{3,4,5}} evidence)")

    out = [None] * len(grid)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(build, r): i for i, r in enumerate(grid)}
        done = 0
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if done % 24 == 0 or done == len(grid):
                print(f"  {done}/{len(grid)}")

    with open(OUT, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(out)} -> {OUT} | by method:", dict(Counter(r['method'] for r in out)),
          "| empty:", sum(1 for r in out if not (r.get('final_answer') or '').strip()))


if __name__ == "__main__":
    main()
