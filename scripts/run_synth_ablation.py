"""Synthesis-hardening ablation (C3): re-synthesize the tau=5 answers from the SAME
saved evidence using a hardened synthesis prompt (v1 or v2).

Retrieval is held FIXED -- the only change vs `refanout_k4_t5` is the synthesis prompt.
Isolates "is the answer synthesis-bound?": if answer quality rises with identical
evidence, the bottleneck was synthesis, not retrieval.

Writes a SEPARATE runs file per variant so existing data is untouched. Score each via its
own config (`configs/adaptive_refanout_synthhard[_v2].yaml`).

  v1: SYNTHESIS_PROMPT_HARDENED_V1 -> refanout_k4_t5_hard    -> runs_synthhard.jsonl
  v2: SYNTHESIS_PROMPT_HARDENED_V2 -> refanout_k4_t5_hard_v2 -> runs_synthhard_v2.jsonl

Run:
  GEMINI_MAX_RPM=120 PYTHONPATH=src conda run -n eacl-search --no-capture-output \
      python -u scripts/run_synth_ablation.py --variant v2
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from search_agent import meta_prompt as MP
from search_agent.schemas import Persona, SearchResult
from search_agent.synthesize import _format_evidence, _persona_block
from search_agent.llm_gemini import call_gemini

SYN_MODEL = "gemini-3.5-flash"   # same synthesizer model as the grid
SEED = 42
IN = os.path.join(_ROOT, "outputs/adaptive_refanout_v1/runs_tau_grid.jsonl")

VARIANTS = {
    "v1": dict(template="SYNTHESIS_PROMPT_HARDENED_V1", method="refanout_k4_t5_hard",    out="runs_synthhard.jsonl"),
    "v2": dict(template="SYNTHESIS_PROMPT_HARDENED_V2", method="refanout_k4_t5_hard_v2", out="runs_synthhard_v2.jsonl"),
}


def _sr(d, i):
    return SearchResult(
        title=d.get("title", ""), url=d.get("url", ""), content=d.get("content", ""),
        score=d.get("score"), rank=d.get("rank", i + 1),
        branch_type=d.get("branch_type", "generic"), branch_query=d.get("branch_query", ""),
        is_duplicate_url=d.get("is_duplicate_url", False),
    )


def synth_hard(run, template):
    persona = Persona.from_dict(run["persona"]) if run.get("persona") else None
    evidence = [_sr(d, i) for i, d in enumerate(run.get("exact_synthesis_evidence", []))]
    prompt = template.format(
        user_query=run["user_query"],
        persona_block=_persona_block(persona),
        evidence_block=_format_evidence(evidence),
    )
    return call_gemini(prompt, model=SYN_MODEL, temperature=0.4, seed=SEED)


def build(run, template, method, prompt_version):
    ans = synth_hard(run, template)
    rec = copy.deepcopy(run)                 # inherit query/persona/evidence/rubric-link fields
    rec["run_id"] = run["run_id"] + "_" + method.split("_t5_")[-1]
    rec["method"] = rec["variant"] = method
    rec["final_answer"] = ans                # <-- ONLY the answer changes (same evidence)
    rec["synthesis_prompt_version"] = prompt_version
    rec["events"] = [{"event_type": "synth_ablation", "base_run_id": run["run_id"],
                      "evidence_source": "refanout_k4_t5", "synthesis_prompt": prompt_version}]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="v1")
    args = ap.parse_args()
    v = VARIANTS[args.variant]
    template = getattr(MP, v["template"])
    method, out = v["method"], os.path.join(_ROOT, "outputs/adaptive_refanout_v1", v["out"])

    rows = [json.loads(l) for l in open(IN) if l.strip()]
    t5 = [r for r in rows if r.get("method") == "refanout_k4_t5"]
    print(f"{len(t5)} tau=5 records -> {v['template']} re-synthesis (evidence held fixed) -> {method}")

    result = [None] * len(t5)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(build, r, template, method, v["template"]): i for i, r in enumerate(t5)}
        done = 0
        for f in as_completed(futs):
            result[futs[f]] = f.result()
            done += 1
            if done % 12 == 0 or done == len(t5):
                print(f"  {done}/{len(t5)}")

    with open(out, "w", encoding="utf-8") as fh:
        for r in result:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    empty = sum(1 for r in result if not (r.get("final_answer") or "").strip())
    print(f"wrote {len(result)} -> {out} | empty_answers={empty}")


if __name__ == "__main__":
    main()
