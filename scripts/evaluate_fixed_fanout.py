"""Evaluation script for fixed_fanout_scaling_v1.

Runs fanout, retrieval, and final answer evaluation across logged runs, safely reusing judge prompts and rubrics while guaranteeing unbiased retrieval evidence sampling for variable fanout.

Usage:
    python scripts/evaluate_fixed_fanout.py --config configs/fixed_fanout_scaling_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from search_agent.evidence import sample_retrieval_evidence_for_evaluator
from search_agent.llm_gemini import call_gemini
from search_agent.meta_prompt import (
    FANOUT_JUDGE_PROMPT_TEMPLATE,
    FINAL_RESPONSE_ANSWER_QUALITY_JUDGE_PROMPT_TEMPLATE,
    FINAL_RESPONSE_EVIDENCE_FAITHFULNESS_JUDGE_PROMPT_TEMPLATE,
    RETRIEVAL_JUDGE_PROMPT_TEMPLATE,
)
from search_agent.rubrics import (
    FANOUT_RUBRIC_FIELDS,
    FINAL_RUBRIC_FIELDS,
    RETRIEVAL_RUBRIC_FIELDS,
    format_latent_profile,
    format_rubric,
    load_rubrics,
)


def _clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def evaluate_fanout_for_run(run: Dict[str, Any], rubrics: Dict[str, Any], model: str) -> Dict[str, Any]:
    rubric = rubrics.get(run.get("query_id"), {})
    prompt = FANOUT_JUDGE_PROMPT_TEMPLATE.format(
        query=run.get("user_query", run.get("query", "")),
        task_type=run.get("task_type", "unknown"),
        task_category=run.get("task_category", "unknown"),
        macro_domain=run.get("macro_domain", "unknown"),
        search_required=run.get("search_required", True),
        expected_personalization_stage=run.get("expected_personalization_stage", "unknown"),
        persona_relevant_dimensions=run.get("persona_relevant_dimensions", []),
        rubric_block=format_rubric(rubric, FANOUT_RUBRIC_FIELDS),
        profile_block="",
        fanout_branches=json.dumps(run.get("fanout_branches", []), indent=2),
    )

    result = {
        "run_id": run.get("run_id", "unknown"),
        "experiment_name": run.get("experiment_name", "unknown"),
        "variant": run.get("variant", run.get("method", "unknown")),
        "method": run.get("method", run.get("variant", "unknown")),
        "persona_id": run.get("persona_id", "unknown"),
        "query_id": run.get("query_id", "unknown"),
        "task_type": run.get("task_type", "unknown"),
        "task_category": run.get("task_category", "unknown"),
        "scores": {},
        "rationale": {},
    }

    try:
        raw = call_gemini(prompt, model=model, temperature=0.1)
        res_json = json.loads(_clean_json_response(raw))
        result["scores"] = res_json.get("scores", {})
        result["rationale"] = res_json.get("rationale", {})
    except Exception as e:
        result["error"] = str(e)
    return result


def evaluate_retrieval_for_run(
    run: Dict[str, Any],
    rubrics: Dict[str, Any],
    model: str,
    sampling_mode: str = "top_m_per_branch",
    top_m_per_branch: int = 3,
) -> Dict[str, Any]:
    rubric = rubrics.get(run.get("query_id"), {})
    raw_results = run.get("raw_search_results", [])
    
    # Sample evidence cleanly so all branches up to k=8 are represented
    from search_agent.schemas import SearchResult
    results_objs = [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            score=r.get("score"),
            rank=r.get("rank", idx + 1),
            branch_type=r.get("branch_type", "generic"),
            branch_query=r.get("branch_query", ""),
        )
        for idx, r in enumerate(raw_results)
    ]
    sampled_objs = sample_retrieval_evidence_for_evaluator(
        results_objs, mode=sampling_mode, top_m_per_branch=top_m_per_branch
    )
    sampled_dicts = [s.as_dict() for s in sampled_objs]

    prompt = RETRIEVAL_JUDGE_PROMPT_TEMPLATE.format(
        query=run.get("user_query", run.get("query", "")),
        task_type=run.get("task_type", "unknown"),
        task_category=run.get("task_category", "unknown"),
        macro_domain=run.get("macro_domain", "unknown"),
        search_required=run.get("search_required", True),
        expected_personalization_stage=run.get("expected_personalization_stage", "unknown"),
        persona_relevant_dimensions=run.get("persona_relevant_dimensions", []),
        rubric_block=format_rubric(rubric, RETRIEVAL_RUBRIC_FIELDS),
        profile_block="",
        fanout_branches=json.dumps(run.get("fanout_branches", []), indent=2),
        search_results=json.dumps(sampled_dicts, indent=2),
    )

    result = {
        "run_id": run.get("run_id", "unknown"),
        "experiment_name": run.get("experiment_name", "unknown"),
        "variant": run.get("variant", run.get("method", "unknown")),
        "method": run.get("method", run.get("variant", "unknown")),
        "persona_id": run.get("persona_id", "unknown"),
        "query_id": run.get("query_id", "unknown"),
        "task_type": run.get("task_type", "unknown"),
        "task_category": run.get("task_category", "unknown"),
        "scores": {},
        "rationale": {},
    }

    try:
        raw = call_gemini(prompt, model=model, temperature=0.1)
        res_json = json.loads(_clean_json_response(raw))
        result["scores"] = res_json.get("scores", {})
        result["rationale"] = res_json.get("rationale", {})
    except Exception as e:
        result["error"] = str(e)
    return result


def evaluate_final_response_for_run(run: Dict[str, Any], rubrics: Dict[str, Any], model: str) -> Dict[str, Any]:
    rubric = rubrics.get(run.get("query_id"), {})
    
    # Quality prompt
    q_prompt = FINAL_RESPONSE_ANSWER_QUALITY_JUDGE_PROMPT_TEMPLATE.format(
        query=run.get("user_query", run.get("query", "")),
        task_type=run.get("task_type", "unknown"),
        task_category=run.get("task_category", "unknown"),
        macro_domain=run.get("macro_domain", "unknown"),
        rubric_block=format_rubric(rubric, FINAL_RUBRIC_FIELDS),
        profile_block="",
        final_answer=run.get("final_answer", ""),
    )

    # Faithfulness prompt
    synthesis_evidence = run.get("exact_synthesis_evidence") or run.get("raw_search_results", [])[:5]
    f_prompt = FINAL_RESPONSE_EVIDENCE_FAITHFULNESS_JUDGE_PROMPT_TEMPLATE.format(
        query=run.get("user_query", run.get("query", "")),
        search_results=json.dumps(synthesis_evidence, indent=2),
        final_answer=run.get("final_answer", ""),
    )

    result = {
        "run_id": run.get("run_id", "unknown"),
        "experiment_name": run.get("experiment_name", "unknown"),
        "variant": run.get("variant", run.get("method", "unknown")),
        "method": run.get("method", run.get("variant", "unknown")),
        "persona_id": run.get("persona_id", "unknown"),
        "query_id": run.get("query_id", "unknown"),
        "task_type": run.get("task_type", "unknown"),
        "task_category": run.get("task_category", "unknown"),
        "scores": {},
        "rationale": {},
    }

    try:
        q_raw = call_gemini(q_prompt, model=model, temperature=0.1)
        q_json = json.loads(_clean_json_response(q_raw))
        if "scores" in q_json:
            result["scores"].update(q_json["scores"])
            result["rationale"].update(q_json.get("rationale", {}))
    except Exception as e:
        result["error_quality"] = str(e)

    try:
        f_raw = call_gemini(f_prompt, model=model, temperature=0.1)
        f_json = json.loads(_clean_json_response(f_raw))
        if "scores" in f_json:
            result["scores"].update(f_json["scores"])
            result["rationale"].update(f_json.get("rationale", {}))
    except Exception as e:
        result["error_faithfulness"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed fanout scaling experiment runs.")
    parser.add_argument("--config", default="configs/fixed_fanout_scaling_v1.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_cfg = config.get("outputs", {})
    runs_path = out_cfg.get("runs_path", "outputs/fixed_fanout_scaling_v1/runs.jsonl")
    fanout_scores_path = out_cfg.get("fanout_scores_path", "outputs/fixed_fanout_scaling_v1/fanout_scores.jsonl")
    retrieval_scores_path = out_cfg.get("retrieval_scores_path", "outputs/fixed_fanout_scaling_v1/retrieval_scores.jsonl")
    final_scores_path = out_cfg.get("final_response_scores_path", "outputs/fixed_fanout_scaling_v1/final_response_scores.jsonl")
    queries_path = config.get("data", {}).get("queries_path")

    if not os.path.isabs(runs_path): runs_path = os.path.join(_PROJECT_ROOT, runs_path)
    if not os.path.isabs(fanout_scores_path): fanout_scores_path = os.path.join(_PROJECT_ROOT, fanout_scores_path)
    if not os.path.isabs(retrieval_scores_path): retrieval_scores_path = os.path.join(_PROJECT_ROOT, retrieval_scores_path)
    if not os.path.isabs(final_scores_path): final_scores_path = os.path.join(_PROJECT_ROOT, final_scores_path)
    if queries_path and not os.path.isabs(queries_path): queries_path = os.path.join(_PROJECT_ROOT, queries_path)

    model = args.model or config.get("models", {}).get("evaluator") or "gemini-flash-latest"

    if not os.path.exists(runs_path):
        print(f"Error: Runs file not found at {runs_path}")
        sys.exit(1)

    rubrics = load_rubrics(queries_path)
    runs = []
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))

    if args.limit:
        runs = runs[:args.limit]

    print("\n======================================================================")
    print(" [STAGE 2/3] EVALUATION: Fanout, Retrieval & Final Response Judging")
    print("======================================================================")
    print(f"Loaded {len(runs)} runs to evaluate using evaluator model '{model}'.")

    # Run evaluations with thread pool pacing
    evaluator_max_workers = 10
    pacing_delay = 0.2

    fanout_results = []
    retrieval_results = []
    final_results = []

    t_eval_start = time.time()

    print("\n--- Sub-stage 2a: Fanout Query Evaluation ---")
    with ThreadPoolExecutor(max_workers=evaluator_max_workers) as executor:
        futures = [executor.submit(evaluate_fanout_for_run, run, rubrics, model) for run in runs]
        for idx, f in enumerate(as_completed(futures), start=1):
            res = f.result()
            fanout_results.append(res)
            if idx % 10 == 0 or idx == len(runs):
                print(f"  [Fanout Query Eval Progress] {idx}/{len(runs)} complete ({idx/len(runs)*100:.1f}%)")
            time.sleep(pacing_delay)

    print("\n--- Sub-stage 2b: Retrieval Evidence Evaluation ---")
    with ThreadPoolExecutor(max_workers=evaluator_max_workers) as executor:
        futures = [executor.submit(evaluate_retrieval_for_run, run, rubrics, model) for run in runs]
        for idx, f in enumerate(as_completed(futures), start=1):
            res = f.result()
            retrieval_results.append(res)
            if idx % 20 == 0 or idx == len(runs):
                print(f"  [Retrieval Evidence Eval Progress] {idx}/{len(runs)} complete ({idx/len(runs)*100:.1f}%)")
            time.sleep(pacing_delay)

    print("\n--- Sub-stage 2c: Final Response Evaluation ---")
    with ThreadPoolExecutor(max_workers=evaluator_max_workers) as executor:
        futures = [executor.submit(evaluate_final_response_for_run, run, rubrics, model) for run in runs]
        for idx, f in enumerate(as_completed(futures), start=1):
            res = f.result()
            final_results.append(res)
            if idx % 20 == 0 or idx == len(runs):
                print(f"  [Final Response Eval Progress] {idx}/{len(runs)} complete ({idx/len(runs)*100:.1f}%)")
            time.sleep(pacing_delay)

    # Save output JSONL files
    os.makedirs(os.path.dirname(fanout_scores_path), exist_ok=True)
    with open(fanout_scores_path, "w", encoding="utf-8") as f:
        for r in fanout_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(retrieval_scores_path, "w", encoding="utf-8") as f:
        for r in retrieval_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(final_scores_path, "w", encoding="utf-8") as f:
        for r in final_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    t_eval_elapsed = time.time() - t_eval_start
    print(f"\n[STAGE 2/3 COMPLETE] Evaluation finished in {t_eval_elapsed/60.0:.2f} minutes. Scores saved:\n  - {fanout_scores_path}\n  - {retrieval_scores_path}\n  - {final_scores_path}")


if __name__ == "__main__":
    main()
