import argparse
import json
import os
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from search_agent.llm_gemini import call_gemini
from search_agent.meta_prompt import (
    FINAL_RESPONSE_ANSWER_QUALITY_JUDGE_PROMPT_TEMPLATE,
    FINAL_RESPONSE_EVIDENCE_FAITHFULNESS_JUDGE_PROMPT_TEMPLATE,
)
from search_agent.rubrics import (
    FINAL_RUBRIC_FIELDS,
    format_latent_profile,
    format_rubric,
    load_rubrics,
)


def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def evaluate_run(run, rubrics, model="gemini-flash-latest", include_latent_profile=False):
    rubric = rubrics.get(run.get("query_id"), {})

    # 1) Build quality prompt
    profile_block = ""
    if include_latent_profile:
        profile_block = (
            "\nPRIVILEGED USER PROFILE (A/B mode only — curated answer key; "
            "extra ground truth):\n"
            f"{format_latent_profile(run.get('persona') or {})}\n"
        )
    quality_prompt = FINAL_RESPONSE_ANSWER_QUALITY_JUDGE_PROMPT_TEMPLATE.format(
        query=run.get("user_query", run.get("query", "")),
        task_type=run.get("task_type", "unknown"),
        task_category=run.get("task_category", "unknown"),
        macro_domain=run.get("macro_domain", "unknown"),
        rubric_block=format_rubric(rubric, FINAL_RUBRIC_FIELDS),
        profile_block=profile_block,
        final_answer=run.get("final_answer", ""),
    )

    # 2) Build faithfulness prompt (does not see profile or rubric)
    faithfulness_prompt = FINAL_RESPONSE_EVIDENCE_FAITHFULNESS_JUDGE_PROMPT_TEMPLATE.format(
        query=run.get("user_query", run.get("query", "")),
        search_results=json.dumps(run.get("raw_search_results", [])[:3], indent=2),  # truncated to top 3
        final_answer=run.get("final_answer", ""),
    )

    result = {
        "run_id": run.get("run_id", "unknown"),
        "experiment_name": run.get("experiment_name", "unknown"),
        "variant": run.get("variant", "unknown"),
        "persona_id": run.get("persona_id", "unknown"),
        "query_id": run.get("query_id", "unknown"),
        "task_type": run.get("task_type", "unknown"),
        "task_category": run.get("task_category", "unknown"),
        "rubric_found": bool(rubric),
        "scores": {},
        "rationale": {},
    }

    # Call Gemini for Quality
    try:
        q_response = call_gemini(quality_prompt, model=model, temperature=0.1, throttle=False)
        q_json = json.loads(clean_json_response(q_response))
        if "scores" in q_json and "rationale" in q_json:
            result["scores"].update(q_json["scores"])
            result["rationale"].update(q_json["rationale"])
        else:
            raise ValueError("Invalid Quality JSON response structure")
    except Exception as e:
        result["error_quality"] = str(e)

    # Call Gemini for Faithfulness
    try:
        f_response = call_gemini(faithfulness_prompt, model=model, temperature=0.1, throttle=False)
        f_json = json.loads(clean_json_response(f_response))
        if "scores" in f_json and "rationale" in f_json:
            result["scores"].update(f_json["scores"])
            result["rationale"].update(f_json["rationale"])
        else:
            raise ValueError("Invalid Faithfulness JSON response structure")
    except Exception as e:
        result["error_faithfulness"] = str(e)

    # General error handling if either/both failed
    if "error_quality" in result or "error_faithfulness" in result:
        result["error"] = f"Quality error: {result.get('error_quality')}; Faithfulness error: {result.get('error_faithfulness')}"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default="gemini-flash-latest")
    parser.add_argument(
        "--include-latent-profile",
        action="store_true",
        help="A/B mode: also feed the curated persona answer key (latent_profile + "
        "description) to the judge. Default off = leak-free, rubric-only grading.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    runs_path = config.get("outputs", {}).get("runs_path")
    out_path = config.get("outputs", {}).get("final_response_scores_path")
    queries_path = config.get("data", {}).get("queries_path")

    if not os.path.isabs(runs_path): runs_path = os.path.join(_PROJECT_ROOT, runs_path)
    if not os.path.isabs(out_path): out_path = os.path.join(_PROJECT_ROOT, out_path)
    if queries_path and not os.path.isabs(queries_path):
        queries_path = os.path.join(_PROJECT_ROOT, queries_path)

    if not os.path.exists(runs_path):
        print(f"Error: Runs file not found at {runs_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rubrics = load_rubrics(queries_path)
    print(f"Loaded frozen rubrics for {len(rubrics)} queries from {queries_path}.")
    if args.include_latent_profile:
        print("[A/B MODE] include_latent_profile=True — re-introducing the persona answer key.")

    runs = []
    with open(runs_path, "r") as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))

    if args.limit:
        runs = runs[:args.limit]

    missing = sum(1 for r in runs if r.get("query_id") not in rubrics)
    if missing:
        print(f"[WARNING] {missing}/{len(runs)} runs had no matching rubric (graded with empty rubric).")

    evaluator_rpm = 250
    evaluator_max_workers = 15
    pacing_delay = 60.0 / evaluator_rpm

    results_map = {}

    with ThreadPoolExecutor(max_workers=evaluator_max_workers) as executor:
        future_to_idx = {}
        for idx, run in enumerate(runs):
            print(f"Submitting final response evaluation {idx+1}/{len(runs)} (run_id: {run.get('run_id')})...")
            future = executor.submit(evaluate_run, run, rubrics, args.model, args.include_latent_profile)
            future_to_idx[future] = idx
            time.sleep(pacing_delay)

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
                results_map[idx] = res
                print(f"Completed final response evaluation {idx+1}/{len(runs)} (run_id: {res.get('run_id')}).")
            except Exception as e:
                print(f"Error in final response evaluation {idx+1}: {e}")

    results = [results_map[k] for k in sorted(results_map.keys())]

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Evaluated {len(results)} runs. Final response scores saved to {out_path}")


if __name__ == "__main__":
    main()
