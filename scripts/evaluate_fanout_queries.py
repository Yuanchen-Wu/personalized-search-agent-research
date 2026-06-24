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
from search_agent.rubrics import (
    FANOUT_RUBRIC_FIELDS,
    format_latent_profile,
    format_rubric,
    load_rubrics,
)

JUDGE_PROMPT_TEMPLATE = """You are an expert AI judge evaluating a search-augmented agent's query fanout generation.

You do NOT have access to the user's private profile, demographics, or hidden intent. You are given only the visible user query and a FROZEN evaluation rubric authored for this query before any search ran. Judge the fan-out ONLY against this rubric and the visible query.

Return your evaluation as a strict JSON object. Do not include markdown formatting like ```json or any other text outside the JSON object.

Input Data:
User Query: {query}
Task Type: {task_type}
Task Category: {task_category}
Macro Domain: {macro_domain}
Search Required: {search_required}
Expected Personalization Stage: {expected_personalization_stage}
Persona Relevant Dimensions: {persona_relevant_dimensions}

FROZEN EVALUATION RUBRIC (the ground truth for what a good search plan should target):
{rubric_block}
{profile_block}
Generated Fanout Branches to Evaluate:
{fanout_branches}

Metrics to score on a 1-5 scale:
- persona_field_use: Does the fan-out translate the rubric's `positive_persona_signals` / `must_use` constraints into the search queries? (Higher is better)
- query_specificity: Are the subqueries specific enough to retrieve useful evidence? (Higher is better)
- query_diversity: Do the subqueries cover meaningfully different aspects? (Higher is better)
- search_realism: Do the subqueries look like realistic search queries? (Higher is better)
- faithfulness_to_user_query: Do the subqueries preserve the original user intent? (Higher is better)
- overpersonalization_risk: Do the subqueries chase the rubric's `distractor_signals_to_ignore` or inject persona details the rubric does not call for? (1 = no problematic overpersonalization, 5 = severe overpersonalization)

Important Instructions:
Do not reward a fan-out merely because it repeats persona-like words. Reward only usage of signals the rubric marks relevant (`positive_persona_signals` / `must_use`).
For retrieval_sensitive tasks, reward fan-out that translates the rubric's required constraints into useful search queries.
For synthesis_sensitive tasks, do not require heavy persona-specific fan-out. Reward faithful, search-worthy, evidence-grounding queries. Penalize fan-out that over-personalizes by narrowing the search away from the user's actual information need or by chasing distractors.

Your output MUST be exactly in this JSON format:
{{
  "scores": {{
    "persona_field_use": 1,
    "query_specificity": 1,
    "query_diversity": 1,
    "search_realism": 1,
    "faithfulness_to_user_query": 1,
    "overpersonalization_risk": 1
  }},
  "rationale": {{
    "persona_field_use": "...",
    "query_specificity": "...",
    "query_diversity": "...",
    "search_realism": "...",
    "faithfulness_to_user_query": "...",
    "overpersonalization_risk": "..."
  }}
}}
"""


def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def build_prompt(run, rubric, include_latent_profile=False):
    """Render the fan-out judge prompt for one run. Pure (no model call)."""
    profile_block = ""
    if include_latent_profile:
        profile_block = (
            "\nPRIVILEGED USER PROFILE (A/B mode only — curated answer key; "
            "extra ground truth):\n"
            f"{format_latent_profile(run.get('persona') or {})}\n"
        )
    return JUDGE_PROMPT_TEMPLATE.format(
        query=run.get("user_query", run.get("query", "")),
        task_type=run.get("task_type", "unknown"),
        task_category=run.get("task_category", "unknown"),
        macro_domain=run.get("macro_domain", "unknown"),
        search_required=run.get("search_required", True),
        expected_personalization_stage=run.get("expected_personalization_stage", "unknown"),
        persona_relevant_dimensions=run.get("persona_relevant_dimensions", []),
        rubric_block=format_rubric(rubric, FANOUT_RUBRIC_FIELDS),
        profile_block=profile_block,
        fanout_branches=json.dumps(run.get("fanout_branches", []), indent=2),
    )


def evaluate_fanout(run, rubrics, model="gemini-flash-latest", include_latent_profile=False):
    rubric = rubrics.get(run.get("query_id"), {})
    prompt = build_prompt(run, rubric, include_latent_profile=include_latent_profile)

    result = {
        "run_id": run.get("run_id", "unknown"),
        "experiment_name": run.get("experiment_name", "unknown"),
        "variant": run.get("variant", "unknown"),
        "persona_id": run.get("persona_id", "unknown"),
        "query_id": run.get("query_id", "unknown"),
        "task_type": run.get("task_type", "unknown"),
        "task_category": run.get("task_category", "unknown"),
        "rubric_found": bool(rubric),
    }

    try:
        response = call_gemini(prompt, model=model, temperature=0.1)
        response_json = json.loads(clean_json_response(response))
        if "scores" not in response_json or "rationale" not in response_json:
            raise ValueError("Missing 'scores' or 'rationale' in JSON output.")
        result["scores"] = response_json["scores"]
        result["rationale"] = response_json["rationale"]
    except Exception as e:
        result["error"] = str(e)
        result["scores"] = {}
        result["rationale"] = {}

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=str, default="gemini-flash-latest")
    parser.add_argument(
        "--include-latent-profile",
        action="store_true",
        help="A/B mode: also feed the curated persona answer key to the judge. "
        "Default off = leak-free, rubric-only grading.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    runs_path = config.get("outputs", {}).get("runs_path")
    out_path = config.get("outputs", {}).get("fanout_scores_path")
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

    evaluator_rpm = 250
    evaluator_max_workers = 15
    pacing_delay = 60.0 / evaluator_rpm

    results_map = {}

    with ThreadPoolExecutor(max_workers=evaluator_max_workers) as executor:
        future_to_idx = {}
        for idx, run in enumerate(runs):
            if "fanout_branches" in run and len(run["fanout_branches"]) > 0:
                print(f"Submitting fanout evaluation {idx+1}/{len(runs)} (run_id: {run.get('run_id')})...")
                future = executor.submit(evaluate_fanout, run, rubrics, args.model, args.include_latent_profile)
                future_to_idx[future] = idx
                time.sleep(pacing_delay)

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
                results_map[idx] = res
                print(f"Completed fanout evaluation {idx+1}/{len(runs)} (run_id: {res.get('run_id')}).")
            except Exception as e:
                print(f"Error in fanout evaluation {idx+1}: {e}")

    results = [results_map[k] for k in sorted(results_map.keys())]

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Evaluated {len(results)} runs. Fanout scores saved to {out_path}")


if __name__ == "__main__":
    main()
