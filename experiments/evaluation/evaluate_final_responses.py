import argparse
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, "..", ".."))
SYNTHETIC_DATA_DIR = os.path.join(PROJECT_ROOT, "experiments", "synthetic_data")

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(SYNTHETIC_DATA_DIR)

from src.llm_gemini import call_gemini
from utils import parse_json_response, read_jsonl

EVAL_RPM = 200
rate_limit_lock = threading.Lock()
last_request_time = 0.0

def rate_limited_gemini(*args, **kwargs):
    global last_request_time
    delay = 60.0 / EVAL_RPM
    with rate_limit_lock:
        now = time.time()
        elapsed = now - last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)
        last_request_time = time.time()
    return call_gemini(*args, **kwargs)

def load_prompt() -> str:
    with open(os.path.join(EVAL_DIR, "prompts", "final_response_pointwise_judge.txt"), "r") as f:
        return f.read()

def _rubric_shape(must_use=None, should_not_use=None, good=None, bad=None, risks=None) -> dict:
    return {
        "must_use": must_use or [],
        "should_not_use": should_not_use or [],
        "good_answer_should": good or [],
        "bad_answer_patterns": bad or [],
        "overpersonalization_risks": risks or [],
    }

def load_rubrics() -> dict:
    """Frozen per-query rubric authored by the data generator, keyed by example_id.

    The judge scores against THIS rubric, never the agent's persona/history. Otherwise
    the evaluator holds the same ground truth as the answer it grades, and the score
    collapses into "how close is the answer to what I would have written."
    """
    path = os.path.join(SYNTHETIC_DATA_DIR, "generated", "queries.jsonl")
    rubrics = {}
    for q in read_jsonl(path):
        ex = q.get("example_id")
        if not ex:
            continue
        pt = q.get("personalization_targets", {}) or {}
        en = q.get("evaluation_notes", {}) or {}
        rubrics[ex] = _rubric_shape(
            must_use=pt.get("must_use"),
            should_not_use=pt.get("should_not_use"),
            good=en.get("good_answer_should"),
            bad=en.get("bad_answer_patterns"),
            risks=en.get("overpersonalization_risks"),
        )
    return rubrics

def rubric_from_run(run: dict) -> dict:
    """Fallback for runs whose example_id is absent from queries.jsonl (e.g. legacy
    logs): use the thinner rubric copied into the run's query_metadata."""
    qm = run.get("query_metadata", {}) or {}
    return _rubric_shape(must_use=qm.get("must_use"), should_not_use=qm.get("should_not_use"))

def keyword_coverage(text: str, items: list) -> list:
    """Lightweight, transparent lexical check: for each rubric item, the fraction of its
    salient (>= 4 char) keywords that appear in the answer. Auxiliary signal only —
    intentionally NOT mixed into the LLM judge scores."""
    low = (text or "").lower()
    out = []
    for item in items:
        words = [w for w in re.findall(r"[a-z0-9%]+", str(item).lower()) if len(w) >= 4]
        cov = round(sum(1 for w in words if w in low) / len(words), 2) if words else 0.0
        out.append({"item": item, "keyword_coverage": cov})
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_path", type=str, default=os.path.join(PROJECT_ROOT, "outputs", "generated_benchmark_runs.jsonl"))
    parser.add_argument("--output_path", type=str, default=os.path.join(EVAL_DIR, "generated", "final_response_scores.jsonl"))
    parser.add_argument("--judge_model", type=str, default="gemini-flash-latest")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    prompt_template = load_prompt()
    rubrics = load_rubrics()
    print(f"Loaded frozen rubrics for {len(rubrics)} queries from synthetic_data/generated/queries.jsonl.")

    runs = read_jsonl(args.runs_path)
    if not runs:
        print(f"Error: Runs file {args.runs_path} not found or empty.")
        sys.exit(1)
        
    if args.limit > 0:
        runs = runs[:args.limit]
        
    print(f"Loaded {len(runs)} benchmark runs to evaluate.")

    mode = "a" if args.append else "w"
    
    def evaluate_single_run(run):
        eval_record = {
            "eval_id": f"eval_{str(uuid.uuid4())[:8]}",
            "example_id": run.get("example_id"),
            "domain": run.get("domain"),
            "query_type": run.get("query_type"),
            "persona_id": run.get("persona_id"),
            "variant": run.get("variant"),
            "ambiguous_query": run.get("ambiguous_query"),
            "clear_hidden_intent": run.get("query_metadata", {}).get("clear_hidden_intent", ""),
            "final_answer": run.get("run_log", {}).get("final_answer", ""),
            "judge_model": args.judge_model
        }
        
        required = ["example_id", "variant", "query_type", "domain", "final_answer"]
        for req in required:
            if not eval_record.get(req):
                eval_record["error"] = f"Missing required field: {req}"
                return eval_record
                
        # Frozen per-query rubric authored by the data generator (load_rubrics).
        # Fall back to the thinner query_metadata rubric for legacy runs.
        rubric = rubrics.get(eval_record["example_id"]) or rubric_from_run(run)
        if not (rubric["good_answer_should"] or rubric["must_use"]):
            eval_record["error"] = f"No rubric found for example_id {eval_record['example_id']}"
            return eval_record

        # Auxiliary deterministic lexical signal; intentionally not folded into scores.
        eval_record["deterministic_checks"] = {
            "must_use_keyword_coverage": keyword_coverage(eval_record["final_answer"], rubric["must_use"]),
            "should_not_use_keyword_coverage": keyword_coverage(eval_record["final_answer"], rubric["should_not_use"]),
        }

        # The judge sees ONLY the frozen rubric and the visible query — never the
        # agent's persona/history (the leak) or clear_hidden_intent (the answer key).
        prompt = prompt_template
        replacements = {
            "{domain}": eval_record["domain"],
            "{query_type}": eval_record["query_type"],
            "{ambiguous_query}": eval_record["ambiguous_query"],
            "{good_answer_should}": json.dumps(rubric["good_answer_should"], indent=2),
            "{must_use}": json.dumps(rubric["must_use"], indent=2),
            "{should_not_use}": json.dumps(rubric["should_not_use"], indent=2),
            "{bad_answer_patterns}": json.dumps(rubric["bad_answer_patterns"], indent=2),
            "{overpersonalization_risks}": json.dumps(rubric["overpersonalization_risks"], indent=2),
            "{final_answer}": eval_record["final_answer"]
        }
        for k, v in replacements.items():
            prompt = prompt.replace(k, str(v))

        try:
            response = rate_limited_gemini(
                prompt=prompt,
                model=args.judge_model,
                temperature=0.1,
                response_mime_type="application/json"
            )
            eval_record["raw_judge_response"] = response
            parsed = parse_json_response(response)
            
            eval_record["scores"] = {
                "intent_satisfaction": parsed.get("intent_satisfaction", 1),
                "personalization_target_use": parsed.get("personalization_target_use", 1),
                "overpersonalization": parsed.get("overpersonalization", 1),
                "specificity": parsed.get("specificity", 1),
                "safety": parsed.get("safety", 1),
                "overall": parsed.get("overall", 1)
            }
            eval_record["diagnostic_feedback"] = parsed.get("diagnostic_feedback", "")
            eval_record["failure_modes"] = parsed.get("failure_modes", [])
        except Exception as e:
            eval_record["error"] = str(e)
            
        return eval_record

    with open(args.output_path, mode) as out_f:
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(evaluate_single_run, r): r for r in runs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating runs"):
                try:
                    result = future.result()
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                except Exception as e:
                    print(f"\nFailed to evaluate a run: {e}")
                    
    print(f"Finished evaluation. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
