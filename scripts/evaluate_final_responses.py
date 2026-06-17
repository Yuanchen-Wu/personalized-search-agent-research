import argparse
import json
import os
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from search_agent.llm_gemini import call_gemini

# Mocking a basic evaluator for the sake of the ablation structure
def evaluate_run(run, config):
    metrics = config.get("metrics", {}).get("final_response", [
        "intent_satisfaction", "personalization_target_use", 
        "overpersonalization", "specificity", "safety"
    ])
    
    # Normally we'd call Gemini here to score 1-5 for each metric based on the prompt.
    # For now, we simulate evaluation.
    scores = {}
    for m in metrics:
        scores[m] = 4.0 # Dummy score
        
    # Adding extra if specified
    if "groundedness" not in scores: scores["groundedness"] = 4.0
    if "non_genericness" not in scores: scores["non_genericness"] = 4.0

    return {
        "run_id": run.get("run_id", "unknown"),
        "experiment_name": run.get("experiment_name", "unknown"),
        "variant": run.get("variant", "unknown"),
        "persona_id": run.get("persona_id", "unknown"),
        "query_id": run.get("query_id", "unknown"),
        "task_type": run.get("task_type", "unknown"),
        "task_category": run.get("task_category", "unknown"),
        "scores": scores
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    runs_path = config.get("outputs", {}).get("runs_path")
    out_path = config.get("outputs", {}).get("final_response_scores_path")
    
    if not os.path.isabs(runs_path): runs_path = os.path.join(_PROJECT_ROOT, runs_path)
    if not os.path.isabs(out_path): out_path = os.path.join(_PROJECT_ROOT, out_path)
    
    if not os.path.exists(runs_path):
        print(f"Error: Runs file not found at {runs_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    results = []
    with open(runs_path, "r") as f:
        runs = [json.loads(l) for l in f if l.strip()]
        
    for run in runs:
        results.append(evaluate_run(run, config))
        
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Evaluated {len(results)} runs. Final response scores saved to {out_path}")

if __name__ == "__main__":
    main()
