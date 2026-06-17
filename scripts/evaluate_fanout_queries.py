import argparse
import json
import os
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from search_agent.llm_gemini import call_gemini

def evaluate_fanout(run, config):
    metrics = config.get("metrics", {}).get("fanout", [
        "persona_field_use", "query_specificity", "query_diversity", 
        "search_realism", "faithfulness_to_user_query", "overpersonalization_risk"
    ])
    
    # Mock scoring for structure
    scores = {}
    for m in metrics:
        scores[m] = 4.0

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
    out_path = config.get("outputs", {}).get("fanout_scores_path")
    
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
        # Only evaluate if fanout queries exist
        if "fanout_branches" in run and len(run["fanout_branches"]) > 0:
            results.append(evaluate_fanout(run, config))
        
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Evaluated {len(results)} runs. Fanout scores saved to {out_path}")

if __name__ == "__main__":
    main()
