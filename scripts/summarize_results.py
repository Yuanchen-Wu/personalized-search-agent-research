import argparse
import json
import os
import sys
import yaml
import csv
from collections import defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def compute_mean(scores):
    if not scores: return 0.0
    return sum(scores) / len(scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    final_scores_path = config.get("outputs", {}).get("final_response_scores_path")
    fanout_scores_path = config.get("outputs", {}).get("fanout_scores_path")
    
    if not os.path.isabs(final_scores_path): final_scores_path = os.path.join(_PROJECT_ROOT, final_scores_path)
    if not os.path.isabs(fanout_scores_path): fanout_scores_path = os.path.join(_PROJECT_ROOT, fanout_scores_path)
    
    # Load scores
    runs_data = defaultdict(lambda: {"final": {}, "fanout": {}, "meta": {}})
    
    if os.path.exists(final_scores_path):
        with open(final_scores_path, "r") as f:
            for line in f:
                d = json.loads(line)
                runs_data[d["run_id"]]["final"] = d["scores"]
                runs_data[d["run_id"]]["meta"] = d
                
    if os.path.exists(fanout_scores_path):
        with open(fanout_scores_path, "r") as f:
            for line in f:
                d = json.loads(line)
                runs_data[d["run_id"]]["fanout"] = d["scores"]
                runs_data[d["run_id"]]["meta"].update(d)
                
    # Groupings
    by_variant = defaultdict(lambda: defaultdict(list))
    by_variant_task_type = defaultdict(lambda: defaultdict(list))
    by_variant_task_cat = defaultdict(lambda: defaultdict(list))
    
    for rid, data in runs_data.items():
        meta = data["meta"]
        var = meta.get("variant")
        tt = meta.get("task_type")
        tc = meta.get("task_category")
        
        flat_scores = {}
        for k, v in data["final"].items(): flat_scores[f"final_{k}"] = v
        for k, v in data["fanout"].items(): flat_scores[f"fanout_{k}"] = v
            
        for k, v in flat_scores.items():
            by_variant[var][k].append(v)
            by_variant_task_type[(var, tt)][k].append(v)
            by_variant_task_cat[(var, tc)][k].append(v)
            
    # Write CSVs
    def write_csv(path, group_dict, key_names):
        if not group_dict: return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Collect all columns
        all_cols = set()
        for v in group_dict.values():
            all_cols.update(v.keys())
        all_cols = sorted(list(all_cols))
        
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(key_names + [c + "_mean" for c in all_cols])
            
            for keys, scores in group_dict.items():
                if not isinstance(keys, tuple): keys = (keys,)
                row = list(keys)
                for c in all_cols:
                    row.append(f"{compute_mean(scores.get(c, [])):.2f}")
                writer.writerow(row)
                
    summary_by_variant_path = config.get("outputs", {}).get("summary_by_variant_path")
    summary_by_task_type_path = config.get("outputs", {}).get("summary_by_variant_task_type_path")
    summary_by_task_cat_path = config.get("outputs", {}).get("summary_by_variant_task_category_path")
    
    if not os.path.isabs(summary_by_variant_path): summary_by_variant_path = os.path.join(_PROJECT_ROOT, summary_by_variant_path)
    if not os.path.isabs(summary_by_task_type_path): summary_by_task_type_path = os.path.join(_PROJECT_ROOT, summary_by_task_type_path)
    if not os.path.isabs(summary_by_task_cat_path): summary_by_task_cat_path = os.path.join(_PROJECT_ROOT, summary_by_task_cat_path)
    
    write_csv(summary_by_variant_path, by_variant, ["variant"])
    write_csv(summary_by_task_type_path, by_variant_task_type, ["variant", "task_type"])
    write_csv(summary_by_task_cat_path, by_variant_task_cat, ["variant", "task_category"])
    
    print(f"Saved summaries to {os.path.dirname(summary_by_variant_path)}")

if __name__ == "__main__":
    main()
