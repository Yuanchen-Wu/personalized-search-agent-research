import argparse
import os
import sys
import yaml
import json
from collections import Counter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from search_agent.schemas import QueryRecord, VARIANTS, Persona

def check_path(path_str, name):
    if not os.path.isabs(path_str):
        path_str = os.path.join(_PROJECT_ROOT, path_str)
    exists = os.path.exists(path_str)
    print(f"[{'PASS' if exists else 'FAIL'}] {name}: {path_str}")
    return exists, path_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print(f"Validating experiment setup using config: {args.config}\n")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    print("1. Checking Paths")
    print("-" * 40)
    data_config = config.get("data", {})
    outputs_config = config.get("outputs", {})
    
    queries_ok, queries_path = check_path(data_config.get("queries_path", ""), "Queries file")
    personas_ok, personas_path = check_path(data_config.get("personas_path", ""), "Personas file")
    
    # Optional outputs
    out_dir = os.path.dirname(outputs_config.get("runs_path", "outputs/test.jsonl"))
    if not os.path.isabs(out_dir): out_dir = os.path.join(_PROJECT_ROOT, out_dir)
    print(f"[INFO] Outputs will be written to: {out_dir}")
    
    print("\n2. Checking Variants")
    print("-" * 40)
    variants = config.get("variants", [])
    for v in variants:
        if v in VARIANTS:
            print(f"[PASS] Valid variant: {v}")
        else:
            print(f"[FAIL] Invalid variant: {v}")
            
    print("\n3. Checking Queries Distribution")
    print("-" * 40)
    if queries_ok:
        records = []
        with open(queries_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(QueryRecord.from_dict(json.loads(line)))
        print(f"Loaded {len(records)} queries.")
        
        task_types = Counter([r.task_type for r in records])
        print("\nBy task_type:")
        for k, v in task_types.items():
            print(f"  {k}: {v}")
            
        task_categories = Counter([r.task_category for r in records])
        print("\nBy task_category:")
        for k, v in task_categories.items():
            print(f"  {k}: {v}")
    else:
        print("Skipping queries distribution (file not found).")
        
    print("\n4. Checking Personas")
    print("-" * 40)
    if personas_ok:
        personas = []
        with open(personas_path, "r") as f:
            for line in f:
                if line.strip():
                    personas.append(Persona.from_dict(json.loads(line)))
        print(f"Loaded {len(personas)} personas.")
    else:
        print("Skipping personas check (file not found).")

if __name__ == "__main__":
    main()
