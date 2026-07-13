"""Validator script for fixed_fanout_scaling_v1 setup.

Usage:
    python scripts/validate_fixed_fanout_setup.py --config configs/fixed_fanout_scaling_v1.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

# Load .env if present
env_path = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))



def validate_setup(config_path: str) -> List[str]:
    """Validate configuration file and environment setup for fixed_fanout_scaling_v1."""
    errors: List[str] = []

    if not os.path.exists(config_path):
        return [f"Config file not found: {config_path}"]

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Experiment name
    if config.get("experiment_name") != "fixed_fanout_scaling_v1":
        errors.append(f"Invalid experiment_name: {config.get('experiment_name')!r} (expected 'fixed_fanout_scaling_v1')")

    # 2. k values & fixed_fanout section
    ff = config.get("fixed_fanout", {})
    k_values = ff.get("k_values", [])
    if not k_values or not isinstance(k_values, list):
        errors.append("k_values must be a non-empty list of integers.")
    else:
        for k in k_values:
            if not isinstance(k, int) or k <= 0:
                errors.append(f"Invalid k value in k_values: {k!r}")
        if k_values != sorted(list(set(k_values))):
            errors.append(f"k_values must be sorted and unique, got: {k_values}")

    candidate_pool_size = ff.get("candidate_pool_size", 8)
    max_k = max(k_values) if k_values else 8
    if candidate_pool_size < max_k:
        errors.append(f"candidate_pool_size ({candidate_pool_size}) is smaller than maximum k ({max_k}).")

    if not ff.get("use_nested_prefixes", False):
        errors.append("use_nested_prefixes must be set to true for this experiment.")

    # 3. Evidence budget mode
    synth = config.get("synthesis", {})
    eb_mode = synth.get("evidence_budget_mode")
    if eb_mode not in ("all", "fixed_document_budget"):
        errors.append(f"Invalid evidence_budget_mode: {eb_mode!r}. Must be 'all' or 'fixed_document_budget'.")

    # 4. Data paths
    data_cfg = config.get("data", {})
    p_path = data_cfg.get("personas_path")
    q_path = data_cfg.get("queries_path")

    if p_path and not os.path.isabs(p_path):
        p_path = os.path.join(_PROJECT_ROOT, p_path)
    if q_path and not os.path.isabs(q_path):
        q_path = os.path.join(_PROJECT_ROOT, q_path)

    if not p_path or not os.path.exists(p_path):
        errors.append(f"Personas data file missing or invalid path: {p_path}")
    if not q_path or not os.path.exists(q_path):
        errors.append(f"Queries data file missing or invalid path: {q_path}")

    # 5. Output paths check for conflicts
    outputs = config.get("outputs", {})
    paths = list(outputs.values())
    if len(paths) != len(set(paths)):
        errors.append("Conflicting (duplicate) output paths detected in configuration.")

    # 6. Environment variables
    gemini_key = os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not gemini_key:
        errors.append("Missing required environment variable: GEMINI_API_KEY")
    if not tavily_key:
        errors.append("Missing required environment variable: TAVILY_API_KEY")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate fixed fanout scaling setup.")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    errors = validate_setup(args.config)
    if errors:
        print("[VALIDATION FAILURE] Found setup errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("[VALIDATION SUCCESS] Configuration and setup are valid.")


if __name__ == "__main__":
    main()
