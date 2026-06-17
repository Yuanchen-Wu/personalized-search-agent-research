# Personalization Placement in Query Fan-out (EACL 2027 prototype)

A simple, inspectable search-based AI agent for studying **where personalization should be applied** in a search-augmented LLM pipeline.

## Motivation & Project Goal
When building a retrieval-augmented agent, you can inject the user's persona/context at different stages:
- Only at **final answer synthesis**
- During **query fan-out** (the search queries themselves)
- At **both** stages
- Via a **mixed** fan-out that explicitly seeks generic, personalized, constraint, and *disconfirming* evidence.

**Research question:** does personalization help more at synthesis, at fan-out, at both, or in a mixed/disconfirming fan-out design?

The goal of this repository is to provide a cleaner research experiment structure for a personalization-placement ablation study, while preserving existing working functionality. The main goal is to make the experiment reproducible, well-logged, and easy to analyze.

**Explicit Warning:** `task_type` is used for analysis only. It must NOT alter generation behavior. We want to test whether the same personalization-placement variants naturally perform differently across task types. Do not write generation logic that branches on `task_type`.

## Pipeline
```
user query (+ optional persona)
  -> query fan-out generation        (Gemini; variant-dependent)
  -> Tavily Search API calls         (one call per branch)
  -> collect top results per branch  (normalized, duplicates flagged)
  -> final answer synthesis          (Gemini; persona-dependent)
  -> structured JSONL log            (outputs/placement_ablation_v1/runs.jsonl)
```

## Running a Single Query
You can run the agent locally on a single query via the CLI:
```bash
export PYTHONPATH=src
python -m search_agent.run_agent \
    --query "What laptop should I buy for ML research?" \
    --persona_id ml_phd_budget \
    --variant V4_mixed_fanout
```

## Variants
| Variant | Fan-out | Persona in fan-out? | Persona in synthesis? |
|---|---|---|---|
| `V0_generic_single` | raw query as one branch | no | no |
| `V1_generic_fanout` | 3–5 generic queries | no | no |
| `V2_synthesis_only_personalization` | generic queries (same as V1) | no | **yes** |
| `V3_personalized_fanout` | personalized queries | **yes** | **yes** |
| `V4_mixed_fanout` | generic + personalized + constraint + disconfirming | **yes** | **yes** |

## Directory Structure
```
EACL_2027_search_agent/
  README.md
  requirements.txt
  .env.example
  .gitignore

  configs/
    placement_ablation_v1.yaml
  src/
    search_agent/
      __init__.py, config.py, schemas.py, ...
  data/
    personas/, queries/, generated/
  scripts/
    generate_synthetic_data.py
    run_benchmark.py
    evaluate_final_responses.py
    evaluate_fanout_queries.py
    summarize_results.py
  outputs/
    placement_ablation_v1/
  notebooks/
    analysis.ipynb
  reports/
    notes/experiment_spec.md
```

## Setup
1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys (copy `.env.example` to `.env`):
```bash
cp .env.example .env
```

## Running the Pipeline

Follow these steps sequentially to run the full pipeline:

### 1. Generate Synthetic Data
Generate the synthetic user personas and search queries:
```bash
python scripts/generate_synthetic_data.py
```

### 2. Validate Experiment Setup
Run a smoke test to ensure config, paths, and distributions are valid:
```bash
python scripts/validate_experiment_setup.py --config configs/placement_ablation_v1.yaml
```

### 3. Run the Benchmark
Run the variants against the data:
```bash
python scripts/run_benchmark.py --config configs/placement_ablation_v1.yaml --limit 10
```

### 4. Evaluate Fan-out Queries
Evaluate the generated sub-queries:
```bash
python scripts/evaluate_fanout_queries.py --config configs/placement_ablation_v1.yaml
```

### 5. Evaluate Final Responses
Evaluate the final answers:
```bash
python scripts/evaluate_final_responses.py --config configs/placement_ablation_v1.yaml
```

### 6. Summarize Results
Aggregate scores into CSV files:
```bash
python scripts/summarize_results.py --config configs/placement_ablation_v1.yaml
```

## Analysis Outputs
The summarized CSV files are saved in `outputs/placement_ablation_v1/`. Use `notebooks/analysis.ipynb` to analyze the results.

### Legacy Scripts
Note: The old `/experiments/` directory scripts have been refactored or moved to `scripts/`. Please use the commands above.
