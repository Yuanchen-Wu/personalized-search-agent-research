# Personalization Placement in Query Fan-out (EACL 2027 prototype)

A simple, inspectable search-based AI agent for studying **where personalization
should be applied** in a search-augmented LLM pipeline.

## Motivation

When you build a retrieval-augmented agent, you can inject the user's persona /
context at different stages:

- only at **final answer synthesis**, or
- during **query fan-out** (the search queries themselves), or
- at **both** stages, or
- via a **mixed** fan-out that explicitly seeks generic, personalized,
  constraint, and *disconfirming* evidence.

**Research question:** does personalization help more at synthesis, at fan-out,
at both, or in a mixed/disconfirming fan-out design?

This first iteration is deliberately minimal: plain Python, clean logging, no
reranking or fusion. The goal is **ablation control and research transparency**,
not production performance.

## Pipeline

```
user query (+ optional persona)
  -> query fan-out generation        (Gemini; variant-dependent)
  -> Tavily Search API calls         (one call per branch)
  -> collect top results per branch  (normalized, duplicates flagged)
  -> final answer synthesis          (Gemini; persona-dependent)
  -> structured JSONL log            (outputs/runs.jsonl)
```

We use **Tavily only for search evidence** — its generated `answer` field is
never used. All synthesis is done by Gemini so the experiment cleanly isolates
personalization placement.

## Variants

| Variant | Fan-out | Persona in fan-out? | Persona in synthesis? |
|---|---|---|---|
| `V0_generic_single` | raw query as one branch | no | no |
| `V1_generic_fanout` | 3–5 generic queries | no | no |
| `V2_synthesis_only_personalization` | generic queries (same as V1) | no | **yes** |
| `V3_personalized_fanout` | personalized queries | **yes** | **yes** |
| `V4_mixed_fanout` | generic + personalized + constraint + disconfirming | **yes** | **yes** |

No fusion, reranking, or Reciprocal Rank Fusion is implemented in this version.

## Project structure

```
project/
  README.md
  .env.example
  requirements.txt
  src/
    config.py          # env vars, defaults, paths
    llm_gemini.py      # call_gemini(...)
    search_tavily.py   # search_tavily(...), collect_search_results(...)
    fanout.py          # generate_fanout_queries(...)
    synthesize.py      # synthesize_answer(...)
    run_agent.py       # orchestration + CLI
    schemas.py         # dataclass schemas
    logging_utils.py   # JSONL logging
  experiments/
    sample_queries.jsonl
    sample_personas.jsonl
    run_batch.py
  outputs/
    runs.jsonl         # appended run logs
```

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure API keys. **Keys are read only from environment variables; nothing
   is hardcoded.** Copy the example file and fill in your real keys:

```bash
cp .env.example .env
# then edit .env:
#   GEMINI_API_KEY=...
#   TAVILY_API_KEY=...
```

`.env` is git-ignored. You can also export the variables directly instead of
using a `.env` file:

```bash
export GEMINI_API_KEY=...
export TAVILY_API_KEY=...
```

<!-- ## Run one query

```bash
python src/run_agent.py \
  --query "What laptop should I buy for ML research?" \
  --persona_id ml_phd_budget \
  --variant V4_mixed_fanout
```

This prints:

- the fan-out branches (with type, query, rationale, persona fields used),
- the top Tavily results per branch,
- the final synthesized answer,
- a cost proxy,

and appends one JSON line to `outputs/runs.jsonl`.

Useful flags: `--variant` (one of the five), `--persona_id` (from
`experiments/sample_personas.jsonl`, optional), `--max_results_per_branch`,
`--model`, `--no_log`. -->

## Reproducing the EACL 2027 Evaluation

To reproduce the full evaluation pipeline and generate the final visualizations, follow these steps sequentially:

### 1. Generate Synthetic Data
First, generate the synthetic user personas and their corresponding ambiguous search queries:
```bash
.venv/bin/python experiments/synthetic_data/generate_synthetic_data.py
```
This populates the `experiments/synthetic_data/generated/` folder with profiles and queries designed specifically to test personalization boundaries.

### 2. Run the Benchmark
Execute all five agent variants (`V0` through `V4`) against the generated queries:
```bash
.venv/bin/python experiments/run_generated_benchmark.py
```
*(Tip: Use `--limit N` to run a smaller subset for quick testing)*
This orchestrates the full pipeline and saves the raw search trajectories and final answers to `outputs/generated_benchmark_runs.jsonl`.

### 3. Run the LLM Judge
Evaluate the final answers using our pointwise LLM-as-a-judge:
```bash
.venv/bin/python experiments/evaluation/evaluate_final_responses.py
```
This critically scores each answer on Intent Satisfaction, Personalization Target Use, Overpersonalization, Specificity, and Safety, saving the raw scores to `experiments/evaluation/generated/final_response_scores.jsonl`.

### 4. Summarize Results
Aggregate the raw scores into clean CSVs and Markdown tables:
```bash
.venv/bin/python experiments/evaluation/summarize_eval_results.py
```

### 5. Visualize the Findings
Finally, open the provided Jupyter Notebook to view the performance charts and our final conclusions:
- Open `analysis.ipynb` in your preferred Jupyter environment.
- Run all cells.
- The notebook will automatically load the generated CSVs and render clean, publication-ready visualizations comparing the architectural variants.

<!-- ## Notes & limitations

- No reranking / fusion yet (planned for later iterations).
- Duplicate URLs are kept but flagged, not removed.
- `cost_proxy` is a transparency aid, not real billing.
- Soft-fails on a single bad search/branch so a batch run keeps going. -->
