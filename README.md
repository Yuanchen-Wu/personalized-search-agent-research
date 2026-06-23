# Personalization Placement in Query Fan-out

A simple, inspectable search-based AI agent for studying **where personalization should be applied** in a search-augmented LLM pipeline.

## Motivation & Project Goal
When building a retrieval-augmented agent, you can inject the user's persona/context at different stages:
- Only at **final answer synthesis** (`V2_synthesis_only_personalization`)
- During **query fan-out** (the search queries themselves) (`V3_personalized_fanout`)
- At **both** stages
- Via a **mixed** fan-out that explicitly seeks generic, personalized, constraint, and *disconfirming* evidence (`V4_mixed_fanout`).

**Research question:** does personalization help more at synthesis, at fan-out, at both, or in a mixed/disconfirming fan-out design?

The goal of this repository is to study the personalization-placement ablation across three macro-domains:
1. **Education** (learning resources, explanations, travel/shopping recommendations)
2. **Legal Information** (tenancy, immigration, contracts, custody, labor, copyright)
3. **Personal Finance** (budgeting, credit, student loans, homebuying, retirement, taxes)

---

## Experimental Design Constraints

### 1. Level-2 Under-specified Queries
Surface queries are intentionally written to be natural and under-specified (e.g., *"What GPU should I buy for local ML experiments?"* rather than *"What under-$500 GPU should I buy as a budget CS student?"*). The agent must inspect the user's stated demographics and chronological history to infer their latent preferences and constraints.

### 2. Chronological Interleaved Distractor History
User histories include both relevant observable history (e.g., visa timelines, tenant notices) and unrelated distractor history (e.g., air fryer queries, hiking shoes). This tests whether the agent can successfully filter out noise and avoid over-personalizing.

### 3. Domain Safety & Caveats
High-stakes domains (legal and finance) enforce strict safety rules:
- **Legal Info**: Answers must provide educational checklists and jurisdiction-aware caveats, avoid definitive legal conclusions, and recommend consulting qualified legal counsel for high-stakes decisions.
- **Personal Finance**: Answers must explain tradeoffs, avoid guaranteed return claims, acknowledge missing constraints, and avoid pushing specific commercial products.

### 4. Generation Independence
`task_type` (`retrieval_sensitive` or `synthesis_sensitive`) is used purely as a post-hoc analysis label. The agent's generation prompts and behaviors must remain independent of this label to ensure a fair, unbiased evaluation.

---

## Pipeline & Variants

```
user query (+ optional persona)
  -> query fan-out generation        (Gemini; variant-dependent)
  -> Tavily Search API calls         (one call per branch)
  -> collect top results per branch  (normalized, duplicates flagged)
  -> final answer synthesis          (Gemini; persona-dependent)
  -> structured JSONL log            (outputs/placement_ablation_v1/runs.jsonl)
```

| Variant | Fan-out | Persona in fan-out? | Persona in synthesis? |
|---|---|---|---|
| `V0_generic_single` | raw query as one branch | no | no |
| `V1_generic_fanout` | 3–5 generic queries | no | no |
| `V2_synthesis_only_personalization` | generic queries (same as V1) | no | **yes** |
| `V3_personalized_fanout` | personalized queries | **yes** | **yes** |
| `V4_mixed_fanout` | generic + personalized + constraint + disconfirming | **yes** | **yes** |

For **V4 mixed fan-out**, the agent generates exactly four search queries targeting:
1. **Generic**: neutral, broad search.
2. **Personalized**: tailored to the user's inferred preferences.
3. **Constraint**: targeting hard constraints (budget, jurisdiction, visa timelines, risk tolerance).
4. **Disconfirming**: actively searching for caveats, state exceptions, risk disclosures, or tradeoffs.

---

## Running a Single Query
You can run the agent locally on a single query via the CLI:
```bash
export PYTHONPATH=src
python -m search_agent.run_agent \
    --query "What GPU should I buy for local ML experiments?" \
    --persona_id budget_highstem_phd \
    --variant V4_mixed_fanout
```

---

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

---

## Running the Pipeline

Follow these steps sequentially to run the full pipeline:

### 1. Generate Synthetic Data
Generate the synthetic user personas and search queries for all domains:
```bash
python scripts/generate_synthetic_data.py --macro_domain all --num_users 6 --queries_per_category 1
```

### 2. Validate Experiment Setup
Run the diagnostic validator to ensure config, schemas, and stage mappings are correct:
```bash
python scripts/validate_experiment_setup.py --config configs/placement_ablation_v1.yaml
```

### 3. Run the Benchmark
Execute the ablation benchmark across the generated queries:
```bash
python scripts/run_benchmark.py --config configs/placement_ablation_v1.yaml --limit 10
```

### 4. Evaluate Fan-out Queries
Evaluate sub-query diversity, specificity, and over-personalization:
```bash
python scripts/evaluate_fanout_queries.py --config configs/placement_ablation_v1.yaml
```

### 5. Evaluate Retrieval Evidence
Evaluate retrieved search results against persona constraints, source quality, and distractor robustness:
```bash
python scripts/evaluate_retrieval_results.py --config configs/placement_ablation_v1.yaml
```

### 6. Evaluate Final Responses
Evaluate intent satisfaction, personalization utility, groundedness, and domain safety:
```bash
python scripts/evaluate_final_responses.py --config configs/placement_ablation_v1.yaml
```

### 7. Summarize Results
Aggregate scores and compute contrasts across variants, task types, and macro-domains:
```bash
python scripts/summarize_results.py --config configs/placement_ablation_v1.yaml
```

---

## Analysis Outputs
The summarized CSV files are saved in `outputs/placement_ablation_v1/`:
- `summary_by_variant.csv`: Variant-level averages.
- `summary_by_variant_task_type.csv`: Breakdown by task type.
- `summary_by_macro_domain.csv`: Breakdown by macro-domain.
- `contrasts_by_task_type.csv`: Pairwise variant contrast analysis.
- `contrasts_by_macro_domain_task_type.csv`: Contrasts grouped by domain and task type.
