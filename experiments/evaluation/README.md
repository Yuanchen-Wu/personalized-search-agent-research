# Final Response Evaluation

This directory contains the tools to evaluate the final answers of our personalized search-based AI agent.

## Purpose
The purpose of this evaluation is to answer:
> Does personalized fan-out or mixed fan-out improve final response quality compared with generic fan-out and synthesis-only personalization?

This evaluates *only* the final answer. Fan-out branch evaluation, retrieval quality evaluation, and pairwise LLM comparisons are left for future work.

## Pointwise vs. Exact-Match Evaluation
Because this is an open-ended personalized search task, there is no single exact ground-truth string to compare against. Instead, we use an LLM-as-a-judge to evaluate how well the final response satisfies the user's hidden intent while properly utilizing specific required user constraints (and avoiding irrelevant ones).

## Compact Judge Dimensions
The judge evaluates 6 distinct dimensions using an integer scale of 1-5:
1. `intent_satisfaction`: Does it accomplish what a good answer should, for the real need behind the ambiguous query? (Higher is better)
2. `personalization_target_use`: Does it effectively use `must_use` constraints? (Higher is better)
3. `overpersonalization`: Does it misuse `should_not_use` constraints or overly force persona traits? (Lower is better)
4. `specificity`: Is the answer concrete instead of generic? (Higher is better)
5. `safety`: Crucial for health/medical scenarios. Are there unsafe recommendations? (Higher is better)
6. `overall`: Holistic quality score. (Higher is better)

## Judge inputs (leak-free contract)
The judge is shown **only** the visible ambiguous query and the **frozen per-query rubric** authored by the data generator (`personalization_targets` + `evaluation_notes` from `synthetic_data/generated/queries.jsonl`, loaded by `load_rubrics`). It is deliberately **not** shown the user's persona/history, nor the `clear_hidden_intent` answer key. Otherwise the judge would hold the same ground truth as the answer it grades, and `intent_satisfaction` would collapse into "how close is the answer to what I, the judge, would have written." This also keeps `overpersonalization` anchored to the query-specific `should_not_use` / `bad_answer_patterns` / `overpersonalization_risks` lists rather than the judge's own discretion, so legitimate on-topic specificity is no longer penalized as overpersonalization.

Each score record also carries a `deterministic_checks` field (lexical `must_use` / `should_not_use` keyword coverage) as an auxiliary, transparent signal that is **not** mixed into the LLM scores.

*Note on LLM-as-judge limits:* The judge still provides a heuristic evaluation and may misinterpret subtle nuances. The frozen rubric narrows—but does not eliminate—that discretion.

## Workflow

### 1. Generate Benchmark Data
Ensure you have synthetic users and queries generated via `experiments/synthetic_data`.

### 2. Run the Benchmark
Run all variants (V0-V4) on the generated queries. This will produce a JSONL log with the exact queries, personas, variant types, and outputs.

```bash
python experiments/run_generated_benchmark.py --limit 50
```

### 3. Evaluate Final Responses
Run the pointwise judge over the benchmark logs.

```bash
python experiments/evaluation/evaluate_final_responses.py --limit 50
```

### 4. Summarize Results
Aggregate the JSONL scores into readable CSVs and a Markdown summary report.

```bash
python experiments/evaluation/summarize_eval_results.py
```
