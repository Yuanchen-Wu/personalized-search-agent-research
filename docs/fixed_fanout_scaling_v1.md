# Fixed Fanout Scaling Study (`fixed_fanout_scaling_v1`)

## 1. Overview & Research Question

This study investigates how retrieval quality, final-answer quality, information redundancy, API cost, and latency change as the number of search-query fanout branches increases.

**Core Research Question:**
> *How do retrieval quality, final-answer quality, redundancy, cost, and latency change as the number of search-query fanout branches increases?*

We evaluate fixed fanout branch counts $k \in \{1, 2, 4, 8\}$ under strict experimental controls.

---

## 2. Experimental Conditions

The experiment implements four methods:
- `fixed_k1`: Executes priority rank 1 search query ($b_1$).
- `fixed_k2`: Executes priority ranks 1–2 search queries ($b_1, b_2$).
- `fixed_k4`: Executes priority ranks 1–4 search queries ($b_1, b_2, b_3, b_4$).
- `fixed_k8`: Executes priority ranks 1–8 search queries ($b_1, \dots, b_8$).

### Controlled Variables
- **Personalization Placement**: Held constant across all four conditions (Persona context is visible during both fanout planning and final synthesis, using `V5_mixed_fanout` principles).
- **Planner & Synthesizer Models**: Standardized Gemini model defaults across conditions.
- **Search Provider**: Tavily Search API with `search_depth: basic` and `max_results_per_branch: 5`.

---

## 3. Nested-Prefix Methodology

To ensure that differences across $k$ reflect the effect of **fanout capacity** rather than stochastic query quality differences:
1. Exactly **one candidate 8-query plan** is generated per `(query_id, persona_id, planner_model, prompt_version, seed)` tuple.
2. The four fixed-k runs derive their executed queries strictly from nested prefixes of this shared candidate plan:
   $$Q_1 \subset Q_2 \subset Q_4 \subset Q_8$$
3. Search results for each unique query in the 8-query plan are fetched once and cached. Overlapping prefix conditions reuse these cached search results, eliminating temporal search drift.
4. Information isolation is strictly enforced: `fixed_k2` synthesis only receives search results from queries 1–2, and is completely blind to results from queries 3–8.

---

## 4. Planner Behavior, Branch Ordering & Defensive Logic

The ordered fanout planner prompt (`ORDERED_FANOUT_PLANNER_PROMPT_V1`) orders queries by expected marginal retrieval value:
- **Priority Rank 1**: Strongest standalone search (general interpretation).
- **Priority Rank 2**: Tailored to persona-specific preferences or needs.
- **Priority Rank 3**: Hard constraints (budget, jurisdiction, location, timeline, risk tolerance).
- **Priority Rank 4**: Caveats, state exceptions, tradeoffs, or disconfirming evidence.
- **Priority Ranks 5–8**: Nonredundant supplementary information needs.

Allowed branch types: `generic`, `personalized`, `constraint`, `disconfirming`, `supplementary`.

### Validation, Repair, and Fallback
- **Exact & Near-Duplicate Removal**: Near-duplicates are detected via Jaccard & content token overlap.
- **Structured 1-Step Repair**: If fewer than 8 valid branches remain after initial parsing and deduplication, a structured repair call requests only the missing number of branches, explicit about existing queries and needs already covered.
- **Deterministic Fallback**: If 8 valid branches cannot be obtained after repair, deterministic supplementary templates fill remaining positions to guarantee exact length 8.

---

## 5. Synthesis Evidence Modes & Limitations

The study supports two synthesis evidence modes:
- **Mode A (`all`)** *(Default)*: Passes all deduplicated evidence retrieved by the executed prefix to the synthesizer. This models real-world end-to-end performance where larger fanout delivers more context.
- **Mode B (`fixed_document_budget`)**: Passes at most $N$ documents to synthesis for every $k$ using a deterministic, method-independent ranking rule.

### Critical Limitation & Confounder Isolation
Larger fanout naturally increases both search coverage and the volume of context shown to the synthesizer. Mode A evaluates the practical end-to-end effect of scaling fanout, while Mode B serves as a robustness analysis isolating **retrieval coverage** from **synthesis context quantity**.

---

## 6. Evaluation Procedure & Metrics

### Blind Evaluation
The retrieval judge evaluates search evidence using balanced sampling across executed branches (`top_m_per_branch`), and is blind to method names and values of $k$.

### Measured Metrics
- **Final Response Quality**: Intent satisfaction, personalization utility, overpersonalization risk, specificity, groundedness, domain safety, unsupported claim risk.
- **Retrieval Quality**: Evidence relevance, persona fit, constraint coverage, source quality, disconfirming coverage.
- **Fanout Quality**: Query diversity, query specificity, faithfulness, overpersonalization risk.
- **Non-LLM Diagnostics**: Realized fanout count, unique URL count, duplicate URL rate, unique domain count, retrieved context size, synthesis context size, latencies, Tavily calls, Gemini calls.

---

## 7. Paired Statistical Analysis

Because all conditions share the same candidate plan per query/persona pair, we perform **paired comparisons**:
$$\Delta_{k_a \to k_b} = \text{Score}(k_b) - \text{Score}(k_a)$$

For each comparison ($\Delta_{1\to 2}, \Delta_{2\to 4}, \Delta_{4\to 8}, \Delta_{1\to 4}, \Delta_{1\to 8}$):
- Mean paired difference
- Standard error of the mean (SEM)
- Median paired difference
- Bootstrap 95% confidence interval (1,000 iterations, seed 42)
- Percentage of queries improved, unchanged, or worsened.

---

## 8. Reproduction Commands

### 1. Validate Environment & Config
```bash
python scripts/validate_fixed_fanout_setup.py --config configs/fixed_fanout_scaling_v1.yaml
```

### 2. Run Dry-Run Preview
```bash
python scripts/run_fixed_fanout_benchmark.py --config configs/fixed_fanout_scaling_v1.yaml --dry_run
```

### 3. Execute Benchmark Runs
```bash
python scripts/run_fixed_fanout_benchmark.py --config configs/fixed_fanout_scaling_v1.yaml
```

### 4. Evaluate Runs
```bash
python scripts/evaluate_fixed_fanout.py --config configs/fixed_fanout_scaling_v1.yaml
```

### 5. Generate Summaries & Frontier Tables
```bash
python scripts/summarize_fixed_fanout.py --config configs/fixed_fanout_scaling_v1.yaml
```
