# Draft-Answer-Critic Adaptive Loop — Design & Pilot

**Status:** prototype implemented + 20-pair pilot run and scored. Full 72-pair grid **not run** (the pilot answered the primary question). All work is isolated under `outputs/answercritic_v1/`; the earlier adaptive-loop work is untouched.

## Motivation

The coverage-judged adaptive loop significantly improves **retrieval** quality (evidence constraint coverage, source quality) but its effect on **final-answer quality is not significant** — the answer is bounded by synthesis (latent-intent inference), not by evidence coverage. The in-loop **retrieval-coverage judge** only ever answers *"what evidence is still missing?"*, so the loop is a coverage-maximizer whose objective is near-orthogonal to answer quality.

**Question:** if the loop instead judged the **draft answer** — re-fanning on what the *answer* still fails to do — does answer quality improve?

## Design — leak-free draft-answer critic (`src/search_agent/adaptive_answercritic.py`)

Each round:
1. Fan out *k* persona-conditioned queries (round 1 from scratch; later rounds from the critic's feedback).
2. Search all *k* → this round's evidence.
3. **Synthesize a draft answer** from that evidence (hardened synthesis prompt).
4. A **leak-free answer critic** rates the draft (1–5, K-sampled) and reports its gaps + `needs_more_evidence` (retrieval-bound vs synthesis-bound).
5. Approve (that round's draft is final) when the mean score ≥ threshold, else re-fan on the answer's gaps. Capped at `max_rounds`; on exhaustion use the best-scoring round.

**Leak-free contract (load-bearing).** The critic sees only agent-visible inputs — the query, `persona.render_for_agent()`, this round's evidence, and the agent's own draft. It **never** sees the frozen evaluation rubric / gold intent. Using the rubric-aware *final* judge inside the loop would be leakage (the agent optimizing against its own held-out metric); this critic is a legitimate self-assessment, exactly like the retrieval judge. The module imports nothing from `rubrics`.

**Cost note.** Unlike the coverage loop (one synthesis at the end), this synthesizes a draft *every round* — more expensive, and the reason a draft is logged per round.

## Infrastructure (run in parts, money-safe, isolated)

`scripts/run_answercritic_benchmark.py`:
- Appends + `fsync`s each pair immediately (an interruption never loses completed work).
- Skips pairs already saved for the seed on resume → run in parts with `--limit` / `--query_ids`, re-invoke to continue.
- A single pair that errors is logged and skipped (retried next resume), never crashing the run.
- Isolated output dir (`outputs/answercritic_v1/`).

**Threshold nesting ("t=5 trick").** The critic is threshold-blind and the revised fan-out depends only on its threshold-blind feedback, so the per-round trajectory is threshold-independent. Running the max-persistence config (threshold 5, `max_rounds` 6) lets thresholds {3,4,5} be derived post-hoc — and because a draft is synthesized every round, this needs **no re-synthesis** (`scripts/derive_answercritic_grid.py` just reads off each threshold's stop-round draft).

## Pilot — 20 pairs (`gemini-3.5-flash`, seed 42)

Derived the {3,4,5} answer-score grid and scored it with the out-of-loop frozen-rubric evaluators (isolated files).

### Loop dynamics (from the logs, no rubric scoring)
- Critic approves (hits 5.0) for 8/11 early records; by its own measure re-fanning improves the draft **+0.77** (never worse).
- The loop changes the final answer for only **11/20** pairs (t3 ≠ t5); for 9/20 it is idle (t3 ≡ t5).
- The critic flags **~39% of rounds as synthesis-bound** (`needs_more_evidence=False`) — a retrieval-vs-synthesis diagnosis the coverage judge cannot produce. Exhausted pairs are retrieval-bound.

### Answer quality — the result (rubric-scored, paired)
`answercritic` t3 → t5, intent_satisfaction:

| set | Δ intent | 95% CI |
|---|---|---|
| full (n=20) | **+0.053** | [−0.21, +0.37] |
| engaged, t3≠t5 (n=11) | **+0.000** | [−0.40, +0.50] |

The loop's re-fanning converts almost entirely into **specificity / grounding**, not intent — on the engaged set: specificity **+0.50**, groundedness **+0.18**, unsupported-claim risk **−0.18** (all borderline), intent **0.00**.

### Head-to-head vs the coverage loop (same 20 pairs, full persistence)
Nearly identical starting point (t3 baselines differ by 0.06). `answercritic_t5 − coverage_t5`: intent **+0.10 (ns)**, specificity **+0.20**, personalization **−0.25**, groundedness **0.00**, unsupported-claim **−0.05**. A wash — the answer-critic loop trades personalization for specificity.

### Phase comparison — intent gain
On the matched 20 pairs the loop gains are **statistically identical**: coverage loop **+0.000**, answer-critic loop **+0.053** (overlapping CIs). The large intent gain in this line of work (**+0.76**) came from the **synthesis prompt**, which both loops share. The *loop mechanism* adds ~0 intent in both.

## Verdict

Switching the loop's judge from evidence-coverage to answer-critique **does not lift intent** — the retrieval/answer decoupling survives the judge-swap. Answer quality is a **synthesis** lever, not a loop lever, regardless of what the loop judges. The answer-critic loop's genuine value is a **byproduct**: it catches hallucinations (e.g. fabricated statistics) and diagnoses synthesis-bound vs retrieval-bound failures — a faithfulness / observability tool, not an answer-quality mechanism.

**Caveats.** n=20 pilot, single seed, effects borderline-ns. The full 72-pair grid (~$30) would only firm up the weak faithfulness signal; it will not change the intent null (the engaged-set intent gain is already 0.00). Stopped at the pilot.

## Reproduce

```bash
# run in parts (resumable); drop --limit to finish, add --seed for a replicate
GEMINI_MAX_RPM=120 PYTHONPATH=src python scripts/run_answercritic_benchmark.py --config configs/answercritic_v1.yaml --limit 20
PYTHONPATH=src python scripts/derive_answercritic_grid.py
python scripts/evaluate_fixed_fanout.py --config configs/answercritic_score_v1.yaml
```
