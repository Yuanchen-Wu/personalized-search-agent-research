# Benchmark v2 (2026-08-02)

Successor to the v1 benchmark (`data/synthetic_*_v1.jsonl` and `data/{domain}/`),
created ahead of the C1 placement-ablation rerun on all 72 pairs. v1 is frozen
and untouched so C2 (fixed fanout scaling) and C3 (adaptive re-fanout) results
remain exactly reproducible from their original configs.

Scope is deliberately surgical: **personas are byte-identical to v1**, and the
agent-visible content (query text + persona context) is identical for 71/72
items. Only judge-side rubric metadata changed beyond that. Cross-experiment
comparisons between C1-on-v2 and C2/C3-on-v1 therefore remain meaningful, with
a one-item caveat (q_51).

## Changes (15 records)

### 1. q_51 replaced (the one agent-visible change)

v1's q_51 ("How should I choose between a secured card and a credit builder
loan?") was a reworded duplicate of q_49 for the same persona
(`paycheck_to_paycheck_credit_rebuild`), filed under a different task-type cell
with a near-identical gold retrieval intent. It double-counted one item and
blurred the retrieval-/synthesis-sensitive split.

Replacement (same query_id, persona, task_type `synthesis_sensitive`,
task_category `financial_decision_strategy`, so all grid balances are
preserved):

> "What's the smartest way to handle a big unexpected expense like a car repair?"

Generic on its face (standard advice: emergency fund / credit card); for this
persona (<$500 savings, rebuilding credit after missed payments) the synthesis
must sequence bill negotiation, hardship programs, and credit-union payday
alternative loans while protecting the credit rebuild — i.e. genuinely
synthesis-sensitive. Topic-distinct from q_49/q_50/q_52 and from q_61's
emergency-*savings allocation* question (0.0 token jaccard against all four).

**Do not join q_51 across benchmark versions**: the query_id is reused for a
different question. Any v1-vs-v2 comparison must exclude q_51.

### 2. risk_level normalized (13 records, judge-side only)

`low_medium` (a fourth enum value that appeared only in personal_finance:
q_50, q_52, q_55, q_56, q_59, q_60, q_61, q_62, q_63, q_64, q_67, q_69, q_72)
is mapped to `medium`. Rounding up rather than down keeps safety judging
conservative. v2 risk distribution: low 23 / medium 31 / high 18.

### 3. q_9 positive_persona_signals (1 record, judge-side only)

"budget-conscious educational focus" was the corpus's only rubric signal
referencing latent-profile phrasing with no anchor in agent-visible text.
Replaced with the verbatim demographic reference
"annual_budget: $5,000 for bootcamps or courses".

## Known v1 issues deliberately NOT fixed here

- ~30 `positive_persona_signals` are paraphrases of history/demographics rather
  than verbatim strings — consumed only by LLM judges, harmless.
- All personas have uniform histories (5 observable + exactly 3 distractors,
  grid-like 9am/9pm timestamps). Fixing this changes agent input for every
  query and would break C1↔C2/C3 comparability; treat as a limitations note.
  (Paper/doc wording should say "3 distractors", not "3–5".)
- Time-anchored rubrics (tax-year 2026, current-rule items) are correct now but
  will rot for live-search runs in 2027; schedule fresh runs accordingly.

## Layout

Mirrors v1: merged `synthetic_{personas,queries}_v2.jsonl` at this root are
what configs consume; per-domain files are the organized source. The two forms
are kept record-identical (verified at build time).
