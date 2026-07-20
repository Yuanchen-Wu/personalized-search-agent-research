# C3 Re-fanout: Pipeline Hardening & Noise Investigation

**Status:** pipeline changes implemented, unit- + live-verified. Full experiment grid **not yet run**. This note records what we did, what the smoke runs told us, the resulting code changes, and the next step.

## What C3 is investigating

Does adding the **re-fanout-until-good** adaptive loop improve the **final answer quality** of the search agent — and if so, how much, and how does the gain vary with **max_rounds**, the **approval threshold τ** (the numeric coverage bar), and **fan-out breadth k**?

- **Cost is out of scope** for this study (only quality).
- Scope is the **re-fanout** path only (`src/search_agent/adaptive_refanout.py`). The incremental-deepening path (`adaptive_loop.py`, `adaptive_kN`/`adaptive_bN`) is **deliberately parked** (banner-marked in those files) for a possible later study.

The loop, per round: fan out *k* persona-conditioned queries → search → an LLM **retrieval judge** scores coverage → a controller **approves** (synthesize from that round) when `coverage ≥ τ`, else regenerates the whole fan-out from the judge's feedback and retries (rejected rounds discarded), capped at `max_rounds`.

---

## Observations from the smoke runs

### 1. Judge recalibration — verified
The retrieval judge's `coverage_score` anchors were tightened (spread scores down; "a 5 must be rare, most first fan-outs = 3"). Independently re-derived every figure from the raw logs (`runs.jsonl` = before, `runs_recalibrated_smoke.jsonl` = after, 12 pairs, seed 42):

| metric | before | after |
|---|---|---|
| round-1 score histogram | `1:1, 3:1, 4:4, 5:6` | `3:5, 4:4, 5:3` |
| mean rounds | 1.17 | 1.50 |
| retried (>1 round) | 2/12 | 5/12 (4 improved) |

The recalibration works — scores spread, the loop engages ~40% of queries.

### 2. τ=5 confirmation smoke — the loop engages at a high bar
`runs_tau5_smoke.jsonl` (K=1, τ=5, max_rounds=3, 12 pairs): mean rounds **2.08**, round-count `{1:5, 2:1, 3:6}` — **6/12 hit the round-3 cap** (5 exhausted). So `max_rounds` is a live variable at high τ, unlike at τ=4 (which barely passed round 2). Also surfaced **drift**: some retried rounds scored *lower* than round 1 (e.g. `3→3→2`).

### 3. Noise investigation — the key finding
Two independent K=3 runs at different seeds (`runs_k3judge_smoke.jsonl` seed 42, `runs_k3judge_rep2.jsonl` seed 123), plus within-call vote spreads, disentangled the noise:

- **Within-call judge noise is small:** pooled σ ≈ **0.36** (max vote-range 1, zero ≥2 swings). K=3 averaging → σ ≈ 0.21.
- **The planner rewrites queries every run** (0/12 identical fan-outs across seeds) **but the outcome is stable:** cross-seed σ ≈ **0.34** on round-1 coverage, **0.19** on the approved evidence; engagement stable **11/12**.
- The earlier "±2 swings" were **single-shot judge fat tails**, which K=3 smooths — not planner chaos.

**Conclusion:** the pipeline is reproducible enough that **R ≈ 2–3 seeds** is ample (per-pair SE = σ/√R ≈ 0.20 at R=3; aggregate over 72 pairs ≈ 0.04). The one noise source **still unmeasured** is the **out-of-loop answer evaluator** (the judge that scores final-answer quality — the metric we actually report). That measurement is deferred and re-runnable from saved runs (see Next Steps).

---

## Pipeline improvements (implemented + verified)

All in `src/search_agent/adaptive_refanout.py`, wired through `scripts/run_fixed_fanout_benchmark.py` and `configs/adaptive_refanout_v1.yaml`:

1. **K-sample averaged judge** (`judge_samples`, `judge_temperature`): the retrieval judge is polled K times/round on the same evidence and averaged → a **fractional** `coverage_score`. Denoises the judge and gives τ a finer sweep than integers. Raw votes logged in `sample_scores`. Default K=1 preserves old behavior; config sets K=3.
2. **Fractional τ thresholds**: method names accept `_t{N}` with `p` as decimal point, e.g. `refanout_k4_t4p5` → τ=4.5. Threshold is a float end-to-end.
3. **Per-round evidence logging**: each `refanout_round` event now carries that round's retrieved docs (`results`). This makes **`max_rounds` a nested, post-hoc variable** — run once at a high cap and derive every smaller cap's outcome from the saved trace (same idea as the fixed-k nested prefixes), and re-synthesize/re-score later.
4. **Best-round fallback**: on exhaustion the loop synthesizes from the **highest-coverage round**, not the last — re-fanout can drift, so the last round isn't necessarily the best (`fallback_round` logged).

**Verification:** 39/39 unit tests pass (incl. new tests for K-averaging, best-round fallback, per-round evidence, fractional parsing). A 2-pair live pre-flight (`runs_verify_changes.jsonl`, max_rounds=6, τ=4.5) confirmed all four end-to-end. Illustrative — **q_1**: `4.0 → 3.33 → 3.67 → 4.0 → 5.0` (approved at round 5). It persisted through a dip and reached 5.0; capped at 3 it would have fallen back to round 1's 4.0. That single run shows both that **max_rounds matters** and that **best-round-over-drift matters**.

---

## Run inventory (`outputs/adaptive_refanout_v1/`)

| file | what | judge | τ | max_rounds |
|---|---|---|---|---|
| `runs.jsonl` | pre-recalibration smoke (12) | K=1 old | 4 | 3 |
| `runs_recalibrated_smoke.jsonl` | post-recalibration smoke (12) | K=1 | 4 | 3 |
| `runs_tau5_smoke.jsonl` | τ=5 engagement smoke (12) | K=1 | 5 | 3 |
| `runs_k3judge_smoke.jsonl` | K=3 noise-floor smoke (12, seed 42) | K=3 | 4 | 3 |
| `runs_k3judge_rep2.jsonl` | between-seed replicate (12, seed 123) | K=3 | 4 | 3 |
| `runs_verify_changes.jsonl` | 2-pair code pre-flight | K=3 | 4.5 | 6 |

Reproduce via `refanout_k{k}[_t{τ}]` method names + `--seed` (all use `data/synthetic_*_v1.jsonl`, first N pairs via `--limit`).

---

## Next steps (the plan)

1. **Run the generation grid** (generation now, scoring later — the two are decoupled; runs persist `final_answer` + per-round evidence so any out-of-loop judge can be re-run on saved data):
   - `max_rounds=6`, τ ∈ {3 (loop-off baseline), 4p5 (hard-reachable), 5 (forced)}, K=3, **R seeds**, 72 pairs.
   - `max_rounds` is read off post-hoc (nested), so it is **not** a separate sweep.
   - Timing: ~4–6 h per seed at the 15 RPM Gemini free-tier throttle; the throttle (`GEMINI_MAX_RPM`) is the main lever (30–60 RPM → ~1–2 h/seed). Fully resumable on `(query_id, persona, method, seed)`.
2. **Score the saved runs** with the out-of-loop frozen-rubric evaluators — compare **τ=3 (loop off) vs τ=4.5/5** on answer quality (`intent_satisfaction`, `constraint_coverage`, groundedness, …). This is the "does the loop help" read, and the two seed-replicates give a **free measurement of the evaluator's own noise** (the last unquantified piece). Consider K-sampling the evaluator if noisy.
3. **Analyze** the gain and its sensitivity to **max_rounds** (nested), **τ**, and **k** (k as a secondary sweep once τ shows signal). Pick the fractional τ grid from multi-run data, not a single 12-pair run.

### Open design notes
- τ=5 under K=3 needs a *unanimous* 5 → nearly always exhausts (a "forced persistence" regime); τ=4.5 is the hard-but-reachable bar where approval-round varies.
- The out-of-loop evaluators are also LLM judges → the same denoising/repeat logic applies to the measurement side.
