# Re-fanout Retrieval Judge — Why It Needs Recalibration

**Status:** finding from the `adaptive_refanout_v1` pilot smoke (12 runs) — **recalibration applied and verified** (see *Result* below). The judge's `coverage_score` anchors were tightened; the distribution now spreads and the loop engages.

## Background

The re-fanout adaptive loop (C3) runs, per round: fan out *k* queries → search → an LLM **retrieval judge** rates the retrieved evidence with an anchored **`coverage_score` ∈ {1..5}**. A controller **approves** the round (→ synthesize from it) when `coverage_score ≥ approval_threshold (τ)`; otherwise it regenerates the whole fan-out from the judge's feedback and retries (rejected rounds discarded), capped at `max_rounds`.

τ is meant to be the knob that traces the quality–cost frontier: low τ → approve early → cheap (≈ a single fan-out); high τ → keep retrying → more retrieval cost, higher quality ceiling.

## The smoke

`refanout_k4`, τ=4, `max_rounds=3`, 12 (query, persona) pairs. Every round's `coverage_score` was logged.

## What we found

**Round-1 `coverage_score` histogram** (threshold-independent — every run executes round 1):

| score | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| runs | 1 | 0 | 1 | 4 | 6 |

**10 of 12 runs score 4–5 on the first fan-out.** Mean rounds = **1.17**; at τ=4 only **2/12** retried.

**Threshold sweep** — fraction approving on round 1 (i.e. *not* retrying) at each τ:

| τ | approve round-1 | behavior |
|---|---|---|
| 1 | 12/12 | degenerate — never retries (= single fan-out) |
| 2 | 11/12 | ≡ τ=3 in this sample (no run scored 2) |
| 3 | 11/12 | barely retries |
| 4 | 10/12 | — |
| 5 | 6/12 | most retries |

## Why this is a problem

The distribution is **compressed at the top**, with two consequences:

1. **The τ sweep is nearly degenerate.** τ=1 never retries; τ=2 ≡ τ=3 here; only τ=4 vs τ=5 differ. This knob traces at most ~2–3 distinct operating points, and its low-τ end simply reproduces `fixed_k`.
2. **The loop barely engages.** At the default τ=4 it retries in only 2/12 cases (mean 1.17 rounds), so `refanout_k4` ≈ a single fan-out ≈ `fixed_k4` for most queries. There is little "adaptive" behavior to measure, and the arm's whole premise — spending extra rounds to beat a static fan-out — cannot show up because it rarely spends them. Only τ=5 makes it do anything, and even then 6/12 still pass immediately.

The **mechanism itself is fine**: both runs that did retry **improved their score** (feedback → revised fan-out → higher coverage). It is simply **under-triggered**, because the judge calls almost every initial fan-out a 4 or 5.

## Why recalibrate (rather than just fix τ=5)

Pinning τ=5 gives a single, coarse frontier point and still leaves half the runs never retrying. To trace a real quality–cost frontier we need the **scores to spread lower**, so that:

- a meaningful fraction of *initial* fan-outs score 2–3 → the loop retries and we can actually measure the quality gained from re-fanning-out, and
- τ ∈ {3, 4, 5} produce genuinely distinct operating points.

This is a **judge-calibration** fix, not a loop-logic fix.

## Proposed recalibration

Tighten the `coverage_score` anchors so **5 is genuinely rare** and the middle bands are reachable:

- **5** — evidence covers the core ask **and** the user-relevant constraints **and** has corroborating sources **and** (for consequential topics) explicit caveats/exceptions/disconfirming evidence. A single strong source is **not** a 5.
- **4** — core ask + main constraints covered, but missing corroboration or caveats.
- **3** — core ask covered but a materially relevant constraint/nuance is absent (expected to be common for a first, generic fan-out).
- **2** — on-topic but misses the user's actual need or key constraints.
- **1** — off-target.

Also: instruct the judge to **default to the lower band when unsure** (LLM raters drift high), and to score **relative to what a great answer for THIS user would require**, not "is this generally about the topic."

## How we'll verify

Re-run the same 12-pair smoke after recalibration and confirm the round-1 histogram **spreads** (real mass at 2–3), mean rounds rises above ~1.2, and τ ∈ {3, 4, 5} give distinct approve-rates. Then pick the sweep values from the new distribution.

## Result — after recalibration (verified)

Re-ran the **same 12 pairs (seed 42)** with the recalibrated judge (to a scratch file so the "before" data is untouched). It worked — scores moved down and the loop engages:

| metric | before | after |
|---|---|---|
| round-1 score histogram | `1:1, 3:1, 4:4, 5:6` | `3:5, 4:4, 5:3` |
| runs scoring a 5 (round 1) | 6/12 | 3/12 |
| mean rounds | 1.17 | 1.50 |
| mean searches / run | 4.33 | 5.83 |
| retried (>1 round) | 2/12 | 5/12 (4 improved) |
| τ-sweep approve @ round-1 | t3:11 t4:10 t5:6 | t3:12 t4:7 t5:3 |

The top band is now rare (5: 6→3), the generic first-pass band filled in (3: 1→5), and the loop retries for ~40% of queries (2→5). The mechanism holds — 4 of 5 retries improved the score.

**Which thresholds to sweep:** τ ≤ 2 collapses into τ=3, so the usable sweep is **τ ∈ {3, 4, 5}** — three distinct operating points (0 / 5 / 9 retries out of 12, respectively). With `fixed_k4` (static plan, no loop) as the baseline, that is a workable 4-point cost–quality comparison.

**Not over-tightening further:** the distribution is now `{3,4,5}` with nothing at 1–2. Scores of 1–2 would require a genuinely bad first fan-out, which a decent persona-conditioned planner rarely produces; pushing the anchors harder to force 2s risks scoring adequate evidence unfairly. Recommend stopping here and sweeping {3, 4, 5}.

## Caveats

Small sample (12 runs, one seed, `basic` search depth, one persona/query mix). The specific counts are indicative, not final; the low end (scores 1–2) in particular could shift with more data.
