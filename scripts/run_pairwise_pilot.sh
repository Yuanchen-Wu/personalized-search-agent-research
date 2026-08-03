#!/usr/bin/env bash
# Pairwise-judge PILOT: calibrate the instrument on a SMALL set of benchmark
# items before trusting it on any real comparison.
#
#     bash scripts/run_pairwise_pilot.sh              # 16 items x 2 contrasts (~130 flash calls)
#     LIMIT=8 bash scripts/run_pairwise_pilot.sh      # even smaller
#     bash scripts/run_pairwise_pilot.sh --dry-run    # no API: verify data + show commands
#
# Runs two KNOWN contrasts from the C1 gemini-3.6-flash arm:
#   pilot A: V1 (generic fanout)  vs V4 (full personalization)  — true gap +1.18
#   pilot B: V2 (persona@synth)   vs V3 (persona@fanout)        — true gap +0.22, half ties
#
# PASS criteria (printed after each run):
#   A: V4 win-rate >= ~75%  -> the judge detects large true gaps
#   B: V3 win-rate ~50-70%  -> mild lean, matching the small true gap
#   FAIL signals: A near 50% (judge blind) or B >= ~75% (judge over-eager /
#   biased) -> recalibrate the prompt before any real use.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
ARM=outputs/placement_ablation_v2_gemini-3.6-flash/runs.jsonl
QUERIES=data/v2/synthetic_queries_v2.jsonl
OUT_DIR=outputs/pairwise
LIMIT="${LIMIT:-16}"
SAMPLES="${SAMPLES:-2}"
JUDGE="${JUDGE:-gemini-3.5-flash}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

for f in "$ARM" "$QUERIES"; do
  [ -f "$f" ] || { echo "MISSING: $f (need the merged C1 arm on this checkout)"; exit 1; }
done
mkdir -p "$OUT_DIR"

run_contrast() {  # $1 label-a  $2 variant-a  $3 label-b  $4 variant-b  $5 outfile
  echo
  echo "== pilot: $1 vs $3  (limit=$LIMIT, ${SAMPLES}x2 votes/item, judge=$JUDGE)"
  if [ "$DRY" = 1 ]; then
    "$PY" - "$ARM" "$2" "$4" "$LIMIT" <<'EOF'
import json, sys
path, va, vb, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for l in open(path) if l.strip()]
qa = {r["query_id"] for r in rows if r["variant"] == va}
qb = {r["query_id"] for r in rows if r["variant"] == vb}
shared = sorted(qa & qb)[:limit]
print(f"   DRY: {len(shared)} items ready, ~{len(shared) * 4} judge calls; first: {shared[:5]}")
EOF
    return
  fi
  "$PY" scripts/judge_pairwise.py \
    --runs-a "$ARM" --variant-a "$2" --label-a "$1" \
    --runs-b "$ARM" --variant-b "$4" --label-b "$3" \
    --queries "$QUERIES" --out "$OUT_DIR/$5" \
    --model "$JUDGE" --samples "$SAMPLES" --limit "$LIMIT"
}

run_contrast V1 V1_generic_fanout          V4 V4_personalized_fanout           pilot_v1_vs_v4.jsonl
run_contrast V2 V2_synthesis_only_personalization V3 V3_fanout_only_personalization  pilot_v2_vs_v3.jsonl

if [ "$DRY" = 0 ]; then
  echo
  echo "== interpretation"
  echo "   PASS: pilot A shows V4 win-rate >= ~75%  AND  pilot B shows a mild V3 lean (~50-70%)."
  echo "   FAIL: A near 50% (judge can't see large true gaps) or B >= ~75% (over-eager judge)."
  echo "   On PASS: scale up by re-running without LIMIT (resume keeps pilot items), then"
  echo "   use judge_pairwise.py on real comparisons (e.g. fixed_k8 vs a future adaptive arm)."
fi
