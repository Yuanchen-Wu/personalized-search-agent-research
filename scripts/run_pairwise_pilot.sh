#!/usr/bin/env bash
# Pairwise-judge PILOT: calibrate the instrument on a SMALL set of benchmark
# items before trusting it on any real comparison.
#
#     bash scripts/run_pairwise_pilot.sh              # 16 items x 2 contrasts (~130 flash calls)
#     LIMIT=8 bash scripts/run_pairwise_pilot.sh      # even smaller
#     bash scripts/run_pairwise_pilot.sh --dry-run    # no API: verify data + show commands
#
# Runs two KNOWN contrasts from a C2 v2 arm (distribution-matched to the
# instrument's real use: judging fixed_k8 vs a future adaptive arm):
#   pilot A: fixed_k1 vs fixed_k8  — true gap ~+1.0 on intent
#   pilot B: fixed_k4 vs fixed_k8  — true gap ~+0.3 on intent
#
# PASS criteria (printed after each run):
#   A: k8 win-rate >= ~75%  -> the judge detects large true gaps
#   B: k8 win-rate ~55-70%  -> mild lean, matching the small true gap
#   FAIL signals: A near 50% (judge blind) or B >= ~75% (judge over-eager /
#   biased) -> recalibrate the prompt before any real use.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
ARM="${ARM:-outputs/fixed_fanout_scaling_v2_gemini-3.6-flash/runs.jsonl}"
QUERIES=data/synthetic_queries_v1.jsonl   # C2 runs use benchmark v1 rubrics
OUT_DIR=outputs/pairwise
LIMIT="${LIMIT:-16}"
SAMPLES="${SAMPLES:-2}"
JUDGE="${JUDGE:-gemini-3.5-flash}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

for f in "$ARM" "$QUERIES"; do
  [ -f "$f" ] || { echo "MISSING: $f (need a C2 v2 arm here; override with ARM=...)"; exit 1; }
done
mkdir -p "$OUT_DIR"

run_contrast() {  # $1 label-a  $2 method-a  $3 label-b  $4 method-b  $5 outfile
  echo
  echo "== pilot: $1 vs $3  (limit=$LIMIT, ${SAMPLES}x2 votes/item, judge=$JUDGE)"
  if [ "$DRY" = 1 ]; then
    "$PY" - "$ARM" "$2" "$4" "$LIMIT" <<'EOF'
import json, sys
path, va, vb, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for l in open(path) if l.strip()]
qa = {r["query_id"] for r in rows if r.get("method") == va}
qb = {r["query_id"] for r in rows if r.get("method") == vb}
shared = sorted(qa & qb)[:limit]
print(f"   DRY: {len(shared)} items ready, ~{len(shared) * 4} judge calls; first: {shared[:5]}")
EOF
    return
  fi
  "$PY" scripts/judge_pairwise.py \
    --runs-a "$ARM" --method-a "$2" --label-a "$1" \
    --runs-b "$ARM" --method-b "$4" --label-b "$3" \
    --queries "$QUERIES" --out "$OUT_DIR/$5" \
    --model "$JUDGE" --samples "$SAMPLES" --limit "$LIMIT"
}

run_contrast k1 fixed_k1 k8 fixed_k8 pilot_k1_vs_k8.jsonl
run_contrast k4 fixed_k4 k8 fixed_k8 pilot_k4_vs_k8.jsonl

if [ "$DRY" = 0 ]; then
  echo
  echo "== interpretation"
  echo "   PASS: pilot A shows k8 win-rate >= ~75%  AND  pilot B shows a mild k8 lean (~55-70%)."
  echo "   FAIL: A near 50% (judge can't see large true gaps) or B >= ~85% (over-eager judge)."
  echo "   On PASS: scale up by re-running without LIMIT (resume keeps pilot items), then"
  echo "   use judge_pairwise.py for the real comparison: fixed_k8 vs the adaptive arm."
fi
