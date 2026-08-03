#!/usr/bin/env bash
# Pairwise-judge PILOT: calibrate the instrument on a SMALL set of benchmark
# items before trusting it on any real comparison.
#
#     bash scripts/run_pairwise_pilot.sh              # 16 items x 2 contrasts (~130 flash calls, <1 min)
#     bash scripts/run_pairwise_pilot.sh --dry-run    # no API: verify data + show plan
#
# Judge defaults to gemini-3.6-flash at WORKERS=24 (sized for a 1k-RPM tier;
# calls are unthrottled with retry/backoff, so 24 concurrent ~= 500-700 RPM).
# The pilot performs NO searches — Tavily limits are irrelevant here.
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
JUDGE="${JUDGE:-gemini-3.6-flash}"
WORKERS="${WORKERS:-24}"

DRY=0 VERDICT_ONLY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
[ "${1:-}" = "--verdict" ] && VERDICT_ONLY=1

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
    --model "$JUDGE" --samples "$SAMPLES" --limit "$LIMIT" --workers "$WORKERS"
}

if [ "$VERDICT_ONLY" = 0 ]; then
  run_contrast k1 fixed_k1 k8 fixed_k8 pilot_k1_vs_k8.jsonl
  run_contrast k4 fixed_k4 k8 fixed_k8 pilot_k4_vs_k8.jsonl
fi

if [ "$DRY" = 0 ]; then
  echo
  "$PY" - "$OUT_DIR/pilot_k1_vs_k8.jsonl" "$OUT_DIR/pilot_k4_vs_k8.jsonl" <<'PYEOF'
import json, os, sys

def rate(path):
    if not os.path.exists(path):
        return None, 0
    rows = [json.loads(l) for l in open(path) if l.strip()]
    nb = sum(1 for r in rows if r["majority"] == "b")
    dec = nb + sum(1 for r in rows if r["majority"] == "a")
    return (nb / dec if dec else None), len(rows)

ra, n_a = rate(sys.argv[1])
rb, n_b = rate(sys.argv[2])
print("== PILOT VERDICT")
if ra is None or rb is None:
    print("   INCOMPLETE: pilot output files missing/empty — re-run the pilot.")
    sys.exit(1)
a_pass = ra >= 0.75
b_ok = 0.50 <= rb < 0.85
print(f"   test A (k1 vs k8, known blowout +0.94):   k8 wins {ra:.0%} of decided ({n_a} items) "
      f"-> {'PASS' if a_pass else 'FAIL: judge cannot see large true gaps'}")
print(f"   test B (k4 vs k8, known narrow win +0.30): k8 wins {rb:.0%} of decided ({n_b} items) "
      f"-> {'PASS (mild lean)' if b_ok else 'AMPLIFYING: judge turns small gaps into strong preferences' if rb >= 0.85 else 'FAIL: judge inverted a known direction'}")
if not a_pass or rb < 0.50:
    print("   PILOT: FAIL — do not run the real comparison with this judge.")
    sys.exit(1)
if b_ok:
    print("   PILOT: PASS — judge certified. Interpret real win-rates directly.")
else:
    print("   PILOT: PASS WITH CALIBRATION — the judge amplifies small gaps. Use the")
    print("   anchors when reading the real comparison: ~50% = null, ~"
          f"{rb:.0%} = a +0.3-class effect, ~{ra:.0%} = a +0.9-class effect.")
print("   Next: run the seqpersona_k8 arm, then judge_pairwise.py fixed_k8 vs seqpersona_k8.")
PYEOF
fi
