#!/usr/bin/env bash
# SELF-CONTAINED pilot for the fixed_k8 vs seqpersona_k8 showdown.
#
#     bash scripts/run_seqpersona_pilot.sh              # PAIRS=6 end-to-end (~small spend, see below)
#     bash scripts/run_seqpersona_pilot.sh --dry-run    # no API: verify configs + show the plan
#
# What one invocation does, on any machine with GEMINI_API_KEY + TAVILY_API_KEY:
#   1. Runs seqpersona_k8 on the first PAIRS benchmark pairs — INTO THE CANONICAL
#      output dir (outputs/seqpersona_k8_v1/), so the later full 72-pair run
#      RESUMES past the pilot pairs instead of re-paying them.
#   2. Provides fixed_k8 answers for the same pairs: reuses the canonical C2 v2
#      3.6-flash arm if present on this checkout; otherwise generates a pilot
#      fixed_k8 arm (same pinned models) into an isolated dir.
#   3. Runs the calibrated pairwise judge on the shared pairs and prints the
#      win-rate with the calibration-anchor interpretation.
#
# Cost at PAIRS=6 (worst case, no canonical fixed arm present): ~100 searches,
# ~110 flash calls. With the canonical arm present: ~48 searches, ~80 calls.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PAIRS="${PAIRS:-6}"
WORKERS="${WORKERS:-6}"
export GEMINI_MAX_RPM="${GEMINI_MAX_RPM:-300}"
export TAVILY_MAX_RPM="${TAVILY_MAX_RPM:-90}"
SEQ_CFG=configs/seqpersona_k8_v1.yaml
SEQ_RUNS=outputs/seqpersona_k8_v1/runs.jsonl
CANON_FIXED=outputs/fixed_fanout_scaling_v2_gemini-3.6-flash/runs.jsonl
PILOT_DIR=outputs/seqpersona_pilot
QUERIES=data/synthetic_queries_v1.jsonl

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

for f in "$SEQ_CFG" "$QUERIES" data/synthetic_personas_v1.jsonl; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
mkdir -p "$PILOT_DIR"

# --- fixed_k8 source: reuse canonical arm, else generate a pilot arm ---------
if [ -f "$CANON_FIXED" ]; then
  FIXED_RUNS="$CANON_FIXED"
  FIXED_CFG=""
  echo "== fixed_k8: reusing canonical arm ($CANON_FIXED)"
else
  FIXED_CFG="$PILOT_DIR/fixed_k8_pilot.yaml"
  FIXED_RUNS="$PILOT_DIR/fixed_k8/runs.jsonl"
  "$PY" - "$SEQ_CFG" "$PILOT_DIR" "$FIXED_CFG" <<'EOF'
import sys, yaml
seq_cfg, pilot_dir, dst = sys.argv[1:4]
cfg = yaml.safe_load(open(seq_cfg))          # inherits the pinned models + data paths
cfg["experiment_name"] = "seqpersona_pilot_fixed_k8"
cfg["methods"] = ["fixed_k8"]
cfg["fixed_fanout"] = {"candidate_pool_size": 8, "k_values": [8], "use_nested_prefixes": True}
out = f"{pilot_dir}/fixed_k8"
cfg["outputs"] = {
    "run_dir": out, "runs_path": f"{out}/runs.jsonl",
    "fanout_plans_path": f"{out}/fanout_plans.jsonl",
    "search_cache_path": f"{out}/search_cache.jsonl",
}
with open(dst, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"== fixed_k8: no canonical arm here; pilot arm config -> {dst}")
EOF
fi

if [ "$DRY" = 1 ]; then
  echo "== DRY RUN (no API)"
  "$PY" scripts/run_fixed_fanout_benchmark.py --config "$SEQ_CFG" --limit "$PAIRS" --dry_run | tail -2
  [ -n "$FIXED_CFG" ] && "$PY" scripts/run_fixed_fanout_benchmark.py --config "$FIXED_CFG" --limit "$PAIRS" --dry_run | tail -2
  echo "   then: pairwise judge on the $PAIRS shared pairs"
  exit 0
fi

echo "== [1/3] seqpersona_k8 on $PAIRS pairs (canonical dir; full run resumes past these)"
"$PY" scripts/run_fixed_fanout_benchmark.py --config "$SEQ_CFG" --limit "$PAIRS" --workers "$WORKERS" \
  2>&1 | tee "$PILOT_DIR/seqpersona_runner.log" | tail -3

if [ -n "$FIXED_CFG" ]; then
  echo "== [2/3] pilot fixed_k8 on $PAIRS pairs"
  "$PY" scripts/run_fixed_fanout_benchmark.py --config "$FIXED_CFG" --limit "$PAIRS" --workers "$WORKERS" \
    2>&1 | tee "$PILOT_DIR/fixed_runner.log" | tail -3
else
  echo "== [2/3] fixed_k8: canonical arm reused, nothing to run"
fi

# evidence sanity on the pilot's seqpersona pairs
"$PY" - "$SEQ_RUNS" <<'EOF'
import json, sys
runs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
empty = sum(1 for r in runs if not r.get("raw_search_results"))
print(f"   evidence check: {len(runs) - empty}/{len(runs)} seqpersona runs have search results")
sys.exit(1 if empty else 0)
EOF

echo "== [3/3] pairwise judge (calibrated instrument) on the shared pairs"
"$PY" scripts/judge_pairwise.py \
  --runs-a "$FIXED_RUNS" --method-a fixed_k8 --label-a fixed_k8 \
  --runs-b "$SEQ_RUNS" --method-b seqpersona_k8 --label-b seqpersona \
  --queries "$QUERIES" --limit "$PAIRS" \
  --out "$PILOT_DIR/pairwise_pilot.jsonl"

echo
echo "== interpretation (calibration anchors from run_pairwise_pilot.sh)"
echo "   ~50% seqpersona win-rate = null · ~86% = +0.3-class effect · ~100% = +0.9-class."
echo "   At PAIRS=$PAIRS this is a PLUMBING + direction check, not a verdict — CIs are"
echo "   wide at this n. If it runs clean: full arm = same runner command without"
echo "   --limit (resumes past these pairs), then the same judge command without --limit."
