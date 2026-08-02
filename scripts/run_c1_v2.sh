#!/usr/bin/env bash
# One-command C1 rerun: full 6-variant placement ablation on benchmark v2.
#
# On the experiment laptop (needs GEMINI_API_KEY + TAVILY_API_KEY in .env;
# the one Gemini key serves every Gemini model):
#
#     bash scripts/run_c1_v2.sh                                # baseline agent: gemini-3.5-flash
#     bash scripts/run_c1_v2.sh --agent-model gemini-3.1-pro   # comparison arm, run per model
#
# One invocation = one agent model's full pipeline: bootstraps .venv if
# needed, validates the setup, runs all 432 agent runs (72 pairs x 6
# variants), judges fanout / retrieval / final-response (post-split
# answer-quality + evidence-faithfulness pair), summarizes, and writes a
# provenance manifest. The baseline model lands in
# outputs/placement_ablation_v2/ (what the analysis notebook's RESULTS_DIR
# reads); every other agent model gets its own sibling dir
# outputs/placement_ablation_v2_<model>/ with an auto-generated config, so
# per-model results never mix and each arm resumes independently. That
# per-model-dir layout is what scripts/make_paper_figures.py's MODELS
# placeholder consumes for cross-model figures.
#
# The JUDGE model is deliberately FIXED (gemini-flash-latest) and not a
# parameter: it is the constant measuring stick across C1/C2/C3 and across
# agent models — vary the agent, never the judge.
#
# The command is safe to interrupt and re-invoke at any point (battery death,
# quota exhaustion, Ctrl-C): every completed agent run and every completed
# judge call is already on disk (appended + flushed as it finishes), and the
# next invocation repairs any part-written tail line and resumes from exactly
# what's left instead of re-running or duplicating finished work.
#
# Options:
#   --agent-model NAME   Model that does the search-agent work (fanout,
#                        synthesis). Default gemini-3.5-flash = the baseline
#                        used for C2/C3. Run the script once per model you
#                        want to compare.
#   --smoke              End-to-end shakeout on the first 12 jobs (2 queries
#                        x 6 variants) before committing to the full batch. A
#                        later full run resumes agent and judge stages.
#   --dry-run            Preflight + plan preview only; no API calls.
#   --force-eval         Discard judge scores for this arm and re-judge.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_CONFIG=configs/placement_ablation_v2.yaml
BASE_OUT_DIR=outputs/placement_ablation_v2
PY=.venv/bin/python
BASELINE_AGENT_MODEL="gemini-3.5-flash"
AGENT_MODEL="$BASELINE_AGENT_MODEL"
# Fixed measuring stick across C1/C2/C3 and across agent models. Never vary
# this — the agent model (--agent-model) is the experimental axis.
JUDGE_MODEL="gemini-flash-latest"
MAX_AGENT_PASSES="${MAX_AGENT_PASSES:-3}"

LIMIT="" DRY=0 FORCE_EVAL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --agent-model) [ $# -ge 2 ] || { echo "--agent-model needs a value"; exit 2; }
                   AGENT_MODEL="$2"; shift ;;
    --agent-model=*) AGENT_MODEL="${1#*=}" ;;
    --smoke) LIMIT=12 ;;
    --dry-run) DRY=1 ;;
    --force-eval) FORCE_EVAL=1 ;;
    *) echo "unknown option: $1 (see header of $0)"; exit 2 ;;
  esac
  shift
done

# Baseline model uses the standard dir + config; any other agent model gets a
# sibling dir with a derived config so arms never mix and resume independently.
if [ "$AGENT_MODEL" = "$BASELINE_AGENT_MODEL" ]; then
  OUT_DIR="$BASE_OUT_DIR"
  CONFIG="$BASE_CONFIG"
else
  MODEL_TAG=$(printf '%s' "$AGENT_MODEL" | tr -c 'a-zA-Z0-9._-' '-')
  OUT_DIR="${BASE_OUT_DIR}_${MODEL_TAG}"
  CONFIG="$OUT_DIR/config.yaml"   # generated after the venv exists (preflight)
fi
RUNS="$OUT_DIR/runs.jsonl"

mkdir -p "$OUT_DIR/logs"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo "== [0/5] Preflight"
if [ ! -x "$PY" ]; then
  echo "   .venv missing -> creating and installing requirements"
  python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
fi
"$PY" -c "import yaml, dotenv" 2>/dev/null || "$PY" -m pip install -q -r requirements.txt

for f in "$BASE_CONFIG" data/v2/synthetic_queries_v2.jsonl data/v2/synthetic_personas_v2.jsonl; do
  [ -f "$f" ] || { echo "   MISSING: $f (pull latest main)"; exit 1; }
done

# Non-baseline agent model: derive this arm's config from the base one by
# re-pointing every output path into $OUT_DIR (data paths stay shared).
if [ "$CONFIG" != "$BASE_CONFIG" ]; then
  "$PY" - "$BASE_CONFIG" "$OUT_DIR" "$MODEL_TAG" "$CONFIG" <<'EOF'
import sys, yaml
base_path, out_dir, tag, dst = sys.argv[1:5]
cfg = yaml.safe_load(open(base_path))
base_prefix = cfg["outputs"]["run_dir"]
cfg["experiment_name"] = f'{cfg["experiment_name"]}_{tag}'
cfg["outputs"] = {k: v.replace(base_prefix, out_dir) for k, v in cfg["outputs"].items()}
with open(dst, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"   agent-model arm config -> {dst}")
EOF
fi

# API keys: hard requirement for a real run, warning only for --dry-run.
if ! "$PY" -c "
from search_agent.config import get_gemini_api_key, get_tavily_api_key
get_gemini_api_key(); get_tavily_api_key()
print('   API keys: present (values not shown)')
" 2>/dev/null; then
  if [ "$DRY" = 1 ]; then
    echo "   API keys: NOT set (ok for --dry-run)"
  else
    echo "   API keys missing: put GEMINI_API_KEY and TAVILY_API_KEY in .env"; exit 1
  fi
fi

"$PY" scripts/validate_experiment_setup.py --config "$CONFIG" \
  2>&1 | tee "$OUT_DIR/logs/validate.log" | grep -E "\[FAIL\]|WARNING" || true
if grep -q "\[FAIL\]" "$OUT_DIR/logs/validate.log"; then
  echo "   validation FAILed - see $OUT_DIR/logs/validate.log"; exit 1
fi
echo "   validation: no FAILs"

# Remaining agent jobs = full plan minus (query_id, persona_id, variant) triples
# already in runs.jsonl. Also flags duplicate triples, which should never occur.
remaining_jobs() {
  "$PY" - "$CONFIG" "$RUNS" <<'EOF'
import json, os, sys, yaml
from collections import Counter
cfg = yaml.safe_load(open(sys.argv[1]))
queries = [json.loads(l) for l in open(cfg["data"]["queries_path"]) if l.strip()]
variants = cfg["variants"]
plan = {(q["query_id"], q["persona_id"], v) for q in queries for v in variants}
done = Counter()
if os.path.exists(sys.argv[2]):
    for l in open(sys.argv[2]):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue  # part-written tail line; sanitize_runs repairs the file
        done[(r.get("query_id"), r.get("persona_id"), r.get("variant"))] += 1
dups = {k: c for k, c in done.items() if c > 1 and k in plan}
if dups:
    print(f"[warn] {len(dups)} duplicate (query,persona,variant) triples in runs.jsonl", file=sys.stderr)
print(len(plan - set(done)))
EOF
}

# Repair runs.jsonl in place: drop part-written tail lines (power loss during
# an append) and duplicate (query,persona,variant) records so the evaluators
# and resume logic see a clean file. No-op when the file is already healthy.
sanitize_runs() {
  [ -f "$RUNS" ] || return 0
  "$PY" - "$RUNS" <<'EOF'
import json, os, sys
path = sys.argv[1]
kept, seen, corrupt, dups = [], set(), 0, 0
for line in open(path, encoding="utf-8"):
    s = line.strip()
    if not s:
        continue
    try:
        r = json.loads(s)
    except json.JSONDecodeError:
        corrupt += 1
        continue
    key = (r.get("query_id"), r.get("persona_id"), r.get("variant"))
    if key in seen:
        dups += 1
        continue
    seen.add(key)
    kept.append(s)
if corrupt or dups:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(kept) + ("\n" if kept else ""))
    os.replace(tmp, path)
    print(f"   repaired runs.jsonl: kept {len(kept)} runs, dropped {corrupt} corrupt + {dups} duplicate lines")
EOF
}

if [ "$DRY" = 1 ]; then
  echo "== DRY RUN: plan preview"
  sanitize_runs
  "$PY" scripts/run_benchmark.py --config "$CONFIG" --model "$AGENT_MODEL" \
    --resume --dry-run ${LIMIT:+--limit "$LIMIT"}
  echo "   remaining agent jobs: $(remaining_jobs)"
  exit 0
fi

echo "== [1/5] Agent runs (model=$AGENT_MODEL${LIMIT:+, smoke limit=$LIMIT})"
sanitize_runs
EXPECTED_REMAINING=$(remaining_jobs)
[ -n "$LIMIT" ] && [ "$EXPECTED_REMAINING" -gt "$LIMIT" ] && EXPECTED_REMAINING=$LIMIT
PASS=1
while [ "$EXPECTED_REMAINING" -gt 0 ]; do
  if [ "$PASS" -gt "$MAX_AGENT_PASSES" ]; then
    echo "   still $EXPECTED_REMAINING jobs failing after $MAX_AGENT_PASSES passes."
    echo "   Inspect $OUT_DIR/logs/agent_runs.log, then re-invoke this script to continue."
    exit 1
  fi
  echo "   pass $PASS: $EXPECTED_REMAINING jobs to run"
  "$PY" scripts/run_benchmark.py --config "$CONFIG" --model "$AGENT_MODEL" \
    --resume ${LIMIT:+--limit "$LIMIT"} 2>&1 | tee -a "$OUT_DIR/logs/agent_runs.log" | tail -3
  NEW_REMAINING=$(remaining_jobs)
  [ -n "$LIMIT" ] && break   # smoke mode: one pass is the point
  if [ "$NEW_REMAINING" -ge "$EXPECTED_REMAINING" ]; then
    echo "   pass $PASS made no progress ($NEW_REMAINING remaining) - likely a"
    echo "   persistent error (quota/key). See $OUT_DIR/logs/agent_runs.log"; exit 1
  fi
  EXPECTED_REMAINING=$NEW_REMAINING
  PASS=$((PASS + 1))
done
RUN_COUNT=$(grep -c . "$RUNS")
echo "   agent runs complete: $RUN_COUNT records in runs.jsonl"

# Post-stage gate: complete = one row per run, no embedded per-row error.
# (The evaluators themselves resume incrementally; this is the safety net.)
stage_complete() {  # $1 = scores file
  "$PY" - "$1" "$RUN_COUNT" <<'EOF'
import json, os, sys
path, expected = sys.argv[1], int(sys.argv[2])
if not os.path.exists(path):
    print("missing"); sys.exit(0)
rows = [json.loads(l) for l in open(path) if l.strip()]
errs = sum(1 for r in rows if r.get("error"))
if len(rows) != expected: print(f"incomplete ({len(rows)}/{expected} rows)")
elif errs: print(f"has {errs} error rows")
else: print("ok")
EOF
}

# Judge stages resume internally: previously-scored rows are kept, error rows
# are cleaned and retried, and only what's missing is judged. A fully complete
# stage is a ~2s no-op with zero API calls, so we always invoke them.
run_judge() {  # $1 = step label, $2 = script, $3 = scores file
  local force_flag=""
  if [ "$FORCE_EVAL" = 1 ]; then force_flag="--force"; fi
  echo "== [$1] $2 (model=$JUDGE_MODEL)"
  "$PY" "scripts/$2" --config "$CONFIG" --model "$JUDGE_MODEL" $force_flag \
    2>&1 | tee "$OUT_DIR/logs/$2.log" | tail -2
  local status
  status=$(stage_complete "$3")
  if [ "$status" != "ok" ]; then
    echo "   WARNING: $2 output is '$status' after this pass."
    echo "   Re-invoke this script; the stage resumes and retries only what failed."
    exit 1
  fi
}

run_judge "2/5" evaluate_fanout_queries.py    "$OUT_DIR/fanout_scores.jsonl"
run_judge "3/5" evaluate_retrieval_results.py "$OUT_DIR/retrieval_scores.jsonl"
run_judge "4/5" evaluate_final_responses.py   "$OUT_DIR/final_response_scores.jsonl"

echo "== [5/5] Summaries + manifest"
"$PY" scripts/summarize_results.py --config "$CONFIG" 2>&1 | tee "$OUT_DIR/logs/summarize.log" | tail -2

"$PY" - "$OUT_DIR" "$AGENT_MODEL" "$JUDGE_MODEL" "$STARTED_AT" "$CONFIG" <<'EOF'
import json, os, subprocess, sys, datetime
out_dir, agent_model, judge_model, started_at, config = sys.argv[1:6]
def sh(*a): return subprocess.run(a, capture_output=True, text=True).stdout.strip()
def rows(name):
    p = os.path.join(out_dir, name)
    return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else None
manifest = {
    "experiment": "placement_ablation_v2 (C1 rerun, benchmark v2, 72 pairs x 6 variants)",
    "config": config,
    "git_commit": sh("git", "rev-parse", "HEAD"),
    "git_dirty": bool(sh("git", "status", "--porcelain")),
    "agent_model": agent_model,
    "judge_model": judge_model,
    "started_at_utc": started_at,
    "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "artifact_rows": {n: rows(n) for n in (
        "runs.jsonl", "fanout_scores.jsonl", "retrieval_scores.jsonl", "final_response_scores.jsonl")},
}
path = os.path.join(out_dir, "manifest.json")
json.dump(manifest, open(path, "w"), indent=2)
print(f"   manifest -> {path}")
print(json.dumps(manifest["artifact_rows"], indent=2))
EOF

echo
if [ "$AGENT_MODEL" = "$BASELINE_AGENT_MODEL" ]; then
  echo "== DONE (baseline arm: $AGENT_MODEL). Next: open"
  echo "   notebooks/placement_ablation_analysis_final.ipynb and Run-All"
  echo "   (RESULTS_DIR already points at $OUT_DIR)."
else
  echo "== DONE (comparison arm: $AGENT_MODEL) -> $OUT_DIR"
  echo "   Compare against the baseline by pointing the notebook's RESULTS_DIR"
  echo "   here, or via the MODELS map in scripts/make_paper_figures.py."
fi
