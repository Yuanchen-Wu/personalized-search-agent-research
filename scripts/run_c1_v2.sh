#!/usr/bin/env bash
# One-command C1 rerun: full 6-variant placement ablation on benchmark v2.
#
# On the experiment laptop (needs GEMINI_API_KEY + TAVILY_API_KEY in .env):
#
#     bash scripts/run_c1_v2.sh
#
# That single invocation bootstraps .venv if needed, validates the setup, runs
# all 432 agent runs (72 pairs x 6 variants), judges fanout / retrieval /
# final-response (post-split answer-quality + evidence-faithfulness pair),
# summarizes, and writes a provenance manifest. Everything lands in
# outputs/placement_ablation_v2/, which is exactly what
# notebooks/placement_ablation_analysis_final.ipynb reads (RESULTS_DIR).
#
# The command is safe to interrupt and re-invoke at any point (battery death,
# quota exhaustion, Ctrl-C): every completed agent run and every completed
# judge call is already on disk (appended + flushed as it finishes), and the
# next invocation repairs any part-written tail line and resumes from exactly
# what's left instead of re-running or duplicating finished work.
#
# Options:
#   --smoke        End-to-end shakeout on the first 12 jobs (2 queries x 6
#                  variants) before committing to the full batch. A later full
#                  run resumes both the agent and judge stages from there.
#   --dry-run      Preflight + plan preview only; no API calls.
#   --force-eval   Re-run judge stages even if outputs look complete.
#   AGENT_MODEL=   Agent model (default gemini-3.5-flash, as in C2/C3).
#   JUDGE_MODEL=   Judge model (default gemini-flash-latest, matching all
#                  prior scoring).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/placement_ablation_v2.yaml
OUT_DIR=outputs/placement_ablation_v2
RUNS="$OUT_DIR/runs.jsonl"
PY=.venv/bin/python
AGENT_MODEL="${AGENT_MODEL:-gemini-3.5-flash}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-flash-latest}"
MAX_AGENT_PASSES="${MAX_AGENT_PASSES:-3}"

LIMIT="" DRY=0 FORCE_EVAL=0
for arg in "$@"; do
  case "$arg" in
    --smoke) LIMIT=12 ;;
    --dry-run) DRY=1 ;;
    --force-eval) FORCE_EVAL=1 ;;
    *) echo "unknown option: $arg (see header of $0)"; exit 2 ;;
  esac
done

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

for f in "$CONFIG" data/v2/synthetic_queries_v2.jsonl data/v2/synthetic_personas_v2.jsonl; do
  [ -f "$f" ] || { echo "   MISSING: $f (pull latest main)"; exit 1; }
done

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
echo "== DONE. Next: open notebooks/placement_ablation_analysis_final.ipynb and Run-All"
echo "   (RESULTS_DIR already points at $OUT_DIR)."
