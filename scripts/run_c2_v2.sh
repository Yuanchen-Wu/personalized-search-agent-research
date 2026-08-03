#!/usr/bin/env bash
# One-command C2 rerun: fixed fanout scaling (k = 1,2,4,8, nested prefixes)
# on the ORIGINAL benchmark v1 data, with pinned models.
#
# Design (see configs/fixed_fanout_scaling_v2.yaml): the SYNTHESIZER and the
# JUDGE are both pinned to gemini-3.5-flash — matching what the original C2's
# flash-latest alias resolved to at run/scoring time (pre 2026-07-21) — so the
# PLANNER model (which writes the ordered 8-query fanout plan) is the sole
# experimental variable and results are judge-consistent with the original C2.
#
#     bash scripts/run_c2_v2.sh                                  # baseline planner: gemini-3.5-flash
#     bash scripts/run_c2_v2.sh --planner-model gemini-3.6-flash # comparison arm, one per model
#
# Baseline lands in outputs/fixed_fanout_scaling_v2/; other planner models get
# sibling dirs outputs/fixed_fanout_scaling_v2_<model>/ via derived configs,
# so arms never mix and each resumes independently.
#
# Safe to interrupt and re-invoke at any point: the runner resumes completed
# (query, persona, method, seed) runs, judge sub-stages append per-row and
# resume, part-written tail lines are repaired, and the evidence gate refuses
# to judge an arm whose retrieval silently failed.
#
# Options:
#   --planner-model NAME  Planner model for this arm (default gemini-3.5-flash,
#                         the model the original C2 resolved to).
#   --smoke               3 pairs x 4 k-conditions = 12 runs end-to-end.
#   --dry-run             Preflight + plan preview only; no API calls.
#   --force-eval          Discard this arm's judge scores and re-judge.
#   --retry-empty-searches  Purge failed (empty-result) searches from the
#                         search cache and strip runs missing evidence --
#                         fully evidence-free OR missing an executed branch's
#                         results (partial quota failures) -- plus their stale
#                         judge rows, so resume re-searches and re-runs exactly
#                         those. Combine with --dry-run to preview the surgery
#                         without changing any files.
#   AGENT_WORKERS=N       Concurrent pairs (default 6). Cache and run-log
#                         appends are lock-guarded; per-provider rate budgets
#                         stay correct at any worker count.
#   GEMINI_MAX_RPM=N      Shared pace for planner+synthesis calls (default 150).
#   TAVILY_MAX_RPM=N      Shared pace for searches (default 90 = dev-key safe).
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_CONFIG=configs/fixed_fanout_scaling_v2.yaml
BASE_OUT_DIR=outputs/fixed_fanout_scaling_v2
PY=.venv/bin/python
BASELINE_PLANNER="gemini-3.5-flash"
PLANNER_MODEL="$BASELINE_PLANNER"
MAX_AGENT_PASSES="${MAX_AGENT_PASSES:-3}"
AGENT_WORKERS="${AGENT_WORKERS:-6}"
export GEMINI_MAX_RPM="${GEMINI_MAX_RPM:-150}"
export TAVILY_MAX_RPM="${TAVILY_MAX_RPM:-90}"
EMPTY_EVIDENCE_TOLERANCE_PCT="${EMPTY_EVIDENCE_TOLERANCE_PCT:-2}"

LIMIT="" DRY=0 FORCE_EVAL=0 RETRY_EMPTY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --planner-model) [ $# -ge 2 ] || { echo "--planner-model needs a value"; exit 2; }
                     PLANNER_MODEL="$2"; shift ;;
    --planner-model=*) PLANNER_MODEL="${1#*=}" ;;
    --smoke) LIMIT=3 ;;
    --dry-run) DRY=1 ;;
    --force-eval) FORCE_EVAL=1 ;;
    --retry-empty-searches) RETRY_EMPTY=1 ;;
    *) echo "unknown option: $1 (see header of $0)"; exit 2 ;;
  esac
  shift
done

if [ "$PLANNER_MODEL" = "$BASELINE_PLANNER" ]; then
  OUT_DIR="$BASE_OUT_DIR"; CONFIG="$BASE_CONFIG"
else
  MODEL_TAG=$(printf '%s' "$PLANNER_MODEL" | tr -c 'a-zA-Z0-9._-' '-')
  OUT_DIR="${BASE_OUT_DIR}_${MODEL_TAG}"
  CONFIG="$OUT_DIR/config.yaml"
fi
RUNS="$OUT_DIR/runs.jsonl"
mkdir -p "$OUT_DIR/logs"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo "== [0/4] Preflight (planner=$PLANNER_MODEL, synthesizer+judge=PINNED gemini-3.5-flash)"
if [ ! -x "$PY" ]; then
  python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
fi
for f in "$BASE_CONFIG" data/synthetic_queries_v1.jsonl data/synthetic_personas_v1.jsonl; do
  [ -f "$f" ] || { echo "   MISSING: $f"; exit 1; }
done

if [ "$CONFIG" != "$BASE_CONFIG" ]; then
  "$PY" - "$BASE_CONFIG" "$OUT_DIR" "$MODEL_TAG" "$PLANNER_MODEL" "$CONFIG" <<'EOF'
import sys, yaml
base_path, out_dir, tag, planner, dst = sys.argv[1:6]
cfg = yaml.safe_load(open(base_path))
base_prefix = cfg["outputs"]["run_dir"]
cfg["experiment_name"] = f'{cfg["experiment_name"]}_{tag}'
cfg["models"]["planner"] = planner   # synthesizer + evaluator stay pinned
cfg["outputs"] = {k: v.replace(base_prefix, out_dir) for k, v in cfg["outputs"].items()}
with open(dst, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"   planner-arm config -> {dst}")
EOF
fi

KEYCHECK_RC=0
KEYCHECK_OUT=$("$PY" - 2>&1 <<'EOF'
import sys
sys.path.insert(0, "src")
try:
    from search_agent.config import get_gemini_api_key, get_tavily_api_key
except Exception as e:
    print(f"environment problem (NOT a key problem): {e}"); sys.exit(2)
missing = []
for name, fn in (("GEMINI_API_KEY", get_gemini_api_key), ("TAVILY_API_KEY", get_tavily_api_key)):
    try:
        fn()
    except Exception:
        missing.append(name)
if missing:
    print("missing: " + ", ".join(missing)); sys.exit(1)
print("present (values not shown)")
EOF
) || KEYCHECK_RC=$?
if [ "$KEYCHECK_RC" = 0 ]; then echo "   API keys: $KEYCHECK_OUT"
elif [ "$DRY" = 1 ]; then echo "   API keys: $KEYCHECK_OUT (ok for --dry-run)"
else echo "   API key check failed: $KEYCHECK_OUT"; exit 1; fi

"$PY" scripts/validate_fixed_fanout_setup.py --config "$CONFIG" \
  2>&1 | tee "$OUT_DIR/logs/validate.log" | grep -E "\[FAIL\]|WARNING" || true
if grep -q "\[FAIL\]" "$OUT_DIR/logs/validate.log"; then
  echo "   validation FAILed - see $OUT_DIR/logs/validate.log"; exit 1
fi
echo "   validation: no FAILs"

# Remaining jobs = pairs x methods minus completed (query, persona, method, seed).
remaining_jobs() {
  "$PY" - "$CONFIG" "$RUNS" <<'EOF'
import json, os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
seed = cfg.get("reproducibility", {}).get("seed", 42)
queries = [json.loads(l) for l in open(cfg["data"]["queries_path"]) if l.strip()]
plan = {(q["query_id"], q["persona_id"], m, seed) for q in queries for m in cfg["methods"]}
done = set()
if os.path.exists(sys.argv[2]):
    for l in open(sys.argv[2]):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        done.add((r.get("query_id"), r.get("persona_id"), r.get("method"), r.get("seed")))
print(len(plan - done))
EOF
}

sanitize_runs() {
  [ -f "$RUNS" ] || return 0
  "$PY" - "$RUNS" <<'EOF'
import json, os, sys
path = sys.argv[1]
kept, seen, corrupt, dups = [], set(), 0, 0
for line in open(path, encoding="utf-8"):
    s = line.strip()
    if not s: continue
    try:
        r = json.loads(s)
    except json.JSONDecodeError:
        corrupt += 1; continue
    key = (r.get("query_id"), r.get("persona_id"), r.get("method"), r.get("seed"))
    if key in seen:
        dups += 1; continue
    seen.add(key); kept.append(s)
if corrupt or dups:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(kept) + ("\n" if kept else ""))
    os.replace(tmp, path)
    print(f"   repaired runs.jsonl: kept {len(kept)}, dropped {corrupt} corrupt + {dups} duplicate lines")
EOF
}

retry_empty_searches() {
  local mode="${1:-apply}"
  "$PY" - "$CONFIG" "$mode" <<'EOF'
import json, os, sys, yaml
from collections import Counter

cfg = yaml.safe_load(open(sys.argv[1]))
apply = sys.argv[2] == "apply"
out = cfg["outputs"]
cache_path, runs_path = out["search_cache_path"], out["runs_path"]

def did(past, infinitive):
    return past if apply else f"would {infinitive}"

def rewrite(path, lines):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    os.replace(tmp, path)

# 1. Failed searches (quota exhaustion, rate-limit soft-fails) are cached as
#    empty results and served as cache hits forever after; purge them so
#    resume actually re-searches instead of replaying the failure.
failed_queries, kept_cache = set(), []
if os.path.exists(cache_path):
    for line in open(cache_path, encoding="utf-8"):
        s = line.strip()
        if not s: continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            continue  # torn tail line; dropped by the rewrite
        if rec.get("results"):
            kept_cache.append(s)
        else:
            failed_queries.add(rec.get("query"))
if failed_queries:
    if apply:
        rewrite(cache_path, kept_cache)
    print(f"   {did('purged', 'purge')} {len(failed_queries)} empty (failed-search) cache entries")

# 2. Strip runs with incomplete evidence: no results at all, a branch matching
#    a purged cache entry, or an executed branch that contributed zero results
#    (partial failures -- raw_search_results is attributed per branch_query).
if not os.path.exists(runs_path):
    print("   no runs.jsonl yet; nothing to strip"); raise SystemExit
runs = [json.loads(l) for l in open(runs_path, encoding="utf-8") if l.strip()]

def tainted(r):
    raw = r.get("raw_search_results") or []
    if not raw:
        return True
    branches = r.get("executed_fanout_prefix") or r.get("fanout_branches") or []
    covered = {res.get("branch_query") for res in raw}
    return any(b.get("query") in failed_queries or b.get("query") not in covered
               for b in branches)

keep = [r for r in runs if not tainted(r)]
dropped = [r for r in runs if tainted(r)]
if not dropped:
    print("   no evidence-incomplete runs to strip"); raise SystemExit
if apply:
    rewrite(runs_path, [json.dumps(r, ensure_ascii=False) for r in keep])
by_method = ", ".join(f"{m}:{n}" for m, n in sorted(Counter(r.get("method") for r in dropped).items()))
print(f"   {did('stripped', 'strip')} {len(dropped)} evidence-incomplete runs ({by_method}); "
      f"{len(keep)} kept -- resume re-runs the stripped jobs")

# 3. Judge rows for stripped runs are stale (re-runs get fresh run_ids); purge
#    them so the judge's per-run_id resume re-scores exactly the re-run rows.
valid = {r.get("run_id") for r in keep}
for key in ("fanout_scores_path", "retrieval_scores_path", "final_response_scores_path"):
    p = out.get(key)
    if not p or not os.path.exists(p): continue
    rows, orphans = [], 0
    for l in open(p, encoding="utf-8"):
        if not l.strip(): continue
        try:
            row = json.loads(l)
        except json.JSONDecodeError:
            orphans += 1; continue
        if row.get("run_id") in valid:
            rows.append(l.rstrip("\n"))
        else:
            orphans += 1
    if orphans:
        if apply:
            rewrite(p, rows)
        print(f"   {did('purged', 'purge')} {orphans} stale judge rows from {os.path.basename(p)}")
EOF
}

if [ "$DRY" = 1 ]; then
  echo "== DRY RUN"
  sanitize_runs
  [ "$RETRY_EMPTY" = 1 ] && retry_empty_searches report
  "$PY" scripts/run_fixed_fanout_benchmark.py --config "$CONFIG" --dry_run ${LIMIT:+--limit "$LIMIT"} | tail -5
  echo "   remaining jobs (before any surgery): $(remaining_jobs)"
  exit 0
fi

echo "== [1/4] Runner (planner=$PLANNER_MODEL, workers=$AGENT_WORKERS, gemini rpm=$GEMINI_MAX_RPM, tavily rpm=$TAVILY_MAX_RPM${LIMIT:+, smoke limit=$LIMIT pairs})"
sanitize_runs
[ "$RETRY_EMPTY" = 1 ] && retry_empty_searches apply
EXPECTED_REMAINING=$(remaining_jobs)
PASS=1
while [ "$EXPECTED_REMAINING" -gt 0 ]; do
  if [ "$PASS" -gt "$MAX_AGENT_PASSES" ]; then
    echo "   still $EXPECTED_REMAINING jobs failing after $MAX_AGENT_PASSES passes; see $OUT_DIR/logs/runner.log"
    exit 1
  fi
  echo "   pass $PASS: $EXPECTED_REMAINING jobs to run"
  "$PY" scripts/run_fixed_fanout_benchmark.py --config "$CONFIG" --workers "$AGENT_WORKERS" \
    ${LIMIT:+--limit "$LIMIT"} 2>&1 | tee -a "$OUT_DIR/logs/runner.log" | tail -3
  NEW_REMAINING=$(remaining_jobs)
  [ -n "$LIMIT" ] && break
  if [ "$NEW_REMAINING" -ge "$EXPECTED_REMAINING" ]; then
    echo "   pass $PASS made no progress ($NEW_REMAINING remaining) - see $OUT_DIR/logs/runner.log"
    exit 1
  fi
  EXPECTED_REMAINING=$NEW_REMAINING
  PASS=$((PASS + 1))
done
RUN_COUNT=$(grep -c . "$RUNS")
echo "   runner complete: $RUN_COUNT records in runs.jsonl"

if ! "$PY" - "$RUNS" "$EMPTY_EVIDENCE_TOLERANCE_PCT" <<'EOF'
import json, sys
runs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
def incomplete(r):
    raw = r.get("raw_search_results") or []
    if not raw:
        return True
    covered = {res.get("branch_query") for res in raw}
    branches = r.get("executed_fanout_prefix") or r.get("fanout_branches") or []
    return any(b.get("query") not in covered for b in branches)
empty = sum(1 for r in runs if not (r.get("raw_search_results") or []))
affected = sum(1 for r in runs if incomplete(r))
pct = 100.0 * affected / max(1, len(runs))
print(f"   evidence coverage: {len(runs) - affected}/{len(runs)} runs fully evidenced "
      f"({empty} evidence-free, {affected - empty} missing a searched branch; {pct:.1f}% affected)")
sys.exit(1 if pct > float(sys.argv[2]) else 0)
EOF
then
  echo "   EVIDENCE GATE FAILED (Tavily quota exhausted or rate limited?). Fix the key/quota"
  echo "   or TAVILY_MAX_RPM, then re-invoke with --retry-empty-searches."
  exit 1
fi

echo "== [2/4] Judging (fanout + retrieval + final; resumes internally)"
FORCE_FLAG=""
if [ "$FORCE_EVAL" = 1 ]; then FORCE_FLAG="--force"; fi
"$PY" scripts/evaluate_fixed_fanout.py --config "$CONFIG" $FORCE_FLAG \
  2>&1 | tee "$OUT_DIR/logs/evaluate.log" | tail -4
for f in fanout_scores retrieval_scores final_response_scores; do
  status=$("$PY" - "$OUT_DIR/$f.jsonl" "$RUN_COUNT" <<'EOF'
import json, os, sys
path, expected = sys.argv[1], int(sys.argv[2])
if not os.path.exists(path):
    print("missing"); sys.exit(0)
rows = [json.loads(l) for l in open(path) if l.strip()]
bad = sum(1 for r in rows if r.get("error") or not r.get("scores"))
if len(rows) != expected: print(f"incomplete ({len(rows)}/{expected})")
elif bad: print(f"has {bad} bad rows")
else: print("ok")
EOF
)
  if [ "$status" != "ok" ]; then
    echo "   WARNING: $f is '$status' — re-invoke to resume judging."; exit 1
  fi
done
echo "   judging complete and clean"

echo "== [3/4] Summaries"
"$PY" scripts/summarize_fixed_fanout.py --config "$CONFIG" 2>&1 | tee "$OUT_DIR/logs/summarize.log" | tail -2

echo "== [4/4] Manifest"
"$PY" - "$OUT_DIR" "$CONFIG" "$STARTED_AT" <<'EOF'
import datetime, json, os, subprocess, sys, yaml
out_dir, config, started_at = sys.argv[1:4]
cfg = yaml.safe_load(open(config))
def sh(*a): return subprocess.run(a, capture_output=True, text=True).stdout.strip()
def rows(n):
    p = os.path.join(out_dir, n)
    return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else None
manifest = {
    "experiment": cfg["experiment_name"] + " (C2 rerun, benchmark v1 data, nested-prefix k=1/2/4/8)",
    "config": config,
    "git_commit": sh("git", "rev-parse", "HEAD"),
    "models": cfg["models"],
    "seed": cfg["reproducibility"]["seed"],
    "started_at_utc": started_at,
    "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "artifact_rows": {n: rows(n) for n in (
        "runs.jsonl", "fanout_plans.jsonl", "fanout_scores.jsonl",
        "retrieval_scores.jsonl", "final_response_scores.jsonl")},
}
path = os.path.join(out_dir, "manifest.json")
json.dump(manifest, open(path, "w"), indent=2)
print(f"   manifest -> {path}")
print(json.dumps(manifest["artifact_rows"]))
EOF

echo
echo "== DONE (planner arm: $PLANNER_MODEL) -> $OUT_DIR"
