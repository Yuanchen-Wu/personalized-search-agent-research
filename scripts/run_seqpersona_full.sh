#!/usr/bin/env bash
# One-command FULL seqpersona_k8 vs fixed_k8 experiment (all 72 pairs), with
# BOTH judge systems:
#
#   Judge 1 (absolute, the common instrument used by C1 / C2 / old C3):
#       evaluate_fixed_fanout.py with the PINNED gemini-3.5-flash evaluator ->
#       fanout / retrieval / final 1-5 scores for each arm separately.
#   Judge 2 (comparative, the new instrument):
#       judge_pairwise.py, position-swapped rubric-aware head-to-head between
#       the two arms' final answers (calibration anchors: ~50% = null,
#       ~86% = +0.3-class, ~100% = +0.9-class; see run_pairwise_pilot.sh).
#
# Decision rule this feeds (see final DECISION SUMMARY output): if the
# absolute deltas are small but the pairwise CI clears 50% in seqpersona's
# favor, report the comparison judge only; if both instruments agree, report
# both.
#
#     bash scripts/run_seqpersona_full.sh                # full experiment
#     bash scripts/run_seqpersona_full.sh --dry-run      # no API calls: plan + resume state
#     bash scripts/run_seqpersona_full.sh --smoke        # 3 pairs end-to-end plumbing test
#
# What it reuses (never re-paid):
#   - The 15 pilot seqpersona runs already in outputs/seqpersona_k8_v1/ (the
#     runner resumes past completed (query, persona, method, seed) keys).
#   - The CLEAN fixed_k8 rows of the canonical C2 v2 arm
#     (outputs/fixed_fanout_scaling_v2_gemini-3.6-flash), plus their existing
#     judge scores and fanout plans. The ~15 evidence-incomplete rows from the
#     Tavily quota incident (missing search branches; judging seqpersona
#     against them would be unfairly favorable) are re-run FRESH into this
#     experiment's own dir. Canonical C2 outputs are NEVER modified.
#   - The 15 pilot pairwise verdicts, but ONLY where an answer-length
#     fingerprint proves both underlying answers are unchanged.
#
# Safe to interrupt and re-invoke at any point: every stage resumes (runner by
# run key, judges by run_id / query_id, seeding is key-deduplicated).
#
# Options:
#   --planner-model NAME  Agent-side planner/critic model for BOTH arms
#                         (default gemini-3.6-flash, matching the canonical C2
#                         arm and the pilot). Non-default models get sibling
#                         output dirs and skip canonical-arm reuse unless a
#                         matching fixed arm exists. Synthesizer and absolute
#                         judge stay PINNED to gemini-3.5-flash either way.
#   --smoke               First 3 pairs end-to-end (fresh fixed rows, no
#                         canonical reuse) to prove the plumbing.
#   --dry-run             Preflight + resume/seeding report only; no API calls.
#   --force-eval          Discard both arms' absolute judge scores and re-judge.
#   --retry-empty-searches  Purge failed (empty-result) cached searches and
#                         strip evidence-incomplete runs + their stale judge
#                         rows from BOTH arms' experiment files, so resume
#                         re-searches and re-runs exactly those. Combine with
#                         --dry-run to preview the surgery.
#   AGENT_WORKERS=N       Concurrent pairs for the runner (default 6).
#   GEMINI_MAX_RPM=N      Shared Gemini pacing for planner+synthesis (default 150).
#   TAVILY_MAX_RPM=N      Shared Tavily pacing (default 90 = dev-key safe).
#   PAIRWISE_WORKERS=N    Pairwise judge concurrency (default 12).
#   PAIRWISE_SAMPLES=N    Pairwise votes per presentation order (default 2 -> 4 votes/item).
#   PAIRWISE_JUDGE_MODEL  Pairwise judge model (default gemini-3.6-flash, the
#                         model the instrument was calibrated with).
#   MAX_AGENT_PASSES=N    Runner retry passes per arm (default 3).
#   EMPTY_EVIDENCE_TOLERANCE_PCT  Evidence-gate threshold (default 2).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
BASELINE_PLANNER="gemini-3.6-flash"
PLANNER_MODEL="$BASELINE_PLANNER"
AGENT_WORKERS="${AGENT_WORKERS:-6}"
export GEMINI_MAX_RPM="${GEMINI_MAX_RPM:-150}"
export TAVILY_MAX_RPM="${TAVILY_MAX_RPM:-90}"
PAIRWISE_WORKERS="${PAIRWISE_WORKERS:-12}"
PAIRWISE_SAMPLES="${PAIRWISE_SAMPLES:-2}"
PAIRWISE_JUDGE_MODEL="${PAIRWISE_JUDGE_MODEL:-gemini-3.6-flash}"
MAX_AGENT_PASSES="${MAX_AGENT_PASSES:-3}"
EMPTY_EVIDENCE_TOLERANCE_PCT="${EMPTY_EVIDENCE_TOLERANCE_PCT:-2}"

SEQ_BASE_CONFIG=configs/seqpersona_k8_v1.yaml
SEQ_BASE_DIR=outputs/seqpersona_k8_v1
CANON_FIXED_DIR=outputs/fixed_fanout_scaling_v2_gemini-3.6-flash
EXP_BASE_DIR=outputs/seqpersona_vs_fixed_v1
PILOT_PAIRWISE=outputs/seqpersona_pilot/pairwise_pilot.jsonl
QUERIES=data/synthetic_queries_v1.jsonl
PERSONAS=data/synthetic_personas_v1.jsonl

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

# --- arm layout (sibling dirs for non-default planner, like run_c2_v2.sh) ----
if [ "$PLANNER_MODEL" = "$BASELINE_PLANNER" ]; then
  SEQ_CONFIG="$SEQ_BASE_CONFIG"; SEQ_DIR="$SEQ_BASE_DIR"; EXP_DIR="$EXP_BASE_DIR"
else
  MODEL_TAG=$(printf '%s' "$PLANNER_MODEL" | tr -c 'a-zA-Z0-9._-' '-')
  SEQ_DIR="${SEQ_BASE_DIR}_${MODEL_TAG}"
  SEQ_CONFIG="$SEQ_DIR/config.yaml"
  EXP_DIR="${EXP_BASE_DIR}_${MODEL_TAG}"
  CANON_FIXED_DIR="outputs/fixed_fanout_scaling_v2_${MODEL_TAG}"
fi
SEQ_RUNS="$SEQ_DIR/runs.jsonl"
FIXED_DIR="$EXP_DIR/fixed_k8"
FIXED_CONFIG="$FIXED_DIR/config.yaml"
FIXED_RUNS="$FIXED_DIR/runs.jsonl"
PAIRWISE_OUT="$EXP_DIR/pairwise_seqpersona_vs_fixed.jsonl"
mkdir -p "$EXP_DIR/logs" "$FIXED_DIR" "$SEQ_DIR"
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo "== [0/6] Preflight (planner=$PLANNER_MODEL, synthesizer+absolute judge=PINNED gemini-3.5-flash, pairwise judge=$PAIRWISE_JUDGE_MODEL)"
if [ ! -x "$PY" ]; then
  python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
fi
for f in "$SEQ_BASE_CONFIG" "$QUERIES" "$PERSONAS" scripts/judge_pairwise.py; do
  [ -f "$f" ] || { echo "   MISSING: $f"; exit 1; }
done

# Derived configs. Seqpersona sibling arm for a non-default planner; the fixed
# comparator config is always experiment-local (methods=[fixed_k8], own caches
# and score files) and inherits the canonical arm's config when available so
# the redo rows are produced under the exact same settings.
if [ "$SEQ_CONFIG" != "$SEQ_BASE_CONFIG" ] && [ ! -f "$SEQ_CONFIG" ]; then
  "$PY" - "$SEQ_BASE_CONFIG" "$SEQ_DIR" "$PLANNER_MODEL" "$SEQ_CONFIG" <<'EOF'
import sys, yaml
base_path, out_dir, planner, dst = sys.argv[1:5]
cfg = yaml.safe_load(open(base_path))
base_prefix = cfg["outputs"]["run_dir"]
cfg["experiment_name"] = f'{cfg["experiment_name"]}_{out_dir.rsplit("_", 1)[-1]}'
cfg["models"]["planner"] = planner   # synthesizer + evaluator stay pinned
cfg["outputs"] = {k: v.replace(base_prefix, out_dir) for k, v in cfg["outputs"].items()}
with open(dst, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"   seqpersona planner-arm config -> {dst}")
EOF
fi
"$PY" - "$CANON_FIXED_DIR/config.yaml" configs/fixed_fanout_scaling_v2.yaml \
        "$PLANNER_MODEL" "$FIXED_DIR" "$FIXED_CONFIG" <<'EOF'
import os, sys, yaml
canon_cfg, fallback_cfg, planner, out_dir, dst = sys.argv[1:6]
src = canon_cfg if os.path.exists(canon_cfg) else fallback_cfg
cfg = yaml.safe_load(open(src))
cfg["experiment_name"] = "seqpersona_vs_fixed_v1_fixed_k8"
cfg["methods"] = ["fixed_k8"]
cfg["fixed_fanout"] = {"candidate_pool_size": 8, "k_values": [8], "use_nested_prefixes": True}
cfg["models"]["planner"] = planner   # synthesizer + evaluator stay pinned
base_prefix = cfg["outputs"]["run_dir"]
cfg["outputs"] = {k: v.replace(base_prefix, out_dir) for k, v in cfg["outputs"].items()}
with open(dst, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"   fixed_k8 comparator config -> {dst} (base: {src})")
EOF

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

# --- shared helpers (config-driven; used for both arms) ----------------------
remaining_jobs() {  # $1 = config, $2 = runs.jsonl
  "$PY" - "$1" "$2" <<'EOF'
import json, os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
seed = cfg.get("reproducibility", {}).get("seed", 42)
queries = [json.loads(l) for l in open(cfg["data"]["queries_path"]) if l.strip()]
plan = set()
for q in queries:
    pid = (q.get("metadata") or {}).get("persona_id") or q.get("persona_id")
    for m in cfg["methods"]:
        plan.add((q["query_id"], pid, m, seed))
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

sanitize_runs() {  # $1 = runs.jsonl (drop torn tail lines + key duplicates)
  [ -f "$1" ] || return 0
  "$PY" - "$1" <<'EOF'
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
    print(f"   repaired {path}: kept {len(kept)}, dropped {corrupt} corrupt + {dups} duplicate lines")
EOF
}

retry_empty_searches() {  # $1 = config, $2 = apply|report  (same surgery as run_c2_v2.sh)
  "$PY" - "$1" "$2" <<'EOF'
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
failed_queries, kept_cache = set(), []
if os.path.exists(cache_path):
    for line in open(cache_path, encoding="utf-8"):
        s = line.strip()
        if not s: continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            continue
        if rec.get("results"):
            kept_cache.append(s)
        else:
            failed_queries.add(rec.get("query"))
if failed_queries:
    if apply:
        rewrite(cache_path, kept_cache)
    print(f"   {did('purged', 'purge')} {len(failed_queries)} empty (failed-search) cache entries from {cache_path}")
if not os.path.exists(runs_path):
    print(f"   no runs at {runs_path}; nothing to strip"); raise SystemExit
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
    print(f"   no evidence-incomplete runs in {runs_path}"); raise SystemExit
if apply:
    rewrite(runs_path, [json.dumps(r, ensure_ascii=False) for r in keep])
by_method = ", ".join(f"{m}:{n}" for m, n in sorted(Counter(r.get("method") for r in dropped).items()))
print(f"   {did('stripped', 'strip')} {len(dropped)} evidence-incomplete runs ({by_method}) from {runs_path}; resume re-runs them")
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

evidence_gate() {  # $1 = runs.jsonl
  "$PY" - "$1" "$EMPTY_EVIDENCE_TOLERANCE_PCT" <<'EOF'
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
}

run_arm() {  # $1 = label, $2 = config, $3 = runs.jsonl, $4 = log file
  local label="$1" config="$2" runs="$3" log="$4"
  sanitize_runs "$runs"
  local expected pass=1 new
  expected=$(remaining_jobs "$config" "$runs")
  while [ "$expected" -gt 0 ]; do
    if [ "$pass" -gt "$MAX_AGENT_PASSES" ]; then
      echo "   [$label] still $expected jobs failing after $MAX_AGENT_PASSES passes; see $log"
      exit 1
    fi
    echo "   [$label] pass $pass: $expected jobs to run"
    "$PY" scripts/run_fixed_fanout_benchmark.py --config "$config" --workers "$AGENT_WORKERS" \
      ${LIMIT:+--limit "$LIMIT"} 2>&1 | tee -a "$log" | tail -3
    new=$(remaining_jobs "$config" "$runs")
    [ -n "$LIMIT" ] && break
    if [ "$new" -ge "$expected" ]; then
      echo "   [$label] pass $pass made no progress ($new remaining) - see $log"
      exit 1
    fi
    expected=$new
    pass=$((pass + 1))
  done
  echo "   [$label] runner complete: $(grep -c . "$runs" 2>/dev/null || echo 0) records in $runs"
}

# Seed the experiment-local fixed_k8 arm from the canonical C2 arm: copy CLEAN
# rows (full evidence only), their judge scores, and the shared fanout plans.
# Key-deduplicated and idempotent; never touches the canonical dir. Skipped in
# --smoke so the plumbing test exercises fresh fixed runs.
seed_fixed_arm() {  # $1 = apply|report
  "$PY" - "$CANON_FIXED_DIR" "$FIXED_DIR" "$1" <<'EOF'
import json, os, shutil, sys
canon_dir, fixed_dir, mode = sys.argv[1:4]
apply = mode == "apply"
canon_runs = os.path.join(canon_dir, "runs.jsonl")
if not os.path.exists(canon_runs):
    print(f"   no canonical fixed arm at {canon_runs}; the full fixed_k8 arm will run fresh")
    raise SystemExit
def rows(path):
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path, encoding="utf-8"):
        if l.strip():
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                continue
    return out
def tainted(r):
    raw = r.get("raw_search_results") or []
    if not raw:
        return True
    covered = {res.get("branch_query") for res in raw}
    branches = r.get("executed_fanout_prefix") or r.get("fanout_branches") or []
    return any(b.get("query") not in covered for b in branches)
canon = [r for r in rows(canon_runs) if r.get("method") == "fixed_k8"]
clean = [r for r in canon if not tainted(r)]
bad = [r for r in canon if tainted(r)]
key = lambda r: (r.get("query_id"), r.get("persona_id"), r.get("method"), r.get("seed"))
dst_runs = os.path.join(fixed_dir, "runs.jsonl")
have = {key(r) for r in rows(dst_runs)}
to_add = [r for r in clean if key(r) not in have]
print(f"   canonical fixed_k8: {len(clean)} clean rows reusable, {len(bad)} evidence-incomplete "
      f"(re-run fresh here): {sorted(r.get('query_id') for r in bad)}")
verb = "seeded" if apply else "would seed"
print(f"   {verb} {len(to_add)} clean rows into {dst_runs} ({len(have)} already present)")
if apply and to_add:
    with open(dst_runs, "a", encoding="utf-8") as fh:
        for r in to_add:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
plans_src, plans_dst = (os.path.join(d, "fanout_plans.jsonl") for d in (canon_dir, fixed_dir))
if os.path.exists(plans_src) and not os.path.exists(plans_dst):
    print(f"   {verb} fanout plans (redo pairs reuse the canonical 8-branch plans)")
    if apply:
        shutil.copyfile(plans_src, plans_dst)
present_ids = {r.get("run_id") for r in rows(dst_runs)} | {r.get("run_id") for r in to_add}
for name in ("fanout_scores.jsonl", "retrieval_scores.jsonl", "final_response_scores.jsonl"):
    src, dst = os.path.join(canon_dir, name), os.path.join(fixed_dir, name)
    have_ids = {r.get("run_id") for r in rows(dst)}
    add = [r for r in rows(src)
           if r.get("run_id") in present_ids and r.get("run_id") not in have_ids
           and not r.get("error") and r.get("scores")]
    if add:
        print(f"   {verb} {len(add)} existing judge rows into {name} (absolute judge resumes past them)")
        if apply:
            with open(dst, "a", encoding="utf-8") as fh:
                for r in add:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
EOF
}

# Reuse pilot pairwise verdicts ONLY where both answers are provably unchanged
# (length fingerprint on both arms' current final answers).
seed_pairwise() {  # $1 = apply|report
  "$PY" - "$PILOT_PAIRWISE" "$PAIRWISE_OUT" "$FIXED_RUNS" "$SEQ_RUNS" "$1" <<'EOF'
import json, os, sys
pilot_path, out_path, fixed_runs, seq_runs, mode = sys.argv[1:6]
apply = mode == "apply"
if not os.path.exists(pilot_path):
    print("   no pilot pairwise file; nothing to reuse"); raise SystemExit
def rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
ans = {}
for path, side in ((fixed_runs, "a"), (seq_runs, "b")):
    for r in rows(path):
        if path == fixed_runs and r.get("method") != "fixed_k8":
            continue
        ans.setdefault(r["query_id"], {})[side] = len(r.get("final_answer", ""))
if not any("a" in v for v in ans.values()):
    print("   fixed arm not assembled yet; pilot pairwise verdicts are checked for"
          " reuse when the pairwise stage runs")
    raise SystemExit
have = {r.get("query_id") for r in rows(out_path)}
add = []
for r in rows(pilot_path):
    q = r.get("query_id")
    cur = ans.get(q, {})
    if q not in have and cur.get("a") == r.get("len_a") and cur.get("b") == r.get("len_b"):
        add.append(r)
verb = "reused" if apply else "would reuse"
print(f"   {verb} {len(add)} pilot pairwise verdicts (answer fingerprints unchanged; "
      f"{len(have)} already in {out_path})")
if apply and add:
    with open(out_path, "a", encoding="utf-8") as fh:
        for r in add:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
EOF
}

# --- dry run -----------------------------------------------------------------
if [ "$DRY" = 1 ]; then
  echo "== DRY RUN (no API calls)"
  sanitize_runs "$SEQ_RUNS"; sanitize_runs "$FIXED_RUNS"
  if [ "$RETRY_EMPTY" = 1 ]; then
    retry_empty_searches "$SEQ_CONFIG" report
    retry_empty_searches "$FIXED_CONFIG" report
  fi
  "$PY" scripts/run_fixed_fanout_benchmark.py --config "$SEQ_CONFIG" --dry_run ${LIMIT:+--limit "$LIMIT"} | tail -5
  [ -z "$LIMIT" ] && seed_fixed_arm report
  echo "   seqpersona remaining jobs: $(remaining_jobs "$SEQ_CONFIG" "$SEQ_RUNS")"
  echo "   fixed_k8 remaining jobs (before seeding): $(remaining_jobs "$FIXED_CONFIG" "$FIXED_RUNS")"
  seed_pairwise report
  exit 0
fi

# --- stage 1: seqpersona_k8 arm ---------------------------------------------
echo "== [1/6] seqpersona_k8 arm (workers=$AGENT_WORKERS, gemini rpm=$GEMINI_MAX_RPM, tavily rpm=$TAVILY_MAX_RPM${LIMIT:+, smoke limit=$LIMIT pairs})"
[ "$RETRY_EMPTY" = 1 ] && retry_empty_searches "$SEQ_CONFIG" apply
run_arm seqpersona "$SEQ_CONFIG" "$SEQ_RUNS" "$EXP_DIR/logs/seqpersona_runner.log"
if ! evidence_gate "$SEQ_RUNS"; then
  echo "   EVIDENCE GATE FAILED for seqpersona (Tavily quota/rate-limited?). Fix the key or"
  echo "   TAVILY_MAX_RPM, then re-invoke with --retry-empty-searches."
  exit 1
fi

# --- stage 2: fixed_k8 comparator arm ---------------------------------------
echo "== [2/6] fixed_k8 comparator (reusing clean canonical C2 rows where possible)"
[ "$RETRY_EMPTY" = 1 ] && retry_empty_searches "$FIXED_CONFIG" apply
[ -z "$LIMIT" ] && seed_fixed_arm apply
run_arm fixed_k8 "$FIXED_CONFIG" "$FIXED_RUNS" "$EXP_DIR/logs/fixed_runner.log"
if ! evidence_gate "$FIXED_RUNS"; then
  echo "   EVIDENCE GATE FAILED for fixed_k8. Fix the key or TAVILY_MAX_RPM, then"
  echo "   re-invoke with --retry-empty-searches."
  exit 1
fi

# --- stage 3: absolute judge (the common C1/C2/C3 instrument) ---------------
echo "== [3/6] Absolute judging, both arms (pinned evaluator; resumes per run_id)"
FORCE_FLAG=""
if [ "$FORCE_EVAL" = 1 ]; then FORCE_FLAG="--force"; fi
"$PY" scripts/evaluate_fixed_fanout.py --config "$SEQ_CONFIG" $FORCE_FLAG \
  2>&1 | tee "$EXP_DIR/logs/evaluate_seqpersona.log" | tail -4
"$PY" scripts/evaluate_fixed_fanout.py --config "$FIXED_CONFIG" $FORCE_FLAG \
  2>&1 | tee "$EXP_DIR/logs/evaluate_fixed.log" | tail -4
for cfg in "$SEQ_CONFIG" "$FIXED_CONFIG"; do
  status=$("$PY" - "$cfg" <<'EOF'
import json, os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
out = cfg["outputs"]
runs = [json.loads(l) for l in open(out["runs_path"]) if l.strip()]
ids = {r.get("run_id") for r in runs}
problems = []
for key in ("fanout_scores_path", "retrieval_scores_path", "final_response_scores_path"):
    p = out[key]
    name = os.path.basename(p)
    if not os.path.exists(p):
        problems.append(f"{name} missing"); continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    good = {r.get("run_id") for r in rows if not r.get("error") and r.get("scores")}
    if ids - good:
        problems.append(f"{name} incomplete ({len(good & ids)}/{len(ids)})")
print("; ".join(problems) if problems else "ok")
EOF
)
  if [ "$status" != "ok" ]; then
    echo "   WARNING: $cfg absolute judging is '$status' -- re-invoke to resume."; exit 1
  fi
done
echo "   absolute judging complete and clean"

# --- stage 4: comparative judge (the new instrument) ------------------------
echo "== [4/6] Pairwise judge: fixed_k8 vs seqpersona_k8 (judge=$PAIRWISE_JUDGE_MODEL, $((2 * PAIRWISE_SAMPLES)) votes/item; resumes per query_id)"
seed_pairwise apply
"$PY" scripts/judge_pairwise.py \
  --runs-a "$FIXED_RUNS" --method-a fixed_k8 --label-a fixed_k8 \
  --runs-b "$SEQ_RUNS" --method-b seqpersona_k8 --label-b seqpersona \
  --queries "$QUERIES" --out "$PAIRWISE_OUT" \
  --model "$PAIRWISE_JUDGE_MODEL" --samples "$PAIRWISE_SAMPLES" \
  --workers "$PAIRWISE_WORKERS" ${LIMIT:+--limit "$LIMIT"} \
  2>&1 | tee "$EXP_DIR/logs/pairwise.log"

# --- stage 5: decision summary ----------------------------------------------
echo "== [5/6] Decision summary (absolute deltas vs pairwise favorability)"
"$PY" - "$SEQ_CONFIG" "$FIXED_CONFIG" "$PAIRWISE_OUT" "$EXP_DIR/decision_summary.json" <<'EOF'
import json, os, random, sys, yaml
seq_cfg, fixed_cfg, pairwise_path, out_path = sys.argv[1:5]

def load(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    out = cfg["outputs"]
    runs = {r["run_id"]: r for l in open(out["runs_path"]) if l.strip()
            for r in [json.loads(l)]}
    scores = {}
    for l in open(out["final_response_scores_path"]):
        if not l.strip():
            continue
        row = json.loads(l)
        rid = row.get("run_id")
        if rid in runs and row.get("scores"):
            scores[runs[rid]["query_id"]] = row["scores"]
    return scores

def leaves(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(leaves(v, f"{prefix}{k}." if isinstance(v, dict) else f"{prefix}{k}"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out

seq, fixed = load(seq_cfg), load(fixed_cfg)
shared = sorted(set(seq) & set(fixed))
fields = sorted({f for q in shared for f in leaves(seq[q])} &
                {f for q in shared for f in leaves(fixed[q])})
absolute = {}
for f in fields:
    pairs = [(leaves(seq[q]).get(f), leaves(fixed[q]).get(f)) for q in shared]
    pairs = [(s, x) for s, x in pairs if s is not None and x is not None]
    if not pairs:
        continue
    n = len(pairs)
    ms = sum(s for s, _ in pairs) / n
    mf = sum(x for _, x in pairs) / n
    absolute[f] = {"seqpersona_mean": round(ms, 3), "fixed_k8_mean": round(mf, 3),
                   "paired_delta": round(ms - mf, 3), "n": n,
                   "seq_better": sum(1 for s, x in pairs if s > x),
                   "fixed_better": sum(1 for s, x in pairs if x > s)}

pw = [json.loads(l) for l in open(pairwise_path) if l.strip()]
na = sum(1 for r in pw if r["majority"] == "a")
nb = sum(1 for r in pw if r["majority"] == "b")
nt = len(pw) - na - nb
dec = na + nb
rng = random.Random(0)
if dec:
    outcomes = [1] * nb + [0] * na
    boots = sorted(sum(rng.choices(outcomes, k=dec)) / dec for _ in range(4000))
    lo, hi = boots[100], boots[3899]
else:
    lo = hi = float("nan")
win = nb / dec if dec else float("nan")

agreement = {}
pw_by_q = {r["query_id"]: r["majority"] for r in pw if r["majority"] in ("a", "b")}
for f, st in absolute.items():
    agree = tot = 0
    for q in shared:
        m = pw_by_q.get(q)
        if not m:
            continue
        d = leaves(seq[q]).get(f, 0) - leaves(fixed[q]).get(f, 0)
        if d == 0:
            continue
        tot += 1
        if (d > 0) == (m == "b"):
            agree += 1
    agreement[f] = {"agree": agree, "decided_nonzero": tot,
                    "pct": round(100.0 * agree / tot, 1) if tot else None}

summary = {
    "n_shared_pairs": len(shared),
    "absolute_judge": absolute,
    "pairwise_judge": {
        "items": len(pw), "seqpersona_wins": nb, "fixed_k8_wins": na, "ties": nt,
        "seqpersona_win_rate_decided": round(win, 4) if dec else None,
        "ci95": [round(lo, 4), round(hi, 4)] if dec else None,
        "ci_clears_50pct": bool(dec and (lo > 0.5 or hi < 0.5)),
        "calibration_anchors": {"null": 0.50, "+0.3-class": 0.86, "+0.9-class": 1.00},
    },
    "instrument_agreement_by_field": agreement,
}
with open(out_path, "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"   shared pairs: {len(shared)}")
print(f"   ABSOLUTE (common judge, paired means seqpersona vs fixed_k8):")
for f, st in absolute.items():
    print(f"     {f:<40} {st['seqpersona_mean']:.2f} vs {st['fixed_k8_mean']:.2f} "
          f"(delta {st['paired_delta']:+.2f}; seq better on {st['seq_better']}, "
          f"fixed on {st['fixed_better']} of {st['n']})")
print(f"   PAIRWISE (comparison judge): seqpersona wins {nb}, fixed wins {na}, ties {nt}")
if dec:
    print(f"     win-rate among decided: {win:.1%} [95% CI {lo:.1%}, {hi:.1%}] "
          f"{'-- CI clears 50%' if (lo > 0.5 or hi < 0.5) else '-- CI includes 50%'}")
    print(f"     anchors: ~50% null | ~86% +0.3-class | ~100% +0.9-class")
print(f"   AGREEMENT (sign of absolute delta vs pairwise majority, per field):")
for f, st in agreement.items():
    if st["decided_nonzero"]:
        print(f"     {f:<40} {st['agree']}/{st['decided_nonzero']} ({st['pct']}%)")
print()
print("   Decision rule: small absolute deltas + pairwise CI clearing 50% for")
print("   seqpersona -> report the comparison judge only. Both instruments")
print("   aligned and non-trivial -> report both.")
print(f"   -> full numbers in {out_path}")
EOF

# --- stage 6: manifest -------------------------------------------------------
echo "== [6/6] Manifest"
"$PY" - "$EXP_DIR" "$SEQ_CONFIG" "$FIXED_CONFIG" "$PAIRWISE_OUT" "$STARTED_AT" "$PLANNER_MODEL" "$PAIRWISE_JUDGE_MODEL" <<'EOF'
import datetime, json, os, subprocess, sys, yaml
exp_dir, seq_cfg_p, fixed_cfg_p, pairwise, started_at, planner, pw_model = sys.argv[1:8]
seq_cfg, fixed_cfg = (yaml.safe_load(open(p)) for p in (seq_cfg_p, fixed_cfg_p))
def sh(*a): return subprocess.run(a, capture_output=True, text=True).stdout.strip()
def rows(p): return sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else None
manifest = {
    "experiment": "seqpersona_k8 vs fixed_k8, full 72 pairs, dual judges (absolute pinned 3.5-flash + calibrated pairwise)",
    "planner_model": planner,
    "pairwise_judge_model": pw_model,
    "seq_config": seq_cfg_p, "fixed_config": fixed_cfg_p,
    "models": {"seqpersona": seq_cfg["models"], "fixed_k8": fixed_cfg["models"]},
    "git_commit": sh("git", "rev-parse", "HEAD"),
    "started_at_utc": started_at,
    "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "artifact_rows": {
        "seqpersona_runs": rows(seq_cfg["outputs"]["runs_path"]),
        "fixed_runs": rows(fixed_cfg["outputs"]["runs_path"]),
        "seqpersona_final_scores": rows(seq_cfg["outputs"]["final_response_scores_path"]),
        "fixed_final_scores": rows(fixed_cfg["outputs"]["final_response_scores_path"]),
        "pairwise": rows(pairwise),
    },
}
path = os.path.join(exp_dir, "manifest.json")
json.dump(manifest, open(path, "w"), indent=2)
print(f"   manifest -> {path}")
print(json.dumps(manifest["artifact_rows"]))
EOF

echo
echo "== DONE -> $EXP_DIR (decision_summary.json + manifest.json)"
