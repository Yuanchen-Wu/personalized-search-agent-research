"""Crash-safe resume support for the judge scripts.

The evaluators append one JSONL row per judged run as results complete, so an
interrupted invocation (battery, quota, Ctrl-C) keeps everything already
judged. On the next invocation ``prepare_scores_file`` cleans the output file
(dropping corrupt part-written lines, rows that recorded a per-row ``error``,
and duplicate run_ids) and returns the run_ids that are already done, so only
the remainder is re-judged.
"""

from __future__ import annotations

import json
import os
from typing import Set


def prepare_scores_file(out_path: str, force: bool = False) -> Set[str]:
    """Return run_ids with a good (error-free) row in ``out_path``.

    Cleans the file in place first — corrupt lines (e.g. a write cut off by
    power loss), error rows, and duplicate run_ids are dropped via an atomic
    rewrite, so re-judged rows never coexist with their failed predecessors.
    With ``force=True`` the file is discarded and everything is re-judged.
    """
    if force:
        if os.path.exists(out_path):
            os.remove(out_path)
        return set()
    if not os.path.exists(out_path):
        return set()

    kept, dropped, seen = [], 0, set()
    with open(out_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            run_id = row.get("run_id")
            if row.get("error") or not run_id or run_id in seen:
                dropped += 1
                continue
            seen.add(run_id)
            kept.append(row)

    if dropped:
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)
        print(
            f"[resume] cleaned {os.path.basename(out_path)}: kept {len(kept)} "
            f"scored rows, dropped {dropped} (corrupt/error/duplicate)"
        )
    return seen


def append_score(fh, row: dict) -> None:
    """Write one score row and flush, so progress survives a sudden death."""
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()
