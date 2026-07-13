"""Ordered candidate plan generation, validation, repair, fallback, caching, and prefix slicing for fixed_fanout_scaling_v1.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import DEFAULT_GEMINI_MODEL, DEFAULT_MAX_RESULTS_PER_BRANCH, DEFAULT_SEARCH_DEPTH
from .fanout import _extract_json, _persona_block
from .llm_gemini import call_gemini
from .meta_prompt import (
    ORDERED_FANOUT_PLANNER_PROMPT_V1,
    ORDERED_FANOUT_REPAIR_PROMPT_V1,
)
from .schemas import BRANCH_TYPES, FanoutBranch, Persona, SearchResult
from .search_tavily import search_tavily

PROMPT_VERSION_ORDERED_PLANNER = "V1"


def _normalize_query(query: str) -> str:
    """Normalize a search query string for duplicate checking and caching."""
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


STOPWORDS = {"the", "a", "an", "for", "in", "of", "to", "and", "or", "is", "on", "at", "by", "with"}

def _is_near_duplicate(query: str, existing_queries: List[str], threshold: float = 0.70) -> bool:
    """Detect near-duplicate search queries using Jaccard & content token overlap ratio."""
    norm_q = _normalize_query(query)
    tokens_q = set(w for w in norm_q.split() if w not in STOPWORDS)
    if not tokens_q:
        tokens_q = set(norm_q.split())
    if not tokens_q:
        return True

    for existing in existing_queries:
        norm_e = _normalize_query(existing)
        if norm_q == norm_e:
            return True
        tokens_e = set(w for w in norm_e.split() if w not in STOPWORDS)
        if not tokens_e:
            tokens_e = set(norm_e.split())
        if not tokens_e:
            continue
        intersection = len(tokens_q & tokens_e)
        union = len(tokens_q | tokens_e)
        min_len = min(len(tokens_q), len(tokens_e))
        
        jaccard = intersection / float(union) if union > 0 else 0.0
        containment = intersection / float(min_len) if min_len > 0 else 0.0
        
        if jaccard >= threshold or containment >= threshold:
            return True
    return False



def _generate_deterministic_fallback_branch(
    user_query: str,
    rank: int,
    existing_queries: List[str],
) -> FanoutBranch:
    """Produce a deterministic fallback branch based on rank and missing needs."""
    templates = {
        1: ("generic", "{query}", "general overview of the request"),
        2: ("personalized", "{query} preferences guide", "user-tailored aspect"),
        3: ("constraint", "{query} rules requirements limits eligibility", "hard constraints and requirements"),
        4: ("disconfirming", "{query} risks downsides tradeoffs alternatives", "caveats and disconfirming evidence"),
        5: ("supplementary", "{query} technical breakdown examples", "supplementary technical details"),
        6: ("supplementary", "{query} timeline cost breakdown", "supplementary operational details"),
        7: ("supplementary", "{query} practical tips documentation", "supplementary documentation and edge cases"),
        8: ("supplementary", "{query} comprehensive options review", "supplementary deep-dive synthesis"),
    }
    btype, template, need = templates.get(
        rank,
        ("supplementary", f"{{query}} detail section {rank}", f"supplementary info need {rank}"),
    )
    base_q = template.format(query=user_query)
    fallback_q = base_q
    counter = 1
    while _is_near_duplicate(fallback_q, existing_queries, threshold=0.9):
        fallback_q = f"{base_q} topic {counter}"
        counter += 1

    return FanoutBranch(
        branch_type=btype,
        query=fallback_q,
        rationale=f"deterministic fallback for rank {rank}",
        used_persona_fields=[],
        information_need=need,
        priority_rank=rank,
    )


def _parse_and_validate_branches(
    parsed: Any,
    allowed_types: Tuple[str, ...] = BRANCH_TYPES,
) -> List[FanoutBranch]:
    """Convert parsed JSON into validated FanoutBranch objects."""
    if isinstance(parsed, dict):
        for key in ("branches", "queries", "fanout", "plan", "results"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        return []

    branches: List[FanoutBranch] = []
    for item in parsed:
        if isinstance(item, str):
            branches.append(
                FanoutBranch(
                    branch_type="generic",
                    query=item.strip(),
                    information_need="general search",
                )
            )
            continue
        if not isinstance(item, dict):
            continue
        query = (item.get("query") or item.get("q") or "").strip()
        if not query:
            continue
        btype = (item.get("branch_type") or "generic").strip().lower()
        if btype not in allowed_types:
            btype = "supplementary"
        used_fields = item.get("used_persona_fields") or []
        if not isinstance(used_fields, list):
            used_fields = [str(used_fields)]
        rank = item.get("priority_rank")
        try:
            rank_int = int(rank) if rank is not None else None
        except (ValueError, TypeError):
            rank_int = None

        branches.append(
            FanoutBranch(
                branch_type=btype,
                query=query,
                rationale=(item.get("rationale") or "").strip(),
                used_persona_fields=[str(f) for f in used_fields],
                information_need=(item.get("information_need") or "").strip(),
                priority_rank=rank_int,
            )
        )
    return branches


def generate_ordered_fanout_plan(
    user_query: str,
    persona: Optional[Persona],
    candidate_pool_size: int = 8,
    model: str = DEFAULT_GEMINI_MODEL,
    seed: int = 42,
) -> Tuple[List[FanoutBranch], List[Dict[str, Any]]]:
    """Generate an ordered plan of candidate_pool_size search branches.

    Includes defensive parsing, deduplication, 1-step LLM repair, and deterministic fallback.
    Returns (validated_branches, logged_events).
    """
    events: List[Dict[str, Any]] = []
    prompt = ORDERED_FANOUT_PLANNER_PROMPT_V1.format(
        candidate_pool_size=candidate_pool_size,
        user_query=user_query,
        persona_block=_persona_block(persona),
    )

    t0 = time.time()
    raw_response = call_gemini(
        prompt, model=model, response_mime_type="application/json", seed=seed
    )
    t_llm = time.time() - t0

    events.append(
        {
            "event_type": "initial_planner_call",
            "latency": t_llm,
            "raw_response_length": len(raw_response or ""),
        }
    )

    parsed = _extract_json(raw_response)
    candidate_branches = _parse_and_validate_branches(parsed)

    # Deduplicate candidate branches
    validated_branches: List[FanoutBranch] = []
    seen_queries: List[str] = []

    for b in candidate_branches:
        if not _is_near_duplicate(b.query, seen_queries):
            seen_queries.append(b.query)
            validated_branches.append(b)

    events.append(
        {
            "event_type": "initial_validation",
            "parsed_count": len(candidate_branches),
            "validated_count": len(validated_branches),
        }
    )

    # Step 2: Structured repair attempt if < candidate_pool_size
    if len(validated_branches) < candidate_pool_size:
        missing_count = candidate_pool_size - len(validated_branches)
        existing_queries_block = "\n".join(
            f"  - [{idx+1}] ({b.branch_type}): {b.query}"
            for idx, b in enumerate(validated_branches)
        )
        existing_needs_block = "\n".join(
            f"  - [{idx+1}]: {b.information_need or 'N/A'}"
            for idx, b in enumerate(validated_branches)
        )

        repair_prompt = ORDERED_FANOUT_REPAIR_PROMPT_V1.format(
            candidate_pool_size=candidate_pool_size,
            missing_count=missing_count,
            start_rank=len(validated_branches) + 1,
            user_query=user_query,
            persona_block=_persona_block(persona),
            existing_queries_block=existing_queries_block or "(none)",
            existing_needs_block=existing_needs_block or "(none)",
        )

        t0 = time.time()
        repair_raw = call_gemini(
            repair_prompt,
            model=model,
            response_mime_type="application/json",
            seed=seed,
        )
        t_repair = time.time() - t0

        events.append(
            {
                "event_type": "repair_attempt",
                "latency": t_repair,
                "missing_count": missing_count,
            }
        )

        repair_parsed = _extract_json(repair_raw)
        repair_branches = _parse_and_validate_branches(repair_parsed)

        for b in repair_branches:
            if len(validated_branches) >= candidate_pool_size:
                break
            if not _is_near_duplicate(b.query, seen_queries):
                seen_queries.append(b.query)
                validated_branches.append(b)

    # Step 3: Deterministic fallback if still < candidate_pool_size
    if len(validated_branches) < candidate_pool_size:
        missing = candidate_pool_size - len(validated_branches)
        events.append(
            {
                "event_type": "deterministic_fallback",
                "missing_count": missing,
            }
        )
        while len(validated_branches) < candidate_pool_size:
            next_rank = len(validated_branches) + 1
            fallback_b = _generate_deterministic_fallback_branch(
                user_query, next_rank, seen_queries
            )
            seen_queries.append(fallback_b.query)
            validated_branches.append(fallback_b)

    # Enforce priority ranks 1..candidate_pool_size sequentially
    final_branches: List[FanoutBranch] = []
    for idx, b in enumerate(validated_branches[:candidate_pool_size], start=1):
        b.priority_rank = idx
        final_branches.append(b)

    return final_branches, events


def compute_plan_cache_key(
    query_id: str,
    user_query: str,
    persona: Optional[Persona],
    planner_model: str,
    prompt_version: str = PROMPT_VERSION_ORDERED_PLANNER,
    candidate_pool_size: int = 8,
    seed: int = 42,
) -> str:
    """Compute a stable hash key for candidate plan caching."""
    persona_str = persona.render_for_agent() if persona else ""
    raw_key = (
        f"{query_id}|{user_query.strip().lower()}|{persona_str}|"
        f"{planner_model}|{prompt_version}|{candidate_pool_size}|{seed}"
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


def get_or_create_shared_plan(
    query_id: str,
    user_query: str,
    persona: Optional[Persona],
    candidate_pool_size: int = 8,
    planner_model: str = DEFAULT_GEMINI_MODEL,
    prompt_version: str = PROMPT_VERSION_ORDERED_PLANNER,
    seed: int = 42,
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[str, List[FanoutBranch], List[Dict[str, Any]], bool]:
    """Load or generate the shared ordered candidate plan.

    Returns (plan_id, branches, events, is_cache_hit).
    """
    plan_id = compute_plan_cache_key(
        query_id,
        user_query,
        persona,
        planner_model,
        prompt_version,
        candidate_pool_size,
        seed,
    )

    if use_cache and cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("plan_id") == plan_id:
                    branches = [
                        FanoutBranch(
                            branch_type=b["branch_type"],
                            query=b["query"],
                            rationale=b.get("rationale", ""),
                            used_persona_fields=b.get("used_persona_fields", []),
                            information_need=b.get("information_need", ""),
                            priority_rank=b.get("priority_rank"),
                        )
                        for b in record.get("branches", [])
                    ]
                    events = record.get("events", [])
                    events.append({"event_type": "plan_cache_hit", "plan_id": plan_id})
                    return plan_id, branches, events, True

    # Cache miss or cache disabled
    branches, events = generate_ordered_fanout_plan(
        user_query=user_query,
        persona=persona,
        candidate_pool_size=candidate_pool_size,
        model=planner_model,
        seed=seed,
    )

    if use_cache and cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        record = {
            "plan_id": plan_id,
            "query_id": query_id,
            "user_query": user_query,
            "persona_id": persona.persona_id if persona else None,
            "planner_model": planner_model,
            "prompt_version": prompt_version,
            "candidate_pool_size": candidate_pool_size,
            "seed": seed,
            "branches": [b.as_dict() for b in branches],
            "events": events,
        }
        with open(cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return plan_id, branches, events, False


def compute_search_cache_key(
    query: str,
    provider: str = "tavily",
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    max_results: int = DEFAULT_MAX_RESULTS_PER_BRANCH,
) -> str:
    """Compute a stable hash key for search query caching."""
    norm_q = _normalize_query(query)
    raw_key = f"{norm_q}|{provider}|{search_depth}|{max_results}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


def search_tavily_cached(
    query: str,
    branch_type: str = "generic",
    max_results: int = DEFAULT_MAX_RESULTS_PER_BRANCH,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    cache_path: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[List[SearchResult], bool]:
    """Execute search with caching.

    Returns (normalized_search_results, is_cache_hit).
    """
    key = compute_search_cache_key(query, "tavily", search_depth, max_results)

    if use_cache and cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("cache_key") == key:
                    raw_results = record.get("results", [])
                    results = [
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("url", ""),
                            content=r.get("content", ""),
                            score=r.get("score"),
                            rank=r.get("rank", idx + 1),
                            branch_type=branch_type,
                            branch_query=query,
                            is_duplicate_url=r.get("is_duplicate_url", False),
                        )
                        for idx, r in enumerate(raw_results)
                    ]
                    return results, True

    # Cache miss
    results = search_tavily(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        branch_type=branch_type,
    )

    if use_cache and cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        record = {
            "cache_key": key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "results": [r.as_dict() for r in results],
        }
        with open(cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return results, False
