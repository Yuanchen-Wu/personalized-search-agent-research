"""Thin wrapper around the Tavily Search API.

Important: we use Tavily purely as a search-evidence provider. We deliberately
ignore Tavily's generated ``answer`` field — all answer synthesis happens via
Gemini downstream so that the experiment isolates personalization placement.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .config import (
    DEFAULT_MAX_RESULTS_PER_BRANCH,
    DEFAULT_SEARCH_DEPTH,
    get_tavily_api_key,
)
from .schemas import FanoutBranch, SearchResult

_client = None  # lazily initialized Tavily client


def _get_client():
    """Construct (once) and return the Tavily client."""
    global _client
    if _client is None:
        from tavily import TavilyClient

        _client = TavilyClient(api_key=get_tavily_api_key())
    return _client


def search_tavily(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS_PER_BRANCH,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
    *,
    branch_type: str = "generic",
    max_retries: int = 3,
) -> List[SearchResult]:
    """Run a single Tavily search and normalize the results.

    Args:
        query: The search query string.
        max_results: Max number of results to request.
        search_depth: ``"basic"`` or ``"advanced"``.
        branch_type: Branch label attached to each result for later analysis.
        max_retries: Number of attempts on transient failures.

    Returns:
        A list of :class:`SearchResult`, ordered by Tavily rank (1-indexed).
        Returns an empty list if the search fails after all retries (we log the
        failure rather than crashing an entire batch run).
    """
    client = _get_client()

    last_err: Optional[Exception] = None
    raw_results = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                # We never use Tavily's synthesized answer in this experiment.
                include_answer=False,
            )
            raw_results = response.get("results", []) if response else []
            break
        except Exception as err:  # noqa: BLE001 - broad retry coverage
            last_err = err
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
            else:
                # Soft-fail: return empty so a single bad branch doesn't kill
                # the whole run. The empty result is still visible in logs.
                print(
                    f"[search_tavily] WARNING: search failed for query "
                    f"{query!r} after {max_retries} attempts: {last_err}"
                )
                return []

    normalized: List[SearchResult] = []
    for rank, item in enumerate(raw_results or [], start=1):
        score = item.get("score")
        normalized.append(
            SearchResult(
                title=item.get("title", "") or "",
                url=item.get("url", "") or "",
                content=item.get("content", "") or "",
                score=float(score) if isinstance(score, (int, float)) else None,
                rank=rank,
                branch_type=branch_type,
                branch_query=query,
            )
        )
    return normalized


def collect_search_results(
    fanout_branches: List[FanoutBranch],
    max_results_per_branch: int = DEFAULT_MAX_RESULTS_PER_BRANCH,
    search_depth: str = DEFAULT_SEARCH_DEPTH,
) -> List[SearchResult]:
    """Search every fan-out branch and concatenate the normalized results.

    Behavior (intentionally simple for this first iteration):
      - One Tavily call per branch.
      - Branch metadata (type + query) is attached to each result.
      - Tavily rank is preserved within each branch.
      - Duplicate URLs are KEPT, but flagged via ``is_duplicate_url`` so later
        analysis can decide what to do. No reranking or fusion is performed.

    Args:
        fanout_branches: Branches produced by ``generate_fanout_queries``.
        max_results_per_branch: Max results requested per branch.
        search_depth: Tavily search depth (``"basic"`` or ``"advanced"``).

    Returns:
        A flat list of :class:`SearchResult` across all branches, in branch
        order (and Tavily rank order within each branch).
    """
    all_results: List[SearchResult] = []
    seen_urls: set[str] = set()

    for branch in fanout_branches:
        branch_results = search_tavily(
            query=branch.query,
            max_results=max_results_per_branch,
            search_depth=search_depth,
            branch_type=branch.branch_type,
        )
        for result in branch_results:
            if result.url and result.url in seen_urls:
                result.is_duplicate_url = True
            elif result.url:
                seen_urls.add(result.url)
            all_results.append(result)

    return all_results
