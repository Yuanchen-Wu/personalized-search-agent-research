"""Evidence deduplication, synthesis selection modes, context sizing, and retrieval evaluation sampling.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
from .schemas import SearchResult


def deduplicate_search_results(results: List[SearchResult]) -> List[SearchResult]:
    """Flag duplicate URLs across search results while maintaining order."""
    seen_urls: Set[str] = set()
    deduped: List[SearchResult] = []
    for r in results:
        is_dup = False
        if r.url:
            if r.url in seen_urls:
                is_dup = True
            else:
                seen_urls.add(r.url)
        res_copy = SearchResult(
            title=r.title,
            url=r.url,
            content=r.content,
            score=r.score,
            rank=r.rank,
            branch_type=r.branch_type,
            branch_query=r.branch_query,
            is_duplicate_url=is_dup,
        )
        deduped.append(res_copy)
    return deduped


def filter_unique_documents(results: List[SearchResult]) -> List[SearchResult]:
    """Return only non-duplicate search results by URL."""
    seen_urls: Set[str] = set()
    unique: List[SearchResult] = []
    for r in results:
        if r.url and r.url not in seen_urls:
            seen_urls.add(r.url)
            unique.append(r)
    return unique


def compute_context_character_count(results: List[SearchResult]) -> int:
    """Compute character length of search evidence content."""
    return sum(len(r.content or "") + len(r.title or "") + len(r.url or "") for r in results)


def select_evidence_for_synthesis(
    search_results: List[SearchResult],
    evidence_budget_mode: str = "all",
    max_documents: Optional[int] = None,
    max_context_chars: Optional[int] = None,
) -> List[SearchResult]:
    """Select the exact evidence subset passed to synthesis based on config.

    Modes:
      - 'all': Pass all unique non-duplicate search results.
      - 'fixed_document_budget': Pass at most `max_documents` (default 5) unique results,
        selected deterministically by branch rank and search score.
    """
    unique_results = filter_unique_documents(search_results)

    if evidence_budget_mode == "fixed_document_budget":
        limit = max_documents if max_documents is not None else 5
        # Deterministic, method-independent ranking rule: sort by branch rank, then tavily rank / score
        ranked = sorted(
            unique_results,
            key=lambda r: (r.rank, -(r.score if r.score is not None else 0.0)),
        )
        selected = ranked[:limit]
    else:
        # 'all' mode
        if max_documents is not None:
            selected = unique_results[:max_documents]
        else:
            selected = unique_results

    # Apply optional max context character budget safety cap if configured
    if max_context_chars is not None and max_context_chars > 0:
        capped: List[SearchResult] = []
        current_chars = 0
        for r in selected:
            r_len = len(r.content or "") + len(r.title or "") + len(r.url or "")
            if current_chars + r_len > max_context_chars and capped:
                break
            capped.append(r)
            current_chars += r_len
        selected = capped

    return selected


def sample_retrieval_evidence_for_evaluator(
    raw_search_results: List[SearchResult],
    mode: str = "top_m_per_branch",
    top_m_per_branch: int = 3,
    top_n_global: int = 10,
    max_context_chars: int = 15000,
) -> List[SearchResult]:
    """Select search results to show to the retrieval judge.

    Modes:
      - 'top_m_per_branch': Select up to M non-duplicate results per executed branch.
      - 'all_deduplicated': Pass all unique results.
      - 'top_n_global': Select top N unique results globally across branches.
    """
    if mode == "all_deduplicated":
        results = filter_unique_documents(raw_search_results)
    elif mode == "top_n_global":
        results = filter_unique_documents(raw_search_results)[:top_n_global]
    elif mode == "top_m_per_branch":
        by_branch: Dict[str, List[SearchResult]] = {}
        for r in raw_search_results:
            by_branch.setdefault(r.branch_query, []).append(r)
        
        sampled: List[SearchResult] = []
        seen_urls: Set[str] = set()
        for branch_query, b_results in by_branch.items():
            b_count = 0
            for r in b_results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    sampled.append(r)
                    b_count += 1
                    if b_count >= top_m_per_branch:
                        break
        results = sampled
    else:
        results = filter_unique_documents(raw_search_results)

    # Context character safety limit for evaluation prompt
    capped: List[SearchResult] = []
    current_chars = 0
    for r in results:
        r_len = len(r.content or "") + len(r.title or "")
        if current_chars + r_len > max_context_chars and capped:
            break
        capped.append(r)
        current_chars += r_len

    return capped
