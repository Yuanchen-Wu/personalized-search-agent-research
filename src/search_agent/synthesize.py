"""Final answer synthesis via Gemini.

The synthesis step is deliberately conservative: it answers the user query using
the retrieved evidence, only leans on persona context when relevant, and is
explicit about uncertainty and the non-exhaustiveness of the retrieved sources.

Whether persona context is passed in is controlled by the variant (handled by
the caller in ``run_agent.py``), but ``synthesize_answer`` also defensively
ignores the persona when ``persona is None``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

from .config import (
    DEFAULT_GEMINI_MODEL,
    MAX_RESULTS_PER_BRANCH_FOR_SYNTHESIS,
)
from .llm_gemini import call_gemini
from .meta_prompt import SYNTHESIS_PROMPT_TEMPLATE
from .schemas import Persona, SearchResult


def _select_results_for_synthesis(
    search_results: List[SearchResult],
    max_per_branch: int = MAX_RESULTS_PER_BRANCH_FOR_SYNTHESIS,
) -> List[SearchResult]:
    """Cap evidence at top-N per branch query to keep context manageable."""
    by_branch: dict[str, List[SearchResult]] = defaultdict(list)
    for result in search_results:
        by_branch[result.branch_query].append(result)

    selected: List[SearchResult] = []
    for branch_query, results in by_branch.items():
        ranked = sorted(results, key=lambda r: r.rank)
        selected.extend(ranked[:max_per_branch])
    return selected


def _format_evidence(results: List[SearchResult]) -> str:
    """Render evidence into a compact, citable text block."""
    if not results:
        return "(no search evidence was retrieved)"

    lines: List[str] = []
    for idx, r in enumerate(results, start=1):
        dup = " [duplicate-url]" if r.is_duplicate_url else ""
        lines.append(
            f"[{idx}] ({r.branch_type}) {r.title}{dup}\n"
            f"    URL: {r.url}\n"
            f"    {r.content.strip()}"
        )
    return "\n\n".join(lines)


def _persona_block(persona: Optional[Persona]) -> str:
    if persona is None:
        return ""
    return (
        "\nWhat we know about the user (stated details + recent search history; "
        "some history may be unrelated — infer what is genuinely relevant and do "
        "NOT over-personalize):\n"
        f"{persona.render_for_agent()}\n"
    )


def synthesize_answer(
    user_query: str,
    persona: Optional[Persona],
    search_results: List[SearchResult],
    variant: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
) -> str:
    """Synthesize a final answer from the user query and retrieved evidence.

    Args:
        user_query: The original user question.
        persona: Persona context, or ``None`` to synthesize without it. For
            V0/V1 the caller passes ``None``; for V2/V3/V4 it passes the persona.
        search_results: Collected evidence across all branches.
        variant: The experimental variant (included for traceability).
        model: Gemini model name.

    Returns:
        The synthesized answer text.
    """
    selected = _select_results_for_synthesis(search_results)
    evidence_block = _format_evidence(selected)
    persona_block = _persona_block(persona)

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        user_query=user_query,
        persona_block=persona_block,
        evidence_block=evidence_block,
    )

    return call_gemini(prompt, model=model, temperature=0.4)
