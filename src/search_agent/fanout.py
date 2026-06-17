"""Query fan-out generation.

This module turns a user query (plus optional persona) into a list of
:class:`FanoutBranch` objects, conditioned on the experimental variant. The
heart of the personalization-placement study lives here: which variants are
allowed to see the persona when generating search queries.

All Gemini calls request strict JSON and include a defensive parser with a
fallback so a malformed model response never crashes a run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import DEFAULT_GEMINI_MODEL
from .llm_gemini import call_gemini
from .schemas import BRANCH_TYPES, FanoutBranch, Persona

# How many branches each variant should aim for. These are soft targets passed
# into the prompt; parsing is tolerant of slightly different counts.
NUM_GENERIC_BRANCHES = 4
NUM_PERSONALIZED_BRANCHES = 4
MIXED_BRANCH_PLAN = {
    "generic": 2,
    "personalized": 2,
}


def _persona_block(persona: Optional[Persona]) -> str:
    """Render the agent-visible user context for prompting.

    Shows stated demographics + raw search history only; the curated
    ``latent_profile`` is withheld so the planner must infer preferences.
    """
    if persona is None:
        return "(no user context provided)"
    return persona.render_for_agent()


def _extract_json(text: str) -> Optional[Any]:
    """Best-effort JSON extraction from a model response.

    Handles raw JSON, fenced ```json blocks, and leading/trailing prose.
    Returns the parsed object or ``None`` if nothing parseable is found.
    """
    if not text:
        return None

    # 1) Direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Strip Markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 3) Grab the first balanced-looking JSON array or object.
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    return None


def _coerce_branches(
    parsed: Any,
    *,
    allowed_types: Tuple[str, ...],
    default_type: str,
) -> List[FanoutBranch]:
    """Convert parsed JSON into validated FanoutBranch objects."""
    if isinstance(parsed, dict):
        # Allow {"branches": [...]} or {"queries": [...]} wrappers.
        for key in ("branches", "queries", "fanout", "results"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        return []

    branches: List[FanoutBranch] = []
    for item in parsed:
        if isinstance(item, str):
            branches.append(
                FanoutBranch(branch_type=default_type, query=item.strip())
            )
            continue
        if not isinstance(item, dict):
            continue
        query = (item.get("query") or item.get("q") or "").strip()
        if not query:
            continue
        btype = (item.get("branch_type") or default_type).strip().lower()
        if btype not in allowed_types:
            btype = default_type
        used_fields = item.get("used_persona_fields") or []
        if not isinstance(used_fields, list):
            used_fields = [str(used_fields)]
        branches.append(
            FanoutBranch(
                branch_type=btype,
                query=query,
                rationale=(item.get("rationale") or "").strip(),
                used_persona_fields=[str(f) for f in used_fields],
            )
        )
    return branches


def _fallback_branches(
    user_query: str, default_type: str = "generic"
) -> List[FanoutBranch]:
    """Deterministic fallback if Gemini JSON parsing fails entirely.

    We return the raw user query as a single branch so the pipeline can still
    produce evidence and a final answer. The rationale flags that this is a
    fallback, which is useful when inspecting logs.
    """
    return [
        FanoutBranch(
            branch_type=default_type,
            query=user_query,
            rationale="fallback: fan-out JSON parsing failed; used raw query",
            used_persona_fields=[],
        )
    ]


def _generate_generic(
    user_query: str, model: str
) -> List[FanoutBranch]:
    """Generate generic (non-personalized) fan-out branches."""
    prompt = f"""You are a search query planner for a retrieval system.
Given a user question, produce {NUM_GENERIC_BRANCHES} diverse, GENERIC web
search queries that together gather broad, high-quality evidence to answer it.
Do NOT assume anything about the specific user. Keep queries concise and
search-engine friendly.

User question: {user_query!r}

Return STRICT JSON: a list of objects, each with fields:
  "branch_type": always "generic",
  "query": the search query string,
  "rationale": one short sentence on what evidence this gathers,
  "used_persona_fields": always an empty list [].

Return ONLY the JSON array, no prose."""
    raw = call_gemini(
        prompt, model=model, response_mime_type="application/json"
    )
    branches = _coerce_branches(
        _extract_json(raw), allowed_types=("generic",), default_type="generic"
    )
    return branches or _fallback_branches(user_query, "generic")


def _generate_personalized(
    user_query: str, persona: Optional[Persona], model: str
) -> List[FanoutBranch]:
    """Generate personalized fan-out branches conditioned on the persona."""
    prompt = f"""You are a search query planner for a PERSONALIZED retrieval
system. You are given a user's question plus a snapshot of what we know about
them: a few stated details and their recent search history. Some history entries
are unrelated to the current question — INFER which of their interests,
preferences, and constraints are actually relevant and ignore the rest. Produce
{NUM_PERSONALIZED_BRANCHES} web search queries tailored to this user's inferred
needs, but keep them realistic search queries (not full sentences about the user).

User question: {user_query!r}

User context:
{_persona_block(persona)}

Return STRICT JSON: a list of objects, each with fields:
  "branch_type": always "personalized",
  "query": the search query string,
  "rationale": one short sentence explaining the personalization,
  "used_persona_fields": list of the user signals you inferred and used (e.g. ["self-paced learner","prefers subscription-free hardware"]); [] if none.

Return ONLY the JSON array, no prose."""
    raw = call_gemini(
        prompt, model=model, response_mime_type="application/json"
    )
    branches = _coerce_branches(
        _extract_json(raw),
        allowed_types=("personalized",),
        default_type="personalized",
    )
    return branches or _fallback_branches(user_query, "personalized")


def _generate_mixed(
    user_query: str, persona: Optional[Persona], model: str
) -> List[FanoutBranch]:
    """Generate mixed branch types for variant V4."""
    plan_desc = ", ".join(
        f"{count} {btype}" for btype, count in MIXED_BRANCH_PLAN.items()
    )
    prompt = f"""You are an advanced search query planner. Produce a MIXED set of
web search queries across two branch types to thoroughly and fairly answer a
user's question.

Branch types:
  - "generic": broad, neutral evidence about the topic (ignore the user context).
  - "personalized": tailored to the user's needs/preferences that you INFER from
    their stated details and recent search history (some history is unrelated —
    use only what is genuinely relevant).

Aim for roughly: {plan_desc}.

User question: {user_query!r}

User context:
{_persona_block(persona)}

Return STRICT JSON: a list of objects, each with fields:
  "branch_type": one of "generic" | "personalized",
  "query": the search query string,
  "rationale": one short sentence,
  "used_persona_fields": list of inferred user signals used (empty list for generic).

Return ONLY the JSON array, no prose."""
    raw = call_gemini(
        prompt, model=model, response_mime_type="application/json"
    )
    branches = _coerce_branches(
        _extract_json(raw),
        allowed_types=BRANCH_TYPES,
        default_type="generic",
    )
    return branches or _fallback_branches(user_query, "generic")


def generate_fanout_queries(
    user_query: str,
    persona: Optional[Persona],
    variant: str,
    *,
    model: str = DEFAULT_GEMINI_MODEL,
) -> List[FanoutBranch]:
    """Generate fan-out branches for the given variant.

    Variant behavior:
      - V0_generic_single: no fan-out; the raw query as one generic branch.
      - V1_generic_fanout / V2_synthesis_only_personalization: generic branches.
      - V3_personalized_fanout: personalized branches (persona-conditioned).
      - V4_mixed_fanout: mixed generic/personalized/constraint/disconfirming.

    Returns at least one branch in all cases (fallback guarantees this).
    """
    if variant == "V0_generic_single":
        return [
            FanoutBranch(
                branch_type="generic",
                query=user_query,
                rationale="V0: use the raw user query as a single branch",
                used_persona_fields=[],
            )
        ]

    if variant in ("V1_generic_fanout", "V2_synthesis_only_personalization"):
        return _generate_generic(user_query, model)

    if variant == "V3_personalized_fanout":
        return _generate_personalized(user_query, persona, model)

    if variant == "V4_mixed_fanout":
        return _generate_mixed(user_query, persona, model)

    raise ValueError(f"Unknown variant: {variant!r}")
