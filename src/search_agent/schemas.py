"""Lightweight dataclass schemas shared across the pipeline.

These are deliberately plain dataclasses (no Pydantic) to keep the prototype
inspectable and dependency-light. Each schema has an ``as_dict`` helper so we
can serialize everything cleanly into JSONL logs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Branch types used throughout the fan-out logic. Kept as a simple tuple so it
# is trivial to validate against and to extend later.
BRANCH_TYPES = ("generic", "personalized")

# Variant identifiers. Centralized here so the CLI, batch runner, and fan-out
# logic all agree on the canonical names.
VARIANTS = (
    "V0_generic_single",
    "V1_generic_fanout",
    "V2_synthesis_only_personalization",
    "V3_personalized_fanout",
    "V4_mixed_fanout",
)


@dataclass
class Persona:
    """A user persona / context bundle used to drive personalization.

    For the personalization-placement study the agent is shown a *realistic*
    view of the user — stated ``demographics`` plus the raw, interleaved
    ``observable_history`` and ``distractor_history`` — and must INFER what is
    relevant. The curated ``latent_profile`` inside ``attributes`` is treated as
    ground truth for evaluation only and is deliberately NOT rendered for the
    agent (see :meth:`render_for_agent`).
    """

    persona_id: str
    description: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    observable_history: List[Dict[str, Any]] = field(default_factory=list)
    distractor_history: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Persona":
        return cls(
            persona_id=data["persona_id"],
            description=data.get("description", ""),
            attributes=data.get("attributes", {}) or {},
            observable_history=data.get("observable_history", []) or [],
            distractor_history=data.get("distractor_history", []) or [],
        )

    def _merged_history(self) -> List[Dict[str, Any]]:
        """Observable + distractor history, interleaved chronologically.

        The two sources are merged and sorted by timestamp so distractor entries
        are indistinguishable from genuine ones: the agent must judge relevance
        from the query text itself, not from a source label.
        """
        merged = list(self.observable_history) + list(self.distractor_history)
        return sorted(merged, key=lambda h: str(h.get("timestamp", "")))

    def render_for_agent(self) -> str:
        """Render the agent-visible view of the user.

        Shows stated demographics and the raw, interleaved search history, and
        deliberately EXCLUDES ``latent_profile`` (curated ground truth reserved
        for evaluation). Falls back to legacy ``attributes`` rendering for
        personas that carry no demographics/history (e.g. hand-made demo data).
        """
        lines: List[str] = []
        demographics = (self.attributes or {}).get("demographics")
        if isinstance(demographics, dict) and demographics:
            lines.append("Stated user details:")
            for key, value in demographics.items():
                lines.append(f"  - {key}: {value}")

        history = self._merged_history()
        if history:
            if lines:
                lines.append("")
            lines.append(
                "Recent search history (chronological; some entries may be "
                "unrelated to the current question — infer what is relevant):"
            )
            for entry in history:
                ts = str(entry.get("timestamp", "")).strip()
                content = str(entry.get("content", "")).strip()
                if not content:
                    continue
                prefix = f"[{ts}] " if ts else ""
                lines.append(f"  - {prefix}{content}")

        if lines:
            return "\n".join(lines)

        # Legacy fallback: no demographics or history (e.g. demo personas).
        return (
            f"description: {self.description}\n"
            f"attributes: {json.dumps(self.attributes, ensure_ascii=False)}"
        )


@dataclass
class QueryRecord:
    """A user query with metadata indicating the intended task type."""

    query: str
    query_id: str = "unknown"
    task_type: str = "unknown"
    task_category: str = "unknown"
    persona_relevant_dimensions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryRecord":
        query_id = data.get("query_id", data.get("id", data.get("example_id", "unknown")))
        
        task_category = data.get("task_category", data.get("category", data.get("domain", "unknown")))
        
        # normalize task_category
        cat_lower = task_category.lower()
        if any(x in cat_lower for x in ["ecommerce", "product", "shopping"]):
            if "tech" in cat_lower or "comparison" in cat_lower:
                task_category = "tech_product_comparison"
            else:
                task_category = "shopping_commerce"
        elif any(x in cat_lower for x in ["textbook", "book", "course", "resource"]):
            task_category = "textbook_or_resource_recommendation"
        elif any(x in cat_lower for x in ["technical", "concept"]):
            task_category = "technical_explanation"
        elif any(x in cat_lower for x in ["career", "education advice", "professional"]):
            task_category = "professional_career_strategy"
        elif any(x in cat_lower for x in ["health", "wellness"]):
            task_category = "health_information"
        elif any(x in cat_lower for x in ["travel", "local"]):
            task_category = "travel_local_planning"

        task_type = data.get("task_type")
        if not task_type:
            q_type = str(data.get("query_type", "")).lower()
            combined_hint = task_category.lower() + " " + q_type
            
            search_keywords = ["shopping", "ecommerce", "product", "textbook", "course", "resource", "travel", "local"]
            synthesis_keywords = ["technical", "career", "education advice", "professional", "background-adaptive", "explanation", "conceptual"]
            
            if any(k in combined_hint for k in search_keywords):
                task_type = "search_native"
            elif any(k in combined_hint for k in synthesis_keywords):
                task_type = "synthesis_native"
            else:
                task_type = "unknown"
                print(f"Warning: Could not infer task_type for query {query_id}. Defaulting to 'unknown'.")
        
        metadata = {}
        for k, v in data.items():
            if k not in ["query", "query_id", "id", "example_id", "task_type", "task_category", "category", "domain", "persona_relevant_dimensions"]:
                metadata[k] = v

        return cls(
            query=data.get("query", ""),
            query_id=query_id,
            task_type=task_type,
            task_category=task_category,
            persona_relevant_dimensions=data.get("persona_relevant_dimensions", []),
            metadata=metadata
        )



@dataclass
class FanoutBranch:
    """A single search branch produced during query fan-out."""

    branch_type: str  # one of BRANCH_TYPES
    query: str
    rationale: str = ""
    used_persona_fields: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """A single normalized search result from Tavily plus branch metadata."""

    title: str
    url: str
    content: str
    score: Optional[float]
    rank: int
    branch_type: str
    branch_query: str
    is_duplicate_url: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CostProxy:
    """Cheap, transparent accounting of how much work a run did."""

    num_gemini_calls: int = 0
    num_tavily_calls: int = 0
    num_fanout_branches: int = 0
    num_raw_results: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunLog:
    """The full structured record of a single agent run. ok."""

    run_id: str
    experiment_name: str
    timestamp: str
    variant: str
    user_query: str
    query_id: str
    task_type: str
    task_category: str
    persona_relevant_dimensions: List[str]
    persona_id: Optional[str]
    persona: Optional[Dict[str, Any]]
    fanout_branches: List[Dict[str, Any]]
    raw_search_results: List[Dict[str, Any]]
    final_answer: str
    cost_proxy: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
