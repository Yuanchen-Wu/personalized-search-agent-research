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
BRANCH_TYPES = (
    "generic",
    "personalized",
    "constraint",
    "disconfirming",
    "supplementary",
)

FIXED_FANOUT_METHODS = ("fixed_k1", "fixed_k2", "fixed_k4", "fixed_k8")

# Variant identifiers. Centralized here so the CLI, batch runner, and fan-out
# logic all agree on the canonical names.
VARIANTS = (
    "V0_generic_single",
    "V1_generic_fanout",
    "V2_synthesis_only_personalization",
    "V3_fanout_only_personalization",
    "V4_personalized_fanout",
    "V5_mixed_fanout",
) + FIXED_FANOUT_METHODS


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
    macro_domain: str = "education"
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
            macro_domain=data.get("macro_domain", "education"),
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
    macro_domain: str = "education"
    persona_relevant_dimensions: List[str] = field(default_factory=list)
    search_required: bool = True
    expected_personalization_stage: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryRecord":
        query_id = data.get("query_id", data.get("id", data.get("example_id", "unknown")))
        
        task_category = data.get("task_category", data.get("category", data.get("domain", "unknown")))
        task_type = data.get("task_type", "unknown")
        macro_domain = data.get("macro_domain", "education")
        
        # Backward compatibility mapping
        cat_lower = task_category.lower()
        tt_lower = task_type.lower()
        
        if tt_lower == "search_native":
            task_type = "retrieval_sensitive"
        elif tt_lower == "synthesis_native":
            task_type = "synthesis_sensitive"

        # Check if it's already a known new category
        known_retrieval = {
            "travel_dining", "shopping_product_recommendation",
            "jurisdiction_resource_lookup", "form_policy_deadline_lookup",
            "product_or_program_comparison", "current_rule_limit_lookup"
        }
        known_synthesis = {
            "technical_explanation", "personal_decision_strategy",
            "legal_issue_explanation", "legal_decision_strategy",
            "financial_concept_explanation", "financial_decision_strategy"
        }

        if task_category in known_retrieval:
            task_type = "retrieval_sensitive"
        elif task_category in known_synthesis:
            task_type = "synthesis_sensitive"
        else:
            # Fallback mappings for education and travel-like legacies
            if "travel" in cat_lower or "local" in cat_lower or "dining" in cat_lower:
                task_category = "travel_dining"
                task_type = "retrieval_sensitive"
            elif any(x in cat_lower for x in ["shopping", "commerce", "product", "textbook", "resource", "course"]):
                task_category = "shopping_product_recommendation"
                task_type = "retrieval_sensitive"
            elif "technical" in cat_lower and "explanation" in cat_lower:
                task_category = "technical_explanation"
                task_type = "synthesis_sensitive"
            elif any(x in cat_lower for x in ["professional", "career", "education", "decision", "strategy"]):
                task_category = "personal_decision_strategy"
                task_type = "synthesis_sensitive"
            else:
                # Fallbacks for unknown categories
                if task_type == "retrieval_sensitive":
                    task_category = "shopping_product_recommendation"
                elif task_type == "synthesis_sensitive":
                    task_category = "technical_explanation"
                else:
                    task_type = "unknown"
                    print(f"Warning: Could not map task_type/category for query {query_id}. Defaulting to unknown.")

        expected_stage = data.get("expected_personalization_stage")
        if not expected_stage or expected_stage == "unknown":
            if task_type == "retrieval_sensitive":
                expected_stage = "fanout_retrieval"
            elif task_type == "synthesis_sensitive":
                expected_stage = "final_synthesis"
            else:
                expected_stage = "unknown"

        search_req = data.get("search_required", True)

        metadata = {}
        for k, v in data.items():
            if k not in ["query", "query_id", "id", "example_id", "task_type", "task_category", "category", "domain", "persona_relevant_dimensions", "search_required", "expected_personalization_stage", "macro_domain"]:
                metadata[k] = v

        # Flatten nested metadata if present
        if "metadata" in metadata and isinstance(metadata["metadata"], dict):
            nested = metadata.pop("metadata")
            metadata.update(nested)

        return cls(
            query=data.get("query", ""),
            query_id=query_id,
            task_type=task_type,
            task_category=task_category,
            macro_domain=macro_domain,
            persona_relevant_dimensions=data.get("persona_relevant_dimensions", []),
            search_required=search_req,
            expected_personalization_stage=expected_stage,
            metadata=metadata
        )



@dataclass
class FanoutBranch:
    """A single search branch produced during query fan-out."""

    branch_type: str  # one of BRANCH_TYPES
    query: str
    rationale: str = ""
    used_persona_fields: List[str] = field(default_factory=list)
    information_need: str = ""
    priority_rank: Optional[int] = None

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
    """The full structured record of a single agent run."""

    run_id: str
    experiment_name: str
    timestamp: str
    variant: str
    user_query: str
    query_id: str
    task_type: str
    task_category: str
    macro_domain: str
    persona_relevant_dimensions: List[str]
    search_required: bool
    expected_personalization_stage: str
    persona_id: Optional[str]
    persona: Optional[Dict[str, Any]]
    fanout_branches: List[Dict[str, Any]]
    raw_search_results: List[Dict[str, Any]]
    final_answer: str
    cost_proxy: Dict[str, Any]
    method: str = ""
    seed: Optional[int] = None
    planner_model: Optional[str] = None
    synthesis_model: Optional[str] = None
    requested_fanout_count: Optional[int] = None
    realized_fanout_count: Optional[int] = None
    full_candidate_plan_id: Optional[str] = None
    executed_fanout_prefix: List[Dict[str, Any]] = field(default_factory=list)
    branch_types_executed: List[str] = field(default_factory=list)
    information_needs_executed: List[str] = field(default_factory=list)
    priority_ranks_executed: List[int] = field(default_factory=list)
    deduplicated_search_results: List[Dict[str, Any]] = field(default_factory=list)
    exact_synthesis_evidence: List[Dict[str, Any]] = field(default_factory=list)
    num_planner_calls: int = 0
    num_synthesis_calls: int = 0
    num_tavily_calls: int = 0
    num_cache_hits: int = 0
    num_cache_misses: int = 0
    num_raw_results: int = 0
    num_unique_results: int = 0
    total_retrieved_context_size: int = 0
    total_synthesis_context_size: int = 0
    planner_latency: float = 0.0
    search_latency: float = 0.0
    synthesis_latency: float = 0.0
    total_latency: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d.get("method"):
            d["method"] = d.get("variant", "")
        return d
