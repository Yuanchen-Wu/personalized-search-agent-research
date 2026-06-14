"""Lightweight dataclass schemas shared across the pipeline.

These are deliberately plain dataclasses (no Pydantic) to keep the prototype
inspectable and dependency-light. Each schema has an ``as_dict`` helper so we
can serialize everything cleanly into JSONL logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Branch types used throughout the fan-out logic. Kept as a simple tuple so it
# is trivial to validate against and to extend later.
BRANCH_TYPES = ("generic", "personalized", "constraint", "disconfirming")

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
    """A user persona / context bundle used to drive personalization."""

    persona_id: str
    description: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Persona":
        return cls(
            persona_id=data["persona_id"],
            description=data.get("description", ""),
            attributes=data.get("attributes", {}) or {},
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
    timestamp: str
    variant: str
    user_query: str
    persona_id: Optional[str]
    persona: Optional[Dict[str, Any]]
    fanout_branches: List[Dict[str, Any]]
    raw_search_results: List[Dict[str, Any]]
    final_answer: str
    cost_proxy: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
