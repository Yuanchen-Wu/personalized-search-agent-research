"""Centralized configuration and environment variable handling.

API keys are read exclusively from environment variables. We optionally load a
local ``.env`` file (via python-dotenv) for convenience during development, but
no secret is ever hardcoded, logged, or printed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Project paths (computed early so dotenv can target the repo root reliably).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

try:
    # Load .env from the project root (not cwd), so batch/CLI work from any dir.
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    pass


# Default model and search parameters. Kept here so experiments stay consistent.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
DEFAULT_MAX_RESULTS_PER_BRANCH = 5
DEFAULT_SEARCH_DEPTH = "basic"
MAX_RESULTS_PER_BRANCH_FOR_SYNTHESIS = 5
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_RUNS_LOG = os.path.join(OUTPUTS_DIR, "placement_ablation_v1", "runs.jsonl")


class MissingAPIKeyError(RuntimeError):
    """Raised when a required API key is not set in the environment."""


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings. Never store the actual key values in logs."""

    gemini_model: str = DEFAULT_GEMINI_MODEL
    max_results_per_branch: int = DEFAULT_MAX_RESULTS_PER_BRANCH
    search_depth: str = DEFAULT_SEARCH_DEPTH


def get_gemini_api_key() -> str:
    """Return the Gemini API key from the environment or raise a clear error."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or export GEMINI_API_KEY in your shell."
        )
    return key


def get_tavily_api_key() -> str:
    """Return the Tavily API key from the environment or raise a clear error."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "TAVILY_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or export TAVILY_API_KEY in your shell."
        )
    return key
