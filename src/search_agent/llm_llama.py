"""Meta Llama API provider (OpenAI-compatible).

Meta's first-party Llama API (https://llama.developer.meta.com) exposes an
OpenAI-compatible endpoint, so this provider reuses :class:`OpenAIClient`'s
request/parse logic and only swaps the base URL + key. The default base URL
targets Meta's compatibility endpoint; override ``LLAMA_BASE_URL`` to point at any
other OpenAI-compatible Llama host (Ollama, Groq, Together, vLLM, ...).

Model ids look like ``Llama-4-Maverick-17B-128E-Instruct-FP8``,
``Llama-4-Scout-17B-16E-Instruct-FP8`` or ``Llama-3.3-70B-Instruct``.

Note: the exact compatibility base URL should be confirmed against your Llama API
dashboard; it is exposed as ``LLAMA_BASE_URL`` precisely so it is trivial to
correct without code changes. This provider is structurally complete but has not
been live-tested here (no Meta key was available).
"""

from __future__ import annotations

import os

from .config import get_llama_api_key
from .llm_openai import OpenAIClient

# Meta Llama API OpenAI-compatibility base URL (host api.llama.com + /compat/v1).
# Override via LLAMA_BASE_URL (also lets you target Ollama/Groq/Together/vLLM).
DEFAULT_LLAMA_BASE_URL = "https://api.llama.com/compat/v1"


class LlamaClient(OpenAIClient):
    """Llama backend via an OpenAI-compatible endpoint (Meta's API by default)."""

    name = "llama"
    default_max_rpm = 30.0  # override with LLAMA_MAX_RPM
    # The compat endpoint's support for OpenAI's response_format JSON param is
    # unverified, so we rely on the prompt's explicit "strict JSON" instruction
    # plus the callers' tolerant parser (json_mode is a no-op here).
    supports_response_format = False

    _sdk_client = None  # own cache, distinct from OpenAIClient's

    def _client(self):
        def factory():
            from openai import OpenAI  # reuse the OpenAI SDK against a custom base_url

            base_url = os.environ.get("LLAMA_BASE_URL", DEFAULT_LLAMA_BASE_URL)
            return OpenAI(
                api_key=get_llama_api_key(),
                base_url=base_url,
                timeout=self._timeout_seconds(),
            )

        return self._get_or_build_sdk(factory)
