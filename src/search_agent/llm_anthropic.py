"""Anthropic (Claude) provider for the LLM client abstraction.

Implements :class:`~llm_client.LLMClient` for the Anthropic Messages API. The SDK
client is built lazily, so importing this module requires neither the API key nor
the ``anthropic`` package -- only an actual call does. Use model ids like
``claude-opus-4-8``.
"""

from __future__ import annotations

from typing import Optional

from .config import get_anthropic_api_key
from .llm_client import LLMClient, retry_after_from_error


class AnthropicClient(LLMClient):
    """Claude backend via the Anthropic Messages API."""

    name = "anthropic"
    default_max_rpm = 50.0  # override with ANTHROPIC_MAX_RPM

    _sdk_client = None  # class-level shared SDK client

    def _client(self):
        def factory():
            import anthropic  # lazy: only needed when actually calling Claude

            return anthropic.Anthropic(
                api_key=get_anthropic_api_key(), timeout=self._timeout_seconds()
            )

        return self._get_or_build_sdk(factory)

    def _raw_generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        json_mode: bool,
        seed: Optional[int] = None,
    ) -> str:
        # TODO(M3): the Messages API has no JSON mode, so `json_mode` is a no-op
        # here -- we rely on the prompt's "return strict JSON" instruction plus the
        # callers' tolerant parser (fanout._extract_json / utils.parse_json_response).
        # Verify output parses cleanly on the first real Claude run (not yet
        # live-tested). Anthropic has no seed parameter, so `seed` is ignored.
        # The Messages API *requires* max_tokens, so fall back to 8192 when
        # LLM_MAX_OUTPUT_TOKENS is unset (override the env var to change it).
        max_tokens = self._max_output_tokens()
        if max_tokens is None:
            max_tokens = 8192
        response = self._client().messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        return "".join(parts).strip()

    def _retry_after_seconds(self, err: Exception, attempt: int) -> float:
        retry_after = retry_after_from_error(err)
        if retry_after is not None:
            return min(retry_after + 1.0, 90.0)
        return min(2.0 ** (attempt - 1), 30.0)
