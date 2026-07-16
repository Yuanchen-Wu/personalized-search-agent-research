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
        # The Messages API has no native JSON mode, so when json_mode is set we
        # steer output with a system instruction and lean on the caller's tolerant
        # parser as a backstop (not yet live-tested on Claude -- see M3). Anthropic
        # has no seed parameter, so `seed` is ignored. The API *requires*
        # max_tokens, so fall back to 8192 when LLM_MAX_OUTPUT_TOKENS is unset.
        max_tokens = self._max_output_tokens()
        if max_tokens is None:
            max_tokens = 8192
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["system"] = (
                "Respond with only valid JSON -- no prose, no explanation, and no "
                "markdown code fences."
            )
        response = self._client().messages.create(**kwargs)
        parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError(
                "Anthropic returned no text content "
                f"(stop_reason={getattr(response, 'stop_reason', None)!r})"
            )
        return text

    def _retry_after_seconds(self, err: Exception, attempt: int) -> float:
        retry_after = retry_after_from_error(err)
        if retry_after is not None:
            return min(retry_after + 1.0, 90.0)
        return min(2.0 ** (attempt - 1), 30.0)
