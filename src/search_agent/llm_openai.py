"""OpenAI provider for the LLM client abstraction.

Implements :class:`~llm_client.LLMClient` for the OpenAI Chat Completions API. The
SDK client is built lazily, so importing this module requires neither the API key
nor the ``openai`` package -- only an actual call does. Use model ids like
``gpt-4o`` (or your target ``gpt-*`` model).

Note: some reasoning models (``o1``/``o3``/``gpt-5`` family) constrain or rename
parameters (e.g. fixed temperature, ``max_completion_tokens`` instead of
``max_tokens``). If you target one, adjust ``_raw_generate`` accordingly; the
standard Chat Completions call below covers the common case.
"""

from __future__ import annotations

from typing import Optional

from .config import get_openai_api_key
from .llm_client import LLMClient


class OpenAIClient(LLMClient):
    """OpenAI backend via the Chat Completions API."""

    name = "openai"
    default_max_rpm = 60.0  # override with OPENAI_MAX_RPM
    # Some OpenAI-compatible endpoints (e.g. the Llama subclass) don't support the
    # response_format JSON param; subclasses set this False to skip it.
    supports_response_format = True

    _sdk_client = None  # class-level shared SDK client

    def _client(self):
        def factory():
            from openai import OpenAI  # lazy import

            return OpenAI(
                api_key=get_openai_api_key(), timeout=self._timeout_seconds()
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
        messages = [{"role": "user", "content": prompt}]
        if json_mode:
            # Steer JSON with a system message. The lowercase "json" also satisfies
            # the literal-word requirement of OpenAI's json_object mode, and gives
            # response_format-less endpoints (Llama) a nudge; the caller's tolerant
            # parser remains the backstop.
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "Respond with only valid json — no prose, no markdown fences.",
                },
            )
        kwargs = {"model": model, "messages": messages, "temperature": temperature}
        max_output_tokens = self._max_output_tokens()
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
        effective_seed = seed if seed is not None else self._seed()
        if effective_seed is not None:
            kwargs["seed"] = effective_seed
        if json_mode and self.supports_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client().chat.completions.create(**kwargs)
        # Surface an empty/refused/truncated-to-nothing completion instead of
        # silently returning "" (which the caller would treat as a valid answer).
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            raise RuntimeError(
                "OpenAI returned no text content "
                f"(finish_reason={getattr(choice, 'finish_reason', None)!r})"
            )
        return content.strip()
