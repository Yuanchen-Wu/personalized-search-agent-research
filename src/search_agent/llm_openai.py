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

from .config import get_openai_api_key
from .llm_client import LLMClient, retry_after_from_error


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
        self, prompt: str, *, model: str, temperature: float, json_mode: bool
    ) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self._max_output_tokens(),
        }
        seed = self._seed()
        if seed is not None:
            kwargs["seed"] = seed
        if json_mode and self.supports_response_format:
            # TODO(M3): json_object mode requires the literal word "json" in the
            # prompt; ours use "JSON" (uppercase). Verify acceptance on the first
            # real OpenAI run (not yet live-tested).
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client().chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()

    def _retry_after_seconds(self, err: Exception, attempt: int) -> float:
        retry_after = retry_after_from_error(err)
        if retry_after is not None:
            return min(retry_after + 1.0, 90.0)
        return min(2.0 ** (attempt - 1), 30.0)
