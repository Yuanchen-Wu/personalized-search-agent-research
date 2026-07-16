"""Google Gemini provider for the LLM client abstraction.

Implements :class:`~llm_client.LLMClient` for Gemini (google-genai). The SDK
client is built lazily so importing this module never requires the API key or the
package to be present. A backward-compatible ``call_gemini`` shim is retained for
callers not yet migrated to ``llm_client.generate``; it simply routes through the
abstraction (and therefore picks up multi-provider routing + shared throttling).
"""

from __future__ import annotations

import re
from typing import Optional

from .config import DEFAULT_GEMINI_MODEL, get_gemini_api_key
from .llm_client import LLMClient, get_client


class GeminiClient(LLMClient):
    """Gemini backend via the google-genai SDK."""

    name = "gemini"
    default_max_rpm = 15.0  # free-tier-friendly; override with GEMINI_MAX_RPM

    _sdk_client = None  # class-level: all instances share one google-genai client

    def _client(self):
        def factory():
            from google import genai  # lazy import keeps module import cheap
            from google.genai import types

            return genai.Client(
                api_key=get_gemini_api_key(),
                http_options=types.HttpOptions(
                    timeout=int(self._timeout_seconds() * 1000)  # genai timeout is ms
                ),
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
        from google.genai import types

        config_kwargs = {"temperature": temperature}
        max_output_tokens = self._max_output_tokens()
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens
        effective_seed = seed if seed is not None else self._seed()
        if effective_seed is not None:
            config_kwargs["seed"] = effective_seed
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        response = self._client().models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = getattr(response, "text", None)
        return text.strip() if text else ""

    def _retry_after_seconds(self, err: Exception, attempt: int) -> float:
        """Honor Gemini's ``retryDelay`` body hint on 429s; else defer to the base
        (``Retry-After`` header / capped exponential backoff)."""
        text = str(err)
        if "429" in text or "RESOURCE_EXHAUSTED" in text:
            m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", text)
            if m:
                return min(float(m.group(1)) + 1.0, 90.0)
        return super()._retry_after_seconds(err, attempt)


def call_gemini(
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    *,
    max_retries: int = 6,
    temperature: float = 0.7,
    response_mime_type: Optional[str] = None,
    seed: Optional[int] = None,
    throttle: bool = True,
) -> str:
    """Backward-compatible shim. Prefer ``llm_client.generate``.

    Routes through the provider abstraction so existing callers transparently get
    multi-provider support and shared rate limiting. ``response_mime_type=
    "application/json"`` maps to ``json_mode=True``. ``seed`` and ``throttle`` are
    forwarded to :meth:`LLMClient.generate` — the per-call ``seed`` overrides
    ``LLM_SEED``, and ``throttle=False`` skips client-side pacing for callers that
    manage their own rate (e.g. the self-pacing judge harness).
    """
    return get_client(model).generate(
        prompt,
        model=model,
        temperature=temperature,
        json_mode=(response_mime_type == "application/json"),
        max_retries=max_retries,
        seed=seed,
        throttle=throttle,
    )
