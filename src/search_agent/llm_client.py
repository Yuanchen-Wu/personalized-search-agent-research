"""Provider-agnostic LLM client abstraction.

This is the single integration point for every LLM call in the project. We
deliberately hand-roll it (instead of depending on a multi-provider library) so
the exact request/response behavior is transparent and reproducible: there is no
hidden prompt rewriting, message reformatting, or response post-processing beyond
what is written in these files.

Design
------
- :class:`LLMClient` is an abstract base that owns the *shared* behavior every
  provider needs — proactive per-provider rate limiting and quota-aware retries —
  and delegates the one provider-specific thing (``_raw_generate``) to subclasses
  in ``llm_gemini`` / ``llm_anthropic`` / ``llm_openai``.
- :func:`get_client` routes a model id to a provider by prefix (``gemini-*`` /
  ``claude-*`` / ``gpt-*``); :func:`generate` is the convenience entrypoint.

Rate limiting is *per provider*: each provider paces against its own
``<PROVIDER>_MAX_RPM`` env knob (e.g. ``GEMINI_MAX_RPM``), so running one model
never throttles against another's budget. A client-side limiter cannot raise your
real quota — it only prevents bursting and rides out transient limits.
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional

# Per-provider pacing state, shared across all instances of a provider so a
# thread pool (e.g. the judge's ThreadPoolExecutor) shares one pace per provider.
_throttle_lock = threading.Lock()
_next_allowed_at: Dict[str, float] = {}

# Serializes lazy construction of each provider's SDK client (build-once).
_sdk_build_lock = threading.Lock()


def retry_after_from_error(err: Exception) -> Optional[float]:
    """Best-effort ``Retry-After`` (seconds) from a provider error's HTTP headers.

    Anthropic/OpenAI SDK errors expose the upstream response; on a 429 it usually
    carries a ``retry-after`` header. Returns ``None`` if not present/parseable.
    """
    resp = getattr(err, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        val = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:  # noqa: BLE001 - headers object shape varies by SDK
        return None
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


class LLMClient(ABC):
    """Abstract LLM provider. Subclasses implement ``_raw_generate`` only."""

    #: Short provider id; also the prefix of its ``<NAME>_MAX_RPM`` env knob.
    name: str = "base"
    #: Default requests-per-minute cap when ``<NAME>_MAX_RPM`` is unset.
    default_max_rpm: float = 15.0

    # --- provider-specific hooks -------------------------------------------
    @abstractmethod
    def _raw_generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        json_mode: bool,
        seed: Optional[int],
    ) -> str:
        """Make the actual provider API call and return the response text."""

    def _retry_after_seconds(self, err: Exception, attempt: int) -> float:
        """Seconds to wait before retrying after ``err`` on ``attempt``.

        Default: capped exponential backoff. Providers override to honor a
        server-supplied hint (Gemini ``retryDelay`` / a ``Retry-After`` header).
        """
        return min(2.0 ** (attempt - 1), 30.0)

    # --- shared behavior ----------------------------------------------------
    def _max_rpm(self) -> float:
        raw = os.environ.get(f"{self.name.upper()}_MAX_RPM")
        if raw is None:
            return self.default_max_rpm
        try:
            return float(raw)
        except ValueError:
            return self.default_max_rpm

    def _max_output_tokens(self) -> Optional[int]:
        """Optional hard cap on generated tokens (``LLM_MAX_OUTPUT_TOKENS``).

        Off by default (``None``) so output is bounded only by the model's own
        maximum and long answers are never silently truncated. Set the env var to
        impose a ceiling (e.g. to bound pathological runaway output). Providers
        whose API *requires* a max (Anthropic) supply their own fallback.
        """
        raw = os.environ.get("LLM_MAX_OUTPUT_TOKENS")
        if raw is None:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _timeout_seconds(self) -> float:
        """Per-call API timeout (LLM_TIMEOUT_SECONDS, default 120) so a hung
        request can't stall a whole batch run."""
        raw = os.environ.get("LLM_TIMEOUT_SECONDS")
        if raw is None:
            return 120.0
        try:
            return float(raw)
        except ValueError:
            return 120.0

    def _seed(self) -> Optional[int]:
        """Optional generation seed for best-effort reproducibility (LLM_SEED).

        Applied where the provider supports it (OpenAI/Gemini); Anthropic has no
        seed param. None => provider default (non-deterministic).
        """
        raw = os.environ.get("LLM_SEED")
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @classmethod
    def _get_or_build_sdk(cls, factory):
        """Build this provider's SDK client once, thread-safely (double-checked)."""
        if cls._sdk_client is None:
            with _sdk_build_lock:
                if cls._sdk_client is None:
                    cls._sdk_client = factory()
        return cls._sdk_client

    @staticmethod
    def _is_retryable(err: Exception) -> bool:
        """True for transient failures worth retrying (429 / 5xx / network);
        False for unrecoverable client errors (auth, bad request, missing model)."""
        code = getattr(err, "status_code", None)
        if code is None:
            code = getattr(err, "code", None)
        if isinstance(code, bool):
            code = None  # bool is an int subclass; ignore
        if isinstance(code, int):
            # 429 plus 408/409/425 (request-timeout / conflict / too-early) are
            # transient and worth retrying; other 4xx are client errors.
            if code in (429, 408, 409, 425):
                return True
            if 400 <= code < 500:
                return False
            if code >= 500:
                return True
        s = str(err).lower()
        if any(t in s for t in (
            "429", "resource_exhausted", "rate limit", "ratelimit", "overloaded",
            "timeout", "timed out", "temporarily", "unavailable", "connection",
            "500", "502", "503", "504",
        )):
            return True
        if any(t in s for t in (
            "401", "403", "404", "400", "422", "unauthorized", "permission denied",
            "invalid api key", "api key not valid", "not found", "invalid_request",
        )):
            return False
        return True  # unknown: default to retry (batch robustness)

    def _throttle(self) -> None:
        """Block until this provider's next allowed call time.

        Reserves a slot ``60/RPM`` after the previous one and sleeps *outside* the
        lock, so concurrent callers serialize their slots without one provider's
        sleep blocking another's.
        """
        rpm = self._max_rpm()
        if rpm <= 0:
            return
        min_interval = 60.0 / rpm
        with _throttle_lock:
            now = time.monotonic()
            scheduled = max(now, _next_allowed_at.get(self.name, 0.0))
            _next_allowed_at[self.name] = scheduled + min_interval
            wait = scheduled - now
        if wait > 0:
            time.sleep(wait)

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.7,
        json_mode: bool = False,
        max_retries: int = 6,
        seed: Optional[int] = None,
        throttle: bool = True,
    ) -> str:
        """Generate text, with per-provider pacing and quota-aware retries.

        Args:
            prompt: Full prompt string (a single user turn).
            model: Provider model id (selects the backend via :func:`get_client`).
            temperature: Sampling temperature.
            json_mode: Ask the provider to return strict JSON where supported.
            max_retries: Attempts before giving up.
            seed: Per-call generation seed; takes precedence over the ``LLM_SEED``
                env default. ``None`` => ``LLM_SEED`` or the provider default.
                Ignored by providers without a seed parameter (Anthropic).
            throttle: Apply client-side per-provider pacing before the first
                attempt. Pass ``False`` when the caller already manages its own
                concurrency/rate (e.g. the judge harness paces itself) to avoid
                throttling twice.

        Returns:
            The model's text output (stripped); ``""`` if it produced none.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            # Pace only the first attempt; retries are already spaced by
            # _retry_after_seconds below, so re-throttling here would double the
            # wait and let failed attempts run the shared per-provider RPM cursor
            # away into the future, stalling other threads.
            if throttle and attempt == 1:
                self._throttle()
            try:
                return self._raw_generate(
                    prompt,
                    model=model,
                    temperature=temperature,
                    json_mode=json_mode,
                    seed=seed,
                )
            except Exception as err:  # noqa: BLE001 - broad retry coverage
                last_err = err
                # Only retry transient failures; fail fast on unrecoverable ones
                # (bad key / model / request) instead of burning the retry budget.
                if attempt < max_retries and self._is_retryable(err):
                    time.sleep(self._retry_after_seconds(err, attempt))
                else:
                    break
        raise RuntimeError(
            f"{self.name} call failed after {attempt} attempt(s): {last_err}"
        ) from last_err


# ---------------------------------------------------------------------------
# Model -> provider routing.
# ---------------------------------------------------------------------------

# One cached client per provider (each lazily builds its own SDK client).
_client_cache: Dict[str, LLMClient] = {}


def provider_for(model: str) -> str:
    """Map a model id to its provider name by prefix."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"model must be a non-empty string, got {model!r}.")
    m = model.lower()
    if m.startswith(("gemini", "models/gemini")):
        return "gemini"
    if m.startswith(("claude", "anthropic/")):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4", "openai/")):
        return "openai"
    if m.startswith(("llama", "meta-llama/")):
        return "llama"
    raise ValueError(
        f"No provider registered for model {model!r}. "
        "Expected a prefix of: gemini-*, claude-*, gpt-*/o1-*/o3-*, llama-*."
    )


def get_client(model: str) -> LLMClient:
    """Return the (cached) provider client that serves ``model``."""
    provider = provider_for(model)
    client = _client_cache.get(provider)
    if client is None:
        if provider == "gemini":
            from .llm_gemini import GeminiClient

            client = GeminiClient()
        elif provider == "anthropic":
            from .llm_anthropic import AnthropicClient

            client = AnthropicClient()
        elif provider == "openai":
            from .llm_openai import OpenAIClient

            client = OpenAIClient()
        elif provider == "llama":
            from .llm_llama import LlamaClient

            client = LlamaClient()
        else:  # pragma: no cover - provider_for guards this
            raise ValueError(f"Unhandled provider {provider!r}")
        _client_cache[provider] = client
    return client


def generate(
    prompt: str,
    *,
    model: str,
    temperature: float = 0.7,
    json_mode: bool = False,
    max_retries: int = 6,
    seed: Optional[int] = None,
    throttle: bool = True,
) -> str:
    """Convenience entrypoint: route ``model`` to its provider and generate."""
    return get_client(model).generate(
        prompt,
        model=model,
        temperature=temperature,
        json_mode=json_mode,
        max_retries=max_retries,
        seed=seed,
        throttle=throttle,
    )
