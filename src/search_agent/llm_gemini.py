"""Thin wrapper around the Google Gemini API.

We keep this intentionally minimal: a single ``call_gemini`` function plus a
small retry loop. The client is lazily constructed so that importing this module
does not require the API key to be present (useful for tests / dry runs).
"""

from __future__ import annotations

import time
from typing import Optional

from .config import DEFAULT_GEMINI_MODEL, get_gemini_api_key

_client = None  # lazily initialized google-genai client


def _get_client():
    """Construct (once) and return the google-genai client."""
    global _client
    if _client is None:
        from google import genai  # imported lazily so import-time stays cheap

        _client = genai.Client(api_key=get_gemini_api_key())
    return _client


def call_gemini(
    prompt: str,
    model: str = DEFAULT_GEMINI_MODEL,
    *,
    max_retries: int = 3,
    temperature: float = 0.7,
    response_mime_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """Call Gemini with a single text prompt and return the response text.

    Args:
        prompt: The full prompt string.
        model: Gemini model name (default ``gemini-flash-latest``).
        max_retries: Number of attempts on transient failures.
        temperature: Sampling temperature.
        response_mime_type: If set to ``"application/json"``, asks Gemini to
            return JSON. Useful for structured fan-out generation.
        seed: Random seed for model sampling generation.

    Returns:
        The model's text output (stripped). Returns an empty string if the
        model produced no text.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    from google.genai import types

    client = _get_client()

    config_kwargs = {"temperature": temperature}
    if response_mime_type is not None:
        config_kwargs["response_mime_type"] = response_mime_type
    if seed is not None:
        config_kwargs["seed"] = seed
    gen_config = types.GenerateContentConfig(**config_kwargs)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_config,
            )
            text = getattr(response, "text", None)
            return text.strip() if text else ""
        except Exception as err:  # noqa: BLE001 - we want broad retry coverage
            last_err = err
            if attempt < max_retries:
                # Simple exponential backoff: 1s, 2s, 4s, ...
                time.sleep(2 ** (attempt - 1))
            else:
                break

    raise RuntimeError(
        f"Gemini call failed after {max_retries} attempts: {last_err}"
    ) from last_err
