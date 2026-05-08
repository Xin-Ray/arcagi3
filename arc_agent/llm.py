"""Claude API wrapper — auto-caches the static system prompt, tracks token usage.

Uses the official Anthropic SDK (claude-api skill). The SDK auto-retries 429/5xx
with exponential backoff (max_retries=2 default) — no need to roll our own.

Default model is `claude-opus-4-7` per the claude-api skill. Override via
constructor arg if you want a cheaper dev loop:
    LLMClient(model="claude-haiku-4-5")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic


@dataclass
class LLMResponse:
    """One Claude completion + cost breakdown."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def cached_fraction(self) -> float:
        """Share of input tokens served from cache. Higher = cheaper."""
        total = (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )
        return self.cache_read_input_tokens / total if total else 0.0


# Approximate pricing per million tokens (input / cache_read / output).
# Used only for cost estimation; not billed through this code.
_PRICING: dict[str, tuple[float, float, float]] = {
    "claude-opus-4-7":         (15.0,  1.50,  75.0),
    "claude-sonnet-4-6":       ( 3.0,  0.30,  15.0),
    "claude-haiku-4-5":        ( 0.8,  0.08,   4.0),
    "claude-haiku-4-5-20251001": (0.8, 0.08,   4.0),
}
_DEFAULT_PRICING = (10.0, 1.0, 30.0)  # conservative fallback for unknown models


class LLMClient:
    """Anthropic SDK wrapper with auto-cache and cumulative cost tracking.

    The system prompt is the cacheable prefix (it's the same every step within
    an episode); the user message carries the dynamic per-step state. With a
    long static system prompt this should cut per-step cost ~10×.

    Pure library — caller injects an `Anthropic` instance (good for tests) or
    we build one ourselves (the SDK reads ANTHROPIC_API_KEY from env).
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-7",
        max_tokens: int = 4096,
        client: Optional[Anthropic] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client or Anthropic()
        # cumulative usage across all complete() calls on this instance
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cache_read_tokens: int = 0
        self.total_cache_creation_tokens: int = 0

    def estimated_cost_usd(self) -> float:
        """Estimated total cost in USD based on accumulated token usage."""
        p_in, p_cr, p_out = _PRICING.get(self.model, _DEFAULT_PRICING)
        return (
            self.total_input_tokens * p_in / 1_000_000
            + self.total_cache_read_tokens * p_cr / 1_000_000
            + self.total_cache_creation_tokens * p_in / 1_000_000  # creation billed as input
            + self.total_output_tokens * p_out / 1_000_000
        )

    def complete(self, *, system: str, user: str) -> LLMResponse:
        """One blocking Claude call. Caches the (large, static) `system` prompt.

        Uses top-level `cache_control` for auto-caching of the last cacheable
        block. With our (system, user) shape the system prompt gets cached.
        """
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            cache_control={"type": "ephemeral"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in msg.content if b.type == "text"), "")
        u = msg.usage
        resp = LLMResponse(
            text=text,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_cache_read_tokens += resp.cache_read_input_tokens
        self.total_cache_creation_tokens += resp.cache_creation_input_tokens
        return resp
