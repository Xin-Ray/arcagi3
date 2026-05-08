"""Tests for arc_agent.llm.LLMClient — uses an injected mock client (no real API calls)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from arc_agent.llm import LLMClient, LLMResponse


def _mock_anthropic_response(
    text: str = "ACTION1",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int = 0,
    cache_creation: int = 0,
):
    """Build an object shaped like anthropic.types.Message for tests."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
        ),
    )


def test_complete_returns_text_and_usage() -> None:
    fake = MagicMock()
    fake.messages.create.return_value = _mock_anthropic_response(
        text="hello", input_tokens=20, output_tokens=8, cache_read=15, cache_creation=5
    )
    client = LLMClient(model="claude-haiku-4-5", max_tokens=128, client=fake)

    out = client.complete(system="you are a test agent", user="say hi")

    assert isinstance(out, LLMResponse)
    assert out.text == "hello"
    assert out.input_tokens == 20
    assert out.output_tokens == 8
    assert out.cache_read_input_tokens == 15
    assert out.cache_creation_input_tokens == 5


def test_complete_passes_cache_control_and_system() -> None:
    fake = MagicMock()
    fake.messages.create.return_value = _mock_anthropic_response()
    client = LLMClient(model="claude-haiku-4-5", max_tokens=64, client=fake)

    client.complete(system="static prompt", user="dynamic question")

    kwargs = fake.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["max_tokens"] == 64
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"] == "static prompt"
    assert kwargs["messages"] == [{"role": "user", "content": "dynamic question"}]


def test_complete_handles_missing_text_block() -> None:
    fake = MagicMock()
    fake.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="...")],
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0
        ),
    )
    out = LLMClient(client=fake).complete(system="s", user="u")
    assert out.text == ""


def test_cached_fraction_zero_when_no_input() -> None:
    out = LLMResponse(text="")
    assert out.cached_fraction == 0.0


def test_cached_fraction_high_after_cache_warm() -> None:
    out = LLMResponse(
        text="",
        input_tokens=10,
        cache_read_input_tokens=90,
        cache_creation_input_tokens=0,
    )
    # 90 / (10+90) = 0.9
    assert out.cached_fraction == 0.9


def test_cumulative_token_counters_accumulate() -> None:
    fake = MagicMock()
    fake.messages.create.side_effect = [
        _mock_anthropic_response(input_tokens=10, output_tokens=5, cache_read=20, cache_creation=3),
        _mock_anthropic_response(input_tokens=15, output_tokens=8, cache_read=25, cache_creation=0),
    ]
    client = LLMClient(model="claude-haiku-4-5", client=fake)
    client.complete(system="s", user="u1")
    client.complete(system="s", user="u2")

    assert client.total_input_tokens == 25
    assert client.total_output_tokens == 13
    assert client.total_cache_read_tokens == 45
    assert client.total_cache_creation_tokens == 3


def test_estimated_cost_usd_haiku() -> None:
    fake = MagicMock()
    fake.messages.create.return_value = _mock_anthropic_response(
        input_tokens=1_000_000, output_tokens=1_000_000, cache_read=0, cache_creation=0
    )
    client = LLMClient(model="claude-haiku-4-5", client=fake)
    client.complete(system="s", user="u")
    # haiku: $0.8/MTok input + $4/MTok output = $4.8
    assert abs(client.estimated_cost_usd() - 4.8) < 0.01
