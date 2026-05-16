"""Wrap Tier-1 samples into the ChatML training format expected by SFTTrainer.

The Qwen2.5 tokenizer's `apply_chat_template` returns the final string; for
SFTTrainer we expose the messages list under `messages` so the trainer can
apply the template itself (handles loss-mask on the assistant turn correctly).
See docs/arch_sft_tier1_zh.md §3 / §4.
"""
from __future__ import annotations

from typing import TypedDict

from .synth_tier1 import Sample


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatSample(TypedDict):
    messages: list[ChatMessage]
    task: str


def to_chat(sample: Sample) -> ChatSample:
    """Convert one Sample dict into the SFTTrainer messages format."""
    msgs: list[ChatMessage] = []
    if sample.get("system"):
        msgs.append({"role": "system", "content": sample["system"]})
    msgs.append({"role": "user", "content": sample["user"]})
    msgs.append({"role": "assistant", "content": sample["assistant"]})
    return {"messages": msgs, "task": sample["task"]}


def to_chat_many(samples: list[Sample]) -> list[ChatSample]:
    """Batch helper around `to_chat`."""
    return [to_chat(s) for s in samples]
