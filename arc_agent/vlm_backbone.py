"""Qwen2.5-VL backbone loader + generate wrapper.

Heavy deps (`torch`, `transformers`, `bitsandbytes`, `qwen_vl_utils`, `peft`) are
imported **lazily** inside the functions, so unit tests that only exercise
`VLMAgent` prompt/parse logic can import this module on a machine without a GPU
stack installed.

Three entry points:

- `load_model(model_path, quantize)` — returns `(model, processor)` tuple.
- `generate(model, processor, image, prompt, *, system)` — runs one generation.
- `HFBackbone` — thin wrapper exposing `.generate(image, prompt, system=...)`
  so `VLMAgent` can stay decoupled from the (model, processor) tuple shape and
  accept a fake in tests.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, runtime_checkable


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


@runtime_checkable
class VLMBackbone(Protocol):
    """Single-method interface VLMAgent depends on. Mock in tests."""

    def generate(self, image: Any, prompt: str, *, system: str) -> str: ...


def load_model(
    model_path: str = DEFAULT_MODEL,
    quantize: Optional[str] = "4bit",
) -> Tuple[Any, Any]:
    """Load Qwen2.5-VL with optional bitsandbytes quantization.

    Args:
        model_path: HuggingFace repo id or local checkpoint path.
        quantize: "4bit" (default) | "8bit" | None (full precision).

    Returns:
        (model, processor) tuple, both in eval mode, on the chosen device.

    Heavy imports are deferred to keep this module importable without a GPU
    stack. Raises ImportError with a clear message if `transformers` etc.
    aren't installed.
    """
    try:
        import torch
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )
    except ImportError as e:
        raise ImportError(
            "vlm_backbone.load_model requires `torch` + `transformers`. "
            "Install training deps: pip install torch transformers accelerate "
            "bitsandbytes qwen-vl-utils"
        ) from e

    kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
    if quantize == "4bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "quantize='4bit' needs bitsandbytes — pip install bitsandbytes"
            ) from e
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize is not None:
        raise ValueError(f"quantize must be '4bit' | '8bit' | None, got {quantize!r}")

    if quantize is None:
        kwargs["device_map"] = "auto"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor


def generate(
    model: Any,
    processor: Any,
    image: Any,
    prompt: str,
    *,
    system: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Run one Qwen2.5-VL generation and return decoded text.

    Builds the chat-template messages (system + user-with-image), tokenizes,
    runs `model.generate`, and decodes only the newly generated tokens.
    """
    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise ImportError(
            "vlm_backbone.generate requires torch + qwen_vl_utils — "
            "pip install torch qwen-vl-utils"
        ) from e

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    })

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0]


class HFBackbone:
    """Bundle (model, processor) behind the VLMBackbone Protocol.

    `VLMAgent` only needs `.generate(image, prompt, system=...)`, so it can
    accept either this real backbone or a stub in tests.
    """

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    def generate(self, image: Any, prompt: str, *, system: str = "") -> str:
        return generate(self.model, self.processor, image, prompt, system=system)

    @classmethod
    def load(
        cls,
        model_path: str = DEFAULT_MODEL,
        quantize: Optional[str] = "4bit",
    ) -> "HFBackbone":
        model, processor = load_model(model_path, quantize)
        return cls(model, processor)
