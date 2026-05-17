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
    lora_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load Qwen2.5-VL with optional bitsandbytes quantization + LoRA adapter.

    Args:
        model_path: HuggingFace repo id or local checkpoint path.
        quantize: "4bit" (default) | "8bit" | None (full precision).
        lora_path: optional PEFT LoRA adapter dir; applied on top of the
            base model. Used by `scripts/run_validation.py` to evaluate
            trained checkpoints.

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

    if lora_path:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "lora_path requires `peft` — pip install peft"
            ) from e
        model = PeftModel.from_pretrained(model, lora_path)

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
    temperature: float = 0.0,
    constrained_schema: Optional[dict] = None,
) -> str:
    """Run one Qwen2.5-VL generation and return decoded text.

    Builds the chat-template messages (system + user-with-image), tokenizes,
    runs `model.generate`, and decodes only the newly generated tokens.

    If `constrained_schema` is given AND `lm-format-enforcer` is installed,
    decoding is restricted to outputs that parse against the JSON Schema. If
    the library is missing, falls back to free-form generation with a logged
    warning (so unit tests on machines without the dep still pass). Per
    `docs/ARCHITECTURE_AGENTS.md` §1 A2: this is the path that kills the
    bp35-style 76-row parse blowup.
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

    if image is None:
        # Text-only mode: skip the image content block entirely.
        user_content = [{"type": "text", "text": prompt}]
    else:
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    messages.append({"role": "user", "content": user_content})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    proc_kwargs: dict[str, Any] = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs:
        proc_kwargs["images"] = image_inputs
    if video_inputs:
        proc_kwargs["videos"] = video_inputs
    inputs = processor(**proc_kwargs).to(model.device)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature,
    }

    logits_processor = _build_schema_logits_processor(constrained_schema, processor)
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0]


def _build_schema_logits_processor(
    schema: Optional[dict], processor: Any
) -> Any:
    """Return a HF LogitsProcessorList constrained to `schema`, or None.

    Tries lm-format-enforcer first; absence is non-fatal (returns None and
    logs once). Kept as a helper so unit tests can monkeypatch it.
    """
    if schema is None:
        return None
    # Compat shim: newer transformers moved PreTrainedTokenizerBase to
    # transformers.tokenization_utils_base, but lm-format-enforcer 0.x still
    # imports it from transformers.tokenization_utils. Without this, the
    # integration's ImportError handler swallows the real error and the
    # fall-back path silently picks free-form decoding.
    try:
        import transformers.tokenization_utils as _tu
        import transformers.tokenization_utils_base as _tub
        if not hasattr(_tu, "PreTrainedTokenizerBase"):
            _tu.PreTrainedTokenizerBase = _tub.PreTrainedTokenizerBase
    except ImportError:
        pass

    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
        from transformers import LogitsProcessorList
        from transformers.generation.logits_process import (
            PrefixConstrainedLogitsProcessor,
        )
    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(
            "constrained_schema given but constrained-decoding deps "
            "couldn't be imported (%s) — falling back to free-form decoding. "
            "Install with `pip install lm-format-enforcer`.", e,
        )
        return None

    parser = JsonSchemaParser(schema)
    tokenizer = getattr(processor, "tokenizer", processor)
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    return LogitsProcessorList([PrefixConstrainedLogitsProcessor(prefix_fn, 1)])


class HFBackbone:
    """Bundle (model, processor) behind the VLMBackbone Protocol.

    `VLMAgent` only needs `.generate(image, prompt, system=...)`, so it can
    accept either this real backbone or a stub in tests.
    """

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    def generate(
        self,
        image: Any,
        prompt: str,
        *,
        system: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        constrained_schema: Optional[dict] = None,
    ) -> str:
        return generate(
            self.model, self.processor, image, prompt,
            system=system,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            constrained_schema=constrained_schema,
        )

    @classmethod
    def load(
        cls,
        model_path: str = DEFAULT_MODEL,
        quantize: Optional[str] = "4bit",
        lora_path: Optional[str] = None,
    ) -> "HFBackbone":
        model, processor = load_model(model_path, quantize, lora_path=lora_path)
        return cls(model, processor)


# ─── Causal LM backbone (Phi-4-mini-reasoning / SmolLM3-3B / etc) ─────────


class CausalLMBackbone:
    """Adapter exposing VLMBackbone Protocol over a text-only causal LM.

    Used to plug Phi-4-mini-reasoning, SmolLM3-3B, or any AutoModelForCausalLM
    HF model into the existing v3.2 ActionAgent path (which originally only
    supported Qwen2.5-VL). The `image` argument to `.generate()` is ignored —
    we are text-only by design.
    """

    def __init__(self, model: Any, tokenizer: Any, hf_id: str = "") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.hf_id = hf_id

    def generate(
        self,
        image: Any,
        prompt: str,
        *,
        system: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        constrained_schema: Optional[dict] = None,
    ) -> str:
        """Apply chat template, generate text, decode."""
        import torch

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback for tokenizers without chat template
            prompt_str = (
                (f"{system}\n\n" if system else "")
                + f"User: {prompt}\nAssistant:"
            )

        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    @classmethod
    def load(
        cls,
        model_path: str,
        quantize: Optional[str] = "4bit",
    ) -> "CausalLMBackbone":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if quantize == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantize == "8bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        model.eval()
        return cls(model, tokenizer, hf_id=model_path)


# ─── Factory ─────────────────────────────────────────────────────────────


# Known model IDs and which backbone class to use.
_BACKBONE_REGISTRY: dict[str, str] = {
    "Qwen/Qwen2.5-VL-3B-Instruct": "vl",
    "Qwen/Qwen2.5-VL-7B-Instruct": "vl",
    "microsoft/Phi-4-mini-reasoning": "causal",
    "microsoft/Phi-4-mini-instruct": "causal",
    "HuggingFaceTB/SmolLM3-3B": "causal",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "causal",
}


def make_backbone(
    model_path: str = DEFAULT_MODEL,
    quantize: Optional[str] = "4bit",
    lora_path: Optional[str] = None,
) -> VLMBackbone:
    """Factory: return the right backbone for the given model_path.

    Dispatch by:
      1. exact match in `_BACKBONE_REGISTRY`
      2. heuristic: any "Qwen2.5-VL" / "Qwen2_5_VL" in the path → VL
      3. fallback → CausalLM
    """
    btype = _BACKBONE_REGISTRY.get(model_path)
    if btype is None:
        btype = "vl" if ("Qwen2.5-VL" in model_path or "Qwen2_5_VL" in model_path) else "causal"
    if btype == "vl":
        return HFBackbone.load(model_path, quantize=quantize, lora_path=lora_path)
    return CausalLMBackbone.load(model_path, quantize=quantize)
