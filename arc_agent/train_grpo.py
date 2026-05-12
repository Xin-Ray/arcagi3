"""GRPO training reward function + trainer factory.

Per `docs/ARCHITECTURE_RL.md` §3, the dense reward is:

    r = +1.0 if state == WIN
        +0.2 * f1   if parse_ok
        -0.5        if not parse_ok
        -0.3        if action not in available_actions
        +0.05       if entity recognition self-consistent (optional)

`reward_fn` operates on a `StepRecord` dataclass — pure, no torch. Unit tests
in `tests/test_train_grpo.py` cover each branch independently and the
combined cases. Drift-fix this formula HERE; the trainer just calls.

`build_trainer` lazily imports `trl` so the module is importable on a
machine without the training stack — only invoking the trainer requires it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Coefficients pulled out as named constants so test failures point straight
# at the source of truth.
R_WIN              = 1.0
R_F1_COEFF         = 0.2
R_PARSE_FAIL       = -0.5
R_ILLEGAL_ACTION   = -0.3
R_ENTITY_BONUS     = 0.05


@dataclass
class StepRecord:
    """One rollout step's reward inputs.

    Names match `docs/ARCHITECTURE_RL.md` §3 verbatim so the formula in code
    is grep-able against the spec.
    """

    state: str                                # "WIN" | "GAME_OVER" | "NOT_FINISHED"
    parsed_json_ok: bool                      # JSON parse succeeded
    f1: float = 0.0                            # in [0, 1]; ignored if not parse_ok
    action: Optional[str] = None              # "ACTION1".."ACTION7" | None
    available_actions: list[int] = field(default_factory=list)
    entity_recognition_consistent: bool = False


def reward_fn(step: StepRecord) -> float:
    """Compute the per-step reward exactly per §3.

    Independent terms — added together, no clamping. The natural max is
    1.0 (WIN) + 0.2 (perfect F1) + 0.05 (entity bonus) = 1.25; the natural
    min is -0.5 (parse_fail) + -0.3 (illegal) = -0.8. These bounds inform
    GRPO advantage normalization but aren't enforced here.
    """
    r = 0.0

    if step.state == "WIN":
        r += R_WIN

    if step.parsed_json_ok:
        r += R_F1_COEFF * float(step.f1)
    else:
        r += R_PARSE_FAIL

    if step.action is not None:
        action_id = _action_name_to_id(step.action)
        if action_id is not None and action_id not in step.available_actions:
            r += R_ILLEGAL_ACTION

    if step.entity_recognition_consistent:
        r += R_ENTITY_BONUS

    return r


def _action_name_to_id(name: str) -> Optional[int]:
    """Convert "ACTION3" → 3 (no SDK import — pure). Returns None on garbage."""
    if not isinstance(name, str) or not name.startswith("ACTION"):
        return None
    tail = name[len("ACTION"):]
    try:
        i = int(tail)
    except ValueError:
        return None
    if 1 <= i <= 7:
        return i
    return None


def build_trainer(
    model: Any,
    processor: Any,
    reward_fn: Callable[[StepRecord], float],
    train_games: list[str],
    *,
    val_games: list[str] | None = None,
    output_dir: str = "outputs/grpo",
    learning_rate: float = 5e-6,
    per_device_batch: int = 1,
    grad_accum: int = 4,
    num_generations: int = 4,
    max_steps: int = 500,
    val_every: int = 50,
    **grpo_kwargs: Any,
) -> Any:
    """Build a `trl.GRPOTrainer` for our VLM rollout setup.

    Lazy-imports `trl` so this module loads on machines without the training
    stack. Raises ImportError with a clear hint if `trl` is missing.

    Note: as of TRL 0.11.x the GRPOTrainer interface takes a list-of-string
    reward function `(prompts, completions, **) -> list[float]`. The
    per-step `reward_fn(StepRecord)` in this module is the *core* of that;
    the trl-facing wrapper is built inside this function so the public
    reward signature stays simple and testable.
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise ImportError(
            "build_trainer requires `trl` — pip install trl accelerate "
            "transformers bitsandbytes peft. Install training deps "
            "separately (intentionally not in requirements.txt; see CLAUDE.md)."
        ) from e

    config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_generations,
        max_steps=max_steps,
        logging_steps=10,
        save_steps=val_every,
        **grpo_kwargs,
    )

    # The rollout-shape adapter belongs here when the actual rollout loop
    # lands (scripts/run_grpo.py). For now the function returns a
    # half-built trainer object so the orchestration script can finish
    # wiring with its own rollout-aware reward closure.
    return _PartialTrainer(
        config=config,
        model=model,
        processor=processor,
        core_reward_fn=reward_fn,
        train_games=train_games,
        val_games=val_games or [],
    )


@dataclass
class _PartialTrainer:
    """Pre-trl container the rollout loop can flesh out.

    Holding GRPOTrainer construction here would couple us to the rollout
    interface (still being shaped); this struct documents what the trainer
    needs and lets `scripts/run_grpo.py` snap on a rollout iterator.
    """

    config: Any
    model: Any
    processor: Any
    core_reward_fn: Callable[[StepRecord], float]
    train_games: list[str]
    val_games: list[str]
