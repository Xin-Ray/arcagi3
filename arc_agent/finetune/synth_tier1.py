"""Synthetic data generators for Tier 1 SFT (see docs/arch_sft_tier1_zh.md §3).

Five tasks (T1 direction, T2 arithmetic, T3 relative position, T4 format,
T8 reasoning-action consistency). Pure stdlib + type hints, no LLM, fully
deterministic via a passed-in `random.Random`.

T8 uses in-context binding (token->direction varies per sample) so the
fine-tuned model never learns a fixed `ACTION3=X` prior — action semantics
are per-game in ARC and must be discovered from OutcomeLog (see §3.6).
"""
from __future__ import annotations

import random
from typing import Literal, TypedDict


class Sample(TypedDict):
    system: str
    user: str
    assistant: str
    task: str


Lang = Literal["en", "zh"]

# ── shared vocab ───────────────────────────────────────────────────────────

DIRECTIONS_4: tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
DIRECTIONS_8: tuple[str, ...] = (
    "up", "down", "left", "right",
    "left-up", "right-up", "left-down", "right-down",
)
ACTION_TOKENS: tuple[str, ...] = ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
COLORS_EN: tuple[str, ...] = (
    "yellow", "red", "blue", "green", "purple", "cyan", "gray",
)
SHAPES_EN: tuple[str, ...] = (
    "square", "circle", "L-shape", "triangle", "rectangle",
)
COLORS_ZH: tuple[str, ...] = ("黄", "红", "蓝", "绿", "紫", "青", "灰")
SHAPES_ZH: tuple[str, ...] = ("方块", "圆", "L 形", "三角形", "矩形")
DIR_ZH: dict[str, str] = {"UP": "上", "DOWN": "下", "LEFT": "左", "RIGHT": "右"}


# ── T1: direction from y/x axis convention (50k = 40k en + 10k zh) ────────

def gen_T1(rng: random.Random, lang: Lang = "en") -> Sample:
    """Given a coordinate-axis convention + src/dst y values, output UP or DOWN.

    50% of samples use ARC convention (y down), 50% use math (y up).
    Forces the model to read the prompt instead of relying on a prior.
    """
    screen = rng.random() < 0.5
    src_y = rng.randint(0, 60)
    dst_y = rng.choice([y for y in range(64) if y != src_y])
    delta = dst_y - src_y
    if screen:
        answer = "DOWN" if delta > 0 else "UP"
    else:
        answer = "UP" if delta > 0 else "DOWN"

    if lang == "en":
        convention = (
            "y increases from top to bottom" if screen
            else "y increases from bottom to top"
        )
        return {
            "system": "Answer in one word: UP or DOWN. No explanation.",
            "user": (
                f"{convention}. Object at y={src_y}. Target at y={dst_y}. "
                f"To reach the target, the object must move which direction?"
            ),
            "assistant": answer,
            "task": "T1",
        }
    convention_zh = "y 从上往下递增" if screen else "y 从下往上递增"
    return {
        "system": "用一个英文单词回答:UP 或 DOWN。不要解释。",
        "user": (
            f"{convention_zh}。物体在 y={src_y}。目标在 y={dst_y}。"
            f"物体要去目标处,应该向哪个方向移动?"
        ),
        "assistant": answer,
        "task": "T1",
    }


# ── T2: small-integer division + counting (40k = 30k main + 5k rem + 5k multi)

def gen_T2(rng: random.Random, lang: Lang = "en") -> Sample:
    """Main T2: delta / step -> integer count."""
    step = rng.randint(1, 5)
    n = rng.randint(1, 20)
    delta = n * step
    if lang == "en":
        user = (
            f"Each move shifts the object by {step} cells. "
            f"The object needs to travel {delta} cells in one direction. "
            f"How many moves?"
        )
    else:
        user = f"每次移动 {step} 格,需要走 {delta} 格,几次?"
    return {
        "system": "Answer with a single integer. No explanation.",
        "user": user,
        "assistant": str(n),
        "task": "T2",
    }


def gen_T2_remainder(rng: random.Random) -> Sample:
    """T2 variant: integer quotient + remainder, fixed-token output."""
    step = rng.randint(2, 5)
    q = rng.randint(1, 15)
    r = rng.randint(1, step - 1)
    delta = q * step + r
    return {
        "system": "回答格式严格为 `完整 <Q> 余 <R>`,不要解释。",
        "user": f"delta={delta} step={step},几次完整,余几?",
        "assistant": f"完整 {q} 余 {r}",
        "task": "T2_remainder",
    }


def gen_T2_multi(rng: random.Random) -> Sample:
    """T2 variant: two-axis division, format `x:<n> y:<m>`."""
    step = rng.randint(1, 5)
    nx = rng.randint(1, 12)
    ny = rng.randint(1, 12)
    dx = nx * step
    dy = ny * step
    return {
        "system": "回答格式严格为 `x:<nx> y:<ny>`,不要解释。",
        "user": f"dx={dx} dy={dy} step={step},各几次?",
        "assistant": f"x:{nx} y:{ny}",
        "task": "T2_multi",
    }


# ── T3: 8-direction relative position -> move direction (40k) ────────────

_RELATIONS_EN: tuple[tuple[str, str], ...] = (
    ("top-left",     "right-down"),
    ("top",          "down"),
    ("top-right",    "left-down"),
    ("left",         "right"),
    ("right",        "left"),
    ("bottom-left",  "right-up"),
    ("bottom",       "up"),
    ("bottom-right", "left-up"),
)
_RELATIONS_ZH: tuple[tuple[str, str], ...] = (
    ("左上", "right-down"),
    ("上",   "down"),
    ("右上", "left-down"),
    ("左",   "right"),
    ("右",   "left"),
    ("左下", "right-up"),
    ("下",   "up"),
    ("右下", "left-up"),
)


def gen_T3(rng: random.Random, lang: Lang = "en") -> Sample:
    """Given A is at `<rel>` of B, output the 8-direction move A->B."""
    if lang == "en":
        rel, ans = rng.choice(_RELATIONS_EN)
        c1, c2 = rng.sample(COLORS_EN, 2)
        s1, s2 = rng.choices(SHAPES_EN, k=2)
        user = (
            f"The {c1} {s1} is at the {rel} of the {c2} {s2}. "
            f"To move the {c1} {s1} to reach the {c2} {s2}, output the "
            f"direction (one of: up / down / left / right / "
            f"left-up / right-up / left-down / right-down)."
        )
    else:
        rel, ans = rng.choice(_RELATIONS_ZH)
        c1, c2 = rng.sample(COLORS_ZH, 2)
        s1, s2 = rng.choices(SHAPES_ZH, k=2)
        user = (
            f"{c1}{s1}在{c2}{s2}的{rel}方。"
            f"要把{c1}{s1}移到{c2}{s2},往哪边走?(8 方向之一:"
            f"up / down / left / right / left-up / right-up / "
            f"left-down / right-down)"
        )
    return {
        "system": "Output exactly one direction word. No explanation.",
        "user": user,
        "assistant": ans,
        "task": "T3",
    }


# ── T4: strict-format compliance (30k) ───────────────────────────────────

_T4_TEMPLATES: tuple[tuple[str, str], ...] = (
    (
        "TOTAL_ACTIONS=<N>\nACTION_CHAIN=<a,b,c>",
        "TOTAL_ACTIONS=4\nACTION_CHAIN=ACTION3,ACTION2,ACTION3,ACTION5",
    ),
    (
        "reasoning: <one line>\naction: <token>",
        "reasoning: move yellow object UP\naction: ACTION1",
    ),
    (
        '{"direction": "X", "count": N}',
        '{"direction": "down", "count": 5}',
    ),
    (
        "DIRECTION:<X> COUNT:<N>",
        "DIRECTION:down COUNT:5",
    ),
    (
        "- <line1>\n- <line2>",
        "- yellow 1x1\n- moves UP 3",
    ),
)


def gen_T4(rng: random.Random) -> Sample:
    """Strict format compliance: user prompt is *deliberately unrelated* to
    the system task — teach the model to copy the system-defined format
    rather than 'understand' content. Fixes the prose-everywhere failure
    of probe 1.
    """
    template, target = rng.choice(_T4_TEMPLATES)
    return {
        "system": (
            f"Output format exactly: {template}.\n"
            "Output ONLY the answer line(s). No markdown, no prose, "
            "no numbered steps, no explanation, no extra lines."
        ),
        "user": "Now produce one valid output following the format.",
        "assistant": target,
        "task": "T4",
    }


# ── T8: reasoning ⇄ action consistency, IN-CONTEXT BINDING (40k) ────────
#
# CRITICAL: no fixed ACTION->direction map. Each sample randomly binds a
# token to a direction; the model learns to copy that binding into both
# the reasoning line and the action line. This is the only way to keep
# the LLM compatible with v3's per-game action semantics (see §3.6).

def gen_T8(rng: random.Random, lang: Lang = "en") -> Sample:
    """Reasoning + action must match the user-prompt's per-sample binding."""
    direction = rng.choice(DIRECTIONS_4)
    token = rng.choice(ACTION_TOKENS)
    if lang == "en":
        color = rng.choice(COLORS_EN)
        shape = rng.choice(SHAPES_EN)
        user = (
            f"In this game, {token} moves objects {direction}. "
            f"The {color} {shape} needs to go {direction}."
        )
        assistant = (
            f"reasoning: move the {color} {shape} {direction.lower()} by 3 cells\n"
            f"action: {token}"
        )
    else:
        color = rng.choice(COLORS_ZH)
        shape = rng.choice(SHAPES_ZH)
        dir_zh = DIR_ZH[direction]
        user = (
            f"本游戏中,{token} 让物体向{dir_zh}移动。"
            f"{color}{shape}需要向{dir_zh}走。"
        )
        assistant = (
            f"reasoning: move the {color}{shape} {direction.lower()} by 3 cells\n"
            f"action: {token}"
        )
    return {
        "system": (
            "Output exactly two lines:\n"
            "  reasoning: <one sentence with a subject and direction>\n"
            "  action: <ACTION1..ACTION7>\n"
            "The direction word in `reasoning` must match the action's "
            "direction as defined by the user prompt."
        ),
        "user": user,
        "assistant": assistant,
        "task": "T8",
    }


# ── full Tier-1 mix (used by scripts/gen_tier1_data.py) ──────────────────

# Counts per §3 with §3.3 patch (T2 = 30k main + 5k rem + 5k multi).
_TIER1_COUNTS: dict[str, int] = {
    "T1_en": 40_000,
    "T1_zh": 10_000,
    "T2_en": 15_000,
    "T2_zh": 15_000,
    "T2_remainder": 5_000,
    "T2_multi": 5_000,
    "T3_en": 30_000,
    "T3_zh": 10_000,
    "T4": 30_000,
    "T8_en": 30_000,
    "T8_zh": 10_000,
}


def build_tier1(seed: int = 42) -> list[Sample]:
    """Generate the full ~200k Tier-1 mix, shuffled, deterministic by seed."""
    rng = random.Random(seed)
    out: list[Sample] = []
    for name, n in _TIER1_COUNTS.items():
        for _ in range(n):
            out.append(_dispatch(name, rng))
    rng.shuffle(out)
    return out


def _dispatch(name: str, rng: random.Random) -> Sample:
    if name == "T1_en":         return gen_T1(rng, "en")
    if name == "T1_zh":         return gen_T1(rng, "zh")
    if name == "T2_en":         return gen_T2(rng, "en")
    if name == "T2_zh":         return gen_T2(rng, "zh")
    if name == "T2_remainder":  return gen_T2_remainder(rng)
    if name == "T2_multi":      return gen_T2_multi(rng)
    if name == "T3_en":         return gen_T3(rng, "en")
    if name == "T3_zh":         return gen_T3(rng, "zh")
    if name == "T4":            return gen_T4(rng)
    if name == "T8_en":         return gen_T8(rng, "en")
    if name == "T8_zh":         return gen_T8(rng, "zh")
    raise ValueError(f"unknown generator: {name}")


# ── OOD set (§5.1 / §6.2): paraphrased + extreme deltas, no overlap ─────

def build_ood(seed: int = 1337, n_per_task: int = 1000) -> list[Sample]:
    """OOD probes — single-direction phrasing twists + extreme numbers.

    Separate generators (not the train ones) so we measure real
    generalization, not memorization.
    """
    rng = random.Random(seed)
    out: list[Sample] = []
    for _ in range(n_per_task):
        out.append(_ood_T1(rng))
    for _ in range(n_per_task):
        out.append(_ood_T2(rng))
    for _ in range(n_per_task):
        out.append(_ood_T3(rng))
    for _ in range(n_per_task):
        out.append(_ood_T4(rng))
    for _ in range(n_per_task):
        out.append(_ood_T8(rng))
    rng.shuffle(out)
    return out


def _ood_T1(rng: random.Random) -> Sample:
    """OOD: paraphrased convention + large y deltas (off-board, 100-500)."""
    screen = rng.random() < 0.5
    src_y = rng.randint(100, 400)
    dst_y = rng.choice([y for y in range(100, 500) if y != src_y])
    delta = dst_y - src_y
    if screen:
        answer = "DOWN" if delta > 0 else "UP"
        paraphrase = rng.choice([
            "row index grows downward",
            "down is positive on the y-axis",
            "screen coordinates: y is row, larger = lower on screen",
        ])
    else:
        answer = "UP" if delta > 0 else "DOWN"
        paraphrase = rng.choice([
            "y grows upward like in math class",
            "up is positive on the y-axis",
            "Cartesian convention: larger y = higher",
        ])
    return {
        "system": "Answer in one word: UP or DOWN. No explanation.",
        "user": (
            f"Convention: {paraphrase}. Source y={src_y}, target y={dst_y}. "
            f"Which way to move?"
        ),
        "assistant": answer,
        "task": "T1_ood",
    }


def _ood_T2(rng: random.Random) -> Sample:
    """OOD: extreme step + count outside training range."""
    step = rng.choice([6, 7, 8, 9, 10, 12, 15])
    n = rng.randint(25, 60)
    delta = n * step
    return {
        "system": "Answer with a single integer. No explanation.",
        "user": (
            f"A move covers {step} units. Distance to travel is {delta} units. "
            f"Number of moves?"
        ),
        "assistant": str(n),
        "task": "T2_ood",
    }


def _ood_T3(rng: random.Random) -> Sample:
    """OOD: unseen color×shape combos + paraphrased relation."""
    rel, ans = rng.choice(_RELATIONS_EN)
    novel_colors = ("orange", "magenta", "teal", "olive", "navy")
    novel_shapes = ("hexagon", "star", "plus-shape", "diamond")
    c1, c2 = rng.sample(novel_colors, 2)
    s1, s2 = rng.choices(novel_shapes, k=2)
    rel_paraphrase = rel.replace("-", " and ")
    return {
        "system": "Output exactly one direction word. No explanation.",
        "user": (
            f"You see a {c1} {s1} and a {c2} {s2}. The first is "
            f"{rel_paraphrase} of the second. To move the first onto the "
            f"second, which 8-direction step?"
        ),
        "assistant": ans,
        "task": "T3_ood",
    }


def _ood_T4(rng: random.Random) -> Sample:
    """OOD: novel template not seen in training."""
    direction = rng.choice(["up", "down", "left", "right"])
    count = rng.randint(1, 9)
    return {
        "system": (
            "Output format exactly: <DIR>:<N> (uppercase direction, digit). "
            "ONLY the answer. No prose."
        ),
        "user": "Now produce one valid output following the format.",
        "assistant": f"{direction.upper()}:{count}",
        "task": "T4_ood",
    }


def _ood_T8(rng: random.Random) -> Sample:
    """OOD: in-context binding with ACTION5..ACTION7 tokens (not in train)."""
    direction = rng.choice(DIRECTIONS_4)
    token = rng.choice(["ACTION5", "ACTION6", "ACTION7"])
    color = rng.choice(COLORS_EN)
    shape = rng.choice(SHAPES_EN)
    return {
        "system": (
            "Output exactly two lines:\n"
            "  reasoning: <one sentence with a subject and direction>\n"
            "  action: <ACTION1..ACTION7>\n"
            "The direction word in `reasoning` must match the action's "
            "direction as defined by the user prompt."
        ),
        "user": (
            f"In this game, {token} moves objects {direction}. "
            f"The {color} {shape} needs to go {direction}."
        ),
        "assistant": (
            f"reasoning: move the {color} {shape} {direction.lower()} by 3 cells\n"
            f"action: {token}"
        ),
        "task": "T8_ood",
    }
