"""Persistent per-object confidence map for ACTION6.

Why this exists. v3.2 smoke (`outputs/bug8_9_smoke_20260514-205044`) showed
the Action Agent emitting 219 ACTION6 picks across 376 steps, **zero** of
which changed the frame. 122 of those (56%) hit the same coordinate (5, 60).
None of the reasoning lines mentioned an `obj_*` id. The model was choosing
ACTION6 with no spatial grounding -- "try a different location" meant
"shuffle x,y" rather than "click a different object".

This module replaces the per-step `[CLICK CANDIDATES]` block with a list of
named targets that **remembers** how each one has performed. The Action
Agent picks `obj_NNN`'s known coords instead of inventing x,y; every click
that produces no_op decays that target's confidence; every click that
moves a frame boosts it. Once a target's confidence falls below the noise
floor it stops being suggested. Concrete bandit, deterministic update.

Persistence: ClickTargets live in `Knowledge.click_targets`. They survive
round boundaries via a `signature` key derived from color+shape (obj_ids
are episode-local; signature is stable). When the next round produces a
fresh obj with the same signature, prior confidence / tries / successes
transfer to the new uid.

Module is pure: no I/O, no Knowledge mutation. The orchestrator calls
`update_click_targets(...)` each step and writes the returned list back
into `knowledge.click_targets`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# Hit radius (Euclidean, in grid cells). If an ACTION6 (x,y) is within this
# distance of any target's center, we credit / debit that target. Larger
# than 1 because perception centers are float-rounded and the model often
# clicks a few cells off the geometric center.
DEFAULT_HIT_RADIUS = 3.0

# Confidence multipliers on each outcome.
DEFAULT_DECAY = 0.7   # multiply on no-op hit (so 5 misses -> 0.16)
DEFAULT_BOOST = 2.0   # multiply on success hit (capped at 1.0)

# Maximum entries kept in the prompt-facing list. The orchestrator sorts
# by priority (confidence x (1 - tries/10)) so untried fresh targets always
# bubble above heavily-tried decayed ones.
DEFAULT_MAX_TARGETS = 10


@dataclass
class ClickTarget:
    """One ACTION6 target: which object, where, how it's performed.

    `obj_id` is the current-episode uid; it changes across rounds.
    `signature` (color_name + bbox h x w) is stable across rounds and is
    how we revive prior confidence when the same kind of object reappears.
    """
    obj_id: str
    signature: str
    coords: tuple[int, int]              # (row, col), rounded ints
    color_name: str
    bbox: tuple[int, int, int, int]      # (r0, c0, r1, c1)
    confidence: float = 1.0
    tries: int = 0
    successes: int = 0
    last_seen_step: int = 0
    alive: bool = True

    @property
    def priority(self) -> float:
        """Sort key: fresh untried (tries=0) targets always outrank any
        decayed entry, regardless of confidence. Tried targets are then
        ranked by remaining confidence."""
        return self.confidence * max(0.0, 1.0 - self.tries / 10.0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # tuples become lists in JSON; we'll restore them on load
        d["coords"] = list(self.coords)
        d["bbox"] = list(self.bbox)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClickTarget":
        coords = d.get("coords", [0, 0])
        bbox = d.get("bbox", [0, 0, 0, 0])
        return cls(
            obj_id=str(d.get("obj_id", "")),
            signature=str(d.get("signature", "")),
            coords=(int(coords[0]), int(coords[1])),
            color_name=str(d.get("color_name", "")),
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            confidence=float(d.get("confidence", 1.0)),
            tries=int(d.get("tries", 0)),
            successes=int(d.get("successes", 0)),
            last_seen_step=int(d.get("last_seen_step", 0)),
            alive=bool(d.get("alive", True)),
        )


def _shape_token(bbox: tuple[int, int, int, int]) -> str:
    """Human-friendly shape descriptor like '1x1' or '2x3' for a bbox."""
    r0, c0, r1, c1 = bbox
    return f"{r1 - r0 + 1}x{c1 - c0 + 1}"


def signature_of(color_name: str, bbox: tuple[int, int, int, int]) -> str:
    """Cross-round-stable identifier: color + shape. Two yellow 1x1s in
    different positions share a signature; that's intentional -- bandit
    history transfers across instances."""
    return f"{color_name}_{_shape_token(bbox)}"


def update_click_targets(
    targets: list[ClickTarget],
    alive_tracked: list[Any],
    *,
    last_action: Optional[str],
    last_coords: Optional[tuple[int, int]],
    frame_changed: bool,
    step: int,
    hit_radius: float = DEFAULT_HIT_RADIUS,
    decay: float = DEFAULT_DECAY,
    boost: float = DEFAULT_BOOST,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> list[ClickTarget]:
    """Reconcile `targets` with the latest perception + last action outcome.

    Steps:
      1. For every alive TrackedObject, produce a target carrying any
         prior confidence (looked up first by obj_id, then by signature).
      2. If `last_action == "ACTION6"`, find the target nearest to
         `last_coords` within `hit_radius`; debit on no_op, credit on
         frame_changed. Clicks beyond hit_radius are "wild" and ignored.
      3. Sort by priority and cap at `max_targets`.

    Pure: returns a new list; does NOT mutate the input `targets` or any
    field of the items in it.
    """
    by_uid: dict[str, ClickTarget] = {t.obj_id: t for t in targets}
    by_sig: dict[str, ClickTarget] = {}
    for t in targets:
        prior = by_sig.get(t.signature)
        if prior is None or t.last_seen_step > prior.last_seen_step:
            by_sig[t.signature] = t

    out: list[ClickTarget] = []
    for obj in alive_tracked:
        history = getattr(obj, "history", None) or []
        if not history:
            continue
        snap = history[-1]
        sig = signature_of(snap.color_name, tuple(snap.bbox))
        center = (int(round(snap.center[0])), int(round(snap.center[1])))
        prior = by_uid.get(obj.uid) or by_sig.get(sig)
        if prior is not None:
            new = ClickTarget(
                obj_id=obj.uid,
                signature=sig,
                coords=center,
                color_name=snap.color_name,
                bbox=tuple(snap.bbox),
                confidence=prior.confidence,
                tries=prior.tries,
                successes=prior.successes,
                last_seen_step=step,
                alive=True,
            )
        else:
            new = ClickTarget(
                obj_id=obj.uid,
                signature=sig,
                coords=center,
                color_name=snap.color_name,
                bbox=tuple(snap.bbox),
                last_seen_step=step,
            )
        out.append(new)

    if last_action == "ACTION6" and last_coords is not None and out:
        lx, ly = int(last_coords[0]), int(last_coords[1])
        best: Optional[ClickTarget] = None
        best_d: float = float("inf")
        for t in out:
            tx, ty = t.coords
            d = ((tx - lx) ** 2 + (ty - ly) ** 2) ** 0.5
            if d < best_d:
                best_d, best = d, t
        if best is not None and best_d <= hit_radius:
            best.tries += 1
            if frame_changed:
                best.successes += 1
                best.confidence = min(1.0, best.confidence * boost)
            else:
                best.confidence = best.confidence * decay

    out.sort(key=lambda t: t.priority, reverse=True)
    return out[:max_targets]


def render_click_targets_block(
    targets: list[ClickTarget],
    *,
    max_show: int = DEFAULT_MAX_TARGETS,
) -> str:
    """Render the [CLICK TARGETS] prompt block. Empty string when there
    are no targets -- caller should omit the block entirely in that case.

    Sorts defensively by priority (descending) so the call-site doesn't
    have to. `update_click_targets` already sorts on the orchestrator hot
    path; this is idempotent for that case and a safety net when callers
    (e.g. tests, future code) hand us an unsorted list.
    """
    if not targets:
        return ""
    sorted_targets = sorted(targets, key=lambda t: t.priority, reverse=True)
    lines: list[str] = [
        "[CLICK TARGETS -- pick by obj_id, NOT raw coords]"
    ]
    for t in sorted_targets[:max_show]:
        if t.tries == 0:
            tag = "  <- UNTRIED, prefer"
        elif t.confidence < 0.1:
            tag = "  <- WRITTEN OFF"
        elif t.successes > 0:
            tag = "  <- known interactive"
        else:
            tag = ""
        lines.append(
            f"  {t.obj_id} ({t.color_name} {_shape_token(t.bbox)}) "
            f"at ({t.coords[0]},{t.coords[1]})  "
            f"conf={t.confidence:.2f}  tries={t.tries} ok={t.successes}{tag}"
        )
    lines.append(
        "  Use the OBJECT'S coords. Do NOT invent x,y. Prefer conf >= 0.5."
    )
    lines.append(
        "  If all conf < 0.3, ACTION6 is probably not the right tool -- "
        "pick a different ACTION."
    )
    return "\n".join(lines)


__all__ = [
    "ClickTarget",
    "DEFAULT_BOOST",
    "DEFAULT_DECAY",
    "DEFAULT_HIT_RADIUS",
    "DEFAULT_MAX_TARGETS",
    "render_click_targets_block",
    "signature_of",
    "update_click_targets",
]
