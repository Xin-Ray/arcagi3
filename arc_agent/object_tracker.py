"""Maintain UID-keyed object history across frames in an episode.

scipy's `extract_objects` re-assigns ids each frame (color-major,
component-order). To talk about "object X moved" across multiple steps,
we need stable UIDs that survive the scipy re-numbering.

This module owns that bridge:
  ObjectMemory.update(matches, current_frame_objects)
    - matches from `object_aligner.align_objects(prev_active, current_active)`
    - assigns/propagates UIDs so the same physical object has the same uid
      regardless of where scipy puts it in the list

Per-uid history is a list of ObjectSnapshot, one per step the object
was seen (no entry on steps where it was disappeared / not in ACTIVE).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from arc_agent.object_aligner import Match
from arc_agent.object_extractor import ObjectRecord


@dataclass
class ObjectSnapshot:
    """One observation of an object at a particular step."""
    step: int
    scipy_id: int
    color: int
    color_name: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    size: int


@dataclass
class TrackedObject:
    """A persistent (across-frame) record for one physical object."""
    uid: str
    history: list[ObjectSnapshot] = field(default_factory=list)
    color: int = 0     # latest color
    last_step: int = -1
    alive: bool = True   # set to False when disappeared

    @property
    def descriptor(self) -> str:
        """One-line description for [ACTIVE] prompt block."""
        if not self.history:
            return f"{self.uid}: empty"
        last = self.history[-1]
        r0, c0, r1, c1 = last.bbox
        return (f"{self.uid}: {last.color_name} (size={last.size}, "
                f"bbox=[{r0},{c0},{r1},{c1}])")


class ObjectMemory:
    """Per-episode UID-keyed object memory."""

    def __init__(self) -> None:
        self._tracked: dict[str, TrackedObject] = {}
        self._next_uid = 0
        # Reverse lookup: scipy_id (in PREVIOUS frame) -> uid, so we can
        # follow `before_id` in alignment matches.
        self._prev_id_to_uid: dict[int, str] = {}

    def _new_uid(self, color_name: str) -> str:
        s = f"obj_{self._next_uid:03d}"
        self._next_uid += 1
        return s

    def update(self,
               step: int,
               current_active: list[ObjectRecord],
               matches: list[Match]) -> None:
        """Merge one step of alignment into the memory.

        Args:
            step: episode step index (0-based).
            current_active: ACTIVE objects in the current frame
                (after temporal filtering; scipy_id is the AFTER index).
            matches: from align_objects(prev_active, current_active).
        """
        new_prev_id_to_uid: dict[int, str] = {}
        active_by_id = {o.id: o for o in current_active}
        used_after_ids: set[int] = set()

        for m in matches:
            if m.before_id is None and m.after_id is None:
                continue
            if m.type == "appeared":
                # New object in AFTER frame
                if m.after_id is None:
                    continue
                obj = active_by_id.get(m.after_id)
                if obj is None:
                    continue
                uid = self._new_uid(obj.color_name)
                self._tracked[uid] = TrackedObject(
                    uid=uid, color=obj.color, last_step=step, alive=True,
                    history=[ObjectSnapshot(
                        step=step, scipy_id=obj.id, color=obj.color,
                        color_name=obj.color_name, bbox=obj.bbox,
                        center=obj.center, size=obj.size,
                    )],
                )
                new_prev_id_to_uid[obj.id] = uid
                used_after_ids.add(m.after_id)
                continue
            if m.type == "disappeared":
                # Object in BEFORE that vanished
                if m.before_id is None:
                    continue
                uid = self._prev_id_to_uid.get(m.before_id)
                if uid and uid in self._tracked:
                    self._tracked[uid].alive = False
                continue
            # Paired match (moved / unchanged / reshaped / recolored)
            if m.before_id is None or m.after_id is None:
                continue
            uid = self._prev_id_to_uid.get(m.before_id)
            obj = active_by_id.get(m.after_id)
            if obj is None:
                continue
            if uid is None:
                # First-time observation if BEFORE side wasn't tracked
                uid = self._new_uid(obj.color_name)
                self._tracked[uid] = TrackedObject(
                    uid=uid, color=obj.color, last_step=step, alive=True,
                )
            t = self._tracked[uid]
            t.color = obj.color
            t.last_step = step
            t.alive = True
            t.history.append(ObjectSnapshot(
                step=step, scipy_id=obj.id, color=obj.color,
                color_name=obj.color_name, bbox=obj.bbox,
                center=obj.center, size=obj.size,
            ))
            new_prev_id_to_uid[obj.id] = uid
            used_after_ids.add(m.after_id)

        # Any current-active object NOT covered by matches → treat as new
        for o in current_active:
            if o.id in used_after_ids:
                continue
            if o.id in new_prev_id_to_uid:
                continue
            uid = self._new_uid(o.color_name)
            self._tracked[uid] = TrackedObject(
                uid=uid, color=o.color, last_step=step, alive=True,
                history=[ObjectSnapshot(
                    step=step, scipy_id=o.id, color=o.color,
                    color_name=o.color_name, bbox=o.bbox,
                    center=o.center, size=o.size,
                )],
            )
            new_prev_id_to_uid[o.id] = uid

        self._prev_id_to_uid = new_prev_id_to_uid

    def alive_tracked(self) -> list[TrackedObject]:
        return [t for t in self._tracked.values() if t.alive]

    def get(self, uid: str) -> Optional[TrackedObject]:
        return self._tracked.get(uid)

    def reset(self) -> None:
        self._tracked.clear()
        self._prev_id_to_uid.clear()
        self._next_uid = 0
