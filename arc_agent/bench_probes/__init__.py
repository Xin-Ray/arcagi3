"""Game-content-based spatial-reasoning probes for cross-model benchmarking.

Each probe is multi-choice (A/B/C/D), exercising the *general* spatial
skill needed to operate in ANY ARC-AGI-3 grid game. Categories tested:

  T1 direction         — given coord delta, name the direction
  T2 single-step       — after 1 action, where is the object
  T3 multi-step plan   — sequence of N actions to go A -> B
  T4 boundary          — will the action be a no-op at the edge
  T5 multi-dim plan    — need both x and y movement
  T6 selection         — given click coord, which object gets selected
  T7 inverse plan      — given A & B, what's the first action
  T8 distance          — manhattan / euclidean estimation

Probes are calibrated to the real game mechanics (ar25/bp35/cd82/cn04/dc22
all use ±1 cell per move, 64x64 grid, ACTION1-7 enum) but framed as
abstract questions so the model doesn't need game-specific knowledge.

The right answer for each probe is at `correct` (uppercase letter).
"""
from __future__ import annotations

PROBES: list[dict] = [
    # ── T1 direction (5) ────────────────────────────────────────────────
    {
        "id": "T1-01", "cat": "T1",
        "q": "On a 64x64 grid, y increases downward. An object at row=20 wants to reach row=25. Which direction must it move?",
        "options": {"A": "UP (row decreases)", "B": "DOWN (row increases)",
                    "C": "LEFT (col decreases)", "D": "RIGHT (col increases)"},
        "correct": "B",
    },
    {
        "id": "T1-02", "cat": "T1",
        "q": "On a 64x64 grid, x increases to the right. An object at col=30 wants to reach col=20. Which direction must it move?",
        "options": {"A": "UP", "B": "DOWN", "C": "LEFT (col decreases)", "D": "RIGHT (col increases)"},
        "correct": "C",
    },
    {
        "id": "T1-03", "cat": "T1",
        "q": "An object is at (row=10, col=15). A target is at (row=10, col=8). The shortest direction to move?",
        "options": {"A": "UP", "B": "DOWN", "C": "LEFT", "D": "RIGHT"},
        "correct": "C",
    },
    {
        "id": "T1-04", "cat": "T1",
        "q": "An object is at (row=40, col=12). A target is at (row=35, col=12). The shortest direction to move?",
        "options": {"A": "UP (row decreases)", "B": "DOWN (row increases)",
                    "C": "LEFT", "D": "RIGHT"},
        "correct": "A",
    },
    {
        "id": "T1-05", "cat": "T1",
        "q": "Object at row=5. Target at row=2. Object must move which way (rows count from top, top=row 0)?",
        "options": {"A": "UP (toward row 0)", "B": "DOWN (away from row 0)",
                    "C": "LEFT", "D": "RIGHT"},
        "correct": "A",
    },

    # ── T2 single-step (4) ──────────────────────────────────────────────
    {
        "id": "T2-01", "cat": "T2",
        "q": "ACTION1 moves the active object UP by 1 cell. Object is at (row=10, col=20). After one ACTION1, where is it?",
        "options": {"A": "(row=10, col=19)", "B": "(row=10, col=21)",
                    "C": "(row=9, col=20)", "D": "(row=11, col=20)"},
        "correct": "C",
    },
    {
        "id": "T2-02", "cat": "T2",
        "q": "ACTION4 moves the active object RIGHT by 1 cell. Object at (row=15, col=30). After one ACTION4?",
        "options": {"A": "(row=15, col=29)", "B": "(row=15, col=31)",
                    "C": "(row=14, col=30)", "D": "(row=16, col=30)"},
        "correct": "B",
    },
    {
        "id": "T2-03", "cat": "T2",
        "q": "ACTION2 moves DOWN by 1. Object at (row=8, col=12). After three ACTION2 in a row?",
        "options": {"A": "(row=5, col=12)", "B": "(row=11, col=12)",
                    "C": "(row=8, col=15)", "D": "(row=8, col=9)"},
        "correct": "B",
    },
    {
        "id": "T2-04", "cat": "T2",
        "q": "ACTION3 moves LEFT by 1. Object at (row=20, col=10). After five ACTION3 in a row?",
        "options": {"A": "(row=20, col=5)", "B": "(row=20, col=15)",
                    "C": "(row=15, col=10)", "D": "(row=25, col=10)"},
        "correct": "A",
    },

    # ── T3 multi-step plan (4) ──────────────────────────────────────────
    {
        "id": "T3-01", "cat": "T3",
        "q": "ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT (each moves 1 cell). Object at (row=10, col=20). Target at (row=10, col=23). What is the shortest action sequence?",
        "options": {"A": "ACTION1, ACTION1, ACTION1", "B": "ACTION3, ACTION3, ACTION3",
                    "C": "ACTION4, ACTION4, ACTION4", "D": "ACTION2, ACTION2, ACTION2"},
        "correct": "C",
    },
    {
        "id": "T3-02", "cat": "T3",
        "q": "ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT (1 cell each). Object at (row=12, col=15). Target at (row=15, col=15). What is the shortest action sequence?",
        "options": {"A": "ACTION1, ACTION1, ACTION1", "B": "ACTION2, ACTION2, ACTION2",
                    "C": "ACTION3, ACTION3, ACTION3", "D": "ACTION4, ACTION4, ACTION4"},
        "correct": "B",
    },
    {
        "id": "T3-03", "cat": "T3",
        "q": "ACTION1=UP, ACTION2=DOWN (1 cell each). Object at row=20. Target at row=14. How many ACTION1 calls needed?",
        "options": {"A": "4", "B": "6", "C": "8", "D": "10"},
        "correct": "B",
    },
    {
        "id": "T3-04", "cat": "T3",
        "q": "ACTION3=LEFT, ACTION4=RIGHT (1 cell each). Object at col=8. Target at col=20. How many ACTION4 calls?",
        "options": {"A": "10", "B": "12", "C": "14", "D": "16"},
        "correct": "B",
    },

    # ── T4 boundary check (3) ───────────────────────────────────────────
    {
        "id": "T4-01", "cat": "T4",
        "q": "Grid is 64x64 (rows 0-63, cols 0-63). Object at (row=0, col=10). ACTION1 (UP, row decreases). What happens?",
        "options": {"A": "Object moves to (row=-1, col=10)", "B": "No-op (out of bounds; row=0 is top edge)",
                    "C": "Wraps to row=63", "D": "Moves to col=9 instead"},
        "correct": "B",
    },
    {
        "id": "T4-02", "cat": "T4",
        "q": "Grid is 64x64. Object at (row=20, col=63). ACTION4 (RIGHT, col increases). What happens?",
        "options": {"A": "Object moves to col=64", "B": "No-op (right edge)",
                    "C": "Wraps to col=0", "D": "Moves to row=21"},
        "correct": "B",
    },
    {
        "id": "T4-03", "cat": "T4",
        "q": "Grid is 64x64. Object is at (row=63, col=63). Which actions will result in no-op (no movement)?",
        "options": {"A": "Only ACTION1 (UP)", "B": "Only ACTION2 (DOWN) and ACTION4 (RIGHT)",
                    "C": "Only ACTION3 (LEFT)", "D": "All four directional actions"},
        "correct": "B",
    },

    # ── T5 multi-dim plan (4) ───────────────────────────────────────────
    {
        "id": "T5-01", "cat": "T5",
        "q": "ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT (1 cell each). Object at (row=10, col=10). Target at (row=8, col=12). Total minimum actions?",
        "options": {"A": "2", "B": "3", "C": "4", "D": "5"},
        "correct": "C",
    },
    {
        "id": "T5-02", "cat": "T5",
        "q": "Same mapping as above. Object at (row=20, col=5). Target at (row=15, col=10). Minimum total actions?",
        "options": {"A": "5", "B": "10", "C": "15", "D": "20"},
        "correct": "B",
    },
    {
        "id": "T5-03", "cat": "T5",
        "q": "Same mapping. Object at (10, 10), target at (10, 10). Minimum actions?",
        "options": {"A": "0 (already there)", "B": "1", "C": "2", "D": "4"},
        "correct": "A",
    },
    {
        "id": "T5-04", "cat": "T5",
        "q": "Same mapping. Object at (4, 4), target at (1, 1). Minimum actions and which type?",
        "options": {"A": "3 ACTION1 only", "B": "3 ACTION3 only",
                    "C": "3 ACTION1 + 3 ACTION3 (6 total)", "D": "3 ACTION2 + 3 ACTION4 (6 total)"},
        "correct": "C",
    },

    # ── T6 selection (3) ────────────────────────────────────────────────
    {
        "id": "T6-01", "cat": "T6",
        "q": "ACTION6 takes (x, y) and selects whichever object covers that cell. Object A occupies rows 10-15, cols 20-25. ACTION6 with x=22, y=12 selects which?",
        "options": {"A": "Object A (inside its bounding box)",
                    "B": "No object (outside grid)",
                    "C": "An object at (12, 22) only",
                    "D": "Object A only if it's the active one already"},
        "correct": "A",
    },
    {
        "id": "T6-02", "cat": "T6",
        "q": "Object A spans rows 5-10, cols 10-15. Object B spans rows 30-35, cols 40-45. ACTION6 with x=42, y=32 selects:",
        "options": {"A": "Object A", "B": "Object B", "C": "Neither (between them)", "D": "Both"},
        "correct": "B",
    },
    {
        "id": "T6-03", "cat": "T6",
        "q": "Object A spans rows 0-5, cols 0-5. ACTION6 with x=10, y=10 selects:",
        "options": {"A": "Object A", "B": "Neither (point outside A's bbox)", "C": "An object at (10,10) only", "D": "Cannot tell"},
        "correct": "B",
    },

    # ── T7 inverse plan (3) ─────────────────────────────────────────────
    {
        "id": "T7-01", "cat": "T7",
        "q": "ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT. Object at (20, 20), target at (20, 18). What is the FIRST action?",
        "options": {"A": "ACTION1", "B": "ACTION2", "C": "ACTION3 (toward smaller col)", "D": "ACTION4"},
        "correct": "C",
    },
    {
        "id": "T7-02", "cat": "T7",
        "q": "Same mapping. Object at (10, 10), target at (10, 30). FIRST action (most efficient)?",
        "options": {"A": "ACTION1", "B": "ACTION2", "C": "ACTION3", "D": "ACTION4"},
        "correct": "D",
    },
    {
        "id": "T7-03", "cat": "T7",
        "q": "Same mapping. Object at (15, 10), target at (5, 10). FIRST action?",
        "options": {"A": "ACTION1 (toward smaller row)", "B": "ACTION2", "C": "ACTION3", "D": "ACTION4"},
        "correct": "A",
    },

    # ── T8 distance (3) ─────────────────────────────────────────────────
    {
        "id": "T8-01", "cat": "T8",
        "q": "Manhattan distance between (row=5, col=3) and (row=8, col=7)?",
        "options": {"A": "5", "B": "7", "C": "9", "D": "11"},
        "correct": "B",
    },
    {
        "id": "T8-02", "cat": "T8",
        "q": "Object A at (10, 10), B at (10, 20). Manhattan distance?",
        "options": {"A": "0", "B": "10", "C": "20", "D": "Cannot tell"},
        "correct": "B",
    },
    {
        "id": "T8-03", "cat": "T8",
        "q": "Object A spans rows 5-10, cols 5-10 (so center ≈ (7.5, 7.5)). Object B spans rows 20-25, cols 20-25 (center ≈ (22.5, 22.5)). Approximate manhattan distance between centers?",
        "options": {"A": "15", "B": "20", "C": "30", "D": "45"},
        "correct": "C",
    },
]

# Tally
N_PROBES = len(PROBES)
CATEGORIES = sorted({p["cat"] for p in PROBES})


def get_probes() -> list[dict]:
    """Return a fresh list copy so callers can't mutate the module-level data."""
    return [dict(p, options=dict(p["options"])) for p in PROBES]


__all__ = ["PROBES", "N_PROBES", "CATEGORIES", "get_probes"]
