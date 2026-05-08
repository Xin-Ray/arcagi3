"""
ARC-AGI-3 starter agent — minimal one-game demo using the arc_agent library.

Demonstrates the canonical SDK 0.9.x flow:
  Arcade()  ->  open_scorecard()  ->  make(game_id)  ->  step loop  ->  close_scorecard()

For batch evaluation across many games, use `eval.py` instead.
Requires `ARC_API_KEY` in `.env` — anonymous mode is rejected by the live API.
"""

import os

from dotenv import load_dotenv

from arc_agi import Arcade

from arc_agent.agents.random import RandomAgent
from arc_agent.runner import play_one

load_dotenv()

GAME_ID = "ls20"  # any game ID listed at https://three.arcprize.org/
MAX_ACTIONS = 80  # safety cap (matches official scaffold default)


def main() -> None:
    key = os.getenv("ARC_API_KEY", "")
    if not key or key.startswith("your_"):
        raise RuntimeError(
            "ARC_API_KEY is unset or still the .env.example placeholder. "
            "Put a real key in .env (https://arcprize.org/api-keys). "
            "Anonymous mode is rejected by https://three.arcprize.org as of 2026-04-27."
        )

    arc = Arcade()
    card_id = arc.open_scorecard(tags=["agent_starter", "random"])
    result = play_one(arc, RandomAgent(), GAME_ID, card_id, max_actions=MAX_ACTIONS)
    print(result)
    scorecard = arc.close_scorecard(card_id)
    if scorecard is not None:
        print("Scorecard score:", scorecard.score)


if __name__ == "__main__":
    main()
