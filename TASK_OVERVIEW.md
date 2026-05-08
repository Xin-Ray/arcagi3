# ARC Prize 2026 — ARC-AGI-3 Task Overview

**Competition**: [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
**Host**: ARC Prize / Kaggle
**Prize Pool**: $850,000 total

---

## What Is ARC-AGI-3?

ARC-AGI-3 is the **first fully interactive benchmark** in the ARC-AGI series. Unlike ARC-AGI-1/2 (which tested pattern recognition on static grid puzzles), ARC-AGI-3 presents **turn-based game environments** where an agent must:

- Explore an unknown environment with **no instructions, no rules, and no stated goals**
- Figure out how the environment works through interaction
- Discover what "winning" looks like on its own
- Progress through increasingly difficult levels within each environment

Human performance baseline: **100%**. Best frontier AI as of March 2026: **~0.51%**.

---

## Environment Structure

| Property | Detail |
|---|---|
| Observation space | 64×64 grid, each cell one of 16 possible colors |
| Action space | 5 directional keys + Undo + 1 coordinate selection (cell on grid) |
| Format | Turn-based; agent receives frame(s), submits one action per turn |
| Levels per environment | Minimum 6, with compositional difficulty |
| No instructions | Agent receives no goal or rule description |

### Core Knowledge Assumptions
Environments are grounded only in concepts accessible to all humans regardless of language or culture:
- Objectness and spatial relationships
- Basic geometry and topology
- Intuitive physics
- Agentness (recognizing interactive vs. passive elements)

---

## Dataset Splits

| Split | Environments | Purpose |
|---|---|---|
| Public Demo | 25 | Format exploration; intentionally easier |
| Semi-Private | 55 | External API testing and benchmarking |
| Fully Private | 55 | Official competition evaluation (hidden) |

Total: 135 environments, each with multiple levels (thousands of levels overall).

---

## Evaluation Metric: RHAE (Relative Human Action Efficiency)

### Per-Level Score
```
S = min(1.0, h / a)²
```
- `h` = second-best human action count for that level
- `a` = AI agent action count
- Quadratic penalty: being 2× less efficient yields 0.25, not 0.5

### Per-Environment Aggregation
- Weighted average across levels; level `l` has weight `w_l = l`
- Later levels (harder, requiring deeper understanding) count more

### Total Score
- Mean of all environment scores across the dataset
- Range: 0–100%

### Hard Cutoff
If a human takes `h` actions on average, the agent is cut off after `5h` actions (5× human budget). Exceeding this counts as failure on that level.

### Human Calibration
- 10+ participants per environment, 486 total across 2,893 attempts
- Median attempt duration: 7.4 minutes
- All humans: first-contact, no prior task-specific training

---

## Prize Structure

| Prize | Amount | Condition |
|---|---|---|
| Grand Prize | $700,000 | First agent achieving **100%** on ARC-AGI-3 |
| Top Score 1st | $40,000 | Highest score at competition end |
| Top Score 2nd | $15,000 | |
| Top Score 3rd | $10,000 | |
| Top Score 4th–5th | $5,000 each | |
| Milestone #1 (Jun 30, 2026) | $25K / $10K / $2.5K | Top 3 open-sourced solutions |
| Milestone #2 (Sep 30, 2026) | $25K / $10K / $2.5K | Top 3 open-sourced solutions |

---

## Key Rules

- Submissions must go through **Kaggle** only
- **No internet access** during evaluation
- All code must be **open-sourced** for prize eligibility
- No task-specific or domain-specific optimization (for leaderboard — must be general-purpose)
- No external tools or handcrafted harnesses for official leaderboard
- Hardware/compute limits: to be announced

---

## What Makes This Hard

ARC-AGI-3 tests four capabilities that current LLMs/agents lack:

1. **Exploration** — active information gathering in an unknown state space
2. **Modeling** — building a generalizable world model from sparse experience
3. **Goal-setting** — autonomously identifying what "winning" means
4. **Planning & Execution** — strategic multi-step action sequences with adaptive correction

---

## API & SDK

### Installation
```bash
pip install arc-agi
# or
uv add arc-agi
```

### Authentication
Set your API key (register at https://arcprize.org/api-keys):
```bash
export ARC_API_KEY=your_key_here
# or place in .env file
```

### Basic Interaction Pattern
```python
import arc_agi
from arc_agi import GameAction

arc = arc_agi.Arcade()

env = arc.make("ls20", render_mode="terminal")  # game_id from arcprize.org/tasks
obs, info = env.reset()

done = False
while not done:
    action = GameAction.ACTION1  # replace with your agent logic
    obs, reward, done, truncated, info = env.step(action)

scorecard = arc.get_scorecard()
print(scorecard)
```

### Available Game IDs
Browse all environments at: https://arcprize.org/tasks

---

## Useful Links

- [Kaggle Competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
- [ARC Prize 2026 Overview](https://arcprize.org/competitions/2026)
- [ARC-AGI-3 Launch Blog Post](https://arcprize.org/blog/arc-agi-3-launch)
- [Technical Paper (arXiv)](https://arxiv.org/abs/2603.24621)
- [API Documentation](https://docs.arcprize.org)
- [Browse Tasks](https://arcprize.org/tasks)
- [Leaderboard](https://arcprize.org/leaderboard)
