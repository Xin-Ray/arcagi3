"""Ablation aggregator — turn a directory of per-(agent, game) summary.json files
into the comparison tables specified in `docs/ARCHITECTURE_AGENTS.md` §2.

Inputs are the standard `summary.json` files written by `arc_agent.eval_split.
write_summary` (the same shape `scripts/run_baseline.py` emits). The
aggregator reads them, groups by `agent`, and produces:

- `RHAE table`: one row per agent with mean RHAE, mean F1, parse_rate,
  wall-clock per step, levels_completed, no_op_rate.
- `Per-game breakdown`: agent × game RHAE matrix.

Pure library code — no I/O orchestration. The CLI lives in
`scripts/report_ablation.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


@dataclass
class AgentRow:
    agent: str
    n_games: int = 0
    mean_rhae: float = 0.0
    mean_f1: float = 0.0
    parse_rate: float = 0.0
    mean_actions: float = 0.0
    mean_levels: float = 0.0
    wall_per_step_sec: float = 0.0
    no_op_rate: float = 0.0
    per_game_rhae: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent":            self.agent,
            "n_games":          self.n_games,
            "mean_rhae":        round(self.mean_rhae, 4),
            "mean_f1":          round(self.mean_f1, 4),
            "parse_rate":       round(self.parse_rate, 4),
            "mean_actions":     round(self.mean_actions, 2),
            "mean_levels":      round(self.mean_levels, 2),
            "wall_per_step_sec": round(self.wall_per_step_sec, 3),
            "no_op_rate":       round(self.no_op_rate, 4),
            "per_game_rhae":    {g: round(v, 4) for g, v in self.per_game_rhae.items()},
        }


def load_summary(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def aggregate(summaries: Iterable[dict[str, Any]]) -> list[AgentRow]:
    """Group summaries by `agent` field and return one AgentRow per group.

    Mean is over the games present in each agent's run. RHAE is pulled from
    `per_game[g]["score"]` (the SDK RHAE), falling back to 0 if absent.
    `no_op_rate` is computed from `per_game[g]["actions"]` vs `n_parseable`
    when available; if the per-game dict doesn't carry no-op info it stays 0.
    """
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for s in summaries:
        agent = s.get("agent", "unknown")
        by_agent.setdefault(agent, []).append(s)

    rows: list[AgentRow] = []
    for agent, runs in by_agent.items():
        row = AgentRow(agent=agent)
        rhaes: list[float] = []
        f1s: list[float] = []
        parses: list[float] = []
        actions_list: list[int] = []
        levels_list: list[int] = []
        wall_list: list[float] = []
        no_op_list: list[float] = []
        games: set[str] = set()

        for s in runs:
            f1s.append(float(s.get("mean_f1", 0.0)))
            parses.append(float(s.get("parse_rate", 0.0)))
            per_game = s.get("per_game", {}) or {}
            wall = float(s.get("wall_clock_seconds", 0.0))
            total_steps = sum(int(g.get("actions", 0)) for g in per_game.values())
            if total_steps:
                wall_list.append(wall / total_steps)

            for game_id, gm in per_game.items():
                games.add(game_id)
                score = gm.get("score")
                if isinstance(score, (int, float)):
                    rhaes.append(float(score))
                    row.per_game_rhae[game_id] = float(score)
                actions = int(gm.get("actions", 0))
                actions_list.append(actions)
                levels_list.append(int(gm.get("levels_completed", 0)))
                # No-op proxy: when the agent has a parse_rate but we don't
                # know step-level frame_changed counts in summary.json. Use
                # 1 - parse_rate per game as a rough surrogate ONLY when no
                # other signal is available; this is a known-loose estimate.
                pr = float(gm.get("parse_rate", 0.0))
                no_op_list.append(max(0.0, 1.0 - pr))

        row.n_games = len(games)
        row.mean_rhae = mean(rhaes) if rhaes else 0.0
        row.mean_f1 = mean(f1s) if f1s else 0.0
        row.parse_rate = mean(parses) if parses else 0.0
        row.mean_actions = mean(actions_list) if actions_list else 0.0
        row.mean_levels = mean(levels_list) if levels_list else 0.0
        row.wall_per_step_sec = mean(wall_list) if wall_list else 0.0
        row.no_op_rate = mean(no_op_list) if no_op_list else 0.0
        rows.append(row)

    rows.sort(key=lambda r: r.mean_rhae, reverse=True)
    return rows


def render_table(rows: list[AgentRow]) -> str:
    """ASCII table — `docs/ARCHITECTURE_AGENTS.md` §2 primary view.

    Width auto-fits the agent column so A1/A2/A3 row labels stay
    distinguishable. F1/parse columns show "n/a" for agents whose
    architecture has no F1 path (every row except A2 / "full").
    """
    name_w = max(20, max((len(r.agent) for r in rows), default=20))
    cols = ("agent", "n", "RHAE", "F1", "parse", "actions",
            "levels", "s/step", "no_op")
    widths = [name_w, 4, 8, 8, 8, 8, 8, 8, 8]
    header = "| " + " | ".join(
        f"{c:>{w}}" for c, w in zip(cols, widths)
    ) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [header, sep]
    for r in rows:
        has_f1_path = ":full" in r.agent or r.agent.endswith(":full")
        f1_str = f"{r.mean_f1:>{widths[3]}.3f}" if has_f1_path else f"{'n/a':>{widths[3]}}"
        parse_str = f"{r.parse_rate:>{widths[4]}.3f}" if has_f1_path else f"{'n/a':>{widths[4]}}"
        no_op_str = f"{r.no_op_rate:>{widths[8]}.3f}" if has_f1_path else f"{'n/a':>{widths[8]}}"
        out.append(
            "| " + " | ".join([
                f"{r.agent:>{widths[0]}}",
                f"{r.n_games:>{widths[1]}}",
                f"{r.mean_rhae:>{widths[2]}.3f}",
                f1_str,
                parse_str,
                f"{r.mean_actions:>{widths[5]}.1f}",
                f"{r.mean_levels:>{widths[6]}.2f}",
                f"{r.wall_per_step_sec:>{widths[7]}.2f}",
                no_op_str,
            ]) + " |"
        )
    return "\n".join(out)


def render_per_game(rows: list[AgentRow]) -> str:
    """Agent × game RHAE matrix — `docs/ARCHITECTURE_AGENTS.md` §2 secondary view."""
    games = sorted({g for r in rows for g in r.per_game_rhae})
    if not games:
        return "(no per-game RHAE recorded)"
    name_w = max(20, max((len(r.agent) for r in rows), default=20))
    short_games = [g.split("-", 1)[0] for g in games]  # "ar25-..." -> "ar25"
    header = ["agent"] + short_games
    widths = [name_w] + [10] * len(games)
    out = ["| " + " | ".join(f"{h:>{w}}" for h, w in zip(header, widths)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        cells = [f"{r.agent:>{name_w}}"]
        for g in games:
            v = r.per_game_rhae.get(g)
            cells.append(f"{v:>10.3f}" if isinstance(v, float) else f"{'--':>10}")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def decision_gate(rows: list[AgentRow], *, threshold: float = 0.02) -> str:
    """Return the §5 decision-gate verdict in one sentence.

    > One of A2/A3/A4 beats A1 by ≥ 0.02 RHAE
    > If A1 wins outright, that's an important negative result
    """
    by_agent = {r.agent: r for r in rows}
    a1_keys = [k for k in by_agent if "lite" in k.lower() and ":lite_h0" in k.lower()]
    if not a1_keys:
        a1_keys = [k for k in by_agent if "lite" in k.lower()]
    if not a1_keys:
        return "no A1 (lite) row in summary — can't apply gate"
    a1 = by_agent[a1_keys[0]].mean_rhae
    contenders = {
        k: r.mean_rhae for k, r in by_agent.items()
        if k != a1_keys[0] and "random" not in k.lower()
    }
    if not contenders:
        return f"only A1 (RHAE={a1:.3f}) — need ≥1 scaffolded agent to apply gate"
    best_name = max(contenders, key=contenders.get)
    best_rhae = contenders[best_name]
    delta = best_rhae - a1
    if delta >= threshold:
        return (
            f"{best_name} beats A1 by {delta:+.3f} RHAE (>={threshold:.2f}) — "
            "scaffold is justified; use it as the GRPO base."
        )
    return (
        f"no scaffold beats A1 by >= {threshold:.2f} RHAE (best delta = "
        f"{delta:+.3f}). Per §5: ship A1 to GRPO (simplest wins)."
    )
