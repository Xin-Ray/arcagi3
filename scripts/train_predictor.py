"""Train the frame-change predictor (M1 LogReg + M2/M3 MLPs).

Per docs/arch_predictor_v0_zh.md S5-S7. Reads existing trace.jsonl,
applies per-action class balancing, splits into train / val by game id
(dc22 30 % is val + rest is train), trains three model variants, dumps
metrics + plots to outputs/predictor_v0/.

Usage:
    .venv/Scripts/python.exe scripts/train_predictor.py
        --output outputs/predictor_v0
        --models logreg,mlp_s,mlp_l
        --max-epochs 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from arc_agent.predictor.dataset import (
    Sample, scan_traces, build_dataset, _short_game,
)
from arc_agent.predictor.features import (
    FEATURE_DIM, encode_batch, labels_array,
)
from arc_agent.predictor.models import (
    LogRegPredictor, MLP_S, MLP_L,
)


def _split_by_game(
    samples: list[Sample], val_game: str = "dc22", val_frac: float = 0.3,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """All other games to train; val_game split val_frac to val.

    Returns (train, val).
    """
    rng = np.random.default_rng(seed)
    train, val_pool = [], []
    for s in samples:
        if _short_game(s.game_id) == val_game:
            val_pool.append(s)
        else:
            train.append(s)
    rng.shuffle(val_pool)
    n_val = int(len(val_pool) * val_frac)
    val = val_pool[:n_val]
    train.extend(val_pool[n_val:])
    rng.shuffle(train)
    return train, val


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """ROC-AUC + accuracy + ECE. Pure numpy."""
    from sklearn.metrics import roc_auc_score, accuracy_score
    auc = float("nan")
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        pass
    acc = float(accuracy_score(y, (p >= 0.5).astype(int)))
    # 10-bin ECE
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    n = len(y)
    for i in range(10):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = float(y[mask].mean())
        bin_conf = float(p[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return {"auc": auc, "acc": acc, "ece": float(ece), "n": int(n)}


def _per_action_auc(
    p: np.ndarray, y: np.ndarray, samples: list[Sample],
) -> dict[str, dict]:
    """Per-action AUC over the val set."""
    from sklearn.metrics import roc_auc_score
    out: dict[str, dict] = {}
    actions = sorted(set(s.action for s in samples))
    for a in actions:
        mask = np.array([s.action == a for s in samples])
        if mask.sum() < 5 or y[mask].sum() == 0 or y[mask].sum() == mask.sum():
            out[a] = {"auc": None, "n": int(mask.sum())}
            continue
        try:
            auc = float(roc_auc_score(y[mask], p[mask]))
        except ValueError:
            auc = float("nan")
        out[a] = {"auc": auc, "n": int(mask.sum())}
    return out


def _plot_train_curves(histories: dict[str, dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, hist in histories.items():
        ax1.plot(hist["epoch"], hist["train_loss"], label=f"{name} train")
        ax2.plot(hist["epoch"], hist["val_loss"], label=f"{name} val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train BCE loss")
    ax1.legend(); ax1.set_title("Train loss")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val BCE loss")
    ax2.legend(); ax2.set_title("Val loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_roc(rocs: dict[str, tuple[np.ndarray, np.ndarray]],
              aucs: dict[str, float], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (fpr, tpr) in rocs.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5, label="random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Val ROC")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_per_action(per_action: dict[str, dict[str, dict]],
                     out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    actions = sorted({a for d in per_action.values() for a in d})
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(per_action), 1)
    for i, (name, d) in enumerate(per_action.items()):
        ys = [d.get(a, {}).get("auc") or 0.0 for a in actions]
        x = np.arange(len(actions)) + (i - len(per_action) / 2) * width + width / 2
        ax.bar(x, ys, width=width, label=name)
    ax.set_xticks(np.arange(len(actions)))
    ax.set_xticklabels(actions, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Per-action val AUC")
    ax.legend()
    ax.axhline(0.5, color="k", lw=0.5, ls="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/predictor_v0")
    parser.add_argument("--root", default="outputs", help="Where trace.jsonls live")
    parser.add_argument("--models", default="logreg,mlp_s,mlp_l")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip per-action class balancing.")
    parser.add_argument("--no-game-feature", action="store_true",
                        help="Ablation: drop game-id one-hot from features.")
    args = parser.parse_args()

    out_dir = REPO / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scan] traces in {REPO / args.root}", file=sys.stderr)
    raw = scan_traces(REPO / args.root)
    print(f"[scan] {len(raw)} raw samples", file=sys.stderr)

    samples = build_dataset(
        raw,
        balance="none" if args.no_balance else "per_action",
        action_filter=None,
    )
    print(f"[scan] {len(samples)} samples after balance", file=sys.stderr)

    train, val = _split_by_game(samples, val_game="dc22",
                                 val_frac=0.3, seed=args.seed)
    print(f"[split] train={len(train)} val={len(val)}", file=sys.stderr)

    if args.no_game_feature:
        from arc_agent.predictor.features_no_game import encode_batch_no_game
        X_train = encode_batch_no_game(train); y_train = labels_array(train)
        X_val = encode_batch_no_game(val);     y_val = labels_array(val)
    else:
        X_train = encode_batch(train); y_train = labels_array(train)
        X_val = encode_batch(val);     y_val = labels_array(val)
    print(f"[feat] X_train={X_train.shape} y mean={y_train.mean():.3f}",
          file=sys.stderr)
    print(f"[feat] X_val  ={X_val.shape}   y mean={y_val.mean():.3f}",
          file=sys.stderr)

    model_specs = []
    for name in args.models.split(","):
        n = name.strip()
        if n == "logreg":
            model_specs.append((n, LogRegPredictor()))
        elif n == "mlp_s":
            model_specs.append((n, MLP_S(in_dim=FEATURE_DIM)))
        elif n == "mlp_l":
            model_specs.append((n, MLP_L(in_dim=FEATURE_DIM)))
        else:
            print(f"[warn] unknown model: {n}", file=sys.stderr)

    histories: dict[str, dict] = {}
    val_probs: dict[str, np.ndarray] = {}
    val_metrics_all: dict[str, dict] = {}
    train_metrics_all: dict[str, dict] = {}
    per_action: dict[str, dict[str, dict]] = {}
    rocs: dict[str, tuple] = {}
    aucs: dict[str, float] = {}

    for name, model in model_specs:
        print(f"[fit] {name} ...", file=sys.stderr)
        hist = model.fit(X_train, y_train, X_val, y_val,
                         max_epochs=args.max_epochs, seed=args.seed)
        histories[name] = hist
        val_p = model.predict_proba(X_val)
        train_p = model.predict_proba(X_train)
        val_probs[name] = val_p
        val_metrics_all[name] = _metrics(val_p, y_val)
        train_metrics_all[name] = _metrics(train_p, y_train)
        per_action[name] = _per_action_auc(val_p, y_val, val)
        # ROC
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_val, val_p)
            rocs[name] = (fpr, tpr)
            aucs[name] = val_metrics_all[name]["auc"]
        except (ImportError, ValueError):
            pass
        model.save(out_dir / f"{name}.pt")
        print(f"[fit] {name}  train_auc={train_metrics_all[name]['auc']:.3f} "
              f"val_auc={val_metrics_all[name]['auc']:.3f}  "
              f"val_acc={val_metrics_all[name]['acc']:.3f}", file=sys.stderr)

    # Dump metrics + plots
    metrics = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "train_mean_y": float(y_train.mean()),
        "val_mean_y": float(y_val.mean()),
        "train_metrics": train_metrics_all,
        "val_metrics": val_metrics_all,
        "per_action_val_auc": per_action,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    _plot_train_curves(histories, out_dir / "train_curves.png")
    if rocs:
        _plot_roc(rocs, aucs, out_dir / "roc.png")
    _plot_per_action(per_action, out_dir / "per_action_auc.png")

    print(f"[done] {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
