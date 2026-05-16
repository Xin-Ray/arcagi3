"""Three small predictor models sharing a common interface.

All output a single sigmoid logit per sample (binary classification).

  - LogReg:        sklearn LogisticRegression(class_weight='balanced').
                   Fitted on numpy; no torch. Lightweight baseline.

  - MLP_S / MLP_L: PyTorch MLPs (128 / 512 hidden). Use BCEWithLogitsLoss,
                   class-balanced via pos_weight. Adam + cosine schedule.

  - CNN_small:     Raw-grid PyTorch CNN. Not used in Phase 0 - included
                   here as the next step if MLP saturates.

Models persist via `save(path)` / `load(path)` returning the same interface.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class Predictor(Protocol):
    """All predictors expose this interface for training + inference."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
            y_val: np.ndarray, *, max_epochs: int, seed: int) -> dict: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...


# ---------- sklearn LogReg ----------------------------------------------

@dataclass
class LogRegPredictor:
    name: str = "logreg"
    _model: object = None
    _scaler: object = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
            y_val: np.ndarray, *, max_epochs: int = 200, seed: int = 42) -> dict:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=False)
        Xs = scaler.fit_transform(X)
        Xvs = scaler.transform(X_val)
        model = LogisticRegression(
            class_weight="balanced", max_iter=max_epochs,
            random_state=seed, solver="lbfgs",
        )
        model.fit(Xs, y)
        self._model = model
        self._scaler = scaler
        # Single-row "history" so the training script can plot something
        train_loss = _bce(model.predict_proba(Xs)[:, 1], y)
        val_loss = _bce(model.predict_proba(Xvs)[:, 1], y_val)
        return {"epoch": [0], "train_loss": [float(train_loss)],
                "val_loss": [float(val_loss)]}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(X)
        return self._model.predict_proba(Xs)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"model": self._model, "scaler": self._scaler,
                         "name": self.name}, f)


# ---------- torch MLPs --------------------------------------------------

def _bce(p: np.ndarray, y: np.ndarray) -> float:
    """Numpy BCE for fitting metrics."""
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


class _TorchMLP:
    """Shared training loop for MLP-S / MLP-L. Lazy-imports torch."""
    def __init__(self, name: str, hidden: int, in_dim: int):
        import torch
        import torch.nn as nn
        self.name = name
        self._torch = torch
        self._nn = nn
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        ).to(self._device)
        self.in_dim = in_dim

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
            y_val: np.ndarray, *, max_epochs: int = 40, seed: int = 42) -> dict:
        torch = self._torch
        nn = self._nn
        torch.manual_seed(seed)

        pos_weight = torch.tensor(
            [(len(y) - y.sum()) / max(y.sum(), 1.0)],
            device=self._device, dtype=torch.float32,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

        Xt = torch.tensor(X, dtype=torch.float32, device=self._device)
        yt = torch.tensor(y, dtype=torch.float32, device=self._device)
        Xvt = torch.tensor(X_val, dtype=torch.float32, device=self._device)
        yvt = torch.tensor(y_val, dtype=torch.float32, device=self._device)

        history = {"epoch": [], "train_loss": [], "val_loss": []}
        batch_size = 256
        for epoch in range(max_epochs):
            self.net.train()
            perm = torch.randperm(Xt.shape[0], device=self._device)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, Xt.shape[0], batch_size):
                idx = perm[i:i + batch_size]
                logits = self.net(Xt[idx]).squeeze(-1)
                loss = criterion(logits, yt[idx])
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += float(loss.item())
                n_batches += 1
            sched.step()
            train_loss = total_loss / max(n_batches, 1)

            self.net.eval()
            with torch.no_grad():
                val_logits = self.net(Xvt).squeeze(-1)
                val_loss = float(criterion(val_logits, yvt).item())
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = self._torch
        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=self._device)
            logits = self.net(Xt).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def save(self, path: Path) -> None:
        torch = self._torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(),
                    "name": self.name, "in_dim": self.in_dim}, path)


class MLP_S(_TorchMLP):
    def __init__(self, in_dim: int):
        super().__init__("mlp_s", 128, in_dim)


class MLP_L(_TorchMLP):
    def __init__(self, in_dim: int):
        super().__init__("mlp_l", 512, in_dim)


# ---------- CNN on raw grid + action conditioning ---------------------

class _CNNSmall:
    """Shallow CNN: 16-channel color one-hot grid + action broadcast to 64x64.

    Input shape: (B, 17, 64, 64) -- 16 colors + 1 action channel.

    Architecture:
      conv3x3(17->32) BN ReLU pool2 -> conv3x3(32->64) BN ReLU pool2 ->
      conv3x3(64->64) BN ReLU global-avg-pool -> linear(64->1)

    Designed to be tiny (~80K params) so it trains fast and won't overfit
    our small dataset.
    """
    def __init__(self) -> None:
        import torch
        import torch.nn as nn
        self.name = "cnn_small"
        self._torch = torch
        self._nn = nn
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Conv2d(17, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        ).to(self._device)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
            y_val: np.ndarray, *, max_epochs: int = 30, seed: int = 42) -> dict:
        """X shape: (N, 17, 64, 64) float32."""
        torch = self._torch
        nn = self._nn
        torch.manual_seed(seed)
        if X.ndim != 4:
            raise ValueError(f"CNN expects 4D input, got {X.shape}")

        pos_weight = torch.tensor(
            [(len(y) - y.sum()) / max(y.sum(), 1.0)],
            device=self._device, dtype=torch.float32,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs)

        Xt = torch.tensor(X, dtype=torch.float32, device=self._device)
        yt = torch.tensor(y, dtype=torch.float32, device=self._device)
        Xvt = torch.tensor(X_val, dtype=torch.float32, device=self._device)
        yvt = torch.tensor(y_val, dtype=torch.float32, device=self._device)

        history = {"epoch": [], "train_loss": [], "val_loss": []}
        batch_size = 32
        for epoch in range(max_epochs):
            self.net.train()
            perm = torch.randperm(Xt.shape[0], device=self._device)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, Xt.shape[0], batch_size):
                idx = perm[i:i + batch_size]
                logits = self.net(Xt[idx]).squeeze(-1)
                loss = criterion(logits, yt[idx])
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += float(loss.item())
                n_batches += 1
            sched.step()
            train_loss = total_loss / max(n_batches, 1)

            self.net.eval()
            with torch.no_grad():
                vl = []
                for i in range(0, Xvt.shape[0], batch_size):
                    logits = self.net(Xvt[i:i + batch_size]).squeeze(-1)
                    vl.append(criterion(logits, yvt[i:i + batch_size]).item())
                val_loss = float(sum(vl) / max(len(vl), 1))
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = self._torch
        self.net.eval()
        out = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                Xt = torch.tensor(X[i:i + batch_size], dtype=torch.float32,
                                   device=self._device)
                logits = self.net(Xt).squeeze(-1)
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def save(self, path: Path) -> None:
        torch = self._torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "name": self.name}, path)


def CNN_small() -> _CNNSmall:
    """Factory so train_predictor can list models uniformly."""
    return _CNNSmall()
