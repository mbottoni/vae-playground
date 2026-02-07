"""Model-agnostic trainer with MPS support for Apple Silicon."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from vae_playground.models.base import BaseVAE


def _get_device() -> torch.device:
    """Pick the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Trainer:
    """Train any :class:`BaseVAE` subclass.

    Parameters
    ----------
    model : a ``BaseVAE`` instance.
    lr : learning rate.
    device : ``None`` auto-detects (prefers MPS on Apple Silicon).
    """

    def __init__(
        self,
        model: BaseVAE,
        *,
        lr: float = 1e-3,
        device: torch.device | str | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    ) -> None:
        self.device = torch.device(device) if device else _get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        self.history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        val_loader: DataLoader | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        forward_kwargs: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Run the training loop.

        Returns the full ``history`` dict mapping loss-component names to
        lists of per-epoch values.
        """
        loss_kwargs = loss_kwargs or {}
        forward_kwargs = forward_kwargs or {}

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True,
                                            loss_kwargs=loss_kwargs,
                                            forward_kwargs=forward_kwargs)
            log = f"Epoch {epoch:3d}/{epochs}"
            for k, v in train_metrics.items():
                key = f"train_{k}"
                self.history.setdefault(key, []).append(v)
                log += f"  | {key}: {v:.4f}"

            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, train=False,
                                              loss_kwargs=loss_kwargs,
                                              forward_kwargs=forward_kwargs)
                for k, v in val_metrics.items():
                    key = f"val_{k}"
                    self.history.setdefault(key, []).append(v)
                    log += f"  | {key}: {v:.4f}"

            if verbose:
                print(log)

        return self.history

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        train: bool,
        loss_kwargs: dict[str, Any],
        forward_kwargs: dict[str, Any],
    ) -> dict[str, float]:
        self.model.train() if train else self.model.eval()
        accum: dict[str, float] = {}
        n_batches = 0

        ctx = torch.no_grad() if not train else _nullcontext()
        with ctx:
            for x, labels in loader:
                x = x.to(self.device)
                labels = labels.to(self.device)

                fwd = self.model(x, labels=labels, **forward_kwargs)
                loss_dict = self.model.loss_function(fwd, **loss_kwargs)

                if train:
                    self.optimizer.zero_grad()
                    loss_dict["loss"].backward()
                    self.optimizer.step()

                for k, v in loss_dict.items():
                    accum[k] = accum.get(k, 0.0) + v.item()
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights + optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "model_class": type(self.model).__name__,
                "model_config": {
                    "in_channels": self.model.in_channels,
                    "latent_dim": self.model.latent_dim,
                },
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model weights + optimizer state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt.get("history", {})


class _nullcontext:
    """Minimal no-op context manager (for Python <3.10 compat)."""
    def __enter__(self):
        return self
    def __exit__(self, *_: Any):
        pass
