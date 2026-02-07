"""Quantitative metrics for VAE evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def reconstruction_mse(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> float:
    """Mean-squared reconstruction error over a dataset.

    Parameters
    ----------
    model : a BaseVAE instance (in eval mode).
    dataloader : supplies (images, labels) tuples.
    max_batches : cap on how many batches to evaluate (``None`` = all).
    """
    model.eval()
    total_mse = 0.0
    n_samples = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            fwd = model(x)
            recon = fwd["recon"]
            total_mse += F.mse_loss(recon, x, reduction="sum").item()
            n_samples += x.size(0)
    return total_mse / max(n_samples, 1)


def compute_latent_stats(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
    max_batches: int | None = 20,
) -> dict[str, np.ndarray]:
    """Collect latent codes and labels for visualisation.

    Returns
    -------
    dict with keys ``"z"`` (N, D) and ``"labels"`` (N,).
    """
    model.eval()
    all_z: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            params = model.encode(x)
            mu = params[0]
            all_z.append(mu.cpu())
            all_labels.append(y)
    return {
        "z": torch.cat(all_z).numpy(),
        "labels": torch.cat(all_labels).numpy(),
    }
