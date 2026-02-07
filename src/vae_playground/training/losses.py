"""Loss components shared across VAE variants."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# Reconstruction losses
# ------------------------------------------------------------------

def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    kind: str = "bce",
) -> torch.Tensor:
    """Per-batch reconstruction loss.

    Parameters
    ----------
    kind : ``"bce"`` | ``"mse"``
        ``"bce"`` expects inputs in [0, 1] (sigmoid output).
        ``"mse"`` works with any range.
    """
    if kind == "bce":
        return F.binary_cross_entropy(recon, target, reduction="sum") / target.size(0)
    elif kind == "mse":
        return F.mse_loss(recon, target, reduction="sum") / target.size(0)
    else:
        raise ValueError(f"Unknown reconstruction loss: {kind}")


# ------------------------------------------------------------------
# KL divergence  (for Gaussian posterior)
# ------------------------------------------------------------------

def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL( q(z|x) || p(z) ) for diagonal Gaussian posterior vs N(0,I) prior.

    Returns a scalar (mean over batch).
    """
    return -0.5 * torch.mean(
        torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    )


# ------------------------------------------------------------------
# MMD penalty  (for WAE-MMD)
# ------------------------------------------------------------------

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """RBF (Gaussian) kernel between two sets of vectors."""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)
    return torch.exp(-((tiled_x - tiled_y).pow(2).sum(2)) / (2 * sigma * dim))


def mmd_penalty(z: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Maximum Mean Discrepancy between encoded z and N(0,I) samples."""
    prior = torch.randn_like(z)
    k_zz = _rbf_kernel(z, z, sigma)
    k_pp = _rbf_kernel(prior, prior, sigma)
    k_zp = _rbf_kernel(z, prior, sigma)
    return k_zz.mean() + k_pp.mean() - 2 * k_zp.mean()


# ------------------------------------------------------------------
# VQ loss components
# ------------------------------------------------------------------

def vq_loss(
    z_e: torch.Tensor,
    z_q: torch.Tensor,
    beta: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vector-quantisation loss: codebook + commitment.

    Returns (codebook_loss, commitment_loss).
    """
    codebook_loss = F.mse_loss(z_q, z_e.detach())
    commitment_loss = F.mse_loss(z_e, z_q.detach())
    return codebook_loss, beta * commitment_loss
