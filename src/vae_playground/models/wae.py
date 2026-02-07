"""WAE-MMD — Wasserstein Autoencoder with MMD penalty (Tolstikhin et al., 2018).

Instead of KL divergence, WAE uses the Maximum Mean Discrepancy between
the aggregated posterior q(z) and the prior p(z) = N(0, I).
"""

from __future__ import annotations

from typing import Any

import torch

from vae_playground.models.vanilla_vae import VanillaVAE
from vae_playground.training.losses import reconstruction_loss, mmd_penalty


class WAE_MMD(VanillaVAE):
    """Wasserstein Autoencoder with MMD regulariser.

    Uses the same convolutional architecture as VanillaVAE but replaces
    the KL term with an MMD penalty.

    Parameters
    ----------
    reg_weight : weight of the MMD penalty.
    kernel_sigma : bandwidth for the RBF kernel.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 16,
        hidden_dims: list[int] | None = None,
        image_size: int = 32,
        reg_weight: float = 100.0,
        kernel_sigma: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, latent_dim, hidden_dims, image_size, **kwargs)
        self.reg_weight = reg_weight
        self.kernel_sigma = kernel_sigma

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        mu, log_var = self.encode(x)
        # WAE uses a *deterministic* encoder (just mu), no reparameterisation
        z = mu
        recon = self.decode(z)
        return {"recon": recon, "input": x, "z": z, "mu": mu, "log_var": log_var}

    def loss_function(self, fwd: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        recon = fwd["recon"]
        x = fwd["input"]
        z = fwd["z"]

        reg_weight = kwargs.get("reg_weight", self.reg_weight)

        recon_l = reconstruction_loss(recon, x, kind="mse")
        mmd = mmd_penalty(z, sigma=self.kernel_sigma)
        total = recon_l + reg_weight * mmd
        return {"loss": total, "recon_loss": recon_l, "mmd": mmd}
