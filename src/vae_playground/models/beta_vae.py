"""Beta-VAE — disentangled representation learning (Higgins et al., 2017).

The only change from VanillaVAE is a ``beta`` weight on the KL term.
When ``beta > 1`` the model is encouraged to learn disentangled factors.
"""

from __future__ import annotations

from typing import Any

import torch

from vae_playground.models.vanilla_vae import VanillaVAE
from vae_playground.training.losses import reconstruction_loss, kl_divergence


class BetaVAE(VanillaVAE):
    """VanillaVAE with a tuneable beta coefficient on the KL term.

    Parameters
    ----------
    beta : float
        Weight of the KL divergence in the ELBO.  ``beta = 1`` recovers
        the standard VAE; ``beta > 1`` encourages disentanglement.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 16,
        hidden_dims: list[int] | None = None,
        image_size: int = 32,
        beta: float = 4.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, latent_dim, hidden_dims, image_size, **kwargs)
        self.beta = beta

    def loss_function(self, fwd: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        recon = fwd["recon"]
        x = fwd["input"]
        mu = fwd["mu"]
        log_var = fwd["log_var"]

        # Override beta from kwargs if supplied (e.g. from marimo slider)
        beta = kwargs.get("beta", self.beta)

        recon_l = reconstruction_loss(recon, x, kind="bce")
        kl = kl_divergence(mu, log_var)
        return {"loss": recon_l + beta * kl, "recon_loss": recon_l, "kl": kl, "beta": torch.tensor(beta)}
