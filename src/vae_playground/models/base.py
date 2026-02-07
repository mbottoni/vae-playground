"""Abstract base class for all VAE variants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseVAE(ABC, nn.Module):
    """Every VAE variant must inherit from this class.

    The contract:
    - ``encode``  returns distribution parameters (mu, log_var, or similar).
    - ``decode``  maps a latent vector back to input space.
    - ``forward`` runs the full encode -> reparameterise -> decode pipeline
      and returns a dict with at least ``recon``, ``input``, and whatever
      extra keys the loss function needs.
    - ``loss_function`` takes the dict from ``forward`` and returns a dict
      ``{"loss": total, ...}`` with an itemised breakdown.
    - ``sample``  generates new data by sampling from the prior.
    """

    def __init__(self, in_channels: int, latent_dim: int, **kwargs: Any) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return distribution parameters (e.g. [mu, log_var])."""
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent code *z* back to input space."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Full forward pass.  Must include keys ``recon`` and ``input``."""
        ...

    @abstractmethod
    def loss_function(self, fwd: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute the loss from the dict returned by ``forward``.

        Must return at least ``{"loss": <scalar>}``.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Sample from the prior N(0, I) and decode."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Reconstruct input through the full pipeline."""
        fwd = self.forward(x, **kwargs)
        return fwd["recon"]

    def get_device(self) -> torch.device:
        """Return device of the first parameter."""
        return next(self.parameters()).device
