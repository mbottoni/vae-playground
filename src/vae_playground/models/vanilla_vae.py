"""Standard Variational Autoencoder (Kingma & Welling, 2014)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vae_playground.models.base import BaseVAE
from vae_playground.training.losses import reconstruction_loss, kl_divergence


class VanillaVAE(BaseVAE):
    """Classic VAE with convolutional encoder / decoder.

    Architecture
    ------------
    Encoder: Conv2d blocks -> flatten -> fc_mu, fc_var
    Decoder: fc -> unflatten -> ConvTranspose2d blocks -> sigmoid
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 16,
        hidden_dims: list[int] | None = None,
        image_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, latent_dim)
        self.image_size = image_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        self.hidden_dims = hidden_dims

        # --- Encoder ---
        encoder_layers: list[nn.Module] = []
        ch = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(ch, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
            ])
            ch = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Compute flattened size after encoder
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            enc_out = self.encoder(dummy)
            self._enc_shape = enc_out.shape[1:]  # (C, H, W)
            flat_dim = enc_out.numel()

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_var = nn.Linear(flat_dim, latent_dim)

        # --- Decoder ---
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        decoder_layers: list[nn.Module] = []
        reversed_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_dims) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(reversed_dims[i], reversed_dims[i + 1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(reversed_dims[i + 1]),
                nn.ReLU(inplace=True),
            ])
        # Final layer back to input channels
        decoder_layers.extend([
            nn.ConvTranspose2d(reversed_dims[-1], in_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return [self.fc_mu(h), self.fc_var(h)]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, *self._enc_shape)
        h = self.decoder(h)
        # Crop / interpolate to exact image size
        h = torch.nn.functional.interpolate(h, size=self.image_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(h)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return {"recon": recon, "input": x, "mu": mu, "log_var": log_var, "z": z}

    def loss_function(self, fwd: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        recon = fwd["recon"]
        x = fwd["input"]
        mu = fwd["mu"]
        log_var = fwd["log_var"]

        recon_l = reconstruction_loss(recon, x, kind="bce")
        kl = kl_divergence(mu, log_var)
        return {"loss": recon_l + kl, "recon_loss": recon_l, "kl": kl}
