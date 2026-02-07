"""Conditional VAE — class-conditioned generation (Sohn et al., 2015).

The label is one-hot encoded and concatenated to both the encoder input
(as an extra channel map) and the decoder input (appended to z).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_playground.models.base import BaseVAE
from vae_playground.training.losses import reconstruction_loss, kl_divergence


class ConditionalVAE(BaseVAE):
    """VAE conditioned on class labels.

    Parameters
    ----------
    num_classes : number of classes (10 for MNIST).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 16,
        num_classes: int = 10,
        hidden_dims: list[int] | None = None,
        image_size: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, latent_dim)
        self.num_classes = num_classes
        self.image_size = image_size

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        self.hidden_dims = hidden_dims

        # --- Label embedding (broadcast to spatial map) ---
        self.label_embed = nn.Linear(num_classes, image_size * image_size)

        # --- Encoder (in_channels + 1 for the label channel) ---
        encoder_layers: list[nn.Module] = []
        ch = in_channels + 1  # extra channel for label
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(ch, h_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
            ])
            ch = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels + 1, image_size, image_size)
            enc_out = self.encoder(dummy)
            self._enc_shape = enc_out.shape[1:]
            flat_dim = enc_out.numel()

        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_var = nn.Linear(flat_dim, latent_dim)

        # --- Decoder (z concatenated with one-hot label) ---
        self.fc_decode = nn.Linear(latent_dim + num_classes, flat_dim)

        decoder_layers: list[nn.Module] = []
        reversed_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_dims) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(reversed_dims[i], reversed_dims[i + 1],
                                   3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(reversed_dims[i + 1]),
                nn.ReLU(inplace=True),
            ])
        decoder_layers.append(
            nn.ConvTranspose2d(reversed_dims[-1], in_channels,
                               3, stride=2, padding=1, output_padding=1),
        )
        self.decoder = nn.Sequential(*decoder_layers)

    # ------------------------------------------------------------------

    def _label_to_channel(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert integer labels to a spatial feature map (B, 1, H, W)."""
        one_hot = F.one_hot(labels, self.num_classes).float()
        embedded = self.label_embed(one_hot)  # (B, H*W)
        return embedded.view(-1, 1, self.image_size, self.image_size)

    def encode(self, x: torch.Tensor, **kwargs: Any) -> list[torch.Tensor]:
        labels = kwargs["labels"]
        label_ch = self._label_to_channel(labels)
        h = torch.cat([x, label_ch], dim=1)
        h = self.encoder(h).flatten(1)
        return [self.fc_mu(h), self.fc_var(h)]

    def decode(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        labels = kwargs["labels"]
        one_hot = F.one_hot(labels, self.num_classes).float()
        z_cond = torch.cat([z, one_hot], dim=1)
        h = self.fc_decode(z_cond)
        h = h.view(-1, *self._enc_shape)
        h = self.decoder(h)
        h = torch.nn.functional.interpolate(h, size=self.image_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(h)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        labels = kwargs["labels"]
        mu, log_var = self.encode(x, labels=labels)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, labels=labels)
        return {"recon": recon, "input": x, "mu": mu, "log_var": log_var, "z": z, "labels": labels}

    def loss_function(self, fwd: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        recon_l = reconstruction_loss(fwd["recon"], fwd["input"], kind="bce")
        kl = kl_divergence(fwd["mu"], fwd["log_var"])
        return {"loss": recon_l + kl, "recon_loss": recon_l, "kl": kl}

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str = "cpu",
               labels: torch.Tensor | None = None) -> torch.Tensor:
        """Sample conditioned on labels.  If labels is ``None``, sample uniformly."""
        if labels is None:
            labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, labels=labels)
