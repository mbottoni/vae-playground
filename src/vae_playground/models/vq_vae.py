"""VQ-VAE — Vector-Quantised Variational Autoencoder (van den Oord et al., 2017).

Key difference from continuous VAEs: the latent space is *discrete*.
Encoder outputs are quantised to the nearest codebook vector before decoding.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_playground.models.base import BaseVAE
from vae_playground.training.losses import reconstruction_loss, vq_loss


class VectorQuantizer(nn.Module):
    """Nearest-neighbour lookup into a learnable codebook.

    Parameters
    ----------
    num_embeddings : codebook size K.
    embedding_dim : dimensionality of each code vector.
    beta : commitment loss weight.
    """

    def __init__(self, num_embeddings: int = 64, embedding_dim: int = 64, beta: float = 0.25) -> None:
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise encoder output.

        Parameters
        ----------
        z_e : (B, D, H, W) — continuous encoder output.

        Returns
        -------
        z_q : quantised tensor (same shape), with straight-through gradient.
        codebook_loss, commitment_loss : scalar losses.
        """
        # (B, D, H, W) -> (B*H*W, D)
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.D)

        # Distances to codebook
        dists = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_e_flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )
        indices = dists.argmin(dim=1)
        z_q_flat = self.embedding(indices)

        # Losses
        codebook_loss = F.mse_loss(z_q_flat, z_e_flat.detach())
        commitment_loss = self.beta * F.mse_loss(z_e_flat, z_q_flat.detach())

        # Straight-through estimator
        z_q_flat_st = z_e_flat + (z_q_flat - z_e_flat).detach()

        # Reshape back
        B, D, H, W = z_e.shape
        z_q = z_q_flat_st.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        return z_q, codebook_loss, commitment_loss


class VQVAE(BaseVAE):
    """VQ-VAE with convolutional encoder/decoder and a discrete codebook.

    Parameters
    ----------
    num_embeddings : codebook size.
    embedding_dim : code vector dimensionality.
    beta : commitment cost weight.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,  # used as embedding_dim
        hidden_dims: list[int] | None = None,
        image_size: int = 32,
        num_embeddings: int = 64,
        beta: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, latent_dim)
        self.image_size = image_size
        self.num_embeddings = num_embeddings
        embedding_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64]
        self.hidden_dims = hidden_dims

        # --- Encoder ---
        encoder_layers: list[nn.Module] = []
        ch = in_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(ch, h_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
            ])
            ch = h_dim
        # Project to embedding_dim channels
        encoder_layers.append(nn.Conv2d(ch, embedding_dim, 1))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- Vector quantizer ---
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, beta)

        # --- Decoder ---
        decoder_layers: list[nn.Module] = []
        reversed_dims = list(reversed(hidden_dims))
        decoder_layers.append(nn.Conv2d(embedding_dim, reversed_dims[0], 1))
        decoder_layers.append(nn.ReLU(inplace=True))
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

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return [z_e] (continuous encoder output, before quantisation)."""
        return [self.encoder(x)]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        h = F.interpolate(h, size=self.image_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(h)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        z_e = self.encode(x)[0]
        z_q, codebook_loss, commitment_loss = self.vq(z_e)
        recon = self.decode(z_q)
        return {
            "recon": recon,
            "input": x,
            "z_e": z_e,
            "z_q": z_q,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }

    def loss_function(self, fwd: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        recon_l = reconstruction_loss(fwd["recon"], fwd["input"], kind="bce")
        cb = fwd["codebook_loss"]
        cm = fwd["commitment_loss"]
        total = recon_l + cb + cm
        return {"loss": total, "recon_loss": recon_l, "codebook_loss": cb, "commitment_loss": cm}

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Sample random codebook indices and decode."""
        # Determine spatial dimensions of latent map
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.image_size, self.image_size, device=device)
            z_e = self.encoder(dummy)
            _, _, H, W = z_e.shape
        indices = torch.randint(0, self.num_embeddings, (num_samples, H * W), device=device)
        z_q = self.vq.embedding(indices)  # (B, H*W, D)
        z_q = z_q.view(num_samples, H, W, self.latent_dim).permute(0, 3, 1, 2).contiguous()
        return self.decode(z_q)
