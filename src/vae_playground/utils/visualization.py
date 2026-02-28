"""Visualisation helpers for VAE experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


# ------------------------------------------------------------------
# Reconstruction comparison
# ------------------------------------------------------------------

def plot_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    n: int = 8,
    title: str = "Reconstructions",
) -> go.Figure:
    """Side-by-side grid: top row originals, bottom row reconstructions.

    Parameters
    ----------
    originals, reconstructions : (B, C, H, W) tensors in [0, 1].
    n : how many images to show.
    """
    n = min(n, originals.size(0))
    fig = make_subplots(rows=2, cols=n, vertical_spacing=0.02, horizontal_spacing=0.02)

    for i in range(n):
        for row, imgs in enumerate([originals, reconstructions], start=1):
            img = imgs[i].detach().cpu()
            if img.shape[0] == 1:
                img = img.squeeze(0)
                fig.add_trace(go.Heatmap(z=img.flip(0).numpy(), colorscale="gray", showscale=False), row=row, col=i + 1)
            else:
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                fig.add_trace(go.Image(z=img_np), row=row, col=i + 1)

    fig.update_layout(
        title_text=title,
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


# ------------------------------------------------------------------
# Latent space
# ------------------------------------------------------------------

def plot_latent_space(
    z: np.ndarray,
    labels: np.ndarray | None = None,
    method: str = "tsne",
    title: str = "Latent Space",
) -> go.Figure:
    """2-D scatter of latent codes, optionally coloured by class label.

    Parameters
    ----------
    z : (N, D) array — latent codes.
    labels : (N,) integer labels (optional).
    method : ``"tsne"`` or ``"pca"``.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if z.shape[1] > 2:
        if method == "tsne":
            z_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z)
        else:
            z_2d = PCA(n_components=2, random_state=42).fit_transform(z)
    else:
        z_2d = z

    fig = go.Figure()
    if labels is not None:
        for c in np.unique(labels):
            mask = labels == c
            fig.add_trace(go.Scatter(
                x=z_2d[mask, 0], y=z_2d[mask, 1],
                mode="markers", name=str(c),
                marker=dict(size=3, opacity=0.6),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=z_2d[:, 0], y=z_2d[:, 1],
            mode="markers",
            marker=dict(size=3, opacity=0.5),
        ))

    fig.update_layout(
        title_text=title,
        height=500, width=600,
        margin=dict(l=40, r=10, t=40, b=40),
        xaxis_title="dim 1",
        yaxis_title="dim 2",
    )
    return fig


# ------------------------------------------------------------------
# Loss curves
# ------------------------------------------------------------------

def plot_loss_curves(
    history: dict[str, list[float]],
    title: str = "Training Loss",
) -> go.Figure:
    """Line plot of one or more loss components over epochs."""
    fig = go.Figure()
    for name, values in history.items():
        fig.add_trace(go.Scatter(
            y=values, mode="lines+markers", name=name,
            marker=dict(size=4),
        ))
    fig.update_layout(
        title_text=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400,
        margin=dict(l=50, r=10, t=40, b=40),
    )
    return fig


# ------------------------------------------------------------------
# Prior sampling
# ------------------------------------------------------------------

def plot_samples(
    images: torch.Tensor,
    n: int = 16,
    title: str = "Samples from Prior",
) -> go.Figure:
    """Grid of sampled images."""
    n = min(n, images.size(0))
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, vertical_spacing=0.02, horizontal_spacing=0.02)

    for i in range(n):
        r = i // cols + 1
        c = i % cols + 1
        img = images[i].detach().cpu()
        if img.shape[0] == 1:
            img = img.squeeze(0)
            fig.add_trace(go.Heatmap(z=img.flip(0).numpy(), colorscale="gray", showscale=False), row=r, col=c)
        else:
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            fig.add_trace(go.Image(z=img_np), row=r, col=c)

    fig.update_layout(
        title_text=title,
        height=120 * rows + 40,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


# ------------------------------------------------------------------
# VQ-VAE codebook analysis
# ------------------------------------------------------------------

def plot_codebook_usage(
    usage: np.ndarray,
    title: str = "Codebook Usage",
) -> go.Figure:
    """Bar chart of how frequently each codebook entry was selected.

    Parameters
    ----------
    usage : (K,) integer array — count of selections per code.
    """
    K = len(usage)
    uniform = usage.sum() / K

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(K)),
        y=usage.tolist(),
        marker=dict(
            color=usage.tolist(),
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Count"),
        ),
        name="usage",
    ))
    fig.add_hline(
        y=uniform,
        line_dash="dash",
        line_color="red",
        annotation_text="uniform",
        annotation_position="top right",
    )
    n_active = int((usage > 0).sum())
    fig.update_layout(
        title_text=f"{title}  ({n_active}/{K} codes active)",
        xaxis_title="Codebook index",
        yaxis_title="Selection count",
        height=380,
        margin=dict(l=50, r=10, t=50, b=40),
        showlegend=False,
    )
    return fig


def plot_codebook_grid(
    model: Any,
    n: int = 32,
    device: torch.device | str = "cpu",
    title: str = "Codebook Entries (decoded)",
) -> go.Figure:
    """Decode each codebook entry individually and display as an image grid.

    For each of the first *n* entries in the VQ-VAE codebook, all spatial
    positions of the latent map are set to that entry's embedding vector and
    the result is decoded.

    Parameters
    ----------
    model : VQVAE instance.
    n : number of codebook entries to visualise.
    """
    model.eval()
    n = min(n, model.num_embeddings)

    with torch.no_grad():
        # Determine spatial size of the latent map
        dummy = torch.zeros(1, model.in_channels, model.image_size, model.image_size, device=device)
        z_e = model.encoder(dummy)
        _, _, H, W = z_e.shape

        images_list: list[torch.Tensor] = []
        for k in range(n):
            e_k = model.vq.embedding.weight[k]          # (D,)
            z_q = e_k.view(1, model.latent_dim, 1, 1).expand(1, model.latent_dim, H, W)
            img = model.decode(z_q)                      # (1, C, H, W)
            images_list.append(img.cpu())

    images = torch.cat(images_list, dim=0)               # (n, C, H, W)
    return plot_samples(images, n=n, title=title)
