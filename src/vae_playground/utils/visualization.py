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
