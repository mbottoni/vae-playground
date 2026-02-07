# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo[recommended]>=0.19.9",
#     "torch>=2.5",
#     "torchvision>=0.20",
#     "plotly>=5.24",
#     "scikit-learn>=1.5",
#     "numpy>=2.0",
# ]
# ///

"""VQ-VAE — Vector-Quantised VAE (van den Oord et al., 2017)."""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


# ── Theory ────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # VQ-VAE

    Unlike continuous VAEs, VQ-VAE learns a **discrete** latent space using a
    learnable codebook of embedding vectors.

    $$\mathcal{L} = \| x - \hat{x} \|^2
      + \| \text{sg}[z_e] - e \|^2
      + \beta \| z_e - \text{sg}[e] \|^2$$

    Where:
    - $z_e$ = encoder output (continuous)
    - $e$ = nearest codebook vector
    - $\text{sg}[\cdot]$ = stop-gradient operator
    - The second term updates the **codebook** to move toward encoder outputs
    - The third is the **commitment loss** — keeps the encoder close to chosen codes

    Gradients flow through the quantisation step via the **straight-through estimator**.
    """)
    return (mo,)


# ── Imports ───────────────────────────────────────────────────────────
@app.cell
def _():
    import sys
    from pathlib import Path

    _root = Path("__file__").resolve().parent.parent
    if str(_root / "src") not in sys.path:
        sys.path.insert(0, str(_root / "src"))

    import torch
    import numpy as np

    from vae_playground.models import VQVAE
    from vae_playground.training import Trainer
    from vae_playground.data import get_dataloader
    from vae_playground.utils import (
        plot_reconstructions,
        plot_loss_curves,
        plot_samples,
    )
    from vae_playground.utils.metrics import reconstruction_mse
    return (
        Path, Trainer, VQVAE, get_dataloader, np, plot_loss_curves,
        plot_reconstructions, plot_samples, reconstruction_mse, sys, torch,
    )


# ── Configuration ─────────────────────────────────────────────────────
@app.cell
def _(mo):
    dataset_dd = mo.ui.dropdown(
        options=["mnist", "fashion_mnist", "cifar10"],
        value="mnist",
        label="Dataset",
    )
    embedding_dim_slider = mo.ui.slider(16, 128, value=64, step=16, label="Embedding dim")
    num_embed_slider = mo.ui.slider(16, 256, value=64, step=16, label="Codebook size (K)")
    commit_slider = mo.ui.slider(0.05, 2.0, value=0.25, step=0.05, label="Commitment β")
    lr_slider = mo.ui.slider(1e-4, 1e-2, value=1e-3, step=1e-4, label="Learning rate")
    epochs_slider = mo.ui.slider(1, 50, value=10, step=1, label="Epochs")
    batch_size_slider = mo.ui.slider(32, 512, value=128, step=32, label="Batch size")

    mo.md(f"""
    ## Configuration

    | Parameter | Control |
    |-----------|---------|
    | Dataset | {dataset_dd} |
    | Embedding dim | {embedding_dim_slider} |
    | Codebook size K | {num_embed_slider} |
    | Commitment β | {commit_slider} |
    | Learning rate | {lr_slider} |
    | Epochs | {epochs_slider} |
    | Batch size | {batch_size_slider} |
    """)
    return (
        batch_size_slider, commit_slider, dataset_dd, embedding_dim_slider,
        epochs_slider, lr_slider, num_embed_slider,
    )


# ── Training ──────────────────────────────────────────────────────────
@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="Train Model")
    train_btn
    return (train_btn,)


@app.cell
def _(
    Trainer, VQVAE, batch_size_slider, commit_slider, dataset_dd,
    embedding_dim_slider, epochs_slider, get_dataloader, lr_slider, mo,
    num_embed_slider, torch, train_btn,
):
    mo.stop(not train_btn.value, mo.md("*Click **Train Model** to start.*"))

    _in_ch = 1 if dataset_dd.value in ("mnist", "fashion_mnist") else 3
    _model = VQVAE(
        in_channels=_in_ch,
        latent_dim=int(embedding_dim_slider.value),
        image_size=32,
        num_embeddings=int(num_embed_slider.value),
        beta=commit_slider.value,
    )
    _trainer = Trainer(_model, lr=lr_slider.value)

    _train_loader = get_dataloader(
        dataset_dd.value, train=True, batch_size=int(batch_size_slider.value),
    )
    _val_loader = get_dataloader(
        dataset_dd.value, train=False, batch_size=int(batch_size_slider.value),
    )

    _history = _trainer.fit(_train_loader, epochs=int(epochs_slider.value), val_loader=_val_loader)

    model = _trainer.model
    trainer = _trainer
    history = _history
    train_loader = _train_loader
    val_loader = _val_loader

    mo.md(f"**Training complete** on `{_trainer.device}` — final train loss: {history['train_loss'][-1]:.4f}")
    return history, model, train_loader, trainer, val_loader


# ── Loss curves ───────────────────────────────────────────────────────
@app.cell
def _(history, mo, plot_loss_curves):
    mo.ui.plotly(plot_loss_curves(history, title="VQ-VAE — Training Loss"))
    return


# ── Reconstructions ───────────────────────────────────────────────────
@app.cell
def _(model, mo, plot_reconstructions, torch, val_loader):
    _x, _ = next(iter(val_loader))
    _x = _x.to(model.get_device())
    with torch.no_grad():
        _recon = model.reconstruct(_x)
    mo.ui.plotly(plot_reconstructions(_x, _recon, n=10, title="VQ-VAE — Reconstructions"))
    return


# ── Samples ──────────────────────────────────────────────────────────
@app.cell
def _(model, mo, plot_samples):
    _samples = model.sample(16, device=model.get_device())
    mo.ui.plotly(plot_samples(_samples, n=16, title="VQ-VAE — Samples (random codes)"))
    return


# ── Save checkpoint ──────────────────────────────────────────────────
@app.cell
def _(mo):
    save_btn = mo.ui.run_button(label="Save Checkpoint")
    save_btn
    return (save_btn,)


@app.cell
def _(mo, save_btn, trainer):
    mo.stop(not save_btn.value, mo.md("*Click to save the trained model.*"))
    _path = "checkpoints/vq_vae.pt"
    trainer.save_checkpoint(_path)
    mo.md(f"Checkpoint saved to `{_path}`")
    return


if __name__ == "__main__":
    app.run()
