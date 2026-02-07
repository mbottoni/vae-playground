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

"""WAE-MMD — Wasserstein Autoencoder with MMD penalty (Tolstikhin et al., 2018)."""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


# ── Theory ────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # WAE-MMD (Wasserstein Autoencoder)

    WAE replaces the ELBO's KL divergence with a **Maximum Mean Discrepancy (MMD)**
    penalty that matches the aggregated posterior $q(z) = \int q(z|x) p(x)\,dx$
    to the prior $p(z) = \mathcal{N}(0, I)$:

    $$\mathcal{L} = \mathbb{E}[\| x - \hat{x} \|^2] + \lambda \cdot \text{MMD}(q(z), p(z))$$

    Key differences from VAE:
    - Uses a **deterministic encoder** (no reparameterisation noise).
    - Uses **MSE reconstruction** (not BCE).
    - The MMD penalty uses an RBF kernel to compare distributions without
      requiring an explicit density.
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

    from vae_playground.models import WAE_MMD
    from vae_playground.training import Trainer
    from vae_playground.data import get_dataloader
    from vae_playground.utils import (
        plot_reconstructions,
        plot_latent_space,
        plot_loss_curves,
        plot_samples,
    )
    from vae_playground.utils.metrics import reconstruction_mse, compute_latent_stats
    return (
        Path, Trainer, WAE_MMD, compute_latent_stats, get_dataloader, np,
        plot_latent_space, plot_loss_curves, plot_reconstructions, plot_samples,
        reconstruction_mse, sys, torch,
    )


# ── Configuration ─────────────────────────────────────────────────────
@app.cell
def _(mo):
    dataset_dd = mo.ui.dropdown(
        options=["mnist", "fashion_mnist", "cifar10"],
        value="mnist",
        label="Dataset",
    )
    latent_dim_slider = mo.ui.slider(2, 128, value=16, step=2, label="Latent dim")
    reg_slider = mo.ui.slider(1.0, 500.0, value=100.0, step=10.0, label="MMD weight (λ)")
    lr_slider = mo.ui.slider(1e-4, 1e-2, value=1e-3, step=1e-4, label="Learning rate")
    epochs_slider = mo.ui.slider(1, 50, value=10, step=1, label="Epochs")
    batch_size_slider = mo.ui.slider(32, 512, value=128, step=32, label="Batch size")

    mo.md(f"""
    ## Configuration

    | Parameter | Control |
    |-----------|---------|
    | Dataset | {dataset_dd} |
    | Latent dim | {latent_dim_slider} |
    | **MMD weight (λ)** | {reg_slider} |
    | Learning rate | {lr_slider} |
    | Epochs | {epochs_slider} |
    | Batch size | {batch_size_slider} |
    """)
    return batch_size_slider, dataset_dd, epochs_slider, latent_dim_slider, lr_slider, reg_slider


# ── Training ──────────────────────────────────────────────────────────
@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="Train Model")
    train_btn
    return (train_btn,)


@app.cell
def _(
    Trainer, WAE_MMD, batch_size_slider, dataset_dd, epochs_slider,
    get_dataloader, latent_dim_slider, lr_slider, mo, reg_slider, torch, train_btn,
):
    mo.stop(not train_btn.value, mo.md("*Click **Train Model** to start.*"))

    _in_ch = 1 if dataset_dd.value in ("mnist", "fashion_mnist") else 3
    _model = WAE_MMD(
        in_channels=_in_ch,
        latent_dim=int(latent_dim_slider.value),
        image_size=32,
        reg_weight=reg_slider.value,
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
    mo.ui.plotly(plot_loss_curves(history, title="WAE-MMD — Training Loss"))
    return


# ── Reconstructions ───────────────────────────────────────────────────
@app.cell
def _(model, mo, plot_reconstructions, torch, val_loader):
    _x, _ = next(iter(val_loader))
    _x = _x.to(model.get_device())
    with torch.no_grad():
        _recon = model.reconstruct(_x)
    mo.ui.plotly(plot_reconstructions(_x, _recon, n=10, title="WAE-MMD — Reconstructions"))
    return


# ── Latent space ──────────────────────────────────────────────────────
@app.cell
def _(compute_latent_stats, model, mo, plot_latent_space, val_loader):
    _stats = compute_latent_stats(model, val_loader, device=model.get_device())
    mo.ui.plotly(plot_latent_space(_stats["z"], _stats["labels"], method="tsne",
                                    title="WAE-MMD — Latent Space (t-SNE)"))
    return


# ── Samples ──────────────────────────────────────────────────────────
@app.cell
def _(model, mo, plot_samples):
    _samples = model.sample(16, device=model.get_device())
    mo.ui.plotly(plot_samples(_samples, n=16, title="WAE-MMD — Samples"))
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
    _path = "checkpoints/wae_mmd.pt"
    trainer.save_checkpoint(_path)
    mo.md(f"Checkpoint saved to `{_path}`")
    return


if __name__ == "__main__":
    app.run()
