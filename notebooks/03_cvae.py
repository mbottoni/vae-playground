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

"""Conditional VAE — class-conditioned generation (Sohn et al., 2015)."""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


# ── Theory ────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # Conditional VAE (CVAE)

    The Conditional VAE extends the standard VAE by conditioning both encoder
    and decoder on a label $y$:

    $$\mathcal{L} = \mathbb{E}_{q_\phi(z|x,y)}[\log p_\theta(x|z,y)]
      - D_\text{KL}(q_\phi(z|x,y) \| p(z))$$

    This allows **targeted generation**: you can specify which class to generate.
    The label is one-hot encoded and provided as:
    - An extra spatial channel to the encoder
    - A vector concatenated to $z$ for the decoder
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

    from vae_playground.models import ConditionalVAE
    from vae_playground.training import Trainer
    from vae_playground.data import get_dataloader
    from vae_playground.data.datasets import num_classes as ds_num_classes
    from vae_playground.utils import (
        plot_reconstructions,
        plot_latent_space,
        plot_loss_curves,
        plot_samples,
    )
    from vae_playground.utils.metrics import compute_latent_stats
    return (
        ConditionalVAE, Path, Trainer, compute_latent_stats, ds_num_classes,
        get_dataloader, np, plot_latent_space, plot_loss_curves,
        plot_reconstructions, plot_samples, sys, torch,
    )


# ── Configuration ─────────────────────────────────────────────────────
@app.cell
def _(mo):
    dataset_dd = mo.ui.dropdown(
        options=["mnist", "fashion_mnist"],
        value="mnist",
        label="Dataset",
    )
    latent_dim_slider = mo.ui.slider(2, 128, value=16, step=2, label="Latent dim")
    lr_slider = mo.ui.slider(1e-4, 1e-2, value=1e-3, step=1e-4, label="Learning rate")
    epochs_slider = mo.ui.slider(1, 50, value=10, step=1, label="Epochs")
    batch_size_slider = mo.ui.slider(32, 512, value=128, step=32, label="Batch size")

    mo.md(f"""
    ## Configuration

    | Parameter | Control |
    |-----------|---------|
    | Dataset | {dataset_dd} |
    | Latent dim | {latent_dim_slider} |
    | Learning rate | {lr_slider} |
    | Epochs | {epochs_slider} |
    | Batch size | {batch_size_slider} |
    """)
    return batch_size_slider, dataset_dd, epochs_slider, latent_dim_slider, lr_slider


# ── Training ──────────────────────────────────────────────────────────
@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="Train Model")
    train_btn
    return (train_btn,)


@app.cell
def _(
    ConditionalVAE, Trainer, batch_size_slider, dataset_dd, ds_num_classes,
    epochs_slider, get_dataloader, latent_dim_slider, lr_slider, mo, torch, train_btn,
):
    mo.stop(not train_btn.value, mo.md("*Click **Train Model** to start.*"))

    _in_ch = 1 if dataset_dd.value in ("mnist", "fashion_mnist") else 3
    _nc = ds_num_classes(dataset_dd.value)
    _model = ConditionalVAE(
        in_channels=_in_ch,
        latent_dim=int(latent_dim_slider.value),
        num_classes=_nc,
        image_size=32,
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
    mo.ui.plotly(plot_loss_curves(history, title="CVAE — Training Loss"))
    return


# ── Reconstructions ───────────────────────────────────────────────────
@app.cell
def _(model, mo, plot_reconstructions, torch, val_loader):
    _batch = next(iter(val_loader))
    _x, _y = _batch[0].to(model.get_device()), _batch[1].to(model.get_device())
    with torch.no_grad():
        _fwd = model(_x, labels=_y)
    mo.ui.plotly(plot_reconstructions(_x, _fwd["recon"], n=10, title="CVAE — Reconstructions"))
    return


# ── Latent space ──────────────────────────────────────────────────────
@app.cell
def _(compute_latent_stats, model, mo, plot_latent_space, val_loader):
    # CVAE encode needs labels — override compute_latent_stats inline
    import torch as _torch

    _all_z, _all_y = [], []
    model.eval()
    with _torch.no_grad():
        for i, (_x, _y) in enumerate(val_loader):
            if i >= 20:
                break
            _x = _x.to(model.get_device())
            _y_dev = _y.to(model.get_device())
            _mu, _ = model.encode(_x, labels=_y_dev)
            _all_z.append(_mu.cpu())
            _all_y.append(_y)

    _z_np = _torch.cat(_all_z).numpy()
    _y_np = _torch.cat(_all_y).numpy()
    mo.ui.plotly(plot_latent_space(_z_np, _y_np, method="tsne", title="CVAE — Latent Space (t-SNE)"))
    return


# ── Conditional generation ────────────────────────────────────────────
@app.cell
def _(mo):
    class_slider = mo.ui.slider(0, 9, value=3, step=1, label="Generate class")
    class_slider
    return (class_slider,)


@app.cell
def _(class_slider, model, mo, plot_samples, torch):
    _labels = torch.full((16,), int(class_slider.value), dtype=torch.long, device=model.get_device())
    _samples = model.sample(16, device=model.get_device(), labels=_labels)
    mo.ui.plotly(plot_samples(_samples, n=16, title=f"CVAE — Class {class_slider.value} Samples"))
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
    _path = "checkpoints/cvae.pt"
    trainer.save_checkpoint(_path)
    mo.md(f"Checkpoint saved to `{_path}`")
    return


if __name__ == "__main__":
    app.run()
