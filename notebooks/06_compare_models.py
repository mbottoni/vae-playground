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

"""Compare VAE variants side-by-side.

Load saved checkpoints and compare reconstruction quality, latent spaces,
loss curves, and sample quality across different VAE families.
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")


# ── Intro ─────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo

    mo.md("""
    # VAE Model Comparison

    Select two or more trained models (from saved checkpoints) to compare them
    on the same test set. The comparison includes:

    1. **Reconstruction quality** — side-by-side image grids
    2. **Latent space structure** — t-SNE / PCA projections
    3. **Loss curves** — from training history
    4. **Sample quality** — images sampled from the prior
    5. **Quantitative metrics** — reconstruction MSE
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

    from vae_playground.models import MODEL_REGISTRY
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
        MODEL_REGISTRY, Path, Trainer, compute_latent_stats, get_dataloader, np,
        plot_latent_space, plot_loss_curves, plot_reconstructions, plot_samples,
        reconstruction_mse, sys, torch,
    )


# ── Discover checkpoints ─────────────────────────────────────────────
@app.cell
def _(Path, mo):
    _ckpt_dir = Path("__file__").resolve().parent.parent / "checkpoints"
    _ckpt_files = sorted(_ckpt_dir.glob("*.pt")) if _ckpt_dir.exists() else []
    _options = {p.stem: str(p) for p in _ckpt_files}

    if not _options:
        mo.md("""
        **No checkpoints found.** Train models in the individual notebooks first
        and click "Save Checkpoint" to persist them, then come back here.
        """).callout(kind="warn")

    ckpt_select = mo.ui.multiselect(
        options=_options,
        label="Select models to compare",
    )
    ckpt_select
    return (ckpt_select,)


# ── Dataset selector ──────────────────────────────────────────────────
@app.cell
def _(mo):
    dataset_dd = mo.ui.dropdown(
        options=["mnist", "fashion_mnist", "cifar10"],
        value="mnist",
        label="Test dataset",
    )
    dataset_dd
    return (dataset_dd,)


# ── Load models ───────────────────────────────────────────────────────
@app.cell
def _(mo):
    load_btn = mo.ui.run_button(label="Load & Compare")
    load_btn
    return (load_btn,)


@app.cell
def _(
    MODEL_REGISTRY, Trainer, ckpt_select, dataset_dd, get_dataloader,
    load_btn, mo, torch,
):
    mo.stop(not load_btn.value, mo.md("*Select checkpoints and click **Load & Compare**.*"))
    mo.stop(len(ckpt_select.value) < 2, mo.md("*Please select at least **2** models to compare.*"))

    _test_loader = get_dataloader(dataset_dd.value, train=False, batch_size=128)

    _loaded: dict = {}
    for _path in ckpt_select.value:
        _ckpt = torch.load(_path, map_location="cpu", weights_only=False)
        _cls_name = _ckpt["model_class"]
        _cfg = _ckpt["model_config"]
        _model_cls = MODEL_REGISTRY[_cls_name]
        _model = _model_cls(**_cfg)
        _model.load_state_dict(_ckpt["model_state_dict"])
        _model.eval()

        _name = Path(_path).stem
        _loaded[_name] = {
            "model": _model,
            "history": _ckpt.get("history", {}),
        }

    loaded_models = _loaded
    test_loader = _test_loader

    _names = ", ".join(f"**{n}**" for n in loaded_models)
    mo.md(f"Loaded {len(loaded_models)} models: {_names}")
    return loaded_models, test_loader


# ── Reconstruction comparison ─────────────────────────────────────────
@app.cell
def _(loaded_models, mo, plot_reconstructions, test_loader, torch):
    _x, _ = next(iter(test_loader))
    _tabs = {}
    for _name, _info in loaded_models.items():
        _m = _info["model"]
        with torch.no_grad():
            _recon = _m.reconstruct(_x)
        _fig = plot_reconstructions(_x, _recon, n=10, title=f"{_name} — Reconstructions")
        _tabs[_name] = mo.ui.plotly(_fig)
    mo.ui.tabs(_tabs)
    return


# ── Latent space comparison ──────────────────────────────────────────
@app.cell
def _(compute_latent_stats, loaded_models, mo, plot_latent_space, test_loader, torch):
    _tabs = {}
    for _name, _info in loaded_models.items():
        _m = _info["model"]
        try:
            _stats = compute_latent_stats(_m, test_loader, device="cpu")
            _fig = plot_latent_space(_stats["z"], _stats["labels"], method="tsne",
                                      title=f"{_name} — Latent Space (t-SNE)")
            _tabs[_name] = mo.ui.plotly(_fig)
        except Exception:
            _tabs[_name] = mo.md(f"*Latent space not available for {_name}*")
    mo.ui.tabs(_tabs)
    return


# ── Loss curves comparison ───────────────────────────────────────────
@app.cell
def _(loaded_models, mo, plot_loss_curves):
    _tabs = {}
    for _name, _info in loaded_models.items():
        _hist = _info.get("history", {})
        if _hist:
            _fig = plot_loss_curves(_hist, title=f"{_name} — Loss Curves")
            _tabs[_name] = mo.ui.plotly(_fig)
        else:
            _tabs[_name] = mo.md(f"*No training history for {_name}*")
    mo.ui.tabs(_tabs)
    return


# ── Samples comparison ───────────────────────────────────────────────
@app.cell
def _(loaded_models, mo, plot_samples, torch):
    _tabs = {}
    for _name, _info in loaded_models.items():
        _m = _info["model"]
        try:
            with torch.no_grad():
                _s = _m.sample(16, device="cpu")
            _fig = plot_samples(_s, n=16, title=f"{_name} — Samples")
            _tabs[_name] = mo.ui.plotly(_fig)
        except Exception:
            _tabs[_name] = mo.md(f"*Sampling not available for {_name}*")
    mo.ui.tabs(_tabs)
    return


# ── Quantitative comparison ──────────────────────────────────────────
@app.cell
def _(loaded_models, mo, reconstruction_mse, test_loader):
    _rows = []
    for _name, _info in loaded_models.items():
        _m = _info["model"]
        try:
            _mse = reconstruction_mse(_m, test_loader, device="cpu", max_batches=10)
            _rows.append({"Model": _name, "Recon MSE": f"{_mse:.6f}"})
        except Exception as e:
            _rows.append({"Model": _name, "Recon MSE": f"error: {e}"})

    mo.md("## Quantitative Comparison")
    mo.ui.table(_rows) if _rows else mo.md("*No metrics computed.*")
    return


if __name__ == "__main__":
    app.run()
