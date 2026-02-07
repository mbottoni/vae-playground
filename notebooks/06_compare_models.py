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
        MODEL_REGISTRY,
        Path,
        compute_latent_stats,
        get_dataloader,
        plot_latent_space,
        plot_loss_curves,
        plot_reconstructions,
        plot_samples,
        reconstruction_mse,
        torch,
    )


@app.cell
def _(Path, mo):
    _root = Path(__file__).resolve().parent.parent
    _notebooks = _root / "notebooks"
    _ckpt_dir = _notebooks / "checkpoints"
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


@app.cell
def _(mo):
    dataset_dd = mo.ui.dropdown(
        options=["mnist", "fashion_mnist", "cifar10"],
        value="mnist",
        label="Test dataset",
    )
    dataset_dd
    return (dataset_dd,)


@app.cell
def _(mo):
    load_btn = mo.ui.run_button(label="Load & Compare")
    load_btn
    return (load_btn,)


@app.cell
def _(
    MODEL_REGISTRY,
    Path,
    ckpt_select,
    dataset_dd,
    get_dataloader,
    load_btn,
    mo,
    torch,
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


@app.cell
def _(compute_latent_stats, loaded_models, mo, plot_latent_space, test_loader):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _model_names = list(loaded_models.keys())
    _n_models = len(_model_names)
    _cols = min(3, _n_models)
    _rows = (_n_models + _cols - 1) // _cols

    _fig = make_subplots(
        rows=_rows, cols=_cols,
        subplot_titles=[f"{name} — Latent Space (t-SNE)" for name in _model_names],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    for _idx, (_name, _info) in enumerate(loaded_models.items()):
        _m = _info["model"]
        _row = _idx // _cols + 1
        _col = _idx % _cols + 1
    
        try:
            _stats = compute_latent_stats(_m, test_loader, device="cpu")
            _latent_fig = plot_latent_space(_stats["z"], _stats["labels"], method="tsne",
                                            title=f"{_name} — Latent Space (t-SNE)")
        
            # Add traces from the latent space plot to the subplot
            for _trace in _latent_fig.data:
                _fig.add_trace(_trace, row=_row, col=_col)
        except Exception:
            # Add a text annotation for models where latent space is not available
            _fig.add_annotation(
                text=f"Latent space not available",
                xref=f"x{_idx+1}" if _idx > 0 else "x",
                yref=f"y{_idx+1}" if _idx > 0 else "y",
                x=0.5, y=0.5,
                showarrow=False,
                row=_row, col=_col
            )

    _fig.update_layout(height=400 * _rows, showlegend=True)
    mo.ui.plotly(_fig)
    return (make_subplots,)


@app.cell
def _(loaded_models, make_subplots, mo, plot_loss_curves):
    _model_names = list(loaded_models.keys())
    _n_models = len(_model_names)
    _cols = min(3, _n_models)
    _rows = (_n_models + _cols - 1) // _cols

    _fig = make_subplots(
        rows=_rows, cols=_cols,
        subplot_titles=[f"{name} — Loss Curves" for name in _model_names],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    for _idx, (_name, _info) in enumerate(loaded_models.items()):
        _hist = _info.get("history", {})
        _row = _idx // _cols + 1
        _col = _idx % _cols + 1
    
        if _hist:
            _loss_fig = plot_loss_curves(_hist, title=f"{_name} — Loss Curves")
        
            # Add traces from the loss curves plot to the subplot
            for _trace in _loss_fig.data:
                _fig.add_trace(_trace, row=_row, col=_col)
        else:
            # Add a text annotation for models where history is not available
            _fig.add_annotation(
                text=f"No training history",
                xref=f"x{_idx+1}" if _idx > 0 else "x",
                yref=f"y{_idx+1}" if _idx > 0 else "y",
                x=0.5, y=0.5,
                showarrow=False,
                row=_row, col=_col
            )

    _fig.update_layout(height=400 * _rows, showlegend=True)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(loaded_models, make_subplots, mo, plot_samples, torch):
    _model_names = list(loaded_models.keys())
    _n_models = len(_model_names)
    _cols = min(3, _n_models)
    _rows = (_n_models + _cols - 1) // _cols

    _fig = make_subplots(
        rows=_rows, cols=_cols,
        subplot_titles=[f"{name} — Samples" for name in _model_names],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    for _idx, (_name, _info) in enumerate(loaded_models.items()):
        _m = _info["model"]
        _row = _idx // _cols + 1
        _col = _idx % _cols + 1
    
        try:
            with torch.no_grad():
                _s = _m.sample(16, device="cpu")
            _sample_fig = plot_samples(_s, n=16, title=f"{_name} — Samples")
        
            # Add traces from the samples plot to the subplot
            for _trace in _sample_fig.data:
                _fig.add_trace(_trace, row=_row, col=_col)
        except Exception:
            # Add a text annotation for models where sampling is not available
            _fig.add_annotation(
                text=f"Sampling not available",
                xref=f"x{_idx+1}" if _idx > 0 else "x",
                yref=f"y{_idx+1}" if _idx > 0 else "y",
                x=0.5, y=0.5,
                showarrow=False,
                row=_row, col=_col
            )

    _fig.update_layout(height=400 * _rows, showlegend=False)
    mo.ui.plotly(_fig)
    return


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


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
