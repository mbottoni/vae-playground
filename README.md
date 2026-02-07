# VAE Playground

A modular playground for experimenting with different **Variational Autoencoder** families using PyTorch and interactive [marimo](https://marimo.io/) notebooks.

## VAE Variants

| Model | Key Idea | Reference |
|-------|----------|-----------|
| **Vanilla VAE** | Standard ELBO (recon + KL) | Kingma & Welling, 2014 |
| **Beta-VAE** | Disentangled latents via β > 1 on KL | Higgins et al., 2017 |
| **Conditional VAE** | Class-conditioned generation | Sohn et al., 2015 |
| **VQ-VAE** | Discrete latent codes with codebook | van den Oord et al., 2017 |
| **WAE-MMD** | Wasserstein distance + MMD penalty | Tolstikhin et al., 2018 |

## Project Structure

```
vae-playground/
├── src/vae_playground/
│   ├── models/          # BaseVAE + all 5 variants
│   ├── training/        # Trainer, losses, callbacks
│   ├── data/            # Dataset loaders (MNIST, FashionMNIST, CIFAR-10)
│   └── utils/           # Visualization (plotly) + metrics
├── notebooks/           # Marimo interactive notebooks
│   ├── 01-05            # One per VAE variant
│   └── 06_compare       # Side-by-side comparison
├── configs/             # Default hyperparameters
└── checkpoints/         # Saved model weights (gitignored)
```

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+. Tested on Apple Silicon (M1/M2) with MPS acceleration.

```bash
# Install dependencies
uv sync

# Launch a notebook (e.g. vanilla VAE)
uv run marimo edit notebooks/01_vanilla_vae.py

# Compare trained models
uv run marimo edit notebooks/06_compare_models.py
```

## Notebooks

Each notebook has:

1. **Theory** — brief explanation of the variant with LaTeX math
2. **Configuration** — interactive sliders/dropdowns for hyperparameters
3. **Training** — run button to train with live progress
4. **Reconstruction** — original vs reconstructed images
5. **Latent Space** — 2D t-SNE/PCA visualization colored by class
6. **Sampling** — generate new images from the prior
7. **Save** — persist model for the comparison notebook

The **comparison notebook** (`06_compare_models.py`) loads saved checkpoints and shows tabbed views of reconstructions, latent spaces, loss curves, samples, and a quantitative MSE table.

## Datasets

All datasets are lightweight and downloaded automatically via torchvision:

- **MNIST** — handwritten digits (1×28×28, resized to 32×32)
- **Fashion-MNIST** — clothing items (same dimensions)
- **CIFAR-10** — tiny color images (3×32×32)

## Hardware

The trainer auto-detects the best device:
- **Apple Silicon**: uses MPS (Metal Performance Shaders)
- **NVIDIA GPU**: uses CUDA
- **Fallback**: CPU
