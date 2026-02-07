"""Default hyperparameters for each VAE variant.

Import and override in notebooks as needed.
"""

VANILLA_VAE = dict(
    in_channels=1,
    latent_dim=16,
    hidden_dims=[32, 64, 128],
    image_size=32,
)

BETA_VAE = dict(
    **VANILLA_VAE,
    beta=4.0,
)

CONDITIONAL_VAE = dict(
    in_channels=1,
    latent_dim=16,
    num_classes=10,
    hidden_dims=[32, 64, 128],
    image_size=32,
)

VQ_VAE = dict(
    in_channels=1,
    latent_dim=64,
    hidden_dims=[32, 64],
    image_size=32,
    num_embeddings=64,
    beta=0.25,
)

WAE_MMD = dict(
    **VANILLA_VAE,
    reg_weight=100.0,
    kernel_sigma=1.0,
)

TRAINING = dict(
    lr=1e-3,
    epochs=10,
    batch_size=128,
    dataset="mnist",
    image_size=32,
)
