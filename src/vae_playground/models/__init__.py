from vae_playground.models.base import BaseVAE
from vae_playground.models.vanilla_vae import VanillaVAE
from vae_playground.models.beta_vae import BetaVAE
from vae_playground.models.cvae import ConditionalVAE
from vae_playground.models.vq_vae import VQVAE
from vae_playground.models.wae import WAE_MMD

MODEL_REGISTRY: dict[str, type[BaseVAE]] = {
    "VanillaVAE": VanillaVAE,
    "BetaVAE": BetaVAE,
    "ConditionalVAE": ConditionalVAE,
    "VQVAE": VQVAE,
    "WAE_MMD": WAE_MMD,
}

__all__ = [
    "BaseVAE",
    "VanillaVAE",
    "BetaVAE",
    "ConditionalVAE",
    "VQVAE",
    "WAE_MMD",
    "MODEL_REGISTRY",
]
