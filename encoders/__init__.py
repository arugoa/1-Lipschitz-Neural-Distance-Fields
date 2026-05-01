from .base import BaseEncoder
from .cjepa import CJEPAEncoder
from .autoencoder import AutoencoderEncoder
from .dreamer import DreamerEncoder

ENCODER_REGISTRY = {
    'cjepa':       CJEPAEncoder,
    'autoencoder': AutoencoderEncoder,
    'dreamer':     DreamerEncoder,
}

def build_encoder(name: str, **kwargs) -> BaseEncoder:
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[name](**kwargs)
