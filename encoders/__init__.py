from .base import BaseEncoder

ENCODER_REGISTRY = {
    'cjepa':       ('encoders.cjepa',       'CJEPAEncoder'),
    'autoencoder': ('encoders.autoencoder', 'AutoencoderEncoder'),
    'dreamer':     ('encoders.dreamer',     'DreamerEncoder'),
    'lewm':        ('encoders.lewm',        'LEWMEncoder'),
}

def build_encoder(name: str, **kwargs) -> BaseEncoder:
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(ENCODER_REGISTRY)}")
    module_path, class_name = ENCODER_REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    cls    = getattr(module, class_name)
    return cls(**kwargs)