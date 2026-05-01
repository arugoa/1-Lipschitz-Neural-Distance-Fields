"""Custom autoencoder encoder (original train_lip.py)."""

import numpy as np
import torch

from .base import BaseEncoder


class AutoencoderEncoder(BaseEncoder):
    def __init__(self, checkpoint_path: str):
        import sys
        sys.path.insert(0, '..')
        from training.train_autoencoder import load_autoencoder
        self.model = load_autoencoder(checkpoint_path)
        self.model.eval()

    def output_dim(self) -> int:
        return 5 * 128  # 640 — 5 slots × 128 dims

    def encode(self, imgs_np: np.ndarray, device: str) -> np.ndarray:
        """imgs_np: (T, H, W, C) uint8"""
        imgs_t = torch.from_numpy(imgs_np[None]).float().to(device)
        # actions not available at test time — pass zeros
        acts_t = torch.zeros(1, imgs_np.shape[0], 1).to(device)
        with torch.no_grad():
            enc = self.model.encode(imgs_t, acts_t)
        enc_np = enc[0].cpu().float().numpy().reshape(imgs_np.shape[0], 640)
        del imgs_t, acts_t, enc
        torch.cuda.empty_cache()
        return enc_np
