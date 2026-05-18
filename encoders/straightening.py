"""
Temporal Straightening encoder wrapper (DinoV2Encoder backend).

Encoder forward: (B, C, H, W) -> (B, num_patches, emb_dim) or (B, 1, emb_dim)
We mean-pool the patch dim to get (B, emb_dim), then process T frames -> (T, emb_dim).
"""

import numpy as np
import torch
from torchvision.transforms import v2 as transforms
from .base import BaseEncoder


class TSEncoder(BaseEncoder):
    def __init__(
        self,
        checkpoint_path: str,
        img_size: int = 224,
        img_mean: tuple = (0.485, 0.456, 0.406),
        img_std:  tuple = (0.229, 0.224, 0.225),
    ):
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.')))
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
        hub_path = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        if os.path.exists(hub_path) and hub_path not in sys.path:
            sys.path.insert(0, hub_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert "encoder" in ckpt, (
            f"'encoder' key not found. Available: {list(ckpt.keys())}"
        )

        self.encoder = ckpt["encoder"].eval().cuda()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=list(img_mean), std=list(img_std)),
            transforms.Resize(size=img_size),
        ])

        # Probe output dim
        dummy = torch.zeros(1, 3, img_size, img_size).cuda()
        with torch.no_grad():
            out = self.encoder(dummy)    # (1, num_patches, D) or (1, 1, D)
        # mean-pool patch dim → (1, D)
        self._output_dim = out.mean(dim=1).shape[-1]
        print(f"StraighteningEncoder: raw output {out.shape} → pooled dim {self._output_dim}")

    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, imgs_np: np.ndarray, device: str) -> np.ndarray:
        """
        imgs_np: (T, H, W, C) uint8
        returns: (T, emb_dim) float32 numpy
        """
        frames = [self.transform(imgs_np[t]) for t in range(len(imgs_np))]
        imgs_t = torch.stack(frames).to(device)   # (T, C, H, W)

        with torch.no_grad():
            out = self.encoder(imgs_t)             # (T, num_patches, D) or (T, 1, D)
            enc = out.mean(dim=1)                  # (T, D) — mean-pool patches

        result = enc.cpu().float().numpy()
        del imgs_t, out, enc
        torch.cuda.empty_cache()
        return result