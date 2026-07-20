"""
Temporal Straightening encoder wrapper (DinoV2Encoder backend).

Uses encoder.forward(x, return_agg=True), which runs the trained agg_mlp
head (the same aggregation the "aggcos" curvature loss shaped at train
time) to collapse the patch grid: (B, C, H, W) -> (B, agg_out_dim). Input
is resized to match the patch grid agg_mlp was actually trained on (see
self.encoder_image_size in __init__) before encoding.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from .base import BaseEncoder


class TSEncoder(BaseEncoder):
    def __init__(
        self,
        checkpoint_path: str,
        img_size: int = 224,
        img_mean: tuple = (0.485, 0.456, 0.406),
        img_std:  tuple = (0.229, 0.224, 0.225),
        return_agg: bool = True,
    ):
        # return_agg=True runs the trained aggregation head (agg_mlp for the
        # patch/channel encoder, global-token agg for projglobal). Set False to
        # take the raw encoder output instead.
        self.return_agg = return_agg
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

        # VWorldModel resizes visual input to a size whose patch grid matches
        # what encoder.agg_mlp was actually built (and trained, via the
        # "aggcos" curvature loss) for: (image_size // 16) * patch_size —
        # see models/visual_world_model.py's `self.encoder_transform`. Must
        # replicate that resize here, or agg_mlp's fixed input layer won't
        # match the patch count from a raw img_size input.
        decoder_scale = 16
        num_side_patches = img_size // decoder_scale
        self.encoder_image_size = num_side_patches * self.encoder.patch_size

        # Probe output dim (use the trained aggregation head, not a naive mean-pool)
        dummy = torch.zeros(1, 3, self.encoder_image_size, self.encoder_image_size).cuda()
        with torch.no_grad():
            out = self.encoder(dummy, return_agg=self.return_agg)    # (1, agg_out_dim)
        self._output_dim = out.shape[-1]
        print(f"StraighteningEncoder: aggregated output {out.shape} → dim {self._output_dim} "
              f"(encoder input resized to {self.encoder_image_size}x{self.encoder_image_size})")

    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, imgs_np: np.ndarray, device: str) -> np.ndarray:
        """
        imgs_np: (T, H, W, C) uint8  OR  (T, C, H, W) float tensor/array
        returns: (T, emb_dim) float32 numpy
        """
        # TS dataset returns (T, C, H, W) float tensors — detect and handle both
        if torch.is_tensor(imgs_np):
            imgs_t = imgs_np.float().to(device)   # already (T, C, H, W)
            if imgs_t.max() > 1.0:
                imgs_t = imgs_t / 255.0
        elif imgs_np.ndim == 4 and imgs_np.shape[1] in (1, 3):
            # (T, C, H, W) numpy — already channels-first
            imgs_t = torch.from_numpy(imgs_np).float().to(device)
            if imgs_t.max() > 1.0:
                imgs_t = imgs_t / 255.0
        else:
            # (T, H, W, C) uint8 numpy — apply full transform
            frames = [self.transform(imgs_np[t]) for t in range(len(imgs_np))]
            imgs_t = torch.stack(frames).to(device)

        # match encoder.agg_mlp's expected patch grid (see __init__)
        if imgs_t.shape[-1] != self.encoder_image_size:
            imgs_t = F.interpolate(
                imgs_t, size=(self.encoder_image_size, self.encoder_image_size),
                mode="bilinear", align_corners=False,
            )

        with torch.no_grad():
            out = self.encoder(imgs_t, return_agg=self.return_agg)
            enc = out

        # projglobal returns (T, 1, D) (singleton token dim); projchannel returns
        # (T, D). Collapse the singleton so both give (T, D).
        if enc.dim() == 3 and enc.shape[1] == 1:
            enc = enc[:, 0, :]

        result = enc.cpu().float().numpy()
        del imgs_t, out, enc
        torch.cuda.empty_cache()
        return result