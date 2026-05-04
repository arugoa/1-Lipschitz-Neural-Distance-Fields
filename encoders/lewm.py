import sys
import numpy as np
import torch
from .base import BaseEncoder


class LEWMEncoder(BaseEncoder):
    def __init__(self, checkpoint_path: str, img_size: int = 224):
        sys.path.append('/home/arihant/projects/le-wm')
        import stable_worldmodel as swm
        from torchvision.transforms import v2 as transforms
        import stable_pretraining as spt

        self.model = swm.policy.AutoCostModel(checkpoint_path)
        self.model = self.model.eval().cuda()
        self.model.requires_grad_(False)
        self.model.interpolate_pos_encoding = True

        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ])

        # Probe — images must be (B, T, C, H, W)
        dummy = {'images': torch.zeros(1, 1, 3, img_size, img_size).cuda()}
        with torch.no_grad():
            raw = self.model.encode(dummy)  # returns dict
        self._output_dim = raw['emb'].shape[-1]  # 192
        print(f"LEWMEncoder output dim: {self._output_dim}")

    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, imgs_np: np.ndarray, device: str) -> np.ndarray:
        """imgs_np: (T, H, W, C) uint8 → (T, 192) float32"""
        frames = [self.transform(imgs_np[t]) for t in range(len(imgs_np))]
        imgs_t = torch.stack(frames).to(device)           # (T, C, H, W)
        imgs_t = imgs_t.unsqueeze(0)                      # (1, T, C, H, W)

        with torch.no_grad():
            out = self.model.encode({'images': imgs_t})   # returns dict
            enc = out['emb'].squeeze(0)                   # (T, 192)

        result = enc.cpu().float().numpy()
        del imgs_t, out, enc
        torch.cuda.empty_cache()
        return result