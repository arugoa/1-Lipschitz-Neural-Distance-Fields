"""DreamerV3 MultiEncoder wrapper."""

import sys
import os
import numpy as np
import torch

from .base import BaseEncoder


class DreamerEncoder(BaseEncoder):
    def __init__(self, checkpoint_path: str, config):
        """
        config: the dreamer config namespace (from get_args in dreamer scripts),
                must have .encoder, .device, .task, etc.
        """
        dreamer_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../dreamerv3-torch')
        )
        sys.path.append(dreamer_dir)
        import networks
        import gymnasium

        env    = gymnasium.make(config.task, params=[config])
        shapes = {k: tuple(v.shape) for k, v in env.observation_space_full.spaces.items()}
        enc    = networks.MultiEncoder(shapes, **config.encoder)
        enc.to(config.device)

        ckpt       = torch.load(checkpoint_path, weights_only=False)
        state_dict = {
            k.split("_wm.encoder", 1)[1]: v
            for k, v in ckpt['agent_state_dict'].items()
            if '_wm.encoder' in k
        }
        enc.load_state_dict(state_dict, strict=False)
        enc.eval()
        self.model  = enc
        self._dim   = 8192

    def output_dim(self) -> int:
        return self._dim

    def encode(self, imgs_np: np.ndarray, device: str) -> np.ndarray:
        """imgs_np: (T, H, W, C) uint8"""
        imgs_t = torch.from_numpy(imgs_np[None]).float().to(device) / 255.0
        acts_t = torch.zeros(1, imgs_np.shape[0], 1).to(device)
        obs    = {"image": imgs_t, "actions": acts_t}
        with torch.no_grad():
            encoded = self.model(obs)
        enc_np = encoded[0].detach().cpu().float().numpy()
        del imgs_t, acts_t, encoded
        torch.cuda.empty_cache()
        return enc_np
