"""SAVi (C-JEPA) encoder."""

import sys
import numpy as np
import torch

from .base import BaseEncoder

_SAVI_CONFIG = dict(
    resolution=(128, 128),
    clip_len=6,
    slot_dict=dict(num_slots=7, slot_size=128, slot_mlp_size=256,
                   num_iterations=2, kernel_mlp=True),
    enc_dict=dict(enc_channels=(3, 64, 64, 64, 64), enc_ks=5,
                  enc_out_channels=128, enc_norm=''),
    dec_dict=dict(dec_channels=(128, 64, 64, 64, 64), dec_resolution=(8, 8),
                  dec_ks=5, dec_norm=''),
    pred_dict=dict(pred_type='transformer', pred_rnn=True, pred_norm_first=True,
                   pred_num_layers=2, pred_num_heads=4, pred_ffn_dim=512,
                   pred_sg_every=None),
    loss_dict=dict(use_post_recon_loss=True, kld_method='var-0.01'),
)

NUM_SLOTS = 7
SLOT_DIM  = 128


class CJEPAEncoder(BaseEncoder):
    def __init__(self, checkpoint_path: str = 'clevrer_savi_model.pth'):
        sys.path.append('src/third_party/slotformer')
        from base_slots.models.savi import StoSAVi

        model = StoSAVi(**_SAVI_CONFIG)
        ckpt  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        model.testing = True  # skip decoding
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model.cuda()

    def output_dim(self) -> int:
        return NUM_SLOTS * SLOT_DIM  # 896

    def encode(self, imgs_np: np.ndarray, device: str) -> np.ndarray:
        """imgs_np: (T, H, W, C) uint8"""
        t = torch.from_numpy(imgs_np).float().to(device) / 255.0  # (T, H, W, C)
        t = t.permute(0, 3, 1, 2).unsqueeze(0)                    # (1, T, C, H, W)
        with torch.no_grad():
            out   = self.model({'img': t})
            slots = out['post_slots'].squeeze(0)       # (T, 7, 128)
            slots = slots.reshape(slots.shape[0], -1)  # (T, 896)
        enc = slots.cpu().float().numpy()
        del t, slots
        torch.cuda.empty_cache()
        return enc
