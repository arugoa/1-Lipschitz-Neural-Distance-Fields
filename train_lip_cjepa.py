import torch
import torch.nn as nn
import numpy as np
import glob
import os
import sys
import argparse
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.models import *
from common.visualize import point_cloud_from_arrays
from common.training import Trainer
from common.utils import get_device
from common.callback import *

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# ── SAVi encoder ───────────────────────────────────────────────────────────
sys.path.append('src/third_party/slotformer')
from base_slots.models.savi import StoSAVi

# ── Your SDF utilities (unchanged) ────────────────────────────────────────
# These are assumed to exist in your codebase:
#   get_device, select_model, count_parameters,
#   point_cloud_from_arrays, LoggerCB, CheckpointCB,
#   UpdateHkrRegulCB, Trainer, save_model


# ── Args ───────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()

    # dataset / output
    parser.add_argument("dataset", type=str, default="../dataset-good",
                        help="path to dataset folder")
    parser.add_argument("-o", "--output-name", type=str, default="output")
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("-p", "--pca-dim", type=int, default=2)

    # model
    parser.add_argument("-model", "--model", choices=["ortho", "sll"],
                        default="sll")
    parser.add_argument("-n-layers", "--n-layers", type=int, default=20)
    parser.add_argument("-n-hidden", "--n-hidden", type=int, default=128)

    # optimization
    parser.add_argument("-ne", "--epochs", type=int, default=200)
    parser.add_argument('-bs', "--batch-size", type=int, default=200)
    parser.add_argument("-tbs", "--test-batch-size", type=int, default=5000)
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-4)
    parser.add_argument("-lm", "--loss-margin", type=float, default=1e-2)
    parser.add_argument("-lmbd", "--loss-lambda", type=float, default=100.)

    # misc
    parser.add_argument("-cp", "--checkpoint-freq", type=int, default=10)
    parser.add_argument("-cpu", action="store_true")

    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────

def pad(arr, target_len=150):
    """Pad or truncate (T, ...) array to target_len along axis 0."""
    T = arr.shape[0]
    if T >= target_len:
        return arr[:target_len]
    pad_block = np.repeat(arr[-1:], target_len - T, axis=0)
    return np.concatenate([arr, pad_block], axis=0)


def imgs_to_tensor(imgs_np, device):
    """
    imgs_np: (T, H, W, C) uint8 numpy
    returns: (1, T, C, H, W) float32 cuda tensor in [0, 1]
    """
    t = torch.from_numpy(imgs_np).float().to(device) / 255.0  # (T, H, W, C)
    t = t.permute(0, 3, 1, 2)                                  # (T, C, H, W)
    t = t.unsqueeze(0)                                          # (1, T, C, H, W)
    return t


def encode(savi_model, imgs_np, device):
    """Run SAVi encoder on a single episode. Returns (T, 896) numpy array."""
    imgs_t = imgs_to_tensor(imgs_np, device)
    with torch.no_grad():
        out   = savi_model({'img': imgs_t})
        slots = out['post_slots'].squeeze(0)      # (T, 7, 128)
        slots = slots.reshape(slots.shape[0], -1) # (T, 896)
    enc_np = slots.detach().cpu().float().numpy()
    del imgs_t, slots
    torch.cuda.empty_cache()
    return enc_np


class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, *paths, device="cpu"):
        self.arrays = [np.load(p, mmap_mode="r") for p in paths]
        self.device = device

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, idx):
        return tuple(
            torch.from_numpy(np.array(a[idx])).to(self.device)
            for a in self.arrays
        )

# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    args = get_args()

    device = get_device(args.cpu)

    # ── 1. Load SAVi model ─────────────────────────────────────────────────
    savi_model = StoSAVi(
        resolution=(128, 128),
        clip_len=6,
        slot_dict=dict(
            num_slots=7,
            slot_size=128,
            slot_mlp_size=256,
            num_iterations=2,
            kernel_mlp=True,
        ),
        enc_dict=dict(
            enc_channels=(3, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=128,
            enc_norm='',
        ),
        dec_dict=dict(
            dec_channels=(128, 64, 64, 64, 64),
            dec_resolution=(8, 8),
            dec_ks=5,
            dec_norm='',
        ),
        pred_dict=dict(
            pred_type='transformer',
            pred_rnn=True,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=512,
            pred_sg_every=None,
        ),
        loss_dict=dict(
            use_post_recon_loss=True,
            kld_method='var-0.01',
        ),
    )

    savi_ckpt  = torch.load('clevrer_savi_model.pth', map_location='cpu', weights_only=False)
    state_dict = savi_ckpt.get('state_dict', savi_ckpt)
    savi_model.load_state_dict(state_dict, strict=False)
    savi_model = savi_model.eval().cuda()
    savi_model.testing = True

    for p in savi_model.parameters():
        p.requires_grad = False

    # ── 2. Config ──────────────────────────────────────────────────────────
    NUM_SLOTS = 7
    SLOT_DIM  = 128
    ENC_DIM   = NUM_SLOTS * SLOT_DIM  # 896

    config = SimpleNamespace(
        signed         = not args.unsigned,
        device         = device,
        n_epochs       = args.epochs,
        checkpoint_freq= args.checkpoint_freq,
        batch_size     = args.batch_size,
        test_batch_size= args.test_batch_size,
        loss_margin    = args.loss_margin,
        loss_regul     = args.loss_lambda,
        optimizer      = "adam",
        learning_rate  = args.learning_rate,
        output_folder  = os.path.join(
            "output",
            args.output_name if args.output_name else args.dataset
        ),
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    # ── 3. Scan dataset ────────────────────────────────────────────────────
    files     = sorted(glob.glob(os.path.join(args.dataset, "*.npz")))
    num_train = int(len(files) * 0.7)
    print(f"Found {len(files)} episodes ({num_train} train).")

    # Count samples for memmap pre-allocation
    print("Counting samples...")
    n_in = n_out = n_test = 0
    for i, fp in enumerate(files):
        d = np.load(fp, allow_pickle=True)["dones"]
        d = np.where(d == 0, 1, -1)
        d = pad(d)
        if i < num_train:
            n_in  += int((d ==  1).sum())
            n_out += int((d != 1).sum())
        else:
            n_test += 150
    print(f"n_in={n_in}, n_out={n_out}, n_test={n_test}")

    # ── 4. Fit scaler + PCA (streaming) ───────────────────────────────────
    pca_dim = args.pca_dim
    ipca    = IncrementalPCA(n_components=pca_dim, batch_size=1024)
    scaler  = StandardScaler()

    print("Fitting scaler + PCA...")
    for i, fp in enumerate(files[:num_train]):
        if i % 100 == 0:
            print(f"  PCA fit episode {i}/{num_train}")
        imgs_np = pad(np.load(fp, allow_pickle=True)["images"])  # (150, H, W, C)
        enc_np  = encode(savi_model, imgs_np, device)             # (150, 896)
        scaler.partial_fit(enc_np)
        ipca.partial_fit(scaler.transform(enc_np))

    print(f"Explained variance: {ipca.explained_variance_ratio_.sum():.4f}")
    DIM = pca_dim

    # ── 5. Allocate memmaps ────────────────────────────────────────────────
    mm_path_in   = os.path.join(config.output_folder, "X_train_in.npy")
    mm_path_out  = os.path.join(config.output_folder, "X_train_out.npy")
    mm_path_test = os.path.join(config.output_folder, "X_test.npy")
    mm_path_yt   = os.path.join(config.output_folder, "y_test.npy")

    mm_in   = np.lib.format.open_memmap(mm_path_in,   mode="w+", dtype="float32", shape=(n_in,   DIM))
    mm_out  = np.lib.format.open_memmap(mm_path_out,  mode="w+", dtype="float32", shape=(n_out,  DIM))
    mm_test = np.lib.format.open_memmap(mm_path_test, mode="w+", dtype="float32", shape=(n_test, DIM))
    mm_yt   = np.lib.format.open_memmap(mm_path_yt,   mode="w+", dtype="float32", shape=(n_test,))

    # ── 6. Encode → scale → PCA → write ───────────────────────────────────
    print("Encoding and writing memmaps...")
    idx_in = idx_out = idx_test = 0

    for i, fp in enumerate(files):
        if i % 100 == 0:
            print(f"  Episode {i}/{len(files)}")

        file    = np.load(fp, allow_pickle=True)
        imgs_np = pad(file["images"])          # (150, H, W, C)
        d       = np.where(file["dones"] == 0, 1, -1)
        d       = pad(d)                       # (150,)

        enc_np = encode(savi_model, imgs_np, device)  # (150, 896)
        enc_np = ipca.transform(scaler.transform(enc_np))  # (150, pca_dim)

        if i < num_train:
            mask_in  = (d == 1)
            mask_out = ~mask_in
            n_i = mask_in.sum();  n_o = mask_out.sum()
            mm_in [idx_in :idx_in  + n_i] = enc_np[mask_in]
            mm_out[idx_out:idx_out + n_o] = enc_np[mask_out]
            idx_in  += n_i
            idx_out += n_o
        else:
            chunk = len(d)
            mm_test[idx_test:idx_test + chunk] = enc_np
            mm_yt  [idx_test:idx_test + chunk] = d.astype("float32")
            idx_test += chunk

    del mm_in, mm_out, mm_test, mm_yt
    print(f"Done → in: {idx_in}, out: {idx_out}, test: {idx_test}")

    # ── 7. DataLoaders ────────────────────────────────────────────────────
    loader_in   = DataLoader(MemmapDataset(mm_path_in,  device=device),
                             batch_size=config.batch_size, shuffle=True)
    loader_out  = DataLoader(MemmapDataset(mm_path_out, device=device),
                             batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(MemmapDataset(mm_path_test, mm_path_yt, device=device),
                             batch_size=config.test_batch_size)

    # ── 8. SDF model ──────────────────────────────────────────────────────
    model = select_model(args.model, DIM, args.n_layers, args.n_hidden).to(device)
    print("PARAMETERS:", count_parameters(model))

    # ── 9. Point cloud export ─────────────────────────────────────────────
    if config.signed:
        X_pc_in  = torch.from_numpy(np.load(mm_path_in))
        X_pc_out = torch.from_numpy(np.load(mm_path_out))
        pc = point_cloud_from_arrays((X_pc_in, -1.), (X_pc_out, 1.))
        del X_pc_in, X_pc_out
    else:
        X_pc_out = torch.from_numpy(np.load(mm_path_out))
        pc = point_cloud_from_arrays((X_pc_out, 1.))
        del X_pc_out

    torch.save(pc, os.path.join(config.output_folder, "pc_0.pt"))
    del pc

    # ── 10. Training ──────────────────────────────────────────────────────
    callbacks = [LoggerCB(os.path.join(config.output_folder, "log.csv"))]
    if config.checkpoint_freq > 0:
        callbacks.append(CheckpointCB(
            [x for x in range(0, config.n_epochs, config.checkpoint_freq) if x > 0]
        ))
    callbacks.append(UpdateHkrRegulCB(
        {1: 1., 5: 10., 10: 100., 30: config.loss_regul}
    ))

    if config.signed:
        trainer = Trainer((loader_in, loader_out), test_loader, config)
    else:
        trainer = Trainer((loader_out,), test_loader, config)

    trainer.add_callbacks(*callbacks)

    if config.signed:
        trainer.train_lip(model)
    else:
        trainer.train_lip_unsigned(model)

    save_model(model, os.path.join("output", f"model_hkr_loss_{pca_dim}.pt"))