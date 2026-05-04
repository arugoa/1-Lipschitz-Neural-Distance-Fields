"""
Unified SDF training script.

Usage examples:
    python train_lip.py ../dataset-good/ --encoder cjepa --pca-dim 3
    python train_lip.py ../dataset-good/ --encoder dreamer --pca-dim 5 --dreamer-ckpt path/to/ckpt
    python train_lip.py ../dataset-good/ --encoder autoencoder --autoencoder-ckpt path/to/ckpt
"""

import os
import sys
import glob
import argparse
import pickle
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from encoders import build_encoder
from common.models import *
from common.visualize import point_cloud_from_arrays
from common.training import Trainer
from common.utils import get_device
from common.callback import *


# ── Args ───────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description="Unified SDF training")

    # dataset
    parser.add_argument("dataset", type=str, help="Path to dataset folder")
    parser.add_argument("-o", "--output-name", type=str, default="output")
    parser.add_argument("--unsigned", action="store_true")

    # encoder selection
    parser.add_argument("--encoder", choices=["cjepa", "dreamer", "autoencoder", "lewm"],
                        default="cjepa", help="Which encoder to use")
    parser.add_argument("--cjepa-ckpt", type=str, default="clevrer_savi_model.pth")
    parser.add_argument("--dreamer-ckpt", type=str, default=None)
    parser.add_argument("--autoencoder-ckpt", type=str, default=None)
    parser.add_argument("--lewm-ckpt", type=str, default=None)

    # PCA — can pass multiple dims to train one model per dim
    parser.add_argument("-p", "--pca-dims", type=int, nargs="+", default=[3],
                        help="PCA dimension(s). Pass multiple to train several models.")

    # SDF model
    parser.add_argument("-model", "--model", choices=["ortho", "sll"], default="sll")
    parser.add_argument("-n-layers", "--n-layers", type=int, default=20)
    parser.add_argument("-n-hidden", "--n-hidden", type=int, default=128)

    # optimization
    parser.add_argument("-ne", "--epochs", type=int, default=200)
    parser.add_argument("-bs", "--batch-size", type=int, default=200)
    parser.add_argument("-tbs", "--test-batch-size", type=int, default=5000)
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-4)
    parser.add_argument("-lm", "--loss-margin", type=float, default=1e-2)
    parser.add_argument("-lmbd", "--loss-lambda", type=float, default=100.)
    parser.add_argument("-cp", "--checkpoint-freq", type=int, default=10)
    parser.add_argument("-cpu", action="store_true")

    return parser.parse_args()


class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, *paths, device="cpu"):
        self.arrays = [np.load(p, mmap_mode="r") for p in paths]
        self.device = device

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, idx):
        return tuple(torch.from_numpy(np.array(a[idx])).to(self.device)
                     for a in self.arrays)


def run_pca_dim(args, encoder, files, num_train, device, pca_dim, config):
    """Run the full pipeline for one PCA dimension."""

    run_name    = f"{args.encoder}_pca{pca_dim}"
    out_folder  = os.path.join("output", args.output_name, run_name)
    os.makedirs(out_folder, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Encoder: {args.encoder}   PCA dim: {pca_dim}")
    print(f"  Output:  {out_folder}")
    print(f"{'='*60}")

    # ── 1. Count samples ──────────────────────────────────────────────────
    n_in = n_out = n_test = 0
    for i, fp in enumerate(files):
        d = np.where(np.load(fp, allow_pickle=True)["dones"] == 0, 1, -1)
        if i < num_train:
            n_in  += int((d ==  1).sum())
            n_out += int((d != 1).sum())
        else:
            n_test += 150
    print(f"Samples → in: {n_in}, out: {n_out}, test: {n_test}")

    # ── 2. Fit scaler + PCA ───────────────────────────────────────────────
    ipca   = IncrementalPCA(n_components=pca_dim, batch_size=1024)
    scaler = StandardScaler()

    print("Fitting scaler + PCA...")
    for i, fp in enumerate(files[:num_train]):
        if i % 100 == 0:
            print(f"  PCA fit {i}/{num_train}")
        imgs_np = np.load(fp, allow_pickle=True)["images"]
        enc_np  = encoder.encode(imgs_np, device)
        scaler.partial_fit(enc_np)
        ipca.partial_fit(scaler.transform(enc_np))

    print(f"Explained variance: {ipca.explained_variance_ratio_.sum():.4f}")

    # Save PCA pipeline for use in test_sdf.py
    pca_path = os.path.join(out_folder, "pca_pipeline.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump({"scaler": scaler, "ipca": ipca}, f)
    print(f"PCA pipeline saved to {pca_path}")

    # ── 3. Allocate memmaps ───────────────────────────────────────────────
    mm = {k: os.path.join(out_folder, f"{k}.npy") for k in
          ["X_train_in", "X_train_out", "X_test", "y_test"]}

    mm_in   = np.lib.format.open_memmap(mm["X_train_in"],  mode="w+", dtype="float32", shape=(n_in,   pca_dim))
    mm_out  = np.lib.format.open_memmap(mm["X_train_out"], mode="w+", dtype="float32", shape=(n_out,  pca_dim))
    mm_test = np.lib.format.open_memmap(mm["X_test"],      mode="w+", dtype="float32", shape=(n_test, pca_dim))
    mm_yt   = np.lib.format.open_memmap(mm["y_test"],      mode="w+", dtype="float32", shape=(n_test,))

    # ── 4. Encode → PCA → write ───────────────────────────────────────────
    print("Encoding + writing memmaps...")
    idx_in = idx_out = idx_test = 0

    for i, fp in enumerate(files):
        if i % 200 == 0:
            print(f"  Episode {i}/{len(files)}")
        file    = np.load(fp, allow_pickle=True)
        imgs_np = file["images"]
        d       = np.where(file["dones"] == 0, 1, -1)
        enc_np  = ipca.transform(scaler.transform(encoder.encode(imgs_np, device)))

        if i < num_train:
            mask_in  = (d == 1);  mask_out = ~mask_in
            n_i = mask_in.sum();  n_o = mask_out.sum()
            mm_in [idx_in :idx_in  + n_i] = enc_np[mask_in]
            mm_out[idx_out:idx_out + n_o] = enc_np[mask_out]
            idx_in += n_i;  idx_out += n_o
        else:
            chunk = len(d)
            mm_test[idx_test:idx_test + chunk] = enc_np
            mm_yt  [idx_test:idx_test + chunk] = d.astype("float32")
            idx_test += chunk

    del mm_in, mm_out, mm_test, mm_yt
    print(f"Done → in: {idx_in}, out: {idx_out}, test: {idx_test}")

    # ── 5. DataLoaders ────────────────────────────────────────────────────
    n_safe   = idx_in   # frames where d == 1
    n_unsafe = idx_out  # frames where d == -1

    # Each safe frame gets weight 1/n_safe, each unsafe gets 1/n_unsafe
    weights_in  = torch.full((n_safe,),   1.0 / n_safe)
    weights_out = torch.full((n_unsafe,), 1.0 / n_unsafe)

    sampler_in  = WeightedRandomSampler(weights_in,  num_samples=int(n_safe),   replacement=True)
    sampler_out = WeightedRandomSampler(weights_out, num_samples=int(n_unsafe), replacement=True)
    loader_in   = DataLoader(MemmapDataset(mm["X_train_in"],  device=device), batch_size=config.batch_size, shuffle=True, sampler=sampler_in)
    loader_out  = DataLoader(MemmapDataset(mm["X_train_out"], device=device), batch_size=config.batch_size, shuffle=True, sampler=sampler_out)
    test_loader = DataLoader(MemmapDataset(mm["X_test"], mm["y_test"], device=device), batch_size=config.test_batch_size)

    # ── 6. SDF model ──────────────────────────────────────────────────────
    model = select_model(args.model, pca_dim, args.n_layers, args.n_hidden).to(device)
    print(f"SDF parameters: {count_parameters(model)}")

    # ── 7. Point cloud export ─────────────────────────────────────────────
    if config.signed:
        X_in  = torch.from_numpy(np.load(mm["X_train_in"]))
        X_out = torch.from_numpy(np.load(mm["X_train_out"]))
        pc = point_cloud_from_arrays((X_in, -1.), (X_out, 1.))
        del X_in, X_out
    else:
        X_out = torch.from_numpy(np.load(mm["X_train_out"]))
        pc    = point_cloud_from_arrays((X_out, 1.))
        del X_out
    torch.save(pc, os.path.join(out_folder, "pc_0.pt"))
    del pc

    # ── 8. Train ──────────────────────────────────────────────────────────
    config.output_folder = out_folder  # point callbacks to per-run folder
    callbacks = [LoggerCB(os.path.join(out_folder, "log.csv"))]
    if config.checkpoint_freq > 0:
        callbacks.append(CheckpointCB(
            [x for x in range(0, config.n_epochs, config.checkpoint_freq) if x > 0]
        ))
    callbacks.append(UpdateHkrRegulCB({1: 1., 5: 10., 10: 100., 30: config.loss_regul}))

    trainer = Trainer((loader_in, loader_out) if config.signed else (loader_out,),
                      test_loader, config)
    trainer.add_callbacks(*callbacks)
    if config.signed:
        trainer.train_lip(model)
    else:
        trainer.train_lip_unsigned(model)

    model_path = os.path.join(out_folder, f"model_hkr_loss_{pca_dim}.pt")
    save_model(model, model_path)
    print(f"Model saved to {model_path}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args   = get_args()
    device = get_device(args.cpu)

    # Build encoder
    enc_kwargs = {}
    if args.encoder == "cjepa":
        enc_kwargs["checkpoint_path"] = args.cjepa_ckpt
    elif args.encoder == "dreamer":
        enc_kwargs["checkpoint_path"] = args.dreamer_ckpt
    elif args.encoder == "autoencoder":
        enc_kwargs["checkpoint_path"] = args.autoencoder_ckpt
    elif args.encoder == "lewm":
        enc_kwargs["checkpoint_path"] = args.lewm_ckpt

    print(f"Loading encoder: {args.encoder} ...")
    encoder = build_encoder(args.encoder, **enc_kwargs)

    files     = sorted(glob.glob(os.path.join(args.dataset, "*.npz")))
    num_train = int(len(files) * 0.7)
    print(f"Found {len(files)} episodes ({num_train} train).")

    # Shared config (output_folder overridden per run)
    config = SimpleNamespace(
        signed          = not args.unsigned,
        device          = device,
        n_epochs        = args.epochs,
        checkpoint_freq = args.checkpoint_freq,
        batch_size      = args.batch_size,
        test_batch_size = args.test_batch_size,
        loss_margin     = args.loss_margin,
        loss_regul      = args.loss_lambda,
        optimizer       = "adam",
        learning_rate   = args.learning_rate,
        output_folder   = None,
    )

    for pca_dim in args.pca_dims:
        run_pca_dim(args, encoder, files, num_train, device, pca_dim, config)

    print("\nAll runs complete.")