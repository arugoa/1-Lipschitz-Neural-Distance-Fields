"""
Unified SDF training script.

Usage examples:
    python train_lip_cool.py ../dataset-good/ --encoder cjepa --pca-dims 3
    python train_lip_cool.py ../dataset-good/ --encoder dreamer --pca-dims 5 --dreamer-ckpt path/to/ckpt
    python train_lip_cool.py ../dataset-good/ --encoder autoencoder --autoencoder-ckpt path/to/ckpt
    python train_lip_cool.py ../dataset-good/ --encoder autoencoder --autoencoder-ckpt path/to/ckpt --no-pca
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
    parser.add_argument("dataset", type=str, help="Path to dataset folder", default="../../sold-sam/dataset/")
    parser.add_argument("-o", "--output-name", type=str, default="output")
    parser.add_argument("--unsigned", action="store_true")

    # encoder selection
    parser.add_argument("--encoder", choices=["cjepa", "dreamer", "autoencoder", "lewm"],
                        default="cjepa", help="Which encoder to use")
    parser.add_argument("--cjepa-ckpt", type=str, default="clevrer_savi_model.pth")
    parser.add_argument("--dreamer-ckpt", type=str, default=None)
    parser.add_argument("--dreamer-configs", type=str, default="../configs.yaml")
    parser.add_argument("--autoencoder-ckpt", type=str, default=None)
    parser.add_argument("--lewm-ckpt", type=str, default=None)

    # PCA
    parser.add_argument("-p", "--pca-dims", type=int, nargs="+", default=[3],
                        help="PCA dimension(s). Pass multiple to train several models. Ignored if --no-pca.")
    parser.add_argument("--no-pca", action="store_true",
                        help="Skip PCA and use raw encoder output directly.")
    parser.add_argument("--force-encode", action="store_true",
                        help="Re-encode even if memmaps already exist.")

    # SDF model
    parser.add_argument("-model", "--model", choices=["ortho", "sll", "mlp"], default="sll")
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


def transform_enc(enc_np, scaler, ipca, no_pca):
    """Apply scaler + PCA, or return raw if no_pca."""
    if no_pca:
        return enc_np.astype("float32")
    return ipca.transform(scaler.transform(enc_np)).astype("float32")


# ── Main training function ─────────────────────────────────────────────────

def run_pca_dim(args, encoder, files, device, pca_dim, config):
    """Run the full pipeline for one PCA dimension (or no PCA)."""

    if args.no_pca:
        pca_dim  = encoder.output_dim()
        run_name = f"{args.encoder}_nopca_{args.model}"
    else:
        run_name = f"{args.encoder}_pca{pca_dim}_{args.model}"

    out_folder = os.path.join("output", args.output_name, run_name)
    os.makedirs(out_folder, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Encoder: {args.encoder}   PCA dim: {pca_dim}   no_pca: {args.no_pca}")
    print(f"  Output:  {out_folder}")
    print(f"{'='*60}")

    mm = {k: os.path.join(out_folder, f"{k}.npy") for k in
          ["X_train_in", "X_train_out", "X_test", "y_test"]}
    pca_path = os.path.join(out_folder, "pca_pipeline.pkl")

    data_ready = (not args.force_encode) and all(os.path.exists(p) for p in [*mm.values(), pca_path])

    if data_ready:
        print(f"Found existing data in {out_folder}, skipping encoding.")
        n_in   = len(np.load(mm["X_train_in"],  mmap_mode="r"))
        n_out  = len(np.load(mm["X_train_out"], mmap_mode="r"))
        idx_in, idx_out = n_in, n_out
        print(f"Train → safe: {n_in}, unsafe: {n_out}")
        print(f"Test  → {len(np.load(mm['X_test'], mmap_mode='r'))} frames")

    else:
        # ── 1. Count total safe/unsafe frames across all episodes ──────────
        print("Counting frames...")
        n_safe_total = n_unsafe_total = 0
        for fp in files:
            d = np.load(fp, allow_pickle=True)["dones"]
            n_safe_total   += int((d == 0).sum())
            n_unsafe_total += int((d != 0).sum())
        print(f"Total frames → safe: {n_safe_total}, unsafe: {n_unsafe_total}")

        # 70% of each class goes to train, 30% to test
        n_safe_train   = int(n_safe_total   * 0.7)
        n_unsafe_train = int(n_unsafe_total * 0.7)
        n_safe_test    = n_safe_total   - n_safe_train
        n_unsafe_test  = n_unsafe_total - n_unsafe_train
        n_test         = n_safe_test + n_unsafe_test
        print(f"Train → safe: {n_safe_train}, unsafe: {n_unsafe_train}")
        print(f"Test  → safe: {n_safe_test},  unsafe: {n_unsafe_test}")

        # ── 2. Fit scaler + PCA on all episodes (skipped if --no-pca) ─────
        if args.no_pca:
            scaler, ipca = None, None
        else:
            ipca   = IncrementalPCA(n_components=pca_dim, batch_size=1024)
            scaler = StandardScaler()
            print("Fitting scaler + PCA...")
            for i, fp in enumerate(files):
                if i % 100 == 0:
                    print(f"  PCA fit {i}/{len(files)}")
                imgs_np = np.load(fp, allow_pickle=True)["image"]
                enc_np  = encoder.encode(imgs_np, device)
                scaler.partial_fit(enc_np)
                ipca.partial_fit(scaler.transform(enc_np))
            print(f"Explained variance: {ipca.explained_variance_ratio_.sum():.4f}")

        with open(pca_path, "wb") as f:
            pickle.dump({"scaler": scaler, "ipca": ipca, "no_pca": args.no_pca}, f)
        print(f"PCA pipeline saved to {pca_path}")

        # ── 3. Allocate memmaps ───────────────────────────────────────────
        mm_in   = np.lib.format.open_memmap(mm["X_train_in"],  mode="w+", dtype="float32", shape=(n_safe_train,   pca_dim))
        mm_out  = np.lib.format.open_memmap(mm["X_train_out"], mode="w+", dtype="float32", shape=(n_unsafe_train, pca_dim))
        mm_test = np.lib.format.open_memmap(mm["X_test"],      mode="w+", dtype="float32", shape=(n_test,         pca_dim))
        mm_yt   = np.lib.format.open_memmap(mm["y_test"],      mode="w+", dtype="float32", shape=(n_test,))

        # ── 4. Encode → (PCA) → write, filling train first then test ──────
        # We fill train slots until each class hits its 70% quota,
        # then overflow goes to test.
        print("Encoding + writing memmaps...")
        idx_in = idx_out = idx_test = 0

        for i, fp in enumerate(files):
            if i % 200 == 0:
                print(f"  Episode {i}/{len(files)}")
            file    = np.load(fp, allow_pickle=True)
            imgs_np = file["image"]
            d       = np.where(file["dones"] == 0, 1, -1)  # 1=safe, -1=unsafe
            enc_np  = transform_enc(encoder.encode(imgs_np, device), scaler, ipca, args.no_pca)

            safe_mask   = (d ==  1)
            unsafe_mask = (d == -1)

            safe_enc   = enc_np[safe_mask]
            unsafe_enc = enc_np[unsafe_mask]

            # Safe frames: fill train first, overflow to test
            for chunk, enc in [("safe", safe_enc), ("unsafe", unsafe_enc)]:
                if chunk == "safe":
                    n_train_quota = n_safe_train
                    idx_train     = idx_in
                    label_val     = 1.0
                    mm_train      = mm_in
                else:
                    n_train_quota = n_unsafe_train
                    idx_train     = idx_out
                    label_val     = -1.0
                    mm_train      = mm_out

                if len(enc) == 0:
                    continue

                train_space = n_train_quota - idx_train
                n_to_train  = min(len(enc), train_space)
                n_to_test   = len(enc) - n_to_train

                if n_to_train > 0:
                    mm_train[idx_train:idx_train + n_to_train] = enc[:n_to_train]

                if n_to_test > 0:
                    mm_test[idx_test:idx_test + n_to_test] = enc[n_to_train:]
                    mm_yt  [idx_test:idx_test + n_to_test] = label_val

                if chunk == "safe":
                    idx_in    += n_to_train
                    idx_test  += n_to_test
                else:
                    idx_out   += n_to_train
                    idx_test  += n_to_test

        del mm_in, mm_out, mm_test, mm_yt
        print(f"Done → train_in: {idx_in}, train_out: {idx_out}, test: {idx_test}")

    # ── 5. DataLoaders ────────────────────────────────────────────────────
    n_safe     = int(idx_in)
    n_unsafe   = int(idx_out)
    n_balanced = min(n_safe, n_unsafe)
    print(f"Balancing samplers: safe={n_safe}, unsafe={n_unsafe}, {n_balanced} each per epoch")

    weights_in  = torch.full((n_safe,),   1.0 / n_safe)
    weights_out = torch.full((n_unsafe,), 1.0 / n_unsafe)

    sampler_in  = WeightedRandomSampler(weights_in,  num_samples=n_balanced, replacement=True)
    sampler_out = WeightedRandomSampler(weights_out, num_samples=n_balanced, replacement=True)

    loader_in   = DataLoader(MemmapDataset(mm["X_train_in"],  device=device),
                             batch_size=config.batch_size, sampler=sampler_in)
    loader_out  = DataLoader(MemmapDataset(mm["X_train_out"], device=device),
                             batch_size=config.batch_size, sampler=sampler_out)
    test_loader = DataLoader(MemmapDataset(mm["X_test"], mm["y_test"], device=device),
                             batch_size=config.test_batch_size)

    # ── 6. SDF model ──────────────────────────────────────────────────────
    model = select_model(args.model, pca_dim, args.n_layers, args.n_hidden).to(device)
    print(f"SDF parameters: {count_parameters(model)}")

    # ── 8. Train ──────────────────────────────────────────────────────────
    config.output_folder = out_folder
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

    model_path = os.path.join(out_folder, f"model_hkr_loss_{pca_dim}_{args.model}.pt")
    save_model(model, model_path)
    print(f"Model saved to {model_path}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args   = get_args()
    device = get_device(args.cpu)

    enc_kwargs = {}
    if args.encoder == "cjepa":
        enc_kwargs["checkpoint_path"] = args.cjepa_ckpt
    elif args.encoder == "dreamer":
        enc_kwargs["checkpoint_path"] = args.dreamer_ckpt
        enc_kwargs["configs_path"]    = args.dreamer_configs
    elif args.encoder == "autoencoder":
        enc_kwargs["checkpoint_path"] = args.autoencoder_ckpt
    elif args.encoder == "lewm":
        enc_kwargs["checkpoint_path"] = args.lewm_ckpt

    print(f"Loading encoder: {args.encoder} ...")
    encoder = build_encoder(args.encoder, **enc_kwargs)

    files = sorted(glob.glob(os.path.join(args.dataset, "*.npz")))
    print(f"Found {len(files)} episodes.")

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

    if args.no_pca:
        run_pca_dim(args, encoder, files, device, None, config)
    else:
        for pca_dim in args.pca_dims:
            run_pca_dim(args, encoder, files, device, pca_dim, config)

    print("\nAll runs complete.")