"""
Unified SDF evaluation script.

Usage:
    python test_sdf.py ../dataset-good/ --encoder cjepa --run-dir output/output/cjepa_pca3 --model output/output/cjepa_pca3/model_hkr_loss_3.pt
"""

import os
import sys
import glob
import argparse
import pickle
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf

import numpy as np
import torch

from encoders import build_encoder
from common.models import load_model
from common.utils import get_device


def get_args():
    parser = argparse.ArgumentParser(description="Unified SDF evaluation")

    parser.add_argument("dataset", type=str, help="Path to dataset folder", default="../../sold-sam/dataset/")
    parser.add_argument("--dataset-mode", choices=["npz", "ts", "wall"], default="npz")
    parser.add_argument("--state-key",      type=str, default="state")
    parser.add_argument("--wall-obses-dir", type=str, default=None)
    parser.add_argument("--wall-states",    type=str, default=None)
    parser.add_argument("--wall-locs",      type=str, default=None)
    parser.add_argument("--door-locs",      type=str, default=None)
    parser.add_argument("--wall-config",    type=str, default=None)
    parser.add_argument("--margin",         type=float, default=1.5)

    parser.add_argument("--encoder", choices=["cjepa", "dreamer", "autoencoder", "lewm", "ts", "gt-state"],
                        default="cjepa")
    parser.add_argument("--cjepa-ckpt", type=str, default="clevrer_savi_model.pth")
    parser.add_argument("--dreamer-ckpt", type=str, default=None)
    parser.add_argument("--autoencoder-ckpt", type=str, default=None)
    parser.add_argument("--lewm-ckpt", type=str, default=None)
    parser.add_argument("--ts-ckpt", type=str, default=None)
    parser.add_argument("--ts-img-size", type=int, default=224)
    parser.add_argument("--ts-config", type=str, default=None)
    parser.add_argument("--num-hist", type=int, default=1)
    parser.add_argument("--num-pred", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--split", choices=["train", "valid"], default="valid")

    # Point at the run directory produced by train_lip.py
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Output dir from train_lip.py (contains pca_pipeline.pkl)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved SDF model (.pt)")

    parser.add_argument("-tbs", "--test-batch-size", type=int, default=5000)
    parser.add_argument("-cpu", action="store_true")

    return parser.parse_args()


class TemporalStraighteningFrameDataset(torch.utils.data.Dataset):
    def __init__(self, ts_dataset):
        self.ts_dataset = ts_dataset

    def __len__(self):
        return len(self.ts_dataset)

    def __getitem__(self, idx):
        obs, act, state = self.ts_dataset[idx]

        imgs = obs["visual"]

        if torch.is_tensor(imgs):
            imgs = imgs.cpu().numpy()

        dones = np.zeros(len(imgs), dtype=np.int32)

        return {
            "image": imgs,
            "dones": dones,
        }


def load_ts_dataset(args):
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="train")

    datasets, traj_dsets = hydra.utils.call(
        cfg.env.dataset,
        num_hist=args.num_hist,
        num_pred=args.num_pred,
        frameskip=args.frameskip,
    )
    dataset = datasets[args.split]

    print(f"Loaded TS dataset split={args.split}")
    print(f"Dataset size: {len(dataset)}")
    return TemporalStraighteningFrameDataset(dataset)


def evaluate(preds, y_test):
    tp = ((preds > 0) & (y_test > 0)).sum().item()
    fp = ((preds > 0) & (y_test <= 0)).sum().item()
    fn = ((preds <= 0) & (y_test > 0)).sum().item()
    tn = ((preds <= 0) & (y_test <= 0)).sum().item()

    accuracy  = (torch.sign(preds) == torch.sign(y_test)).float().mean().item()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Accuracy : {accuracy * 100:.4f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nConfusion Matrix")
    print(f"TP: {tp}  FP: {fp}")
    print(f"FN: {fn}  TN: {tn}")

    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


if __name__ == "__main__":
    args   = get_args()
    device = get_device(args.cpu)

    # ── Load encoder ──────────────────────────────────────────────────────
    enc_kwargs = {}
    if args.encoder == "cjepa":
        enc_kwargs["checkpoint_path"] = args.cjepa_ckpt
    elif args.encoder == "dreamer":
        enc_kwargs["checkpoint_path"] = args.dreamer_ckpt
    elif args.encoder == "autoencoder":
        enc_kwargs["checkpoint_path"] = args.autoencoder_ckpt
    elif args.encoder == "lewm":
        enc_kwargs["checkpoint_path"] = args.lewm_ckpt
    elif args.encoder == "ts":
        enc_kwargs["checkpoint_path"] = args.ts_ckpt
        enc_kwargs["img_size"]        = args.ts_img_size

    if args.encoder == "gt-state":
        encoder = None
        print("gt-state encoder: skipping image encoder build.")
    else:
        print(f"Loading encoder: {args.encoder} ...")
        encoder = build_encoder(args.encoder, **enc_kwargs)

    if args.dataset_mode == "npz":
        files = sorted(glob.glob(os.path.join(args.dataset, "*.npz")))
        num_train = int(len(files) * 0.7)
        dataset_source = files[num_train:]
        print(f"Evaluating on {len(dataset_source)} npz test episodes...")
    elif args.dataset_mode == "wall":
        from train_lip_cool import (WallDatasetSource, load_wall_config,
                                    compute_dones_from_geometry, is_near_wall,
                                    check_wall_intersect)
        cfg = load_wall_config(args.wall_config)
        dataset_source = WallDatasetSource(
            obses_dir       = args.wall_obses_dir,
            states_path     = args.wall_states,
            wall_path       = args.wall_locs,
            door_path       = args.door_locs,
            ball_radius     = cfg.env.get("ball_radius",     1.0),
            door_half_w     = cfg.env.get("door_space",      4.0),
            env_size        = cfg.env.get("img_size",        64.0),
            wall_width      = cfg.env.get("wall_width",      4.0),
            border_wall_loc = cfg.env.get("border_wall_loc", 2.0),
        )
        print(f"Evaluating on {len(dataset_source)} wall episodes...")
    else:
        dataset_source = load_ts_dataset(args)
        print(f"Evaluating on {len(dataset_source)} TS episodes...")

    # ── Load PCA pipeline saved by train_lip.py ───────────────────────────
    pca_path = os.path.join(args.run_dir, "pca_pipeline.pkl")
    print(f"Loading PCA pipeline from {pca_path} ...")
    with open(pca_path, "rb") as f:
        pca_data = pickle.load(f)
    scaler = pca_data["scaler"]
    ipca   = pca_data["ipca"]

    # ── Load SDF model ────────────────────────────────────────────────────
    print(f"Loading SDF model from {args.model} ...")
    sdf = load_model(args.model, device)
    sdf.eval()

    # ── Load saved test set from train output ─────────────────────────────
    X_test = np.load(os.path.join(args.run_dir, "X_test.npy"), mmap_mode="r")
    y_test = np.load(os.path.join(args.run_dir, "y_test.npy"), mmap_mode="r")
    print(f"Loaded test set: {X_test.shape[0]} frames")

    enc_t  = torch.from_numpy(np.array(X_test, dtype="float32"))
    labels = torch.from_numpy(np.array(y_test, dtype="float32"))

    # ── Run inference in batches ──────────────────────────────────────────
    preds_list = []
    for start in range(0, len(enc_t), args.test_batch_size):
        batch = enc_t[start : start + args.test_batch_size].to(device)
        with torch.no_grad():
            preds_list.append(-sdf(batch).squeeze(-1).cpu())
    preds = torch.cat(preds_list)

    # ── Save for visualize_test.py ────────────────────────────────────────
    np.save(os.path.join(args.run_dir, "test_preds.npy"),  preds.numpy())
    np.save(os.path.join(args.run_dir, "test_labels.npy"), labels.numpy())
    print(f"Saved test_preds.npy and test_labels.npy to {args.run_dir}")

    print(f"\n── Results: {args.encoder} | {args.run_dir} ──")
    evaluate(preds, labels)
