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
    parser.add_argument("--dataset-mode", choices=["npz", "ts"], default="npz")
    parser.add_argument("--encoder", choices=["cjepa", "dreamer", "autoencoder", "lewm", "ts"],
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

    print(f"Loading encoder: {args.encoder} ...")
    encoder = build_encoder(args.encoder, **enc_kwargs)

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

    # ── Encode test episodes on-the-fly ───────────────────────────────────
    if args.dataset_mode == "npz":
        files         = sorted(glob.glob(os.path.join(args.dataset, "*.npz")))
        num_train     = int(len(files) * 0.7)
        dataset_source = files[num_train:]
        print(f"Evaluating on {len(dataset_source)} npz test episodes...")
    else:
        dataset_source = load_ts_dataset(args)
        print(f"Evaluating on {len(dataset_source)} TS episodes...")
 
    # ── Single evaluation loop ────────────────────────────────────────────
    preds_all  = []
    labels_all = []
 
    for i in range(len(dataset_source)):
        if i % 100 == 0:
            print(f"  Episode {i}/{len(dataset_source)}")
 
        if args.dataset_mode == "npz":
            file    = np.load(dataset_source[i], allow_pickle=True)
            imgs_np = file["image"]
            d       = np.where(file["dones"] == 0, 1, -1)
        else:
            sample  = dataset_source[i]
            imgs_np = sample["image"]
            if torch.is_tensor(imgs_np):
                imgs_np = imgs_np.cpu().numpy()
            d = np.where(sample["dones"] == 0, 1, -1)
 
        if ipca is None:
            enc_np = encoder.encode(imgs_np, device)
        else:
            enc_np = ipca.transform(scaler.transform(encoder.encode(imgs_np, device)))
 
        enc_t = torch.from_numpy(enc_np.astype("float32")).to(device)
 
        with torch.no_grad():
            preds = sdf(enc_t).squeeze(-1).cpu()
 
        preds_all.append(preds)
        labels_all.append(torch.from_numpy(d.astype("float32")))
 
    preds  = torch.cat(preds_all)
    labels = torch.cat(labels_all)

    np.save(os.path.join(args.run_dir, "test_preds.npy"),  preds)
    np.save(os.path.join(args.run_dir, "test_labels.npy"), labels)
    print(f"Saved preds and labels to {args.run_dir}")

    print(f"\n── Results: {args.encoder} | {args.run_dir} ──")
    evaluate(preds, labels)
