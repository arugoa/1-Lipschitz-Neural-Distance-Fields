import os
import sys
import glob
from types import SimpleNamespace
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


from training.train_autoencoder import load_autoencoder
from common.models import load_model
from common.utils import get_device


def pad(arr, target_len=150):
    if arr.shape[0] >= target_len:
        return arr[:target_len]
    last = arr[-1:]
    pad = np.repeat(last, target_len - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def build_pca_pipeline(files, autoencoder, device, num_train, pca_dim):
    print("Fitting PCA (same as training)...")

    scaler = [StandardScaler() for _ in range(5)]
    ipca = [IncrementalPCA(n_components=pca_dim, batch_size=1024) for _ in range(5)]

    for i, file_path in enumerate(files):
        if i >= num_train:
            break

        file = np.load(file_path, allow_pickle=True)

        imgs = torch.from_numpy(pad(file["images"])[None]).float().to(device)
        acts = torch.from_numpy(pad(file["actions"])[None]).float().to(device)

        with torch.no_grad():
            enc = autoencoder.encode(imgs, acts)

        enc = enc[0].cpu().numpy().reshape(150, 640)

        for k in range(5):
            chunk = enc[:, 128*k:128*(k+1)]
            scaler[k].partial_fit(chunk)
            chunk = scaler[k].transform(chunk)
            ipca[k].partial_fit(chunk)

    return scaler, ipca


def transform(enc, scaler, ipca):
    parts = []
    for k in range(5):
        chunk = enc[:, 128*k:128*(k+1)]
        chunk = scaler[k].transform(chunk)
        chunk = ipca[k].transform(chunk)
        parts.append(chunk)
    return np.concatenate(parts, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--pca-dim", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = get_device(args.cpu)
    print("DEVICE:", device)

    # --- Load models ---
    autoencoder = load_autoencoder(args.autoencoder)
    autoencoder.eval()

    sdf = load_model(args.model, device)
    sdf.eval()

    # --- Dataset ---
    files = sorted(glob.glob(os.path.join(args.dataset, "*.npz")))
    num_train = int(len(files) * 0.8)

    # --- Fit PCA like training ---
    scaler, ipca = build_pca_pipeline(files, autoencoder, device, num_train, args.pca_dim)

    # --- Evaluate ---
    preds_all = []
    labels_all = []

    print("Evaluating...")

    for i, file_path in enumerate(files[num_train:]):
        file = np.load(file_path, allow_pickle=True)

        imgs = pad(file["images"])
        acts = pad(file["actions"])

        # FIXED LABELS
        d = file["dones"]
        d = np.where(d == 0, -1, 1)
        d = pad(d)

        imgs_t = torch.from_numpy(imgs[None]).float().to(device)
        acts_t = torch.from_numpy(acts[None]).float().to(device)

        with torch.no_grad():
            enc = autoencoder.encode(imgs_t, acts_t)

        enc = enc[0].cpu().numpy().reshape(150, 640)
        enc = transform(enc, scaler, ipca)

        # --- Batch inference ---
        enc_t = torch.from_numpy(enc).float().to(device)

        with torch.no_grad():
            preds = sdf(enc_t).squeeze(-1).cpu()

        preds_all.append(preds)
        labels_all.append(torch.from_numpy(d))

    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)

    # --- Metrics ---
    pred_sign = torch.sign(preds)
    label_sign = torch.sign(labels)

    accuracy = (pred_sign == label_sign).float().mean()

    tp = ((preds > 0) & (labels > 0)).sum().item()
    fp = ((preds > 0) & (labels <= 0)).sum().item()
    fn = ((preds <= 0) & (labels > 0)).sum().item()
    tn = ((preds <= 0) & (labels <= 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\nAccuracy : {accuracy.item()*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(f"TP: {tp}  FP: {fp}")
    print(f"FN: {fn}  TN: {tn}")