"""
Visualize training results for a single SDF run.

Produces:
  1. Training loss curve       → training_loss.png
  2. 2D PCA scatter (safe/unsafe in latent space) → pca_scatter_2d.png
  3. 3D PCA scatter (if pca_dim >= 3)             → pca_scatter_3d.png

Usage:
    python visualize_train.py --run-dir output/output/cjepa_pca3
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run directory produced by train_lip.py")
    parser.add_argument("--n-sample", type=int, default=5000,
                        help="Max points per class in scatter plots")
    return parser.parse_args()


def sample(X, n, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    return X[idx]


# ── 1. Training loss curve ─────────────────────────────────────────────────

def plot_loss(run_dir):
    log_path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(log_path):
        print(f"No log.csv found in {run_dir}, skipping loss plot.")
        return

    df = pd.read_csv(log_path)
    print(f"Log columns: {list(df.columns)}")

    fig, axes = plt.subplots(1, len(df.columns) - 1, figsize=(5 * (len(df.columns) - 1), 4))
    if len(df.columns) - 1 == 1:
        axes = [axes]

    epoch_col = df.columns[0]  # assume first col is epoch
    for ax, col in zip(axes, df.columns[1:]):
        series = df[col]
        if series.dtype == object:
            series = series.apply(lambda x: list(eval(x).values())[0] if isinstance(x, str) else list(x.values())[0])
        ax.plot(df[epoch_col], series, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col)
        ax.set_title(col.replace("_", " ").title())
        ax.grid(alpha=0.3)

    name = os.path.basename(run_dir)
    fig.suptitle(f"Training — {name}", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(run_dir, "training_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 2. PCA scatter ─────────────────────────────────────────────────────────

def plot_scatter_2d(run_dir, n_sample):
    X_in  = np.load(os.path.join(run_dir, "X_train_in.npy"),  mmap_mode="r")
    X_out = np.load(os.path.join(run_dir, "X_train_out.npy"), mmap_mode="r")
    pca_dim = X_in.shape[1]

    X_in_s  = sample(X_in,  n_sample)
    X_out_s = sample(X_out, n_sample)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_out_s[:, 0], X_out_s[:, 1], s=5, alpha=0.35,
               color="#F44336", label=f"Unsafe (n={len(X_out):,})", rasterized=True)
    ax.scatter(X_in_s[:, 0],  X_in_s[:, 1],  s=5, alpha=0.35,
               color="#2196F3", label=f"Safe   (n={len(X_in):,})", rasterized=True)
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_title(f"PCA Latent Space — {os.path.basename(run_dir)}\n(dim={pca_dim})", fontsize=11)
    ax.legend(markerscale=3, fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(run_dir, "pca_scatter_2d.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_scatter_3d(run_dir, n_sample):
    X_in  = np.load(os.path.join(run_dir, "X_train_in.npy"),  mmap_mode="r")
    X_out = np.load(os.path.join(run_dir, "X_train_out.npy"), mmap_mode="r")

    if X_in.shape[1] < 3:
        print("PCA dim < 3, skipping 3D scatter.")
        return

    X_in_s  = sample(X_in,  n_sample)
    X_out_s = sample(X_out, n_sample)

    # Plot from 4 angles
    angles = [(20, 30), (20, 120), (20, 210), (20, 300)]
    fig = plt.figure(figsize=(14, 12))
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(X_out_s[:, 0], X_out_s[:, 1], X_out_s[:, 2],
                   s=2, alpha=0.25, color="#F44336", label="Unsafe", rasterized=True)
        ax.scatter(X_in_s[:, 0],  X_in_s[:, 1],  X_in_s[:, 2],
                   s=2, alpha=0.25, color="#2196F3", label="Safe",   rasterized=True)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_zlabel("PC 3")
        ax.view_init(elev=elev, azim=azim)
        if i == 0:
            ax.legend(markerscale=4, fontsize=9)

    fig.suptitle(f"PCA 3D — {os.path.basename(run_dir)}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(run_dir, "pca_scatter_3d.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── 3. Class balance bar ───────────────────────────────────────────────────

def plot_class_balance(run_dir):
    X_in  = np.load(os.path.join(run_dir, "X_train_in.npy"),  mmap_mode="r")
    X_out = np.load(os.path.join(run_dir, "X_train_out.npy"), mmap_mode="r")
    y_test = np.load(os.path.join(run_dir, "y_test.npy"),     mmap_mode="r")

    n_train_safe   = len(X_in)
    n_train_unsafe = len(X_out)
    n_test_safe    = int((y_test > 0).sum())
    n_test_unsafe  = int((y_test <= 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (safe, unsafe, title) in zip(axes, [
        (n_train_safe, n_train_unsafe, "Train split"),
        (n_test_safe,  n_test_unsafe,  "Test split"),
    ]):
        bars = ax.bar(["Safe", "Unsafe"], [safe, unsafe],
                      color=["#2196F3", "#F44336"], width=0.5)
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set_title(title)
        ax.set_ylabel("Frames")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Class Balance — {os.path.basename(run_dir)}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(run_dir, "class_balance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()
    run  = args.run_dir

    assert os.path.isdir(run), f"Run dir not found: {run}"

    print(f"\nVisualizing training for: {run}\n")
    plot_loss(run)
    plot_scatter_2d(run, args.n_sample)
    plot_scatter_3d(run, args.n_sample)
    plot_class_balance(run)
    print("\nDone. All figures saved inside the run directory.")
