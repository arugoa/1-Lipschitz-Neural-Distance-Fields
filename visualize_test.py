"""
Visualize test results for a single SDF run.

Produces:
  1. Metrics summary bar chart          → test_metrics.png
  2. Confusion matrix heatmap           → confusion_matrix.png
  3. SDF score distribution             → sdf_score_dist.png
  4. Decision boundary (2D, if pca=2)   → decision_boundary_2d.png
  5. Decision boundary slice (3D→2D)    → decision_boundary_slice.png

Usage:
    python visualize_test.py \
        --run-dir output/output/cjepa_pca3 \
        --model   output/output/cjepa_pca3/model_hkr_loss_3.pt
"""

import os
import glob
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

from common.models import load_model
from common.utils import get_device


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run directory produced by train_lip.py")
    parser.add_argument("--model", type=str, default=None,
                        help="SDF model path. Auto-detected from run-dir if omitted.")
    parser.add_argument("--n-sample", type=int, default=8000,
                        help="Max points to use in boundary plots")
    parser.add_argument("--state-xy", nargs=2, type=int, default=[0, 1],
                        help="Which two state dims are X, Y for the comparison plot")
    parser.add_argument("-cpu", action="store_true")
    return parser.parse_args()


def load_sdf(run_dir, model_path, device):
    if model_path is None:
        found = glob.glob(os.path.join(run_dir, "model_hkr_loss_*.pt"))
        assert found, f"No model found in {run_dir}"
        model_path = found[-1]
    print(f"Loading SDF from: {model_path}")
    return load_model(model_path, device), model_path


def run_sdf(sdf, X, device, batch_size=5000):
    sdf.eval()
    out = []
    X_t = torch.from_numpy(np.array(X)).float()
    for i in range(0, len(X_t), batch_size):
        with torch.no_grad():
            out.append(sdf(X_t[i:i+batch_size].to(device)).squeeze(-1).cpu())
    return torch.cat(out).numpy()


def compute_metrics(preds, labels):
    p = preds;  l = labels
    tp = int(((p > 0) & (l > 0)).sum())
    fp = int(((p > 0) & (l <= 0)).sum())
    fn = int(((p <= 0) & (l > 0)).sum())
    tn = int(((p <= 0) & (l <= 0)).sum())
    accuracy  = float((np.sign(p) == np.sign(l)).mean())
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return dict(accuracy=accuracy, precision=precision, recall=recall,
                f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


# ── 1. Metrics bar chart ───────────────────────────────────────────────────

def plot_metrics(metrics, run_dir):
    keys   = ["accuracy", "precision", "recall", "f1"]
    values = [metrics[k] for k in keys]
    colors = ["#4c72b0", "#55a868", "#dd8452", "#c44e52"]
    labels = ["Accuracy (%)", "Precision", "Recall", "F1"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
    ax.set_ylim(0, 110 if max(values) > 1 else 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Test Metrics — {os.path.basename(run_dir)}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(run_dir, "test_metrics.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ── 2. Confusion matrix ────────────────────────────────────────────────────

def plot_confusion_matrix(metrics, run_dir):
    cm = np.array([[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Safe", "Pred Unsafe"], fontsize=11)
    ax.set_yticklabels(["Actual Safe", "Actual Unsafe"], fontsize=11)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=13, color="white" if cm[i,j] > cm.max() / 2 else "black")

    ax.set_title(f"Confusion Matrix — {os.path.basename(run_dir)}")
    plt.tight_layout()
    path = os.path.join(run_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ── 3. SDF score distribution ──────────────────────────────────────────────

def plot_score_dist(preds, labels, run_dir):
    safe_preds   = preds[labels > 0]
    unsafe_preds = preds[labels <= 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(preds.min(), preds.max(), 80)
    ax.hist(safe_preds,   bins=bins, alpha=0.6, color="#2196F3",
            label=f"Safe   (n={len(safe_preds):,})",   density=True)
    ax.hist(unsafe_preds, bins=bins, alpha=0.6, color="#F44336",
            label=f"Unsafe (n={len(unsafe_preds):,})", density=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Decision boundary (0)")
    ax.set_xlabel("SDF Score")
    ax.set_ylabel("Density")
    ax.set_title(f"SDF Score Distribution — {os.path.basename(run_dir)}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(run_dir, "sdf_score_dist.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ── 4. Decision boundary (2D) ──────────────────────────────────────────────

def plot_boundary_2d(sdf, X_in, X_out, run_dir, device, n_sample=8000):
    """Only valid when pca_dim == 2."""
    rng = np.random.default_rng(42)

    def s(X):
        idx = rng.choice(len(X), min(n_sample, len(X)), replace=False)
        return X[idx]

    X_in_s  = s(X_in)
    X_out_s = s(X_out)

    all_X   = np.vstack([X_in_s, X_out_s])
    pad     = (all_X.max(0) - all_X.min(0)) * 0.1
    x_min, y_min = all_X.min(0) - pad
    x_max, y_max = all_X.max(0) + pad

    res  = 300
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res),
                         np.linspace(y_min, y_max, res))
    grid = np.c_[xx.ravel(), yy.ravel()].astype("float32")
    zz   = run_sdf(sdf, grid, device).reshape(res, res)

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(xx, yy, zz)
    ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=1.5)
    plt.colorbar(cf, ax=ax, label="SDF score")

    ax.scatter(X_out_s[:, 0], X_out_s[:, 1], s=4, alpha=0.4,
               color="#F44336", label="Unsafe", rasterized=True)
    ax.scatter(X_in_s[:, 0],  X_in_s[:, 1],  s=4, alpha=0.4,
               color="#2196F3", label="Safe",   rasterized=True)

    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title(f"Decision Boundary — {os.path.basename(run_dir)}")
    ax.legend(markerscale=3)
    plt.tight_layout()
    path = os.path.join(run_dir, "decision_boundary_2d.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ── 5. Decision boundary slice (for pca_dim >= 3) ─────────────────────────

def plot_boundary_slice(sdf, X_in, X_out, run_dir, device, n_sample=8000):
    all_X = np.vstack([X_in, X_out])
    
    # Fit PCA on all training data → meaningful 2D projection
    pca = PCA(n_components=2)
    pca.fit(all_X)
    evr = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, total={evr.sum():.3f}")

    rng = np.random.default_rng(42)
    def s(X):
        idx = rng.choice(len(X), min(n_sample, len(X)), replace=False)
        return X[idx]
    X_in_s  = s(X_in)
    X_out_s = s(X_out)

    # Project sampled points into PCA space for scatter
    X_in_2d  = pca.transform(X_in_s)
    X_out_2d = pca.transform(X_out_s)

    # Build grid in PCA space, then invert back to embedding space for SDF query
    all_2d = pca.transform(all_X)
    pad    = (all_2d.max(0) - all_2d.min(0)) * 0.1
    x_min, y_min = all_2d.min(0) - pad
    x_max, y_max = all_2d.max(0) + pad

    res = 250
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res),
                         np.linspace(y_min, y_max, res))
    grid_2d = np.c_[xx.ravel(), yy.ravel()].astype("float32")

    # Invert PCA to get embedding-space points for SDF evaluation
    grid_full = pca.inverse_transform(grid_2d).astype("float32")
    zz = run_sdf(sdf, grid_full, device).reshape(res, res)

    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(xx, yy, zz, levels=50, cmap="RdBu_r", alpha=0.85)
    ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=1.5)
    plt.colorbar(cf, ax=ax, label="SDF score")

    ax.scatter(X_out_2d[:, 0], X_out_2d[:, 1], s=4, alpha=0.35,
               color="#F44336", label="Unsafe", rasterized=True)
    ax.scatter(X_in_2d[:, 0],  X_in_2d[:, 1],  s=4, alpha=0.35,
               color="#2196F3", label="Safe",   rasterized=True)

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=11)
    ax.set_title(f"Decision Boundary (PCA projection)\n{os.path.basename(run_dir)}")
    ax.legend(markerscale=3)
    plt.tight_layout()
    path = os.path.join(run_dir, "decision_boundary_slice.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_state_space_sdf(
    sdf,
    run_dir,
    device,
    state_xy=(0, 1),
    n_sample=20000,
):
    """
    Plot the SDF value of every test point at its ground-truth state
    coordinates.
    """

    X = np.load(os.path.join(run_dir, "X_test.npy"), mmap_mode="r")
    states = np.load(os.path.join(run_dir, "State_test.npy"), mmap_mode="r")
    labels = np.load(os.path.join(run_dir, "y_test.npy"), mmap_mode="r")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(n_sample, len(X)), replace=False)

    X = X[idx]
    states = states[idx]
    labels = labels[idx]

    sdf_vals = run_sdf(sdf, X, device)

    xi, yi = state_xy

    fig, ax = plt.subplots(figsize=(8, 7))

    sc = ax.scatter(
        states[:, xi],
        states[:, yi],
        c=sdf_vals,
        cmap="RdBu_r",
        s=8,
        alpha=0.8,
        rasterized=True,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("SDF value")

    # Optional: draw misclassified samples
    pred_safe = sdf_vals > 0
    true_safe = labels > 0
    wrong = pred_safe != true_safe

    # ax.scatter(
    #     states[wrong, xi],
    #     states[wrong, yi],
    #     marker="x",
    #     s=20,
    #     color="k",
    #     linewidth=0.8,
    #     label=f"Misclassified ({wrong.sum()})",
    # )

    ax.set_xlabel("State X")
    ax.set_ylabel("State Y")
    ax.set_aspect("equal")

    ax.set_title(
        f"SDF values projected on state space\n{os.path.basename(run_dir)}"
    )

    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "state_space_sdf.png"), dpi=200)
    plt.close()

    print("Saved:", os.path.join(run_dir, "state_space_sdf.png"))

# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args   = get_args()
    device = get_device(args.cpu)
    run    = args.run_dir

    assert os.path.isdir(run), f"Run dir not found: {run}"

    print(f"\nVisualizing test results for: {run}\n")

    # Load data
    preds_path  = os.path.join(run, "test_preds.npy")
    labels_path = os.path.join(run, "test_labels.npy")
    assert os.path.exists(preds_path), "Run test_sdf_cool.py first to generate test_preds.npy"
    preds  = np.load(preds_path)
    labels = np.load(labels_path)

    # Compute and print metrics
    metrics = compute_metrics(preds, labels)
    print(f"\n── Metrics ──────────────────────────────")
    print(f"Accuracy : {metrics['accuracy']*100:.4f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"TP: {metrics['tp']}  FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}  TN: {metrics['tn']}")
    print(f"─────────────────────────────────────────\n")

    # Plots
    plot_metrics(metrics, run)
    plot_confusion_matrix(metrics, run)
    plot_score_dist(preds, labels, run)

    X_in  = np.load(os.path.join(run, "X_train_in.npy"),  mmap_mode="r")
    X_out = np.load(os.path.join(run, "X_train_out.npy"), mmap_mode="r")
    pca_dim = X_in.shape[1]

    # Load SDF only for boundary plots
    sdf, _ = load_sdf(run, args.model, device)
    if pca_dim == 2:
        plot_boundary_2d(sdf, X_in, X_out, run, device, args.n_sample)
    elif pca_dim >= 3:
        plot_boundary_slice(sdf, X_in, X_out, run, device, args.n_sample)

    plot_state_space_sdf(sdf, run, device)

    print("\nDone. All figures saved inside the run directory.")
