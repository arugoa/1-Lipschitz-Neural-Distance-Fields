"""
plot_pca_wall_episode.py
------------------------
Adapted from plot_pca_episode.py to load episodes directly from the
temporal-straightening "single_wall" dataset via hydra, instead of
manually-specified .npz/.pth files.

1. Load an episode from the TS wall dataset (obs["visual"], state)
2. Run each frame through an encoder checkpoint (.pth with 'encoder' key)
3. PCA(2) on the features → trajectory plot + MSE-to-target curve
4. Background: alpha-composite overlay of all frames, or last frame + state path

Usage
-----
python plot_pca_wall_episode.py \
    --ts-config   ./conf/train.yaml \
    --split       valid \
    --episode-idx 0 \
    --encoder     ./checkpoints/test/wall_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05/checkpoints/model_latest.pth \
    --bg overlay \
    --label TS-encoder \
    --out pca_wall_episode.png
"""

import argparse
import os
import sys
import contextlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageDraw
from sklearn.decomposition import PCA


# ════════════════════════════════════════════════════════════
#  TS wall dataset episode loading
# ════════════════════════════════════════════════════════════

def load_wall_episode(ts_config_path, split, episode_idx):
    """
    Load one episode from the TS single_wall dataset using hydra.

    Returns
    -------
    images : np.ndarray  (T, H, W, 3)  uint8
    states : np.ndarray  (T, S)  or None
    """
    import hydra
    from omegaconf import OmegaConf

    config_dir  = os.path.dirname(os.path.abspath(ts_config_path))
    config_name = os.path.splitext(os.path.basename(ts_config_path))[0]

    from hydra import initialize_config_dir, compose
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)

    datasets, _ = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    dataset = datasets[split]
    print(f"Loaded TS wall dataset split={split}, size={len(dataset)}")

    obs, act, state = dataset[episode_idx]
    imgs = obs["visual"]   # (T, C, H, W) float tensor, normalised
    if torch.is_tensor(imgs):
        imgs = imgs.cpu().numpy()
    if torch.is_tensor(state):
        state = state.cpu().numpy()

    imgs = _normalise_images(imgs)
    states = _normalise_states(state) if state is not None else None

    print(f"Episode {episode_idx}: images={imgs.shape}"
          + (f", states={states.shape}" if states is not None else ""))
    return imgs, states


def _normalise_states(states):
    if states is None:
        return None
    states = np.array(states, dtype=np.float32)
    if states.ndim > 2:
        orig = states.shape
        states = states.reshape(len(states), -1)
        print(f"  states (original shape {orig}) → flattened to {states.shape}")
    return states


def _normalise_images(arr):
    """Ensure shape (T, H, W, 3), dtype uint8."""
    arr = np.array(arr)
    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
        arr = arr.transpose(0, 2, 3, 1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        if hi <= 1.5:
            # could be ImageNet-normalised — undo if values go negative
            if lo < 0:
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                arr = arr * std + mean
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


# ════════════════════════════════════════════════════════════
#  Encoder loading & inference
# ════════════════════════════════════════════════════════════

def load_encoder(path: str, enc_key: str = "encoder"):
    """
    Load a TS encoder from a checkpoint .pth.
    Sets up sys.path for local TS modules + dinov2 hub cache.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    hub_path = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    if os.path.exists(hub_path) and hub_path not in sys.path:
        sys.path.insert(0, hub_path)

    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        print(f"  checkpoint keys: {list(obj.keys())}")
        if enc_key in obj:
            model = obj[enc_key]
        else:
            model = next((v for v in obj.values() if isinstance(v, nn.Module)), None)
            if model is None:
                raise KeyError(f"No nn.Module found under '{enc_key}'. "
                               f"Available: {list(obj.keys())}")
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, device


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@contextlib.contextmanager
def _no_grad():
    with torch.no_grad():
        yield


def encode_images(model, device, images: np.ndarray,
                  img_size=(224, 224), batch_size=32,
                  normalise=True, pool_mode="mean") -> np.ndarray:
    """
    Encode every frame in `images` (T, H, W, 3) uint8 through `model`.
    Returns (T, D) float32 numpy array.
    """
    T = len(images)
    feats_list = []

    with _no_grad():
        for start in range(0, T, batch_size):
            batch_np = images[start:start + batch_size]
            frames = []
            for img in batch_np:
                pil = PILImage.fromarray(img).resize(img_size, PILImage.BILINEAR)
                arr = np.array(pil, dtype=np.float32) / 255.0
                if normalise:
                    arr = (arr - _MEAN) / _STD
                frames.append(arr.transpose(2, 0, 1))  # (3, H, W)

            tensor = torch.tensor(np.stack(frames)).to(device)
            out = model(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]

            if out.ndim == 2:
                pass
            elif out.ndim == 3:
                if pool_mode == "cls":
                    out = out[:, 0, :]
                elif pool_mode == "mean":
                    out = out.mean(dim=1)
                elif pool_mode == "flat":
                    out = out.reshape(out.shape[0], -1)
                else:
                    raise ValueError(f"Unknown pool_mode '{pool_mode}'")
            elif out.ndim == 4:
                out = out.mean(dim=(2, 3))
            else:
                raise ValueError(f"Unexpected encoder output shape: {out.shape}")

            feats_list.append(out.cpu().float().numpy())
            print(f"  encoded {min(start+batch_size, T)}/{T} frames", end="\r")

    print()
    return np.concatenate(feats_list, axis=0)


# ════════════════════════════════════════════════════════════
#  Background helpers
# ════════════════════════════════════════════════════════════

def make_overlay(images: np.ndarray, fg_threshold: int = 20) -> np.ndarray:
    T, H, W, C = images.shape
    imgs_f = images.astype(np.float32)

    step = max(1, T // 30)
    bg   = np.median(imgs_f[::step], axis=0)

    canvas = imgs_f[-1].copy()
    for t in range(T):
        frame = imgs_f[t]
        diff  = np.linalg.norm(frame - bg, axis=-1)
        mask  = diff > fg_threshold
        if not mask.any():
            continue
        alpha = 0.25 + 0.75 * t / max(T - 1, 1)
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * frame[mask]

    return canvas.clip(0, 255).astype(np.uint8)


def make_state_overlay(images: np.ndarray, states: np.ndarray,
                       state_xy: tuple) -> np.ndarray:
    H, W = images.shape[1], images.shape[2]
    bg = images[-1].copy()
    xi, yi = state_xy

    xs = states[:, xi].copy()
    ys = states[:, yi].copy()
    xs = (-xs + 1) * (W / 2)
    ys = (-ys + 1) * (H / 2)

    canvas = PILImage.fromarray(bg)
    draw   = ImageDraw.Draw(canvas, "RGBA")

    T = len(xs)
    for t in range(T - 1):
        alpha = int(80 + 175 * t / max(T - 2, 1))
        draw.line([(xs[t], ys[t]), (xs[t+1], ys[t+1])],
                  fill=(255, 220, 50, alpha), width=3)
    draw.ellipse([xs[0]-5, ys[0]-5, xs[0]+5, ys[0]+5],
                 fill=(100, 200, 255, 220), outline=(255,255,255,200))
    draw.ellipse([xs[-1]-6, ys[-1]-6, xs[-1]+6, ys[-1]+6],
                 fill=(255, 193, 7, 255), outline=(0,0,0,200))

    return np.array(canvas.convert("RGB"))


# ════════════════════════════════════════════════════════════
#  PCA + MSE
# ════════════════════════════════════════════════════════════

def run_pca(features):
    pca = PCA(n_components=2)
    return pca.fit_transform(features), pca


def mse_to_target(features, target_idx):
    tgt = features[target_idx]
    return ((features - tgt) ** 2).mean(axis=1)


# ════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════

def plot_ground_truth(ax, states, state_xy, target_idx, color):
    """Plot the raw ground-truth XY trajectory from the state array."""
    if states is None:
        ax.text(0.5, 0.5, "No state data available",
                ha="center", va="center", fontsize=9, transform=ax.transAxes)
        ax.set_title("Ground Truth", fontsize=10, fontweight="bold")
        ax.axis("off")
        return

    xi, yi = state_xy
    xs = states[:, xi]
    ys = states[:, yi]
    T  = len(xs)

    for t in range(T - 1):
        alpha = 0.25 + 0.75 * t / max(T - 2, 1)
        ax.plot(xs[t:t+2], ys[t:t+2],
                color=color, alpha=alpha, linewidth=1.4, solid_capstyle="round")

    ax.scatter(xs[0], ys[0], color=color, s=22, zorder=4, alpha=0.8, label="Start")
    ax.scatter(xs[target_idx], ys[target_idx], marker="*", s=260, color="#f5c518",
              edgecolors="black", linewidths=0.6, zorder=5, label="Goal")

    ax.set_xlabel("X", fontsize=8); ax.set_ylabel("Y", fontsize=8)
    ax.set_title("Ground Truth (state space)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)


def plot_episode(bg_img, proj, mse, states, state_xy, label, color, target_idx, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    ax_scene, ax_gt = axes[0, 0], axes[0, 1]
    ax_pca,   ax_mse = axes[1, 0], axes[1, 1]

    # ── Top-left: episode overlay ──────────────────────────────────────
    ax_scene.imshow(bg_img)
    ax_scene.axis("off")
    ax_scene.set_title("Episode (overlay)", fontsize=10, fontweight="bold")

    # ── Top-right: ground truth coordinates ────────────────────────────
    plot_ground_truth(ax_gt, states, state_xy, target_idx, color)

    # ── Bottom-left: PCA trajectory ─────────────────────────────────────
    T = len(proj)
    for t in range(T - 1):
        alpha = 0.25 + 0.75 * t / max(T - 2, 1)
        ax_pca.plot(proj[t:t+2, 0], proj[t:t+2, 1],
                    color=color, alpha=alpha, linewidth=1.4, solid_capstyle="round")
    ax_pca.scatter(*proj[0], color=color, s=22, zorder=4, alpha=0.8, label="Start")
    ax_pca.scatter(*proj[target_idx], marker="*", s=260, color="#f5c518",
                   edgecolors="black", linewidths=0.6, zorder=5, label="Goal")
    ax_pca.set_xlabel("PC1", fontsize=8); ax_pca.set_ylabel("PC2", fontsize=8)
    ax_pca.set_title(f"PCA — {label}", fontsize=10, fontweight="bold")
    ax_pca.legend(fontsize=7)
    ax_pca.tick_params(labelsize=7)

    # ── Bottom-right: MSE to target ──────────────────────────────────────
    ax_mse.plot(mse, color=color, linewidth=1.6)
    ax_mse.set_xlabel("time", fontsize=8); ax_mse.set_ylabel("MSE to target", fontsize=8)
    ax_mse.set_title(f"MSE — {label}", fontsize=10, fontweight="bold")
    ax_mse.tick_params(labelsize=7)

    for ax in (ax_pca, ax_mse):
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)

    plt.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ts-config",   type=str, required=True,
                   help="Path to TS hydra config (e.g. ../conf/train.yaml)")
    p.add_argument("--split",       choices=["train", "valid"], default="valid")
    p.add_argument("--episode-idx", type=int, default=0,
                   help="Which episode index in the dataset to use")
    p.add_argument("--encoder",     type=str, required=True,
                   help=".pth checkpoint containing the encoder")
    p.add_argument("--enc-key",     type=str, default="encoder")
    p.add_argument("--label",       type=str, default="TS-encoder")
    p.add_argument("--bg",          choices=["overlay", "state", "last"], default="overlay")
    p.add_argument("--state-xy",    nargs=2, type=int, default=[0, 1])
    p.add_argument("--target-idx",  type=int, default=-1)
    p.add_argument("--pool",        choices=["cls", "mean", "flat"], default="mean")
    p.add_argument("--img-size",    nargs=2, type=int, default=[224, 224])
    p.add_argument("--out",         default="pca_wall_episode.png")
    return p.parse_args()


def main():
    args = parse_args()

    imgs, states = load_wall_episode(args.ts_config, args.split, args.episode_idx)

    if args.bg == "overlay":
        bg = make_overlay(imgs)
    elif args.bg == "state":
        if states is None:
            print("WARNING: no states found, falling back to last-frame bg")
            bg = imgs[-1]
        else:
            bg = make_state_overlay(imgs, states, tuple(args.state_xy))
    else:
        bg = imgs[-1]

    print(f"Loading encoder: {args.encoder}")
    model, device = load_encoder(args.encoder, args.enc_key)

    feats = encode_images(model, device, imgs,
                          img_size=tuple(args.img_size),
                          pool_mode=args.pool)
    print(f"Features: {feats.shape}")

    proj, _ = run_pca(feats)
    mse     = mse_to_target(feats, args.target_idx)

    plot_episode(bg, proj, mse, states, tuple(args.state_xy),
                args.label, "#3a6bc9", args.target_idx, args.out)


if __name__ == "__main__":
    main()