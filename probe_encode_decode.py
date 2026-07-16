"""
probe_encode_decode.py
-----------------------
Sanity-check the TS encoder's latent space by round-tripping a single state:

1. Pick a random frame from the wall dataset (known ground-truth position).
2. Encode it with the TS encoder.
3. Find its k nearest neighbors in the SDF training embeddings (X_train_in/out
   from --run-dir) and look up their ground-truth positions.
4. "Decode" the embedding back to a position estimate — there is no trained
   latent-to-position network in this codebase, so this uses the same
   distance-weighted k-NN average that plan_sdf_ts.py's
   plot_rrt_in_state_space() already uses as its de facto decoder.
5. Plot the original (ground truth) position, the k nearest-neighbor
   positions, and the decoded estimate together, so encoder quality is
   visible directly as how tightly they cluster.

Usage
-----
python probe_encode_decode.py \
    --run-dir        output/output/ts_nopca_sll_wall \
    --ts-ckpt        ../checkpoints_from_cluster/wall_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05/model_latest.pth \
    --wall-obses-dir ../data/wall_single/obses/ \
    --wall-states    ../data/wall_single/states.pth \
    --k 10 \
    --out probe_encode_decode.png
"""

import os
import sys
import glob
import re
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from encoders import build_encoder
from common.utils import get_device


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, required=True,
                   help="Dir containing X_train_in.npy, X_train_out.npy, State_in.npy, State_out.npy")
    p.add_argument("--ts-ckpt", type=str, required=True)
    p.add_argument("--wall-obses-dir", type=str, required=True)
    p.add_argument("--wall-states", type=str, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--episode-idx", type=int, default=None,
                   help="Fix the episode instead of picking randomly")
    p.add_argument("--frame-idx", type=int, default=None,
                   help="Fix the frame within the episode instead of picking randomly")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, default="probe_encode_decode.png")
    p.add_argument("-cpu", action="store_true")
    return p.parse_args()


def load_random_frame(obses_dir, states_path, episode_idx, frame_idx, rng):
    eps = sorted(
        glob.glob(os.path.join(obses_dir, "*.pth")),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )
    if episode_idx is None:
        episode_idx = int(rng.integers(0, len(eps)))

    imgs = torch.load(eps[episode_idx], map_location="cpu", weights_only=False)
    imgs = imgs.numpy() if torch.is_tensor(imgs) else np.asarray(imgs)  # (T, H, W, C) or (T, C, H, W)
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3):          # (T, C, H, W) -> (T, H, W, C)
        imgs = imgs.transpose(0, 2, 3, 1)
    if imgs.dtype != np.uint8:
        imgs = (imgs * 255.0).clip(0, 255).astype(np.uint8) if imgs.max() <= 1.0 else imgs.astype(np.uint8)

    all_states = torch.load(states_path, map_location="cpu", weights_only=False).numpy()  # (N, T, 2)
    ep_states = all_states[episode_idx]  # (T, 2)

    T = imgs.shape[0]
    if frame_idx is None:
        frame_idx = int(rng.integers(0, T))

    frame_img = imgs[frame_idx]        # (H, W, C) uint8
    frame_state = ep_states[frame_idx]  # (2,)

    print(f"Picked episode {episode_idx}, frame {frame_idx}/{T-1}  "
          f"ground-truth state={frame_state}")
    return frame_img, frame_state


def main():
    args = get_args()
    device = get_device(args.cpu)
    rng = np.random.default_rng(args.seed)

    # ── encode a random (or specified) real frame ──────────────────────────
    frame_img, true_state = load_random_frame(
        args.wall_obses_dir, args.wall_states,
        args.episode_idx, args.frame_idx, rng,
    )

    print(f"\nLoading TS encoder from {args.ts_ckpt} ...")
    encoder = build_encoder("ts", checkpoint_path=args.ts_ckpt)

    imgs_np = frame_img[None]  # (1, H, W, C) — encode() expects a batch of frames
    z_query = encoder.encode(imgs_np, device=str(device))  # (1, D)
    print(f"Encoded embedding shape: {z_query.shape}")

    # ── load SDF training embeddings + their ground-truth positions ────────
    X_safe = np.load(os.path.join(args.run_dir, "X_train_in.npy"))
    X_bad = np.load(os.path.join(args.run_dir, "X_train_out.npy"))
    S_safe = np.load(os.path.join(args.run_dir, "State_in.npy"))
    S_bad = np.load(os.path.join(args.run_dir, "State_out.npy"))
    X_train = np.vstack([X_safe, X_bad])
    S_train = np.vstack([S_safe, S_bad])

    if X_train.shape[1] != z_query.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: X_train has D={X_train.shape[1]} but the "
            f"live-encoded query has D={z_query.shape[1]}. X_train_in/out.npy in "
            f"--run-dir were probably generated before the return_agg fix — "
            f"rerun train_lip_cool.py's data-prep step to regenerate them."
        )

    # ── k nearest neighbors ─────────────────────────────────────────────────
    tree = KDTree(X_train)
    dist, ind = tree.query(z_query, k=args.k)
    dist, ind = dist[0], ind[0]
    nn_positions = S_train[ind]  # (k, 2)

    print(f"\nNearest {args.k} neighbor distances: min={dist.min():.4f}  "
          f"max={dist.max():.4f}  mean={dist.mean():.4f}")

    # ── "decode": distance-weighted average of the k neighbors' positions ──
    # (No trained latent->position network exists in this codebase; this
    # matches the k-NN convention plan_sdf_ts.py's plot_rrt_in_state_space
    # already uses to turn a latent into a state-space estimate.)
    weights = 1.0 / (dist + 1e-8)
    weights /= weights.sum()
    decoded_position = (weights[:, None] * nn_positions).sum(axis=0)

    error = np.linalg.norm(decoded_position - true_state)
    print(f"\nTrue position:    {true_state}")
    print(f"Decoded position: {decoded_position}  (error={error:.4f})")

    # ── plot ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(7, 7))
    plt.scatter(S_train[:, 0], S_train[:, 1], s=2, alpha=0.1, color="gray",
                label="X_train states")
    plt.scatter(nn_positions[:, 0], nn_positions[:, 1], s=60, color="tab:blue",
                edgecolors="k", linewidths=0.5, zorder=10,
                label=f"{args.k} nearest-neighbor positions")
    plt.scatter(*decoded_position, marker="X", s=250, color="tab:orange",
                edgecolors="k", linewidths=1.2, zorder=20,
                label="decoded (k-NN weighted avg)")
    plt.scatter(*true_state, marker="*", s=300, color="tab:green",
                edgecolors="k", linewidths=1.2, zorder=20,
                label="ground truth (original)")

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Encode/decode probe — error={error:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
