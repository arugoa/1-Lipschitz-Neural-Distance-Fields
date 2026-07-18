"""
Latent space planning: SDF-RRT vs World Model rollout.

Uses the temporal straightening world model's proper rollout API alongside
the trained 1-Lipschitz SDF to plan safe paths in latent space.

Two paths are computed and compared:
  1. SDF-RRT:  geometric path in PCA space, constrained to SDF safe set
  2. WM rollout: optimised action sequence rolled out through the world model

Usage:
    python plan.py \
        --model-path  ../checkpoints/test/wall_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05 \
        --model-epoch latest \
        --start-img   path/to/start.png \
        --goal-img    path/to/goal.png  \
        --run-dir     output/output/ts_nopca_sll \
        --sdf-model   output/output/ts_nopca_sll/model_hkr_loss_384_sll.pt
"""

import os
import sys
import argparse
import pickle
import numpy as np
import glob
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
import hydra

from common.models import load_model as load_sdf_model
from common.utils import get_device


# ── Args ───────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser()

    # TS world model
    parser.add_argument("--model-path",  type=str, required=True,
                        help="Path to the TS training output dir (contains hydra.yaml + checkpoints/)")
    parser.add_argument("--model-epoch", type=str, default="latest",
                        help="Epoch to load, e.g. '100' or 'latest'")
    parser.add_argument("--ts-repo",     type=str, default=None,
                        help="Path to temporal-straightening repo root")
    

    # start / goal images
    # add to get_args():
    parser.add_argument("--encoder",     choices=["ts", "gt-state"], default="ts")
    parser.add_argument("--state-start", type=float, nargs=2, default=None,
                        help="Start XY state in world coords, e.g. --state-start 45.0 23.0")
    parser.add_argument("--state-goal",  type=float, nargs=2, default=None,
                        help="Goal XY state in world coords")
    parser.add_argument("--wall-npz",    type=str, default=None,
                        help="Wall .npz episode — uses first/last state as start/goal")

    parser.add_argument("--npz",       type=str, default=None,
                    help="Path to .npz episode — uses first frame as start, last as goal")
    parser.add_argument("--start-img", type=str, default=None,
                        help="Path to start image (overrides --npz)")
    parser.add_argument("--goal-img",  type=str, default=None,
                        help="Path to goal image (overrides --npz)")
    parser.add_argument("--img-size",    type=int, default=224)

    parser.add_argument("--wall-episode-idx", type=int, default=0,
                    help="Which episode index to plan for")
    parser.add_argument("--wall-obses-dir",   type=str, default=None)
    parser.add_argument("--wall-states-path", type=str, default=None,
                        help="Path to states.pth (N, T, 2)")

    # SDF
    parser.add_argument("--run-dir",     type=str, required=True)
    parser.add_argument("--sdf-model",   type=str, required=True)

    # RRT
    parser.add_argument("--rrt-iters",   type=int,   default=5000)
    parser.add_argument("--rrt-step",    type=float, default=0.01)
    parser.add_argument("--sdf-margin",  type=float, default=0.001)
    parser.add_argument("--goal-radius", type=float, default=0.01)

    # WM rollout optimisation
    parser.add_argument("--rollout-steps", type=int,   default=30)
    parser.add_argument("--optim-steps",   type=int,   default=300)
    parser.add_argument("--optim-lr",      type=float, default=1e-2)
    parser.add_argument("--safety-weight", type=float, default=1.0)
    parser.add_argument("--frameskip",     type=int,   default=1)

    parser.add_argument("--out-dir", type=str, default="output/planning")
    parser.add_argument("-cpu", action="store_true")
    return parser.parse_args()


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree


def plot_rrt_in_state_space(
    trajectories,
    run_dir,
    out_dir,
    save_path=None,
    filename="state_space.png",
    k=5,
    show_dataset=True,
    show_numbers=False,
    labels=None,
    gt_trajectories=None,
    gt_labels=None,
):
    """
    Visualize one or more latent-space trajectories in ground-truth state
    coordinates using nearest-neighbor lookup, all overlaid on one plot.

    Parameters
    ----------
    trajectories : (N,D) ndarray, or list of such
        A single latent trajectory, or a list of trajectories to compare on
        the same axes (e.g. an RRT/WM path against that same path replayed
        through the world model with a recovered action sequence).

    run_dir : str
        Directory containing
            X_train_in.npy
            X_train_out.npy
            State_in.npy
            State_out.npy

    save_path : str or None
        Defaults to out_dir/filename

    filename : str
        Used to build save_path when save_path is not given.

    k : int
        Number of nearest neighbors used for interpolation.
        k=1 gives nearest neighbor.
        k=5 usually looks much smoother.

    show_dataset : bool
        Draw all stored states in the background.

    show_numbers : bool
        Draw waypoint indices.

    labels : list of str, optional
        Legend label per trajectory. Defaults to "trajectory 0", "trajectory 1", ...

    gt_trajectories : (N,2) ndarray, or list of such, optional
        Ground-truth trajectories already in real state-space coordinates —
        plotted directly (no NN lookup / no encoder involved), dashed, so you
        can visually compare the encoder+PCA+NN-lookup reconstruction in
        `trajectories` against the true states (embedding-quality check).

    gt_labels : list of str, optional
        Legend label per ground-truth trajectory.

    Returns
    -------
    A single (N,2) ndarray if `trajectories` was a single trajectory,
    otherwise a list of (N_i,2) ndarrays, one per input trajectory.
    Ground-truth trajectories are not included in the return value (they're
    already in state space — nothing to hand back).
    """

    single = isinstance(trajectories, np.ndarray) or (
        len(trajectories) > 0 and np.ndim(trajectories[0]) == 1
    )
    traj_list = [trajectories] if single else list(trajectories)
    traj_list = [np.asarray(t) for t in traj_list]

    if labels is None:
        labels = [f"trajectory {i}" for i in range(len(traj_list))]

    if gt_trajectories is None:
        gt_list = []
    else:
        gt_single = isinstance(gt_trajectories, np.ndarray) or (
            len(gt_trajectories) > 0 and np.ndim(gt_trajectories[0]) == 1
        )
        gt_list = [gt_trajectories] if gt_single else list(gt_trajectories)
        gt_list = [np.asarray(t) for t in gt_list]
    if gt_labels is None:
        gt_labels = [f"ground truth {i}" for i in range(len(gt_list))]

    # --------------------------------------------------
    # load dataset
    # --------------------------------------------------

    X_safe = np.load(os.path.join(run_dir, "X_train_in.npy"), mmap_mode="r")
    X_bad  = np.load(os.path.join(run_dir, "X_train_out.npy"), mmap_mode="r")

    S_safe = np.load(os.path.join(run_dir, "State_in.npy"), mmap_mode="r")
    S_bad  = np.load(os.path.join(run_dir, "State_out.npy"), mmap_mode="r")

    X = np.vstack([X_safe, X_bad])
    S = np.vstack([S_safe, S_bad])

    tree = KDTree(X)

    # --------------------------------------------------
    # plotting
    # --------------------------------------------------

    plt.figure(figsize=(8, 8))

    if show_dataset:
        plt.scatter(
            S_safe[:, 0], S_safe[:, 1],
            s=2, alpha=0.15, color="tab:blue", label="safe states",
        )
        plt.scatter(
            S_bad[:, 0], S_bad[:, 1],
            s=4, alpha=0.25, color="tab:red", label="unsafe states",
        )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    traj_states_list = []

    for i, latents in enumerate(traj_list):
        # --------------------------------------------------
        # nearest-neighbor lookup
        # --------------------------------------------------
        dist, ind = tree.query(latents, k=k)
        print(f"[{labels[i]}] Mean NN distance: {dist.mean():.4f}  "
              f"Max NN distance: {dist.max():.4f}")

        if k == 1:
            traj = S[ind[:, 0]]
        else:
            weights = 1.0 / (dist + 1e-8)
            weights /= weights.sum(axis=1, keepdims=True)
            traj = (weights[:, :, None] * S[ind]).sum(axis=1)

        traj_states_list.append(traj)
        color = colors[i % len(colors)]

        plt.plot(
            traj[:, 0], traj[:, 1], "-",
            linewidth=2, color=color, label=labels[i], zorder=10,
        )
        plt.scatter(
            traj[:, 0], traj[:, 1],
            s=40, color=color, edgecolors="k", linewidths=0.4, zorder=20,
        )
        plt.scatter(
            traj[0, 0], traj[0, 1],
            marker="o", s=180, color=color, edgecolors="k", zorder=30,
        )
        plt.scatter(
            traj[-1, 0], traj[-1, 1],
            marker="*", s=250, color=color, edgecolors="k", zorder=30,
        )

        if show_numbers:
            for j, p in enumerate(traj):
                plt.text(p[0], p[1], str(j), fontsize=8)

    # ground-truth trajectories: already in state space, plotted directly
    # (no NN lookup), dashed + distinct marker so they read as "exact" next
    # to the encoded/projected trajectories above.
    for i, traj in enumerate(gt_list):
        color = colors[(len(traj_list) + i) % len(colors)]

        plt.plot(
            traj[:, 0], traj[:, 1], "--",
            linewidth=2, color=color, label=gt_labels[i], zorder=11,
        )
        plt.scatter(
            traj[:, 0], traj[:, 1],
            marker="x", s=40, color=color, linewidths=1.2, zorder=21,
        )
        plt.scatter(
            traj[0, 0], traj[0, 1],
            marker="o", s=180, facecolors="none", edgecolors=color, linewidths=2, zorder=30,
        )
        plt.scatter(
            traj[-1, 0], traj[-1, 1],
            marker="*", s=250, facecolors="none", edgecolors=color, linewidths=2, zorder=30,
        )

        if show_numbers:
            for j, p in enumerate(traj):
                plt.text(p[0], p[1], str(j), fontsize=8)

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Planner trajectories projected to state space")
    plt.legend()

    if save_path is None:
        save_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved {save_path}")

    return traj_states_list[0] if single else traj_states_list

# ── TS model loading ───────────────────────────────────────────────────────

def setup_ts_paths(ts_repo, model_path):
    """Add TS repo and dinov2 hub cache to sys.path."""
    if ts_repo and ts_repo not in sys.path:
        sys.path.insert(0, ts_repo)
    hub_path = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    if os.path.exists(hub_path) and hub_path not in sys.path:
        sys.path.insert(0, hub_path)


ALL_MODEL_KEYS = [
    "encoder", "predictor", "decoder", "proprio_encoder", "action_encoder",
]


def load_wall_episode(obses_dir, states_path, episode_idx, img_size, device):
    """
    Load a single wall episode for both ts and gt_state encoders.
    
    Returns:
        start_img_t  : (1, 1, C, H, W) float tensor  — for ts encoder
        goal_img_t   : (1, 1, C, H, W) float tensor  — for ts encoder
        start_state  : (2,) numpy                     — for gt_state encoder
        goal_state   : (2,) numpy                     — for gt_state encoder
        all_states   : (T, 2) numpy                   — full trajectory
    """
    import re

    # load images
    eps = sorted(
        glob.glob(os.path.join(obses_dir, "*.pth")),
        key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1))
    )
    imgs = torch.load(eps[episode_idx], map_location="cpu",
                      weights_only=False).float()   # (T, C, H, W) or (T, H, W, C)

    # normalise to (T, C, H, W) float [0, 1]
    if imgs.ndim == 4 and imgs.shape[-1] in (3, 4):
        imgs = imgs.permute(0, 3, 1, 2)             # (T, H, W, C) → (T, C, H, W)
    if imgs.max() > 1.0:
        imgs = imgs / 255.0

    # resize if needed
    if imgs.shape[-1] != img_size:
        imgs = F.interpolate(imgs, size=(img_size, img_size), mode="bilinear",
                             align_corners=False)

    start_img_t = imgs[0].unsqueeze(0).unsqueeze(0).to(device)   # (1,1,C,H,W)
    goal_img_t  = imgs[-1].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,C,H,W)

    # load states
    all_states  = torch.load(states_path, map_location="cpu",
                             weights_only=False).numpy()          # (N, T, 2)
    ep_states   = all_states[episode_idx]                        # (T, 2)
    start_state = ep_states[0].astype("float32")                 # (2,)
    goal_state  = ep_states[-1].astype("float32")                # (2,)

    print(f"  Episode {episode_idx}: {len(ep_states)} steps  "
          f"start={start_state}  goal={goal_state}")

    return start_img_t, goal_img_t, start_state, goal_state, ep_states, imgs.unsqueeze(0).to(device)  # (1, T, C, H, W)


def load_ckpt(snapshot_path, device):
    import sys, os
    # Same path setup as straightening.py
    ts_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ts_repo not in sys.path:
        sys.path.insert(0, ts_repo)
    hub_path = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    if os.path.exists(hub_path) and hub_path not in sys.path:
        sys.path.insert(0, hub_path)

    from models.dino import DinoV2Encoder
    _ = DinoV2Encoder('dinov2_vits14', 'x_norm_patchtokens')

    with open(snapshot_path, "rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_ts_wm(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result.get("decoder", None),
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


def load_ts_model(model_path, model_epoch, device):
    """
    Load the full TS world model using the same logic as plan.py's load_model.
    Returns (wm, train_cfg, dset_info).
    """
    model_path = os.path.abspath(model_path)
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        train_cfg = OmegaConf.load(f)

    epoch_str = model_epoch if model_epoch != "latest" else "latest"
    ckpt_name = f"model_{epoch_str}.pth"
    model_ckpt = Path(model_path) / "checkpoints" / ckpt_name

    num_action_repeat = train_cfg.num_action_repeat
    wm = load_ts_wm(model_ckpt, train_cfg, num_action_repeat, device=device)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    return wm, train_cfg


def load_preprocessor(model_path, train_cfg, device):
    """Load a Preprocessor using the validation dataset stats."""
    from preprocessor import Preprocessor

    _, dsets = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    dset = dsets["valid"]

    preprocessor = Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=dset.transform,
    )
    return preprocessor, dset


# ── Image helpers ──────────────────────────────────────────────────────────

def load_img_tensor(path, img_size, transform, device):
    """
    Load an image and apply the dataset transform.
    Returns (1, 1, C, H, W) tensor — batch=1, T=1.
    """
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    img_np = np.array(img)                    # (H, W, C) uint8
    img_t  = transform(img_np)                # (C, H, W) float
    return img_t.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, C, H, W)


def aggregate_visual(wm, z):
    b, t, p, d = z.shape
    tokens = z.reshape(b * t, p, d)
    agg = wm.encoder.agg(tokens)
    return agg.reshape(b, t, -1)


def img_to_latent(wm, img_t, state_gt):
    """
    img_t: (1, 1, C, H, W)
    Returns z: (1, emb_dim) — aggregated (agg_mlp) embedding
    """
    obs = {"visual": img_t, "proprio": state_gt}
    with torch.no_grad():
        z_dict = wm.encode_obs(obs)
    # z_dict is typically {"visual": (1, 1, num_patches, emb_dim)} or (1, 1, emb_dim)
    z = z_dict["visual"]
    # collapse T and patch dims
    if z.ndim == 4:          # (B, T, patches, D)
        z = aggregate_visual(wm, z)  # (B, T, agg_out_dim)
    z = z.squeeze(1)         # (B, D)
    return z                 # (1, emb_dim)


def encode_episode_latents(wm, imgs, ep_states, preprocessor, device):
    """
    Encode a whole episode (all T frames at once) into per-frame mean-pooled
    visual latents, applying the same photometric transform + proprio
    normalisation the model saw at train time.

    imgs:      (1, T, C, H, W) tensor, resized + scaled to [0,1] but not yet
               photometrically normalised (as returned by load_wall_episode).
    ep_states: (T, 2) numpy raw (unnormalised) states.

    Returns (T, D) numpy latent trajectory.
    """
    imgs = preprocessor.transform(imgs)
    proprio = preprocessor.normalize_proprios(
        torch.from_numpy(ep_states).float()
    ).unsqueeze(0).to(device)  # (1, T, 2)

    obs = {"visual": imgs, "proprio": proprio}
    with torch.no_grad():
        z_dict = wm.encode_obs(obs)
    z = z_dict["visual"]       # (1, T, patches, D) or (1, T, D)
    if z.ndim == 4:
        z = aggregate_visual(wm, z)  # (1, T, agg_out_dim)
    return z.squeeze(0).cpu().numpy()  # (T, D)


def load_start_goal_from_npz(npz_path, img_size):
    """
    Load first and last frames from an npz episode.
    Returns (start_np, goal_np) both (H, W, C) uint8.
    """
    file    = np.load(npz_path, allow_pickle=True)
    images  = file["image"]          # (T, H, W, C)
    start_np = images[0]              # first frame
    goal_np  = images[-1]             # last frame
    start_state_gt = torch.from_numpy(file["state"][0])
    goal_state_gt = torch.from_numpy(file["state"][-1])

    # resize if needed
    if start_np.shape[0] != img_size:
        from PIL import Image
        def resize(arr):
            return np.array(
                Image.fromarray(arr).resize((img_size, img_size))
            )
        start_np = resize(start_np)
        goal_np  = resize(goal_np)

    print(f"Loaded episode: {images.shape[0]} frames, "
          f"start={start_np.shape}, goal={goal_np.shape}")
    return start_np, goal_np, start_state_gt, goal_state_gt


def np_to_img_tensor(arr_np, device):
    """(H, W, C) uint8 numpy → (1, 1, C, H, W) float tensor"""
    t = torch.from_numpy(arr_np).float().permute(2, 0, 1)  # (C, H, W)
    return t.unsqueeze(0).unsqueeze(0).to(device)           # (1, 1, C, H, W)


def state_to_latent(xy):
    """(2,) numpy → (1, 2) float32 numpy — identity for gt_state encoder."""
    return np.array(xy, dtype="float32").reshape(1, -1)

# ── PCA helpers ───────────────────────────────────────────────────────────

def to_pca(z_np, scaler, ipca, no_pca):
    if no_pca or scaler is None:
        return z_np
    return ipca.transform(scaler.transform(z_np))


def to_pca_torch(z, scaler, ipca, no_pca, device):
    """
    Differentiable equivalent of to_pca() for a torch tensor z: (..., D).
    Reimplements StandardScaler + IncrementalPCA.transform with torch ops so
    gradients can flow from the SDF value back through to `z` (and hence to
    the actions that produced it).
    """
    if no_pca or scaler is None:
        return z
    mean  = torch.as_tensor(scaler.mean_,  dtype=z.dtype, device=device)
    scale = torch.as_tensor(scaler.scale_, dtype=z.dtype, device=device)
    z = (z - mean) / scale

    pca_mean   = torch.as_tensor(ipca.mean_,       dtype=z.dtype, device=device)
    components = torch.as_tensor(ipca.components_, dtype=z.dtype, device=device)  # (n_comp, D)
    z = (z - pca_mean) @ components.T
    if getattr(ipca, "whiten", False):
        explained_var = torch.as_tensor(ipca.explained_variance_, dtype=z.dtype, device=device)
        z = z / torch.sqrt(explained_var)
    return z


def from_pca_sdf(pca_pts, sdf, device):
    """Query SDF for a batch of PCA points. Returns numpy (N,)."""
    t = torch.from_numpy(pca_pts.astype("float32")).to(device)
    with torch.no_grad():
        vals = sdf(t).squeeze(-1).cpu().numpy()
    return vals


# ── RRT ───────────────────────────────────────────────────────────────────

class RRTNode:
    def __init__(self, x, parent=None):
        self.x      = np.array(x, dtype="float32")
        self.parent = parent


def edge_safe(sdf, x_a, x_b, device, margin, n=50):
    pts  = np.stack([x_a + t * (x_b - x_a) for t in np.linspace(0, 1, n)])
    vals = from_pca_sdf(pts, sdf, device)
    return np.all(vals > margin)


def rrt(x_start, x_goal, sdf, device, bounds,
        n_iters, step_size, goal_radius, margin):
    nodes = [RRTNode(x_start)]
    dim   = len(x_start)

    for it in range(n_iters):
        x_rand = x_goal if np.random.rand() < 0.1 else np.array([
            np.random.uniform(bounds[d, 0], bounds[d, 1]) for d in range(dim)
        ], dtype="float32")

        dists   = np.linalg.norm(np.stack([n.x for n in nodes]) - x_rand, axis=1)
        nearest = nodes[np.argmin(dists)]

        direction = x_rand - nearest.x
        dist      = np.linalg.norm(direction)
        if dist < 1e-6:
            continue
        x_new = nearest.x + (direction / dist) * min(step_size, dist)

        vals = from_pca_sdf(x_new[None], sdf, device)
        if vals[0] <= margin:
            continue
        if not edge_safe(sdf, nearest.x, x_new, device, margin):
            continue

        node = RRTNode(x_new, parent=nearest)
        nodes.append(node)

        if np.linalg.norm(x_new - x_goal) < goal_radius:
            print(f"  RRT: goal reached at iter {it+1}")
            path = []
            n    = node
            while n:
                path.append(n.x); n = n.parent
            return list(reversed(path)), nodes

        if (it + 1) % 500 == 0:
            closest_dist = np.linalg.norm(
                np.stack([n.x for n in nodes]) - x_goal, axis=1
            ).min()
            print(f"  RRT iter {it+1}/{n_iters}  "
                  f"nodes={len(nodes)}  closest={closest_dist:.4f}")

    print("  RRT: goal not reached — returning best partial path.")
    dists   = np.linalg.norm(np.stack([n.x for n in nodes]) - x_goal, axis=1)
    closest = nodes[np.argmin(dists)]
    path    = []
    n       = closest
    while n:
        path.append(n.x); n = n.parent
    return list(reversed(path)), nodes


# ── WM rollout optimisation ────────────────────────────────────────────────

def optimise_rollout(
    wm, start_img_t, start_proprio, goal_latent, sdf,
    scaler, ipca, no_pca, device,
    rollout_steps, action_dim, frameskip,
    n_optim_steps, lr, safety_weight,
    preprocessor, margin=0.0,
):
    """
    Optimise an action sequence so wm.rollout reaches goal_latent.
    start_img_t:  (1, 1, C, H, W)
    goal_latent:  (1, emb_dim)
    Returns (rollout_pca list, action_seq numpy)
    """
    total_action_dim = action_dim   # = 10
    actions = torch.zeros(1, rollout_steps, total_action_dim,
                        device=device, requires_grad=True)
    optimizer = torch.optim.Adam([actions], lr=lr)

    obs_0 = {"visual": start_img_t, "proprio": start_proprio}  # (1, 1, C, H, W)

    act_mean = preprocessor.action_mean.to(device).repeat(frameskip)  # (10,)
    act_std  = preprocessor.action_std.to(device).repeat(frameskip)   # (10,)
    # print(f"action_mean shape: {act_mean.shape}")
    # print(f"action std shape: {act_std.shape}")

    for step in range(n_optim_steps):
        optimizer.zero_grad()

        # normalise actions before passing to WM
        # acts_norm = preprocessor.normalize_actions(actions)
        acts_norm = (actions - act_mean) / (act_std + 1e-8)

        # rollout: returns z_obses dict and z
        z_obses, _ = wm.rollout(obs_0, acts_norm)

        # z_obses["visual"]: (1, T, patches, D) or (1, T, D)
        z_traj = z_obses["visual"]
        if z_traj.ndim == 4:
            z_traj = aggregate_visual(wm, z_traj)   # (1, T, agg_out_dim)
        z_traj = z_traj.squeeze(0)        # (T, D)
        z_final = z_traj[-1:]             # (1, D)

        # goal loss: MSE against the goal latent, matching this codebase's
        # validated planning objective (planning/objectives.py:objective_fn_last,
        # used by gd.py/mpc.py) — cosine similarity is magnitude-blind and lets
        # the optimiser "solve" the loss by pointing the right direction
        # without actually landing on the goal embedding.
        goal_loss = F.mse_loss(z_final, goal_latent)

        # safety loss: penalise SDF < margin along trajectory (differentiable
        # w.r.t. actions, since to_pca_torch replaces the numpy sklearn calls)
        safety_loss = torch.tensor(0.0, device=device)
        if not no_pca and scaler is not None:
            p_t   = to_pca_torch(z_traj, scaler, ipca, no_pca, device)
            sdf_v = sdf(p_t).squeeze(-1)
            safety_loss = F.relu(margin - sdf_v).mean()

        loss = goal_loss + safety_weight * safety_loss
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"  Optim {step+1}/{n_optim_steps}  "
                  f"goal={goal_loss.item():.4f}  "
                  f"safety={safety_loss.item():.4f}")

    # Extract final trajectory
    with torch.no_grad():
        # acts_norm = preprocessor.normalize_actions(actions)
        acts_norm = (actions - act_mean) / (act_std + 1e-8)
        z_obses, _ = wm.rollout(obs_0, acts_norm)
        z_traj = z_obses["visual"]
        if z_traj.ndim == 4:
            z_traj = aggregate_visual(wm, z_traj)
        z_traj = z_traj.squeeze(0).cpu().numpy()  # (T, D)

    rollout_pca = list(to_pca(z_traj, scaler, ipca, no_pca))
    return rollout_pca, actions.detach().cpu().numpy().squeeze(0)


# Finding actions for states planned

def recover_actions(
    wm,
    start_img_t,
    start_proprio,
    latent_path,
    scaler,
    ipca,
    no_pca,
    action_dim,
    frameskip,
    preprocessor,
    device,
    n_iters=300,
    lr=1e-2,
    max_waypoints=30,
):
    """
    Recover an action sequence that reproduces `latent_path` (waypoint 0 is
    the start state) by rolling the world model forward from
    (start_img_t, start_proprio) and gradient-descending on the actions so
    the rollout matches every waypoint.

    latent_path waypoints may live in PCA space (they are lifted back to the
    pooled visual-latent space wm.rollout operates in before optimisation).

    latent_path can be much longer than the WM's own rollout horizon (e.g. an
    RRT path with a fine step size can have hundreds of waypoints). wm.rollout
    unrolls one full predictor forward pass per waypoint and keeps every
    activation for a single backward() call, so optimising against all of
    them at once is not memory-feasible (this previously caused a CUDA OOM
    for RRT paths). Waypoints are evenly subsampled — always keeping the
    first and last — down to at most `max_waypoints` before optimising;
    matching every dense RRT micro-step is not meaningful anyway since RRT's
    step size is far finer than the WM's actual dynamics granularity.

    Returns
    -------
    actions      : (min(len(latent_path), max_waypoints)-1, action_dim) numpy
    target_path  : the (possibly subsampled) latent_path actually optimised
                   against — pass this to plot_rrt_in_state_space alongside
                   a replay of `actions` for a fair, index-aligned comparison.
    """

    wm.eval()

    latent_path = list(latent_path)
    if max_waypoints is not None and len(latent_path) > max_waypoints:
        idx = np.unique(np.linspace(0, len(latent_path) - 1, max_waypoints).round().astype(int))
        print(f"  recover_actions: subsampling {len(latent_path)} waypoints -> {len(idx)}")
        latent_path = [latent_path[i] for i in idx]

    # lift every waypoint back to the pooled visual-latent space
    targets = []
    for p in latent_path:
        p = np.asarray(p, dtype="float32").reshape(1, -1)
        if no_pca or scaler is None:
            full = p
        else:
            full = scaler.inverse_transform(ipca.inverse_transform(p))
        targets.append(full.reshape(-1))
    targets = torch.tensor(np.stack(targets), dtype=torch.float32, device=device)  # (N, D)

    n_steps = targets.shape[0] - 1
    actions = torch.zeros(1, n_steps, action_dim, device=device, requires_grad=True)
    optimiser = torch.optim.Adam([actions], lr=lr)

    obs_0 = {"visual": start_img_t, "proprio": start_proprio}
    act_mean = preprocessor.action_mean.to(device).repeat(frameskip)
    act_std  = preprocessor.action_std.to(device).repeat(frameskip)

    for step in range(n_iters):
        optimiser.zero_grad()

        acts_norm = (actions - act_mean) / (act_std + 1e-8)
        z_obses, _ = wm.rollout(obs_0, acts_norm)
        z_traj = z_obses["visual"]
        if z_traj.ndim == 4:
            z_traj = aggregate_visual(wm, z_traj)
        z_traj = z_traj.squeeze(0)  # (n_steps+1, D)

        loss = F.mse_loss(z_traj, targets)
        loss.backward()
        optimiser.step()

        if (step + 1) % 50 == 0:
            print(f"  recover_actions {step+1}/{n_iters}  loss={loss.item():.6f}")

    return actions.detach().cpu().numpy().squeeze(0), latent_path  # (n_steps, action_dim)


def build_wall_env(states_path, episode_idx, device="cpu"):
    """
    Construct a WallEnvWrapper configured with episode_idx's *actual*
    door/wall layout (door_locations.pth / wall_locations.pth, siblings of
    states_path — same convention as datasets/wall_dset.py's WallDataset
    and plan.py's env.update_env(env_info)), so ground-truth replay happens
    in the same maze the episode was recorded in, not the wrapper's default.
    """
    from env.wall.wall_env_wrapper import WallEnvWrapper

    data_path = Path(states_path).parent
    door_locations = torch.load(data_path / "door_locations.pth", map_location="cpu", weights_only=False)
    wall_locations = torch.load(data_path / "wall_locations.pth", map_location="cpu", weights_only=False)
    fix_door_location = door_locations[episode_idx][0].item()
    fix_wall_location = wall_locations[episode_idx][0].item()

    env = WallEnvWrapper(
        fix_wall=True,
        fix_wall_location=fix_wall_location,
        fix_door_location=fix_door_location,
        device=device,
    )
    # belt-and-suspenders: also apply via update_env, matching plan.py's
    # pattern, in case __init__ alone doesn't regenerate wall geometry.
    env.update_env({
        "fix_door_location": torch.tensor(fix_door_location),
        "fix_wall_location": torch.tensor(fix_wall_location),
    })
    return env


def replay_actions(env, seed, init_state, actions_np, frameskip, action_dim):
    """
    Replay a recovered *macro* action sequence — (T, frameskip*action_dim),
    the WM's own per-step action bundling (see rearrange("b (t f) d -> b t
    (f d)") in plan.py) — through the actual wall simulator, not the world
    model. This is ground truth: env.step()'s transition is a deterministic
    function of (position, action) (see DotWall._generate_transition), so
    unlike a WM rollout this shows what the recovered actions really do,
    including wall-collision clamping the WM's smooth latent dynamics can't
    represent.

    actions_np: (T, frameskip*action_dim) numpy, raw (unnormalised) macro
                actions, as returned by optimise_rollout/recover_actions.
    Returns the real (T*frameskip + 1, state_dim) ground-truth state
    trajectory (including the start state) — already in state-space
    coordinates, so pass it to plot_rrt_in_state_space via
    gt_trajectories=, not trajectories= (no encoder/NN lookup needed).
    """
    T = actions_np.shape[0]
    raw_actions = actions_np.reshape(T, frameskip, action_dim).reshape(T * frameskip, action_dim)
    _, states = env.rollout(seed, np.asarray(init_state), raw_actions)
    return states  # (T*frameskip + 1, state_dim)


# ── Feasibility check ──────────────────────────────────────────────────────

def check_feasibility(rrt_path, pred_path):
    rrt_arr  = np.stack(rrt_path)
    pred_arr = np.stack(pred_path)
    nn_dists = [np.linalg.norm(pred_arr - wp, axis=1).min() for wp in rrt_arr]
    print(f"\nFeasibility — RRT waypoints → nearest WM state:")
    print(f"  Mean NN dist: {np.mean(nn_dists):.4f}")
    print(f"  Max  NN dist: {np.max(nn_dists):.4f}")
    return nn_dists


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_all(rrt_path, pred_path, x_start, x_goal, sdf, device, out_dir, pca_dim):
    os.makedirs(out_dir, exist_ok=True)
    rrt_arr  = np.stack(rrt_path)
    pred_arr = np.stack(pred_path)

    rrt_sdf  = from_pca_sdf(rrt_arr,  sdf, device)
    pred_sdf = from_pca_sdf(pred_arr, sdf, device)

    # SDF along path
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rrt_sdf,  label="RRT",       color="#2196F3")
    ax.plot(pred_sdf, label="WM rollout", color="#F44336")
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Safety boundary")
    ax.set_xlabel("Waypoint"); ax.set_ylabel("SDF value")
    ax.set_title("SDF values along planned paths")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sdf_along_paths.png"), dpi=150)
    plt.close()

    # 2D scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rrt_arr[:, 0],  rrt_arr[:, 1],  "-o", ms=3, color="#2196F3",
            label="RRT", alpha=0.8)
    ax.plot(pred_arr[:, 0], pred_arr[:, 1], "-o", ms=3, color="#F44336",
            label="WM rollout", alpha=0.8)
    ax.scatter(*x_start[:2], s=150, marker="*", color="green",  zorder=5, label="Start")
    ax.scatter(*x_goal[:2],  s=150, marker="*", color="purple", zorder=5, label="Goal")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title("Paths in PCA latent space"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paths_2d.png"), dpi=150)
    plt.close()

    # 3D scatter
    if pca_dim >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(8, 7))
        ax  = fig.add_subplot(111, projection="3d")
        ax.plot(rrt_arr[:, 0],  rrt_arr[:, 1],  rrt_arr[:, 2],
                "-o", ms=3, color="#2196F3", label="RRT", alpha=0.8)
        ax.plot(pred_arr[:, 0], pred_arr[:, 1], pred_arr[:, 2],
                "-o", ms=3, color="#F44336", label="WM rollout", alpha=0.8)
        ax.scatter(*x_start[:3], s=150, marker="*", color="green",  label="Start")
        ax.scatter(*x_goal[:3],  s=150, marker="*", color="purple", label="Goal")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_zlabel("PC 3")
        ax.set_title("3D latent paths"); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "paths_3d.png"), dpi=150)
        plt.close()

    # Feasibility
    nn_dists = check_feasibility(rrt_path, pred_path)
    fig, ax  = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(nn_dists)), nn_dists, color="#4c72b0")
    ax.set_xlabel("RRT waypoint"); ax.set_ylabel("Dist to nearest WM state")
    ax.set_title("RRT feasibility (lower = more reachable by world model)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feasibility.png"), dpi=150)
    plt.close()

    print(f"\nAll plots saved to {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args   = get_args()
    device = get_device(args.cpu)
    os.makedirs(args.out_dir, exist_ok=True)

    setup_ts_paths(args.ts_repo, args.model_path)

    # ── Load SDF ──────────────────────────────────────────────────────────
    print("Loading SDF...")
    _sdf_raw = load_sdf_model(args.sdf_model, device)
    _sdf_raw.eval()
    # This model's HKR training convention has X_train_in ("safe") scoring
    # negative and X_train_out ("unsafe") scoring positive — the opposite of
    # what rrt()/edge_safe()/optimise_rollout assume (SDF > margin = safe).
    # Negate once here so every downstream caller sees the expected sign.
    sdf = lambda x: -_sdf_raw(x)

    with open(os.path.join(args.run_dir, "pca_pipeline.pkl"), "rb") as f:
        pca_data = pickle.load(f)
    scaler = pca_data["scaler"]
    ipca   = pca_data["ipca"]
    no_pca = pca_data.get("no_pca", False)

    # ── Load episode data ─────────────────────────────────────────────────
    if not args.wall_obses_dir:
        raise ValueError("Provide --wall-obses-dir and --wall-states-path")

    start_img_t, goal_img_t, start_state, goal_state, ep_states, imgs = \
        load_wall_episode(
            obses_dir   = args.wall_obses_dir,
            states_path = args.wall_states_path,
            episode_idx = args.wall_episode_idx,
            img_size    = args.img_size,
            device      = device,
        )

    # ── Encode start/goal ─────────────────────────────────────────────────
    if args.encoder == "gt-state":
        wm      = None
        no_pca  = True
        pca_dim = 2
        scaler  = None
        ipca    = None
        x_start = start_state.copy()   # (2,)
        x_goal  = goal_state.copy()    # (2,)

    else:
        print("Loading TS world model...")
        wm, train_cfg = load_ts_model(args.model_path, args.model_epoch, device)
        preprocessor, dset = load_preprocessor(args.model_path, train_cfg, device)
        action_dim = dset.action_dim * train_cfg.frameskip
        print(f"action_dim={dset.action_dim}, frameskip={train_cfg.frameskip}, total={action_dim}")
        pca_dim    = wm.encoder.emb_dim if no_pca else ipca.n_components_

        # ground-truth simulator, configured to this episode's actual
        # door/wall layout — used by replay_actions() below instead of the
        # world model, so replay reflects real (noise-free-given-actions,
        # but wall-collision-aware) dynamics rather than the WM's estimate.
        wall_env = build_wall_env(args.wall_states_path, args.wall_episode_idx, device="cpu")
        sim_seed = args.wall_episode_idx

        # proprio must be (B, T, D) for the encoder, normalised the same way
        # the dataset normalises it before training (raw world coords in,
        # (x - proprio_mean) / proprio_std out) — the encoder never saw raw
        # coordinates at train time.
        def make_proprio(state_np):
            raw = torch.from_numpy(state_np).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
            return preprocessor.normalize_proprios(raw).to(device)

        # start/goal images are only resized + scaled to [0, 1] so far; the
        # encoder (DinoV2Encoder.forward) applies no normalisation of its own,
        # so we still need the dataset's photometric transform
        # (Resize/CenterCrop/Normalize(0.5,0.5,0.5)) that training images went
        # through, otherwise the encoder sees out-of-distribution inputs.
        start_img_t = preprocessor.transform(start_img_t)
        goal_img_t  = preprocessor.transform(goal_img_t)

        ep_latents = encode_episode_latents(wm, imgs, ep_states, preprocessor, device)
        ep_pca = to_pca(ep_latents, scaler, ipca, no_pca)
        plot_rrt_in_state_space(
            ep_pca,
            run_dir=args.run_dir,
            out_dir=args.out_dir,
            save_path=os.path.join(args.out_dir, "original_episode_path.png"),
            labels=["original episode (encoded, projected)"],
            gt_trajectories=[ep_states],
            gt_labels=["original episode (ground truth)"],
        )

        z_start = img_to_latent(wm, start_img_t, make_proprio(start_state))
        z_goal  = img_to_latent(wm, goal_img_t,  make_proprio(goal_state))
        x_start = to_pca(z_start.cpu().numpy(), scaler, ipca, no_pca).squeeze(0)
        x_goal  = to_pca(z_goal.cpu().numpy(),  scaler, ipca, no_pca).squeeze(0)

    print(f"Start SDF: {from_pca_sdf(x_start[None], sdf, device)[0]:.4f}")
    print(f"Goal  SDF: {from_pca_sdf(x_goal[None],  sdf, device)[0]:.4f}")

    # ── RRT bounds ────────────────────────────────────────────────────────
    X_all  = np.vstack([
        np.load(os.path.join(args.run_dir, "X_train_in.npy"),  mmap_mode="r"),
        np.load(os.path.join(args.run_dir, "X_train_out.npy"), mmap_mode="r"),
    ])
    pad    = (X_all.max(0) - X_all.min(0)) * 0.2
    bounds = np.stack([X_all.min(0) - pad, X_all.max(0) + pad], axis=1)

    # ── RRT ───────────────────────────────────────────────────────────────
    print(f"\nRunning RRT (iters={args.rrt_iters}, step={args.rrt_step})...")
    rrt_path, _ = rrt(
        x_start, x_goal, sdf, device, bounds,
        n_iters     = args.rrt_iters,
        step_size   = args.rrt_step,
        goal_radius = args.goal_radius,
        margin      = args.sdf_margin,
    )
    rrt_sdf = from_pca_sdf(np.stack(rrt_path), sdf, device)
    print(f"RRT path: {len(rrt_path)} waypoints  "
          f"SDF min={rrt_sdf.min():.4f}  mean={rrt_sdf.mean():.4f}")
    
    rrt_actions, rrt_target_path = recover_actions(
        wm, start_img_t, make_proprio(start_state), rrt_path,
        scaler, ipca, no_pca, action_dim, train_cfg.frameskip,
        preprocessor, device,
    )
    print("Actions to get to goal from rrt states: ", rrt_actions)

    # validate: replay the recovered actions through the real simulator (not
    # the WM) and compare against the (subsampled) targets they were
    # optimised to reach — this is ground truth, not a WM self-consistency
    # check.
    rrt_replayed_states = replay_actions(
        wall_env, sim_seed, start_state, rrt_actions,
        train_cfg.frameskip, dset.action_dim,
    )
    traj_states = plot_rrt_in_state_space(
        [rrt_path, rrt_target_path],
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        filename="rrt_state_space.png",
        labels=["RRT path (full)", "RRT target (subsampled)"],
        gt_trajectories=[rrt_replayed_states],
        gt_labels=["RRT replayed (ground-truth sim)"],
        k=5)

    # ── WM rollout ────────────────────────────────────────────────────────
    action_seq = None
    if args.encoder == "gt-state":
        print("\nSkipping WM rollout (gt-state encoder — no world model).")
        rollout_pca = [x_start, x_goal]
    else:
        print(f"\nOptimising WM rollout ({args.rollout_steps} steps)...")
        rollout_pca, action_seq = optimise_rollout(
            wm            = wm,
            start_img_t   = start_img_t,
            start_proprio  = make_proprio(start_state),
            goal_latent   = z_goal,
            sdf           = sdf,
            scaler        = scaler,
            ipca          = ipca,
            no_pca        = no_pca,
            device        = device,
            rollout_steps = args.rollout_steps,
            action_dim    = action_dim,
            frameskip     = train_cfg.frameskip,
            n_optim_steps = args.optim_steps,
            lr            = args.optim_lr,
            safety_weight = args.safety_weight,
            preprocessor  = preprocessor,
            margin        = args.sdf_margin,
        )
        # action_seq (from optimise_rollout) already is the recovered action
        # sequence for rollout_pca — no need to re-derive it. Replay it
        # through the real simulator (ground truth) as a check that
        # rollout_pca's plan actually does what the WM thinks it does.
        print("Actions to get to goal from wm states: ", action_seq)
        wm_replayed_states = replay_actions(
            wall_env, sim_seed, start_state, action_seq,
            train_cfg.frameskip, dset.action_dim,
        )
        traj_states = plot_rrt_in_state_space(
            [rollout_pca],
            run_dir=args.run_dir,
            out_dir=args.out_dir,
            filename="wm_state_space.png",
            labels=["WM optimised rollout"],
            gt_trajectories=[wm_replayed_states],
            gt_labels=["WM replayed (ground-truth sim)"],
            k=5)

    pred_sdf = from_pca_sdf(np.stack(rollout_pca), sdf, device)
    print(f"Path: {len(rollout_pca)} steps  "
          f"SDF min={pred_sdf.min():.4f}  mean={pred_sdf.mean():.4f}")
    print(f"\nGoal distance (latent space):")
    print(f"  RRT end: {np.linalg.norm(rrt_path[-1]    - x_goal):.4f}")
    print(f"  WM end:  {np.linalg.norm(rollout_pca[-1] - x_goal):.4f}")

    # ── Save + plot ───────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, "rrt_path.npy"),       np.stack(rrt_path))
    np.save(os.path.join(args.out_dir, "predictor_path.npy"), np.stack(rollout_pca))
    if action_seq is not None:
        np.save(os.path.join(args.out_dir, "action_seq.npy"), action_seq)

    plot_all(rrt_path, rollout_pca, x_start, x_goal,
             sdf, device, args.out_dir, pca_dim)