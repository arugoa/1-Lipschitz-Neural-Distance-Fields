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
    parser.add_argument("--npz",       type=str, default=None,
                    help="Path to .npz episode — uses first frame as start, last as goal")
    parser.add_argument("--start-img", type=str, default=None,
                        help="Path to start image (overrides --npz)")
    parser.add_argument("--goal-img",  type=str, default=None,
                        help="Path to goal image (overrides --npz)")
    parser.add_argument("--img-size",    type=int, default=128)

    # SDF
    parser.add_argument("--run-dir",     type=str, required=True)
    parser.add_argument("--sdf-model",   type=str, required=True)

    # RRT
    parser.add_argument("--rrt-iters",   type=int,   default=5000)
    parser.add_argument("--rrt-step",    type=float, default=0.05)
    parser.add_argument("--sdf-margin",  type=float, default=0.0)
    parser.add_argument("--goal-radius", type=float, default=0.1)

    # WM rollout optimisation
    parser.add_argument("--rollout-steps", type=int,   default=30)
    parser.add_argument("--optim-steps",   type=int,   default=300)
    parser.add_argument("--optim-lr",      type=float, default=1e-2)
    parser.add_argument("--safety-weight", type=float, default=1.0)
    parser.add_argument("--frameskip",     type=int,   default=1)

    parser.add_argument("--out-dir", type=str, default="output/planning")
    parser.add_argument("-cpu", action="store_true")
    return parser.parse_args()


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


def img_to_latent(wm, img_t):
    """
    img_t: (1, 1, C, H, W)
    Returns z: (1, emb_dim) — CLS or mean-pooled patch token
    """
    obs = {"visual": img_t}
    with torch.no_grad():
        z_dict = wm.encode_obs(obs)
    # z_dict is typically {"visual": (1, 1, num_patches, emb_dim)} or (1, 1, emb_dim)
    z = z_dict["visual"]
    # collapse T and patch dims
    if z.ndim == 4:          # (B, T, patches, D)
        z = z.mean(dim=2)    # (B, T, D)
    z = z.squeeze(1)         # (B, D)
    return z                 # (1, emb_dim)


def load_start_goal_from_npz(npz_path, img_size):
    """
    Load first and last frames from an npz episode.
    Returns (start_np, goal_np) both (H, W, C) uint8.
    """
    file    = np.load(npz_path, allow_pickle=True)
    images  = file["image"]          # (T, H, W, C)
    start_np = images[0]              # first frame
    goal_np  = images[-1]             # last frame

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
    return start_np, goal_np


def np_to_img_tensor(arr_np, device):
    """(H, W, C) uint8 numpy → (1, 1, C, H, W) float tensor"""
    t = torch.from_numpy(arr_np).float().permute(2, 0, 1)  # (C, H, W)
    return t.unsqueeze(0).unsqueeze(0).to(device)           # (1, 1, C, H, W)

# ── PCA helpers ───────────────────────────────────────────────────────────

def to_pca(z_np, scaler, ipca, no_pca):
    if no_pca or scaler is None:
        return z_np
    return ipca.transform(scaler.transform(z_np))


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


def edge_safe(sdf, x_a, x_b, device, margin, n=10):
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
    wm, start_img_t, goal_latent, sdf,
    scaler, ipca, no_pca, device,
    rollout_steps, action_dim, frameskip,
    n_optim_steps, lr, safety_weight,
    preprocessor,
):
    """
    Optimise an action sequence so wm.rollout reaches goal_latent.
    start_img_t:  (1, 1, C, H, W)
    goal_latent:  (1, emb_dim)
    Returns (rollout_pca list, action_seq numpy)
    """
    total_action_dim = action_dim * frameskip
    actions = torch.zeros(1, rollout_steps, total_action_dim,
                          device=device, requires_grad=True)
    optimizer = torch.optim.Adam([actions], lr=lr)

    obs_0 = {"visual": start_img_t}  # (1, 1, C, H, W)

    for step in range(n_optim_steps):
        optimizer.zero_grad()

        # normalise actions before passing to WM
        acts_norm = preprocessor.normalize_actions(actions)

        # rollout: returns z_obses dict and z
        z_obses, _ = wm.rollout(obs_0, acts_norm)

        # z_obses["visual"]: (1, T, patches, D) or (1, T, D)
        z_traj = z_obses["visual"]
        if z_traj.ndim == 4:
            z_traj = z_traj.mean(dim=2)   # (1, T, D)
        z_traj = z_traj.squeeze(0)        # (T, D)
        z_final = z_traj[-1:]             # (1, D)

        # goal loss: cosine distance in full latent space
        goal_loss = (1.0 - F.cosine_similarity(z_final, goal_latent, dim=-1)).mean()

        # safety loss: penalise SDF < margin along trajectory
        safety_loss = torch.tensor(0.0, device=device)
        if not no_pca and scaler is not None:
            z_np  = z_traj.detach().cpu().numpy()
            p_np  = ipca.transform(scaler.transform(z_np))
            p_t   = torch.from_numpy(p_np).float().to(device)
            with torch.no_grad():
                sdf_v = sdf(p_t).squeeze(-1)
            safety_loss = F.relu(-sdf_v + 0.0).mean()

        loss = goal_loss + safety_weight * safety_loss
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"  Optim {step+1}/{n_optim_steps}  "
                  f"goal={goal_loss.item():.4f}  "
                  f"safety={safety_loss.item():.4f}")

    # Extract final trajectory
    with torch.no_grad():
        acts_norm = preprocessor.normalize_actions(actions)
        z_obses, _ = wm.rollout(obs_0, acts_norm)
        z_traj = z_obses["visual"]
        if z_traj.ndim == 4:
            z_traj = z_traj.mean(dim=2)
        z_traj = z_traj.squeeze(0).cpu().numpy()  # (T, D)

    rollout_pca = list(to_pca(z_traj, scaler, ipca, no_pca))
    return rollout_pca, actions.detach().cpu().numpy().squeeze(0)


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

    # ── Setup paths ───────────────────────────────────────────────────────
    setup_ts_paths(args.ts_repo, args.model_path)

    # ── Load TS world model ───────────────────────────────────────────────
    print("Loading TS world model...")
    wm, train_cfg = load_ts_model(args.model_path, args.model_epoch, device)
    preprocessor, dset = load_preprocessor(args.model_path, train_cfg, device)
    action_dim = dset.action_dim

    # ── Load SDF + PCA pipeline ───────────────────────────────────────────
    print("Loading SDF...")
    sdf = load_sdf_model(args.sdf_model, device)
    sdf.eval()

    with open(os.path.join(args.run_dir, "pca_pipeline.pkl"), "rb") as f:
        pca_data = pickle.load(f)
    scaler = pca_data["scaler"]
    ipca   = pca_data["ipca"]
    no_pca = pca_data.get("no_pca", False)
    pca_dim = (wm.encoder.emb_dim if no_pca
               else ipca.n_components_)

    # ── Encode start and goal ─────────────────────────────────────────────
    print("Encoding start and goal...")
    if args.npz:
        start_np, goal_np = load_start_goal_from_npz(args.npz, args.img_size)
        start_img_t = np_to_img_tensor(start_np, device)
        goal_img_t  = np_to_img_tensor(goal_np,  device)
    elif args.start_img and args.goal_img:
        start_img_t = load_img_tensor(args.start_img, args.img_size,
                                    preprocessor.transform, device)
        goal_img_t  = load_img_tensor(args.goal_img,  args.img_size,
                                    preprocessor.transform, device)
    else:
        raise ValueError("Provide either --npz or both --start-img and --goal-img")

    z_start = img_to_latent(wm, start_img_t)   # (1, emb_dim)
    z_goal  = img_to_latent(wm, goal_img_t)    # (1, emb_dim)

    x_start = to_pca(z_start.cpu().numpy(), scaler, ipca, no_pca).squeeze(0)
    x_goal  = to_pca(z_goal.cpu().numpy(),  scaler, ipca, no_pca).squeeze(0)

    print(f"Start SDF: {from_pca_sdf(x_start[None], sdf, device)[0]:.4f}")
    print(f"Goal  SDF: {from_pca_sdf(x_goal[None],  sdf, device)[0]:.4f}")

    # ── RRT bounds from training data ─────────────────────────────────────
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
        n_iters=args.rrt_iters,
        step_size=args.rrt_step,
        goal_radius=args.goal_radius,
        margin=args.sdf_margin,
    )
    rrt_sdf = from_pca_sdf(np.stack(rrt_path), sdf, device)
    print(f"RRT path: {len(rrt_path)} waypoints  "
          f"SDF min={rrt_sdf.min():.4f}  mean={rrt_sdf.mean():.4f}")

    # ── WM rollout ────────────────────────────────────────────────────────
    print(f"\nOptimising WM rollout ({args.rollout_steps} steps)...")
    rollout_pca, action_seq = optimise_rollout(
        wm            = wm,
        start_img_t   = start_img_t,
        goal_latent   = z_goal,
        sdf           = sdf,
        scaler        = scaler,
        ipca          = ipca,
        no_pca        = no_pca,
        device        = device,
        rollout_steps = args.rollout_steps,
        action_dim    = action_dim,
        frameskip     = args.frameskip,
        n_optim_steps = args.optim_steps,
        lr            = args.optim_lr,
        safety_weight = args.safety_weight,
        preprocessor  = preprocessor,
    )
    pred_sdf = from_pca_sdf(np.stack(rollout_pca), sdf, device)
    print(f"WM path: {len(rollout_pca)} steps  "
          f"SDF min={pred_sdf.min():.4f}  mean={pred_sdf.mean():.4f}")

    # Goal distances
    print(f"\nGoal distance (PCA space):")
    print(f"  RRT end:    {np.linalg.norm(rrt_path[-1]    - x_goal):.4f}")
    print(f"  WM end:     {np.linalg.norm(rollout_pca[-1] - x_goal):.4f}")

    # ── Save + plot ───────────────────────────────────────────────────────
    np.save(os.path.join(args.out_dir, "rrt_path.npy"),       np.stack(rrt_path))
    np.save(os.path.join(args.out_dir, "predictor_path.npy"), np.stack(rollout_pca))
    np.save(os.path.join(args.out_dir, "action_seq.npy"),     action_seq)

    plot_all(rrt_path, rollout_pca, x_start, x_goal,
             sdf, device, args.out_dir, pca_dim)