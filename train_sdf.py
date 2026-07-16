"""
Unified SDF training script.

Usage examples:
    python train_sdf.py ../dataset-good/ --encoder cjepa --pca-dims 3
    python train_sdf.py ../dataset-good/ --encoder dreamer --pca-dims 5 --dreamer-ckpt path/to/ckpt
    python train_sdf.py ../dataset-good/ --encoder autoencoder --autoencoder-ckpt path/to/ckpt
    python train_sdf.py ../dataset-good/ --encoder autoencoder --autoencoder-ckpt path/to/ckpt --no-pca
"""

import os
import sys
import glob
import argparse
import pickle
from types import SimpleNamespace
import hydra
from hydra import compose, initialize

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
    parser.add_argument("--dataset-mode", choices=["npz", "ts", "wall"], default="npz",)
    parser.add_argument("-o", "--output-name", type=str, default="output")
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("--wall-obses-dir",  type=str, default=None,
                    help="Folder of per-episode image .pth files")
    parser.add_argument("--wall-states",     type=str, default=None)
    parser.add_argument("--wall-locs",       type=str, default=None)
    parser.add_argument("--door-locs",       type=str, default=None)
    # parser.add_argument("--ball-radius",     type=float, default=1.0)
    # parser.add_argument("--door-half-w",     type=float, default=4.0)
    # parser.add_argument("--wall-width",      type=float, default=4.0,
    #                     help="Thickness of the middle wall in world units")
    # parser.add_argument("--border-wall-loc", type=float, default=2.0,
    #                     help="Thickness of the border walls in world units")
    # parser.add_argument("--env-size",        type=float, default=64.0)
    parser.add_argument("--wall-config", type=str, default=None,
                        help="Path to wall_config.pkl — loads all geometry automatically")

    # encoder selection
    parser.add_argument("--encoder", choices=["cjepa", "dreamer", "autoencoder", "lewm", "ts", "gt_state"],
                        default="cjepa", help="Which encoder to use")
    parser.add_argument("--state-key", type=str, default="state",
                    help="Key for ground-truth state in npz files (gt-state encoder only)")
    parser.add_argument("--cjepa-ckpt", type=str, default="clevrer_savi_model.pth")
    parser.add_argument("--dreamer-ckpt", type=str, default=None)
    parser.add_argument("--dreamer-configs", type=str, default="../configs.yaml")
    parser.add_argument("--autoencoder-ckpt", type=str, default=None)
    parser.add_argument("--lewm-ckpt", type=str, default=None)
    parser.add_argument("--ts-ckpt", type=str, default=None)

    parser.add_argument("--ts-img-size", type=int, default=224)
    parser.add_argument("--ts-config", type=str, default=None,)
    parser.add_argument("--num-hist", type=int, default=1)
    parser.add_argument("--num-pred", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--split", choices=["train", "valid"], default="train",)

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
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        dones = np.zeros(len(imgs), dtype=np.int32)
        return {"image": imgs, "dones": dones, "states": state}


import pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def load_wall_config(config_path):
    # don't unpickle manually — instantiate via hydra the same way train.py does
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import hydra

    # find the conf dir relative to the project root
    conf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../conf"))
    
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name="train")

    print(f"  loaded hydra config: env={cfg.env.name}")
    return cfg


def check_vertical_wall_intersect(pos1, pos2, wall_x, hole_y, door_space):
    check_intersection = (
        torch.sign(pos1[0] - wall_x) * torch.sign(pos2[0] - wall_x)
    ) <= 0.1
    if check_intersection:
        # print("found intersection at", i, j.item())
        d = pos2 - pos1
        # a and b are the line parameters fit to the last step
        a = d[1] / d[0]
        b = pos1[1] - a * pos1[0]
        # y is the intersection point of the wall plane
        y = a * wall_x + b
        # If the intersection point is in the hole, we are good
        # otherwise, we need to move the point back
        if (
            hole_y is None or y < hole_y - door_space or y > hole_y + door_space
        ):  # we're not in the hole
            return torch.tensor([wall_x, y]).to(pos1.device)  # we intersect
        else:
            return None  # we are in the hole and we overlap
    return None


def check_horizontal_wall_intersect(pos1, pos2, wall_y, hole_x, door_space):
    check_intersection = (
        torch.sign(pos1[1] - wall_y) * torch.sign(pos2[1] - wall_y)
    ) <= 0.1
    if check_intersection:
        d = pos2 - pos1
        a = d[1] / d[0]
        b = pos1[1] - a * pos1[0]
        x = (wall_y - b) / a
        if (
            hole_x is None or x < hole_x - door_space or x > hole_x + door_space
        ):  # we're not in the hole
            return torch.tensor([x, wall_y]).to(pos1.device)  # we intersect
        else:
            return None  # we are in the hole and we overlap
    return None


def check_wall_intersect(
    pos1,
    pos2,
    wall_x,
    hole_y,
    wall_width,
    door_space,
    border_wall_loc,
    img_size,
    add_noise=True,
):
    """
    Parameters:
        pos1: [2]
        pos2: [2]
        wall_x: []
        hole_y: []
        wall_width: int
        door_space: int
        border_wall_loc: int
        img_size: int
    Returns:
        intersect: [2]
        intersect_w_noise: [2]
    """

    # first, we check to see if the point bumps into the middle wall's width
    left_wall_corner, right_wall_corner = (
        wall_x - wall_width // 2,
        wall_x + wall_width // 2,
    )
    door_bot, door_top = hole_y - door_space, hole_y + door_space

    # check if it's moving upwards and crosses the door top horizontal line
    if pos2[1] - pos1[1] > 0 and pos2[1] > door_top and pos1[1] < door_top:
        # get x intercept with door top horizontal line
        intersect = check_horizontal_wall_intersect(
            pos1, pos2, door_top, None, door_space
        )
        # if x intercept occurs between the left and right wall
        if (
            intersect is not None
            and left_wall_corner <= intersect[0]
            and intersect[0] <= right_wall_corner
        ):
            # add downward noise and return early
            # noise = torch.randn(2, device=pos1.device) * 0.5
            noise = torch.ones(2, device=pos1.device) * 0.5
            noise[1] = noise[1].abs() * -1
            return intersect, intersect + noise

    # check if it's moving downwards and croseses the door bot horizontal line
    if pos2[1] - pos1[1] < 0 and pos2[1] < door_bot and pos1[1] > door_bot:
        # get x intercept with door bot horizontal line
        intersect = check_horizontal_wall_intersect(
            pos1, pos2, door_bot, None, door_space
        )
        # if x intercept occurs between the left and right wall
        if (
            intersect is not None
            and left_wall_corner <= intersect[0]
            and intersect[0] <= right_wall_corner
        ):
            # add upward noise and return early
            # noise = torch.randn(2, device=pos1.device) * 0.5
            noise = torch.ones(2, device=pos1.device) * 0.5
            noise[1] = noise[1].abs()
            return intersect, intersect + noise

    # next, we check to see if point bumps into border and wall proper
    left_wall, left_hole = border_wall_loc - 1, None
    right_wall, right_hole = img_size - border_wall_loc, None
    if wall_x > pos1[0]:
        right_wall, right_hole = wall_x - wall_width // 2, hole_y
    else:
        left_wall, left_hole = wall_x + wall_width // 2, hole_y

    top_wall, top_hole = border_wall_loc - 1, None
    bot_wall, bot_hole = img_size - border_wall_loc, None

    vertical_intersect = check_vertical_wall_intersect(
        pos1, pos2, left_wall, left_hole, door_space
    )
    if vertical_intersect is None:
        vertical_intersect = check_vertical_wall_intersect(
            pos1, pos2, right_wall, right_hole, door_space
        )

    horizontal_intersect = check_horizontal_wall_intersect(
        pos1, pos2, top_wall, top_hole, door_space
    )
    if horizontal_intersect is None:
        horizontal_intersect = check_horizontal_wall_intersect(
            pos1, pos2, bot_wall, bot_hole, door_space
        )

    if vertical_intersect is not None:
        sign = torch.sign(pos1[0] - vertical_intersect[0])
        # vertical_noise = torch.randn(2, device=pos1.device) * 0.5
        vertical_noise = torch.ones(2, device=pos1.device) * 0.5
        vertical_noise[0] = vertical_noise[0].abs() * sign

    if horizontal_intersect is not None:
        sign = torch.sign(pos1[1] - horizontal_intersect[1])
        # horizontal_noise = torch.randn(2, device=pos1.device) * 0.5
        horizontal_noise = torch.ones(2, device=pos1.device) * 0.5
        horizontal_noise[1] = horizontal_noise[1].abs() * sign

    if vertical_intersect is not None and horizontal_intersect is not None:
        # return the intersection that happens first
        if torch.norm(pos1 - vertical_intersect) < torch.norm(
            pos1 - horizontal_intersect
        ):
            intersect = vertical_intersect
            noise = vertical_noise
        else:
            intersect = horizontal_intersect
            noise = horizontal_noise
    elif vertical_intersect is not None:
        intersect = vertical_intersect
        noise = vertical_noise
    elif horizontal_intersect is not None:
        intersect = horizontal_intersect
        noise = horizontal_noise
    else:
        return None, None

    intersect_w_noise = intersect + noise
    # we make sure after adding noise, we don't cross another wall
    intersect_w_noise[0] = torch.clamp(
        intersect_w_noise[0], min=left_wall, max=right_wall
    )
    intersect_w_noise[1] = torch.clamp(intersect_w_noise[1], min=top_wall, max=bot_wall)

    if intersect_w_noise[0] <= left_wall:
        intersect_w_noise[0] = left_wall + 0.3
    if intersect_w_noise[0] >= right_wall:
        intersect_w_noise[0] = right_wall - 0.3
    if intersect_w_noise[1] <= top_wall:
        intersect_w_noise[1] = top_wall + 0.3
    if intersect_w_noise[1] >= bot_wall:
        intersect_w_noise[1] = bot_wall - 0.3

    return intersect, intersect_w_noise


def is_near_wall(pos, wall_x, hole_y, wall_width, door_space, border_wall_loc, env_size, margin=1.5):
    """
    Returns True if pos is within margin units of any solid wall surface.
    Handles middle wall (excluding door gap) and all 4 border walls.
    """
    x, y = float(pos[0]), float(pos[1])

    # border walls
    if x < border_wall_loc + margin:           return True
    if x > env_size - border_wall_loc - margin: return True
    if y < border_wall_loc + margin:           return True
    if y > env_size - border_wall_loc - margin: return True

    # middle vertical wall — only solid outside the door gap
    left_edge  = wall_x - wall_width / 2
    right_edge = wall_x + wall_width / 2
    near_wall_x = (left_edge - margin) < x < (right_edge + margin)
    in_door_gap = (hole_y - door_space) < y < (hole_y + door_space)

    if near_wall_x and not in_door_gap:        return True

    return False


def compute_dones_from_geometry(states, wall_x_arr, door_y_arr,
                                wall_width, door_space,
                                border_wall_loc, env_size, margin=1.5):
    T = len(states)
    done = np.zeros(T, dtype=np.int32)

    for t in range(T):
        pos = torch.tensor(states[t], dtype=torch.float32)

        # proximity check — catches stuck/repeated states
        if is_near_wall(pos, wall_x_arr[t], door_y_arr[t],
                        wall_width, door_space, border_wall_loc, env_size, margin):
            done[t] = 1
            continue

        # crossing check — catches the step where collision first occurs
        if t < T - 1:
            pos2 = torch.tensor(states[t+1], dtype=torch.float32)
            intersect, _ = check_wall_intersect(
                pos, pos2,
                wall_x          = float(wall_x_arr[t]),
                hole_y          = float(door_y_arr[t]),
                wall_width      = int(wall_width),
                door_space      = door_space,
                border_wall_loc = int(border_wall_loc),
                img_size        = int(env_size),
            )
            if intersect is not None:
                done[t] = 1

    return done


class WallDatasetSource:
    """Mimics a list of npz paths so get_sample/dataset_source[i] still works."""
    def __init__(self, obses_dir, states_path, wall_path, door_path,
                 ball_radius, door_half_w, env_size,  wall_width, border_wall_loc):
        import re
        self.wall_width      = wall_width
        self.border_wall_loc = border_wall_loc
        self.obses_dir   = obses_dir
        self.ball_radius = ball_radius
        self.door_half_w = door_half_w
        self.env_size    = env_size

        self.states    = torch.load(states_path,  map_location="cpu", weights_only=False).numpy()
        self.wall_locs = torch.load(wall_path,    map_location="cpu", weights_only=False).numpy()
        self.door_locs = torch.load(door_path,    map_location="cpu", weights_only=False).numpy()

        # sort episode files so index matches states row
        self.eps = sorted(glob.glob(os.path.join(obses_dir, "*.pth")),
                          key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
        assert len(self.eps) == len(self.states), \
            f"Episode file count ({len(self.eps)}) != states rows ({len(self.states)})"

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, idx):
        imgs = torch.load(self.eps[idx], map_location="cpu",
                          weights_only=False).numpy()   # (T, C, H, W) or (T, H, W, C)
        # bx      = self.states[idx, :, 0]
        # by      = self.states[idx, :, 1]
        # wall_x  = self.wall_locs[idx, :, 0]
        # door_y  = self.door_locs[idx, :, 0]
        dones = compute_dones_from_geometry(
            states          = self.states[idx],        # (T, 2)
            wall_x_arr      = self.wall_locs[idx, :, 0],
            door_y_arr      = self.door_locs[idx, :, 0],
            wall_width      = self.wall_width,
            door_space      = self.door_half_w,
            border_wall_loc = self.border_wall_loc,
            env_size        = self.env_size,
        ).astype(np.int32)

        # trim to same length in case images has an extra frame
        T = min(len(imgs), len(dones))
        return {"image": imgs[:T], "dones": dones[:T], "states": self.states[idx][:T]}


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


def transform_enc(enc_np, scaler, ipca, no_pca):
    """Apply scaler + PCA, or return raw if no_pca."""
    if no_pca:
        return enc_np.astype("float32")
    return ipca.transform(scaler.transform(enc_np)).astype("float32")

# ── Main training function ─────────────────────────────────────────────────
def get_sample(args, dataset_source, idx):
    if args.dataset_mode == "npz":
        fp = dataset_source[idx]
        file = np.load(fp, allow_pickle=True)
        imgs_np = file["images"]
        d = np.where(file["dones"] == 0, 1, -1)
        states = file["states"] if "states" in file else None
    else:
        sample = dataset_source[idx]
        imgs_np = sample["image"]
        if torch.is_tensor(imgs_np):
            imgs_np = imgs_np.cpu().numpy()
        d = np.where(sample["dones"] == 0, 1, -1)
        states = sample.get("states", None)
    return imgs_np, d, states


def run_pca_dim(args, encoder, dataset_source, device, pca_dim, config):
    """Run the full pipeline for one PCA dimension (or no PCA)."""

    # ── gt-state: bypass image encoding entirely ──────────────────────────
    if args.encoder == "gt_state":
        # peek at first episode to get state dimensionality
        _, _, s0 = get_sample(args, dataset_source, 0)
        assert s0 is not None, \
            "gt-state encoder requires states in the dataset. " \
            "Check --state-key for npz, or use --dataset-mode wall."
        state_dim = s0.shape[-1] if s0.ndim > 1 else 1
        args.no_pca = True   # states are already low-dim, skip PCA
        pca_dim     = state_dim

        def encode(imgs_np, device, states_np):
            # flatten (T, N, D) → (T, N*D) if needed, else just (T, D)
            s = np.array(states_np, dtype=np.float32)
            return s.reshape(len(s), -1)

    else:
        def encode(imgs_np, device, states_np):
            return encoder.encode(imgs_np, device)

    if args.no_pca:
        if encoder is None:
            pca_dim = 2
        else:
            pca_dim  = encoder.output_dim()
        run_name = f"{args.encoder}_nopca_{args.model}_{args.dataset_mode}"
    else:
        run_name = f"{args.encoder}_pca{pca_dim}_{args.model}"

    out_folder = os.path.join("output", args.output_name, run_name)
    os.makedirs(out_folder, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Encoder: {args.encoder}   PCA dim: {pca_dim}   no_pca: {args.no_pca}")
    print(f"  Output:  {out_folder}")
    print(f"{'='*60}")

    mm = {k: os.path.join(out_folder, f"{k}.npy") for k in
          ["X_train_in", "State_in", "State_out", "X_train_out", "X_test", "State_test", "y_test"]}
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
        for i in range(len(dataset_source)):
            _, d, _ = get_sample(args, dataset_source, i)
            n_safe_total += int((d == 1).sum())
            n_unsafe_total += int((d == -1).sum())
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
            for i in range(len(dataset_source)):
                if i % 100 == 0:
                    print(f"  PCA fit {i}/{len(dataset_source)}")
                imgs_np, _, _ = get_sample(args, dataset_source, i)
                enc_np = encode(imgs_np, device, None)
                scaler.partial_fit(enc_np)
                ipca.partial_fit(scaler.transform(enc_np))
            print(f"Explained variance: {ipca.explained_variance_ratio_.sum():.4f}")

        with open(pca_path, "wb") as f:
            pickle.dump({"scaler": scaler, "ipca": ipca, "no_pca": args.no_pca}, f)
        print(f"PCA pipeline saved to {pca_path}")

        # ── 3. Allocate memmaps ───────────────────────────────────────────
        mm_in     = np.lib.format.open_memmap(mm["X_train_in"],  mode="w+", dtype="float32", shape=(n_safe_train,   pca_dim))
        mm_s_in   = np.lib.format.open_memmap(mm["State_in"],    mode="w+", dtype="float32", shape=(n_safe_train,   2))
        mm_out    = np.lib.format.open_memmap(mm["X_train_out"], mode="w+", dtype="float32", shape=(n_unsafe_train, pca_dim))
        mm_s_out  = np.lib.format.open_memmap(mm["State_out"],   mode="w+", dtype="float32", shape=(n_unsafe_train,   2))
        mm_test   = np.lib.format.open_memmap(mm["X_test"],      mode="w+", dtype="float32", shape=(n_test,         pca_dim))
        mm_s_test = np.lib.format.open_memmap(mm["State_test"],  mode="w+", dtype="float32", shape=(n_test,   2))
        mm_yt     = np.lib.format.open_memmap(mm["y_test"],      mode="w+", dtype="float32", shape=(n_test,))

        # ── 4. Encode → (PCA) → write, filling train first then test ──────
        # We fill train slots until each class hits its 70% quota,
        # then overflow goes to test.
        print("Encoding + writing memmaps...")
        idx_in = idx_out = idx_test = 0

        for i in range(len(dataset_source)):
            if i % 200 == 0:
                print(f"  Episode {i}/{len(dataset_source)}")
            imgs_np, d, states_np = get_sample(args, dataset_source, i)
            enc_np = transform_enc(encode(imgs_np, device, states_np), scaler, ipca, args.no_pca)

            safe_mask   = (d ==  1)
            unsafe_mask = (d == -1)

            safe_enc   = enc_np[safe_mask]
            unsafe_enc = enc_np[unsafe_mask]

            safe_states   = states_np[safe_mask]
            unsafe_states = states_np[unsafe_mask]

            # Safe frames: fill train first, overflow to test
            for chunk, enc, states in [
                ("safe", safe_enc, safe_states),
                ("unsafe", unsafe_enc, unsafe_states),
            ]:
                if chunk == "safe":
                    n_train_quota = n_safe_train
                    idx_train     = idx_in
                    label_val     = 1.0
                    mm_train      = mm_in
                    mm_s_train    = mm_s_in
                else:
                    n_train_quota = n_unsafe_train
                    idx_train     = idx_out
                    label_val     = -1.0
                    mm_train      = mm_out
                    mm_s_train    = mm_s_out

                if len(enc) == 0:
                    continue

                train_space = n_train_quota - idx_train
                n_to_train  = min(len(enc), train_space)
                n_to_test   = len(enc) - n_to_train

                if n_to_train > 0:
                    mm_train[idx_train:idx_train + n_to_train] = enc[:n_to_train]
                    mm_s_train[idx_train:idx_train + n_to_train] = states[:n_to_train]

                if n_to_test > 0:
                    mm_test[idx_test:idx_test + n_to_test] = enc[n_to_train:]
                    mm_s_test[idx_test:idx_test + n_to_test] = states[n_to_train:n_to_train+n_to_test]
                    mm_yt  [idx_test:idx_test + n_to_test] = label_val

                if chunk == "safe":
                    idx_in    += n_to_train
                    idx_test  += n_to_test
                else:
                    idx_out   += n_to_train
                    idx_test  += n_to_test

        del mm_in, mm_out, mm_test, mm_yt, mm_s_in, mm_s_out, mm_s_test
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
    elif args.encoder == "ts":
        enc_kwargs["checkpoint_path"] = args.ts_ckpt
        enc_kwargs["img_size"]        = args.ts_img_size

    if args.encoder == "gt_state":
        encoder = None
        print("gt-state encoder: skipping image encoder build.")
    else:
        print(f"Loading encoder: {args.encoder} ...")
        encoder = build_encoder(args.encoder, **enc_kwargs)

    if args.dataset_mode == "npz":
        dataset_source = sorted(glob.glob(os.path.join(args.dataset, "*.npz")))
        print(f"Found {len(dataset_source)} episodes.")
    elif args.dataset_mode == "wall":
        if args.wall_config is None:
            # guess default location next to obses dir
            args.wall_config = os.path.join(
                os.path.dirname(args.wall_obses_dir.rstrip("/")),
                "wall_config.pkl"
            )
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
        print(f"Wall dataset: {len(dataset_source)} episodes.")
    elif args.dataset_mode == "ts":
        dataset_source = load_ts_dataset(args)

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
        run_pca_dim(args, encoder, dataset_source, device, None, config)
    else:
        for pca_dim in args.pca_dims:
            run_pca_dim(args, encoder, dataset_source, device, pca_dim, config)

    print("\nAll runs complete.")