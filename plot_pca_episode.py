"""
plot_pca_episode.py  –  v2
--------------------------
1. Load an episode (.npz or .pth) that contains:
       images : (T, H, W, 3) or (T, 3, H, W)   uint8 or float
       states : (T, S)                            optional privileged state

2. Run each image through an encoder (.pth) to get feature vectors (T, D).

3. PCA(2) on the features  →  trajectory plot + MSE-to-target curve.

4. Background visualisation (choose one):
       --bg overlay   : alpha-composite of every frame (shows full path)
       --bg state     : last frame + state XY trajectory drawn on top
       --bg last      : just the last frame (no compositing)

Usage – single method
---------------------
python plot_pca_episode.py \
    --episode  episode.npz \
    --encoder  encoder.pth  \
    --enc_key  encoder \          # key inside the checkpoint dict, or omit for raw model
    --img_key  images \           # key for images inside episode file
    --state_key states \          # key for states (only needed for --bg state)
    --state_xy 0 1 \              # which two state dims map to X, Y in pixel space
    --bg overlay \
    --label DINO \
    --out pca_episode.png

Usage – two methods side by side
---------------------------------
python plot_pca_episode.py \
    --episode  episode.npz  episode.npz \
    --encoder  dino.pth     ours.pth \
    --label    DINO         Ours \
    --bg overlay \
    --out comparison.png
"""

import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.decomposition import PCA


# ════════════════════════════════════════════════════════════
#  Episode loading  (npz / pth)
# ════════════════════════════════════════════════════════════

def load_episode(path: str, img_key: str, state_key: str):
    """
    Returns
    -------
    images : np.ndarray  (T, H, W, 3)  uint8
    states : np.ndarray  (T, S)  or None

    State arrays with shape (T, N, D) — e.g. (T, 4, 2) for 4 objects
    with 2D positions — are reshaped to (T, N*D) automatically.
    Print shows the original shape so you can pick --state_obj.
    """
    if (len(path) == 1):
        path = path[0]
        ext = path.rsplit(".", 1)[-1].lower()
        if ext == "npz":
            data = np.load(path, allow_pickle=True)
            _keys = list(data.keys())
            print(f"  npz keys: {_keys}")
            imgs   = _get_key(data, img_key[0],   _keys, "images/obs/rgb/frames")
            states = _get_key(data, state_key[0], _keys, "states/state/proprio", required=False)

        elif ext in ("pth", "pt"):
            print(path)
            import torch
            data = torch.load(path, map_location="cpu", weights_only=False)
            # if not isinstance(data, dict):
            #     raise TypeError(f"Expected a dict in {path}, got {type(data)}")
            _keys = list(data.keys())
            print(f"  pth keys: {_keys}")
            imgs   = _pth_to_np(_get_key(data, img_key,   _keys, "images/obs/rgb/frames"))
            states = _pth_to_np(_get_key(data, state_key, _keys, "states/state/proprio", required=False))

        else:
            raise ValueError(f"Unsupported extension: .{ext}")

        imgs = _normalise_images(imgs)
        states = _normalise_states(states)
        return imgs, states
    else:
        img_path = path[0]
        state_path = path[1]
        import torch, re
        match = re.search(r'(\d+)', os.path.basename(img_path))
        if match is None:
            raise ValueError(f"Could not parse episode index from filename: {img_path}")
        ep_idx = int(match.group(1)) -1
        print(f"  parsed episode index: {ep_idx}")
        imgs = torch.load(img_path, map_location="cpu", weights_only=False).detach().cpu().numpy()
        states = torch.load(state_path, map_location="cpu", weights_only=False).detach().cpu().numpy()[ep_idx, ...]
        
        imgs = _normalise_images(imgs)
        states = _normalise_states(states)
        
        min_t = min(len(imgs), len(states))
        imgs   = imgs[:min_t]
        states = states[:min_t]
        return imgs, states


def _normalise_states(states):
    """Ensure states are 2-D (T, S). Prints shape before flattening."""
    if states is None:
        return None
    states = np.array(states, dtype=np.float32)
    if states.ndim > 2:
        orig = states.shape
        states = states.reshape(len(states), -1)
        print(f"  states (original shape {orig}) → flattened to {states.shape}")
        print(f"  tip: for --bg state, use --state_obj <0..{orig[1]-1}> to pick one object")
    return states


def _get_key(data, user_key, available, fallback_hints, required=True):
    """Try user_key first; then fall back to hints; then raise helpfully."""
    if user_key and user_key in data:
        return data[user_key]
    # try hints
    for hint in fallback_hints.split("/"):
        if hint in data:
            print(f"    (using key '{hint}')")
            return data[hint]
    if required:
        raise KeyError(
            f"Could not find key '{user_key}' in file.\n"
            f"Available: {available}\n"
            f"Re-run with the correct --img_key or --state_key."
        )
    return None


def _pth_to_np(obj):
    if obj is None:
        return None
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return np.array(obj)


def _normalise_images(arr):
    """Ensure shape (T, H, W, 3), dtype uint8."""
    arr = np.array(arr)
    # (T, C, H, W) → (T, H, W, C)
    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
        arr = arr.transpose(0, 2, 3, 1)
    if arr.shape[-1] == 4:          # RGBA → RGB
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        if hi <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


# ════════════════════════════════════════════════════════════
#  Encoder loading & inference
# ════════════════════════════════════════════════════════════

def load_encoder(path: str, enc_key: str):
    """
    Load a PyTorch encoder from a .pth file.

    The file can be:
      - a raw nn.Module  (torch.save(model, path))
      - a dict with key  enc_key  holding the module or state_dict

    Returns a callable that accepts a (B, 3, H, W) float32 CUDA/CPU tensor
    and returns (B, D) features.
    """
    import torch
    import torch.nn as nn
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
    hub_path = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    if os.path.exists(hub_path) and hub_path not in sys.path:
        sys.path.insert(0, hub_path)

    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        model = obj

    elif isinstance(obj, dict):
        _keys = list(obj.keys())
        print(f"  checkpoint keys: {_keys}")
        if enc_key and enc_key in obj:
            candidate = obj[enc_key]
        else:
            # heuristic: pick the first nn.Module value
            candidate = None
            for v in obj.values():
                if isinstance(v, nn.Module):
                    candidate = v
                    break
            if candidate is None:
                raise KeyError(
                    f"No nn.Module found under key '{enc_key}' in {path}.\n"
                    f"Available keys: {_keys}\n"
                    f"Re-run with --enc_key <correct key>."
                )
        if isinstance(candidate, nn.Module):
            model = candidate
        elif isinstance(candidate, dict):
            # state_dict – user must also supply the architecture
            raise ValueError(
                "Found a state_dict but no architecture. "
                "Pass a full model (not just weights) or subclass the script."
            )
        else:
            raise TypeError(f"Unexpected type under enc_key: {type(candidate)}")

    else:
        raise TypeError(f"Unsupported object type in {path}: {type(obj)}")

    model.eval()
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = model.to(device)
    return model, device


# default ImageNet normalisation used by most vision encoders
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@__import__("contextlib").contextmanager
def _no_grad():
    import torch
    with torch.no_grad():
        yield


def encode_images(model, device, images: np.ndarray,
                  img_size=(224, 224), batch_size=32,
                  normalise=True, pool_mode="cls") -> np.ndarray:
    """
    Encode every frame in `images` (T, H, W, 3) through `model`.

    Returns (T, D) float32 numpy array.

    pool_mode controls how patch-token outputs are reduced to one vector:
      'cls'  – take token 0 (the CLS token, standard for DINOv2 / ViT)
      'mean' – mean-pool all patch tokens
      'flat' – concatenate all tokens (very high-dim, use with care)
    """
    import torch
    from PIL import Image as PILImage

    T = len(images)
    feats_list = []

    with _no_grad():
        for start in range(0, T, batch_size):
            batch_np = images[start : start + batch_size]   # (B, H, W, 3)
            frames = []
            for img in batch_np:
                pil = PILImage.fromarray(img).resize(img_size, PILImage.BILINEAR)
                arr = np.array(pil, dtype=np.float32) / 255.0
                if normalise:
                    arr = (arr - _MEAN) / _STD
                frames.append(arr.transpose(2, 0, 1))       # (3, H, W)

            tensor = torch.tensor(np.stack(frames)).to(device)   # (B, 3, H, W)
            out = model(tensor)

            # handle tuple/list outputs (some encoders return (feats, extras))
            if isinstance(out, (tuple, list)):
                out = out[0]

            # ── reduce to (B, D) ──────────────────────────────────────────
            if out.ndim == 2:
                # already (B, D) – nothing to do
                pass
            elif out.ndim == 3:
                # ViT patch tokens: (B, N, D)
                # token 0 is the CLS token for DINO / DINOv2
                if pool_mode == "cls":
                    out = out[:, 0, :]          # (B, D)
                elif pool_mode == "mean":
                    out = out.mean(dim=1)        # (B, D)
                elif pool_mode == "flat":
                    B = out.shape[0]
                    out = out.reshape(B, -1)     # (B, N*D)  ← very large
                else:
                    raise ValueError(f"Unknown pool_mode '{pool_mode}'")
            elif out.ndim == 4:
                # CNN spatial maps: (B, D, h, w) → global avg pool
                out = out.mean(dim=(2, 3))       # (B, D)
            else:
                raise ValueError(f"Unexpected encoder output shape: {out.shape}")

            feats_list.append(out.cpu().float().numpy())
            print(f"  encoded {min(start+batch_size, T)}/{T} frames", end="\r")

    print()
    return np.concatenate(feats_list, axis=0)   # (T, D)


# ════════════════════════════════════════════════════════════
#  Background image helpers
# ════════════════════════════════════════════════════════════

def make_overlay(images: np.ndarray, fg_threshold: int = 20) -> np.ndarray:
    T, H, W, C = images.shape
    imgs_f = images.astype(np.float32)

    # static background via per-pixel median
    step = max(1, T // 30)
    bg   = np.median(imgs_f[::step], axis=0)   # (H, W, 3)

    # start from last frame, paint older frames first
    canvas = imgs_f[-1].copy()
    for t in range(T):
        frame = imgs_f[t]
        diff  = np.linalg.norm(frame - bg, axis=-1)   # (H, W)
        mask  = diff > fg_threshold                    # foreground pixels only
        if not mask.any():
            continue
        alpha = 0.25 + 0.75 * t / max(T - 1, 1)      # faint early → opaque late
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * frame[mask]

    return canvas.clip(0, 255).astype(np.uint8)


def make_state_overlay(images: np.ndarray, states: np.ndarray,
                       state_xy: tuple, state_obj: int = None) -> np.ndarray:
    """
    Draw the XY state trajectory on top of the last frame.

    states   : (T, S) — already flattened by load_episode.
               If the original was (T, N, 2), flattened is (T, N*2).
               Use state_obj to pick which object's XY to plot:
                 state_obj=0  →  dims [0,1]
                 state_obj=1  →  dims [2,3]  etc.

    state_xy : (xi, yi) flat indices into states[:,xi] / states[:,yi].
               Ignored when state_obj is set.
    """
    H, W = images.shape[1], images.shape[2]
    bg = images[-1].copy()

    if state_obj is not None:
        xi = state_obj * 2 
        yi = state_obj * 2 + 1
        if yi >= states.shape[1]:
            raise ValueError(
                f"--state_obj {state_obj} requires at least {yi+1} state dims, "
                f"but states has shape {states.shape}"
            )
    else:
        xi, yi = state_xy

    xs = states[:, xi]
    ys = states[:, yi]

    xs *= -1
    xs += 1
    xs *= W/2
    
    ys *= -1
    ys += 1
    ys *= H/2

    # draw on a PIL canvas
    from PIL import Image as PILImage, ImageDraw
    canvas = PILImage.fromarray(bg)
    draw   = ImageDraw.Draw(canvas, "RGBA")

    T = len(xs)
    for t in range(T - 1):
        alpha = int(80 + 175 * t / max(T - 2, 1))
        draw.line(
            [(xs[t], ys[t]), (xs[t+1], ys[t+1])],
            fill=(255, 220, 50, alpha),
            width=3,
        )
    # start marker
    draw.ellipse([xs[0]-5, ys[0]-5, xs[0]+5, ys[0]+5],
                 fill=(100, 200, 255, 220), outline=(255,255,255,200))
    # target star (last point)
    draw.ellipse([xs[-1]-6, ys[-1]-6, xs[-1]+6, ys[-1]+6],
                 fill=(255, 193, 7, 255), outline=(0,0,0,200))

    return np.array(canvas.convert("RGB"))


# ════════════════════════════════════════════════════════════
#  PCA + MSE
# ════════════════════════════════════════════════════════════

def run_pca(features: np.ndarray):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)
    return proj, pca


def mse_to_target(features: np.ndarray, target_idx: int):
    tgt = features[target_idx]
    return ((features - tgt) ** 2).mean(axis=1)


# ════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════

METHOD_COLORS = ["#c94040", "#3a6bc9"]


def plot_method(ax_scene, ax_pca, ax_mse,
                bg_img, proj, mse, label, color, target_idx):

    T = len(proj)

    # ── scene / background ───────────────────────────────────
    if bg_img is not None:
        ax_scene.imshow(bg_img)
    ax_scene.axis("off")
    ax_scene.set_title("episode", fontsize=8, pad=2)

    # ── PCA trajectory ───────────────────────────────────────
    for t in range(T - 1):
        alpha = 0.25 + 0.75 * t / max(T - 2, 1)
        ax_pca.plot(proj[t:t+2, 0], proj[t:t+2, 1],
                    color=color, alpha=alpha, linewidth=1.4, solid_capstyle="round")

    ax_pca.scatter(*proj[0], color=color, s=22, zorder=4, alpha=0.8)
    tgt_pt = proj[target_idx]
    ax_pca.scatter(*tgt_pt, marker="*", s=260, color="#f5c518",
                   edgecolors="black", linewidths=0.6, zorder=5)

    ax_pca.set_xlabel("PC1", fontsize=8); ax_pca.set_ylabel("PC2", fontsize=8)
    ax_pca.set_title(f"PCA({label})", fontsize=9, fontweight="bold")
    ax_pca.tick_params(labelsize=7)

    # ── MSE curve ────────────────────────────────────────────
    ax_mse.plot(mse, color=color, linewidth=1.6)
    ax_mse.set_xlabel("time", fontsize=8); ax_mse.set_ylabel("MSE to target", fontsize=8)
    ax_mse.set_title(f"MSE({label})", fontsize=9, fontweight="bold")
    ax_mse.tick_params(labelsize=7)

    for ax in (ax_pca, ax_mse):
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)


def make_figure(configs, bg_mode, state_xy, state_obj, target_idx, pool_mode, out_path):
    """
    configs : list of dicts with keys:
        episode_path, encoder_path, enc_key, img_key, state_key, label
    """
    n = len(configs)
    # layout: [scene | pca | mse] × n  but scene is shared if files are identical
    share_scene = (n > 1 and
                   configs[0]["episode_path"] == configs[1]["episode_path"])

    ncols = (1 + 2 * n) if share_scene else 3 * n
    fig_w = 3.5 + 4.0 * n
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 3.4))
    axes = np.atleast_1d(axes)

    col = 0
    shared_bg = None

    for i, cfg in enumerate(configs):
        print(f"\n[{cfg['label']}]  episode: {cfg['episode_path']}")
        imgs, states = load_episode(
            cfg["episode_path"], cfg["img_key"], cfg["state_key"]
        )
        print(f"  images : {imgs.shape}")
        if states is not None:
            print(f"  states : {states.shape}")

        # background
        if i == 0 or not share_scene:
            if bg_mode == "overlay":
                bg = make_overlay(imgs)
            elif bg_mode == "state":
                if states is None:
                    print("  WARNING: no states found, falling back to last-frame bg")
                    bg = imgs[-1]
                else:
                    bg = make_state_overlay(imgs, states, state_xy, state_obj)
            else:   # "last"
                bg = imgs[-1]

            if share_scene:
                shared_bg = bg

        if share_scene and i == 0:
            axes[col].imshow(shared_bg); axes[col].axis("off")
            axes[col].set_title("episode", fontsize=8, pad=2)
            col += 1

        # encoder
        print(f"  encoder: {cfg['encoder_path']}")
        model, device = load_encoder(cfg["encoder_path"][0], cfg["enc_key"][0])
        feats = encode_images(model, device, imgs, pool_mode=pool_mode)
        print(f"  features: {feats.shape}")

        proj, _ = run_pca(feats)
        mse     = mse_to_target(feats, target_idx)

        ax_scene = axes[col]     if not share_scene else None
        ax_pca   = axes[col + (0 if share_scene else 1)]
        ax_mse   = axes[col + (1 if share_scene else 2)]

        if not share_scene:
            plot_method(ax_scene, ax_pca, ax_mse,
                        bg, proj, mse, cfg["label"],
                        METHOD_COLORS[i % len(METHOD_COLORS)], target_idx)
            col += 3
        else:
            plot_method(None, ax_pca, ax_mse,
                        bg, proj, mse, cfg["label"],
                        METHOD_COLORS[i % len(METHOD_COLORS)], target_idx)
            col += 2

    plt.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.show()


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════

def _expand(lst, n, name):
    if len(lst) == 1 and n > 1:
        return lst * n
    if len(lst) != n:
        raise ValueError(f"--{name}: expected 1 or {n} values, got {len(lst)}")
    return lst


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episode",   nargs="+", required=True,
                   help=".npz/.pth file(s) holding images (and optionally states)")
    p.add_argument("--encoder",   nargs="+", required=True,
                   help=".pth file(s) with the encoder nn.Module")
    p.add_argument("--enc_key",   nargs="+", default=["encoder"],
                   help="Key inside encoder checkpoint (default: 'encoder')")
    p.add_argument("--img_key",   nargs="+", default=["images"],
                   help="Key for image array inside episode file (default: 'images')")
    p.add_argument("--state_key", nargs="+", default=["states"],
                   help="Key for state array inside episode file (default: 'states')")
    p.add_argument("--label",     nargs="+", default=None)
    p.add_argument("--bg",        choices=["overlay", "state", "last"],
                   default="overlay",
                   help="Background mode: 'overlay' composites all frames, "
                        "'state' draws XY trajectory on last frame, "
                        "'last' just shows the final frame")
    p.add_argument("--state_xy",  nargs=2, type=int, default=[0, 1],
                   help="Which two state dims map to X and Y (default: 0 1). "
                        "Only used when --bg state")
    p.add_argument("--target_idx", type=int, default=-1,
                   help="Episode timestep used as the goal frame (-1 = last)")
    p.add_argument("--pool",      choices=["cls", "mean", "flat"], default="cls",
                   help="How to reduce ViT patch tokens to one vector per frame. "
                        "'cls' = take CLS token (index 0, default for DINOv2/ViT), "
                        "'mean' = average all patch tokens, "
                        "'flat' = concatenate all tokens (very high-dim)")
    p.add_argument("--state_obj", type=int, default=None,
                   help="For multi-object states shaped (T, N, 2): which object index "
                        "to use for the --bg state trajectory. "
                        "Default: object 0. Use --state_obj -1 for the last object.")
    p.add_argument("--img_size",  nargs=2, type=int, default=[224, 224],
                   help="Resize each frame to HxW before encoding (default 224 224)")
    p.add_argument("--no_norm",   action="store_true",
                   help="Skip ImageNet normalisation (use if your encoder "
                        "expects raw [0,1] pixels)")
    p.add_argument("--out",       default="pca_episode.png")
    return p.parse_args()


def main():
    args  = parse_args()
    n     = len(args.episode)
    encs  = _expand(args.encoder,   n, "encoder")
    ekeys = _expand(args.enc_key,   n, "enc_key")
    ikeys = _expand(args.img_key,   n, "img_key")
    skeys = _expand(args.state_key, n, "state_key")
    labs  = _expand(
        args.label if args.label else
        [os.path.splitext(os.path.basename(e))[0] for e in encs],
        n, "label"
    )

    # configs = [
    #     dict(episode_path=args.episode[i], encoder_path=encs[i],
    #          enc_key=ekeys[i], img_key=ikeys[i], state_key=skeys[i],
    #          label=labs[i])
    #     for i in range(n)
    # ]
    config = [dict(episode_path=args.episode, encoder_path=args.encoder, enc_key=args.enc_key, img_key=args.img_key, state_key=args.state_key, label=labs[0])]

    make_figure(config, args.bg, tuple(args.state_xy),
                args.state_obj if args.state_obj is not None else None,
                args.target_idx, args.pool, args.out)


if __name__ == "__main__":
    main()