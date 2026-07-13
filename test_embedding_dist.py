"""
test_embedding_distance.py
--------------------------
Validate your temporal-straightening encoder by plotting Euclidean distance
in latent space against ground-truth cartesian coordinates — the Figure 6
diagnostic from the TS paper (Wang et al. 2026).

A well-trained encoder should produce a heatmap that looks like an A* distance
map: a smooth gradient radiating outward from the goal, with walls/obstacles
appearing as hard discontinuities (no "seeing through walls").

Files loaded from --run-dir  (same convention as visualize_train.py):
    X_train_in.npy     (N_in,  D)   safe    train embeddings
    X_train_out.npy    (N_out, D)   unsafe  train embeddings
    X_test.npy         (N_test, D)  test    embeddings  (eps * ep_len rows)
    State_test.npy     (N_test, S)  cartesian state per test frame
    y_test.npy         (N_test,)    test labels  (>0 = safe)

Usage
-----
# Basic — goal = last frame of episode 0:
python test_embedding_distance.py --run-dir output/run0

# Pick a specific episode and goal frame within it:
python test_embedding_distance.py --run-dir output/run0 \
    --ep-len 100 --goal-ep 3 --goal-frame -1

# Check multiple goals in one figure:
python test_embedding_distance.py --run-dir output/run0 \
    --ep-len 100 --multi-goal-eps 0 5 10 --multi-goal-frame -1

Outputs (written inside --run-dir unless --out-dir is set)
-----
    embedding_distance_heatmap.png        main heatmap + episode time-series
    embedding_distance_<N>goals.png       one panel per goal (--multi-goal-eps)
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ════════════════════════════════════════════════════════════
#  Loading
# ════════════════════════════════════════════════════════════

def load_run(run_dir: str):
    """
    Load all standard arrays from a run directory.
    Mirrors the convention in visualize_train.py / visualize_test.py.
    """
    def npy(name):
        path = os.path.join(run_dir, name)
        assert os.path.exists(path), f"Missing expected file: {path}"
        return np.load(path, mmap_mode="r").astype(np.float32)

    X_in    = npy("X_train_in.npy")   # (N_in,  D)
    X_out   = npy("X_train_out.npy")  # (N_out, D)
    X_test  = npy("X_test.npy")       # (N_test, D)  — eps * ep_len rows
    states  = npy("State_test.npy")   # (N_test, S)  — same ordering as X_test
    labels  = npy("y_test.npy")       # (N_test,)

    print(f"X_train_in  : {X_in.shape}")
    print(f"X_train_out : {X_out.shape}")
    print(f"X_test      : {X_test.shape}")
    print(f"State_test  : {states.shape}")
    print(f"y_test      : {labels.shape}")

    assert len(X_test) == len(states) == len(labels), (
        "X_test, State_test, y_test must all have the same first dimension"
    )
    return X_in, X_out, X_test, states, labels


def resolve_episode_structure(N_test: int, ep_len: int | None):
    """
    Return (n_eps, ep_len).  If ep_len is not given, treat the whole
    test set as one episode — the heatmap still works, but the
    time-series panel will span the entire dataset.
    """
    if ep_len is None:
        return 1, N_test
    assert N_test % ep_len == 0, (
        f"N_test={N_test} is not divisible by ep_len={ep_len}. "
        "Pass the correct --ep-len."
    )
    return N_test // ep_len, ep_len


def flat_goal_idx(goal_ep: int, goal_frame: int, n_eps: int, ep_len: int) -> int:
    """Convert (episode, frame) to flat row index in X_test / State_test."""
    ep    = goal_ep % n_eps
    frame = goal_frame % ep_len
    return ep * ep_len + frame


# ════════════════════════════════════════════════════════════
#  Core metric
# ════════════════════════════════════════════════════════════

def dist_to_goal(X: np.ndarray, goal_flat: int) -> np.ndarray:
    """
    Squared Euclidean distance from every embedding to the goal embedding.

        d[i] = ||X[i] - X[goal_flat]||²  / D

    Dividing by D (mean rather than sum) keeps the scale independent of
    embedding dimensionality, matching the MSE convention in the TS paper.
    """
    goal = X[goal_flat]                        # (D,)
    diff = X - goal[None, :]                   # (N, D)
    return (diff ** 2).mean(axis=-1)           # (N,)


# ════════════════════════════════════════════════════════════
#  Diagnostics
# ════════════════════════════════════════════════════════════

def print_diagnostics(distances: np.ndarray, goal_flat: int,
                      n_eps: int, ep_len: int):
    goal_ep    = goal_flat // ep_len
    goal_frame = goal_flat  % ep_len

    # Within-episode monotonicity (most meaningful check)
    ep_dists = distances[goal_ep * ep_len : (goal_ep + 1) * ep_len]
    diffs    = np.diff(ep_dists)
    mono     = (diffs < 0).mean()

    # Goal should be its own nearest neighbour (rank 0)
    rank = int((distances < distances[goal_flat]).sum())

    print("\n── Embedding Distance Diagnostics ─────────────────────────")
    print(f"  N test frames      : {len(distances)}")
    print(f"  Episodes × ep_len  : {n_eps} × {ep_len}")
    print(f"  Goal               : episode {goal_ep}, frame {goal_frame}"
          f"  →  flat idx {goal_flat}")
    print(f"  Distance at goal   : {distances[goal_flat]:.6f}  (should be ~0)")
    print(f"  Mean distance      : {distances.mean():.4f}")
    print(f"  Max  distance      : {distances.max():.4f}")
    print(f"  Monotone decrease  : {mono*100:.1f}%  within goal episode")
    print(f"    (>80% = well-straightened, <50% = trajectory still curved)")
    print(f"  Goal rank          : {rank} / {len(distances)}")
    print(f"    (0 = goal is its own nearest neighbour, as expected)")
    if distances[goal_flat] > 1e-3:
        print("  ⚠  Non-zero self-distance — possible encoder collapse or "
              "the goal frame was not included in X_test.")
    if mono < 0.5:
        print("  ⚠  Low monotonicity — latent space is still highly curved. "
              "Try increasing lambda for the curvature loss.")
    print("───────────────────────────────────────────────────────────\n")


# ════════════════════════════════════════════════════════════
#  Plotting helpers
# ════════════════════════════════════════════════════════════

def _scatter_heatmap(ax, xs, ys, values, cmap, vmax, label_cbar, title,
                     goal_x=None, goal_y=None, n_sample=20_000):
    """Scatter points at (xs, ys) coloured by `values`, add goal star."""
    rng = np.random.default_rng(42)
    N   = len(xs)
    idx = rng.choice(N, min(n_sample, N), replace=False)

    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    sc   = ax.scatter(xs[idx], ys[idx], c=values[idx],
                      cmap=cmap, norm=norm, s=6, alpha=0.8, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label_cbar, fontsize=8)

    if goal_x is not None:
        ax.scatter(goal_x, goal_y, marker="*", s=320, color="#f5c518",
                   edgecolors="black", linewidths=0.8, zorder=5, label="Goal")
        ax.legend(fontsize=8, markerscale=1)

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")


def plot_main_heatmap(
    X_test:   np.ndarray,   # (N, D)
    states:   np.ndarray,   # (N, S)
    labels:   np.ndarray,   # (N,)
    distances: np.ndarray,  # (N,)
    goal_flat: int,
    n_eps: int, ep_len: int,
    state_xy: tuple,
    cmap: str,
    out_path: str,
    n_sample: int = 20_000,
):
    xi, yi = state_xy
    xs_all  = states[:, xi]
    ys_all  = states[:, yi]
    goal_x  = states[goal_flat, xi]
    goal_y  = states[goal_flat, yi]
    vmax    = float(np.percentile(distances, 98))

    goal_ep    = goal_flat // ep_len
    goal_frame = goal_flat  % ep_len

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))

    # ── Panel 1: distance heatmap over all test states ───────────────────
    _scatter_heatmap(
        axes, xs_all, ys_all, distances,
        cmap=cmap, vmax=vmax,
        label_cbar="||w_i − w_goal||²  (lower = closer)",
        title=f"Latent distance heatmap\n(goal: ep {goal_ep}, frame {goal_frame})",
        goal_x=goal_x, goal_y=goal_y,
        n_sample=n_sample,
    )
    axes.set_xlabel(f"State dim {xi}", fontsize=9)
    axes.set_ylabel(f"State dim {yi}", fontsize=9)

    # ── Panel 2: safe / unsafe label overlay ────────────────────────────
    #    (shows spatial coverage of what the SDF was trained on)
    # safe_mask   = labels > 0
    # unsafe_mask = ~safe_mask
    # rng   = np.random.default_rng(0)
    # s_idx = rng.choice(safe_mask.sum(),   min(n_sample // 2, safe_mask.sum()),   replace=False)
    # u_idx = rng.choice(unsafe_mask.sum(), min(n_sample // 2, unsafe_mask.sum()), replace=False)

    # axes[1].scatter(xs_all[unsafe_mask][u_idx], ys_all[unsafe_mask][u_idx],
    #                 s=4, alpha=0.35, color="#F44336",
    #                 label=f"Unsafe ({unsafe_mask.sum():,})", rasterized=True)
    # axes[1].scatter(xs_all[safe_mask][s_idx],   ys_all[safe_mask][s_idx],
    #                 s=4, alpha=0.35, color="#2196F3",
    #                 label=f"Safe ({safe_mask.sum():,})",   rasterized=True)
    # axes[1].scatter(goal_x, goal_y, marker="*", s=280, color="#f5c518",
    #                 edgecolors="black", linewidths=0.8, zorder=5, label="Goal")
    # axes[1].set_title("Safe / Unsafe labels in state space\n(reference)", fontsize=10)
    # axes[1].set_xlabel(f"State dim {xi}", fontsize=9)
    # axes[1].set_ylabel(f"State dim {yi}", fontsize=9)
    # axes[1].set_aspect("equal", adjustable="datalim")
    # axes[1].legend(fontsize=7, markerscale=2)

    # ── Panel 3: distance over time within the goal episode ──────────────
    #    For a well-straightened trajectory this should be monotone decreasing
    # ep_start = goal_ep * ep_len
    # ep_dists = distances[ep_start : ep_start + ep_len]   # (ep_len,)
    # t        = np.arange(ep_len)

    # axes[2].plot(t, ep_dists, color="#3a6bc9", linewidth=1.5, alpha=0.9)
    # axes[2].axvline(goal_frame % ep_len, color="#f5c518", linewidth=1.5,
    #                 linestyle="--", label=f"Goal frame ({goal_frame % ep_len})")
    # axes[2].set_xlabel("Frame within episode", fontsize=9)
    # axes[2].set_ylabel("||w_i − w_goal||²", fontsize=9)
    # axes[2].set_title(f"Distance to goal over time\n"
    #                   f"episode {goal_ep}  (good = monotone ↓)", fontsize=10)
    # axes[2].legend(fontsize=8)
    # axes[2].grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def plot_multi_goal(
    X_test:    np.ndarray,
    states:    np.ndarray,
    goal_eps:  list,
    goal_frame: int,
    n_eps: int, ep_len: int,
    state_xy: tuple,
    cmap: str,
    out_path: str,
    n_sample: int = 8_000,
):
    """
    One panel per episode goal — checks that the distance metric is
    geometrically consistent regardless of where the goal is placed.
    """
    G   = len(goal_eps)
    xi, yi = state_xy
    xs  = states[:, xi]
    ys  = states[:, yi]

    fig, axes = plt.subplots(1, G, figsize=(5 * G, 5), squeeze=False)

    for col, ep in enumerate(goal_eps):
        gflat  = flat_goal_idx(ep, goal_frame, n_eps, ep_len)
        dists  = dist_to_goal(X_test, gflat)
        vmax   = float(np.percentile(dists, 98))

        ax = axes[0, col]
        _scatter_heatmap(
            ax, xs, ys, dists,
            cmap=cmap, vmax=vmax,
            label_cbar="||w_i − w_goal||²",
            title=f"Goal: ep {ep % n_eps}, frame {goal_frame % ep_len}",
            goal_x=states[gflat, xi],
            goal_y=states[gflat, yi],
            n_sample=n_sample,
        )
        ax.set_xlabel(f"State dim {xi}", fontsize=8)
        ax.set_ylabel(f"State dim {yi}", fontsize=8)

    fig.suptitle(
        "Latent distance heatmaps — multi-goal consistency check\n"
        "(each panel should look like a smooth A* distance map)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="TS encoder validation: latent distance heatmap in state space"
    )
    p.add_argument("--run-dir", type=str, required=True,
                   help="Run directory (same as visualize_train.py / visualize_test.py)")
    p.add_argument("--ep-len", type=int, default=None,
                   help="Frames per episode.  If omitted, the whole test set "
                        "is treated as one episode.")

    # ── goal selection ──────────────────────────────────────────────────
    p.add_argument("--goal-ep",    type=int, default=0,
                   help="Which episode the goal comes from (default: 0)")
    p.add_argument("--goal-frame", type=int, default=-1,
                   help="Frame within that episode (default: -1 = last frame)")

    # ── multi-goal consistency check ────────────────────────────────────
    p.add_argument("--multi-goal-eps", type=int, nargs="+", default=None,
                   help="List of episode indices to use as goals, e.g. 0 5 10. "
                        "Each uses --goal-frame as the in-episode index.")

    # ── plotting ────────────────────────────────────────────────────────
    p.add_argument("--state-xy", nargs=2, type=int, default=[0, 1],
                   help="Which two columns of State_test are X and Y (default: 0 1)")
    p.add_argument("--n-sample", type=int, default=20_000,
                   help="Max scatter points per panel (randomly subsampled)")
    p.add_argument("--cmap", type=str, default="RdBu_r",
                   help="Matplotlib colormap  (RdBu_r: red=far, blue=close to goal)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (defaults to --run-dir)")
    return p.parse_args()


def main():
    args    = parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir or run_dir

    assert os.path.isdir(run_dir), f"Run dir not found: {run_dir}"
    os.makedirs(out_dir, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────
    X_in, X_out, X_test, states, labels = load_run(run_dir)
    N_test = len(X_test)

    # ── episode structure ─────────────────────────────────────────────────
    n_eps, ep_len = resolve_episode_structure(N_test, args.ep_len)
    print(f"\nEpisode structure: {n_eps} episodes × {ep_len} frames = {N_test} rows\n")

    # ── single-goal heatmap ───────────────────────────────────────────────
    goal_flat = flat_goal_idx(args.goal_ep, args.goal_frame, n_eps, ep_len)
    distances = dist_to_goal(X_test, goal_flat)

    print_diagnostics(distances, goal_flat, n_eps, ep_len)

    out_main = os.path.join(out_dir, "embedding_distance_heatmap.png")
    plot_main_heatmap(
        X_test=X_test, states=states, labels=labels,
        distances=distances,
        goal_flat=goal_flat,
        n_eps=n_eps, ep_len=ep_len,
        state_xy=tuple(args.state_xy),
        cmap=args.cmap,
        out_path=out_main,
        n_sample=args.n_sample,
    )

    # ── multi-goal consistency check ──────────────────────────────────────
    if args.multi_goal_eps is not None:
        out_multi = os.path.join(
            out_dir,
            f"embedding_distance_{len(args.multi_goal_eps)}goals.png"
        )
        plot_multi_goal(
            X_test=X_test, states=states,
            goal_eps=args.multi_goal_eps,
            goal_frame=args.goal_frame,
            n_eps=n_eps, ep_len=ep_len,
            state_xy=tuple(args.state_xy),
            cmap=args.cmap,
            out_path=out_multi,
            n_sample=args.n_sample,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()