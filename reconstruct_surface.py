import argparse
import numpy as np
import torch
from skimage.measure import marching_cubes
import trimesh
from sklearn.decomposition import PCA

from common.models import load_model
from common.utils import get_device
from common.visualize import reconstruct_surface_marching_cubes

def find_zero_crossing_dims(model, device, latent_dim, samples=50, search_range=1.0):
    print("Searching for latent dims with SDF sign change...")
    base = torch.zeros(1, latent_dim).to(device)
    dims = []
    scores = []

    # fallback tracking
    closest_scores = []
    closest_dims = []

    for d in range(latent_dim):
        vals = torch.linspace(-search_range, search_range, samples)

        pts = base.repeat(samples, 1)
        pts[:, d] = vals
        pts = pts.to(device)

        with torch.no_grad():
            sdf = model(pts).detach().cpu().numpy().reshape(-1)

        mn = sdf.min()
        mx = sdf.max()

        if mn < 0 and mx > 0:
            score = mx - mn
            dims.append(d)
            scores.append(score)

        min_abs = np.min(np.abs(sdf))
        closest_scores.append(min_abs)
        closest_dims.append(d)

    if len(dims) >= 3:
        order = np.argsort(scores)[::-1]
        best_dims = [dims[i] for i in order[:3]]
        print("Selected dims (zero-crossing):", best_dims)
        return best_dims

    print("No sufficient zero-crossing dims, using closest-to-zero dims")
    order = np.argsort(closest_scores)
    best_dims = [closest_dims[i] for i in order[:3]]

    print("Selected dims (closest to zero):", best_dims)
    return best_dims

def reconstruct_surface_first3(
    model,
    device,
    latent_dim,
    resolution=1000,
    iso_values=[0.0],
    batch_size=5000):

    # grid in first 3 dims
    grid = np.linspace(-1, 1, resolution)
    X, Y, Z = np.meshgrid(grid, grid, grid)
    coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # build full latent vectors
    dims = find_zero_crossing_dims(model, device, latent_dim)
    pts = np.zeros((coords.shape[0], latent_dim))
    pts[:, dims[0]] = coords[:, 0]
    pts[:, dims[1]] = coords[:, 1]
    pts[:, dims[2]] = coords[:, 2]
    pts = torch.tensor(pts, dtype=torch.float32).to(device)

    # forward in batches
    sdf_vals = []
    with torch.no_grad():
        for i in range(0, pts.shape[0], batch_size):
            batch = pts[i:i+batch_size]
            val = model(batch)
            sdf_vals.append(val.detach().cpu().numpy())

    sdf_vals = np.concatenate(sdf_vals)
    sdf_vals = sdf_vals.reshape(resolution, resolution, resolution)
    vmin = sdf_vals.min()
    vmax = sdf_vals.max()
    print("SDF range:", vmin, vmax)

    meshes = {}

    for iso in iso_values:
        if (iso < vmin or iso > vmax):
            iso = (vmax + vmin) / 2
        verts, faces, normals, _ = marching_cubes(sdf_vals, level=iso)
        # scale verts to [-1,1]
        verts = -1 + 2 * verts / (resolution - 1)
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals)
        meshes[iso] = mesh

    return meshes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D reconstruction from first 3 latent dims"
    )

    parser.add_argument("model")
    parser.add_argument("-o", "--output-name", default="reconstruction")
    parser.add_argument("-iso", "--isovalues", type=float, nargs="+", default=[0.0])
    parser.add_argument("-res", "--resolution", type=int, default=100)
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-bs", "--batch-size", type=int, default=5000)
    # parser.add_argument("-r", "--range", action="store_true", help="override the -iso argument and run marching cube for each iso in linspace(-0.1, 0.1, 21)")
    parser.add_argument("-latent-dim", type=int, default=640)
    args = parser.parse_args()

    device = get_device(args.cpu)
    print("DEVICE:", device)

    sdf = load_model(args.model, device)

    domain = {
        "mini": np.array([-1., -1., -1.]),
        "maxi": np.array([ 1.,  1.,  1.])
    }
    res = args.resolution
    meshes = reconstruct_surface_first3(
        sdf,
        device,
        latent_dim=args.latent_dim,
        resolution=args.resolution,
        iso_values=args.isovalues,
        batch_size=args.batch_size)

    for iso, mesh in meshes.items():
        path = f"output/{args.output_name}_iso{iso:.3f}.obj"
        mesh.export(path)
        print("Saved:", path)