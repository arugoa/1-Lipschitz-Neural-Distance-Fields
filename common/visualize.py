import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch_geometric.data import Data
from skimage.measure import marching_cubes

from .utils import forward_in_batches

def point_cloud_from_array(X, D=None):
    pos = torch.tensor(X, dtype=torch.float)
    pc = Data(pos=pos)
    if D is not None:
        pc.dist = torch.tensor(D, dtype=torch.float)
    return pc

def point_cloud_from_arrays(*args):
    """
    args : [(points,label), ...]
    """
    clouds, labels = [], []
    for pts, label in args:
        clouds.append(pts)
        labels.append(np.full(pts.shape[0], label))
    X = np.concatenate(clouds, axis=0)
    y = np.concatenate(labels, axis=0)
    pc = Data(pos=torch.tensor(X, dtype=torch.float),
              label=torch.tensor(y, dtype=torch.float))
    return pc

def vector_field_from_array(pos, vec, scale=1.):
    pos = np.asarray(pos)
    vec = np.asarray(vec)
    start = pos
    end = pos + scale * vec
    vertices = np.concatenate([start, end], axis=0)
    n = pos.shape[0]

    edge_index = np.vstack([
        np.arange(n),
        np.arange(n, 2*n)
    ])

    pc = Data(
        pos=torch.tensor(vertices, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(vec, dtype=torch.float)
    )

    return pc

def render_sdf_2d(render_path, contour_path, gradient_path, model, domain,
                  device, res=1000, batch_size=1000):

    X = np.linspace(domain.mini[0], domain.maxi[0], res)
    resY = round(res * domain.span[1]/domain.span[0])
    Y = np.linspace(domain.mini[1], domain.maxi[1], resY)

    pts = np.hstack((np.meshgrid(X,Y))).swapaxes(0,1).reshape(2,-1).T
    if gradient_path is not None:
        dist_values,grad_values = forward_in_batches(
            model, pts, device, 
            compute_grad=True, batch_size=batch_size)
    else:
        dist_values = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)

    img = np.concatenate(dist_values).reshape((res,resY)).T
    img = img[::-1,:]

    vmin = np.amin(img)
    vmax = np.amax(img)
    if vmin>0 or vmax<0:
        vmin,vmax = -1, 1

    if render_path is not None:
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.clf()
        pos = plt.imshow(img, cmap="seismic", norm=norm)
        plt.axis("off")
        plt.colorbar(pos)
        plt.savefig(render_path, bbox_inches="tight", pad_inches=0)

    if contour_path is not None:
        plt.clf()
        norm = colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
        plt.imshow(img, cmap="bwr", norm=norm)
        plt.axis("off")
        plt.contour(img, levels=16, colors="k", linewidths=0.3)
        plt.contour(img, levels=[0.], colors="k", linewidths=0.6)
        plt.savefig(contour_path, bbox_inches="tight", pad_inches=0, dpi=200)

    if gradient_path is not None:
        grad_norms = np.linalg.norm(grad_values, axis=1)
        grad_img = grad_norms.reshape((res, resY)).T
        grad_img = grad_img[::-1, :]

        plt.clf()
        pos = plt.imshow(grad_img, vmin=0.5, vmax=1.5, cmap="bwr")
        plt.contour(img, levels=[0.], colors="k", linewidths=0.6)
        plt.axis("off")
        plt.colorbar(pos)
        plt.savefig(gradient_path, bbox_inches="tight", pad_inches=0)


def parameter_singular_values(model):
    layers = list(model.children())
    data= []
    for layer in layers:
        if hasattr(layer, "weight"):
            w = layer.weight
            u, s, v = torch.linalg.svd(w)
            # data.append(f"{layer}, {s}")
            data.append(f"{layer}, min={s.min()}, max={s.max()}")
    return data


def reconstruct_surface_marching_cubes(model, domain, device, iso=0, res=100, batch_size=5000):
    if isinstance(iso, (int,float)): iso = [iso]
    
    ### Feed grid to model
    L = [np.linspace(domain["mini"][i], domain["maxi"][i], res) for i in range(3)]
    pts = np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(3,-1).T
    dist_values = forward_in_batches(model, pts, device, compute_grad=False, batch_size=batch_size)
    dist_values = dist_values.reshape((res,res,res))

    meshes = {}

    for ioff,off in enumerate(iso):
        try:
            verts, faces, normals, values = marching_cubes(dist_values, level=off)
            data = Data(
                pos=torch.tensor(verts, dtype=torch.float),
                face=torch.tensor(faces.T, dtype=torch.long),
                normal=torch.tensor(normals, dtype=torch.float),
                value=torch.tensor(values[:, None], dtype=torch.float)
            )

            meshes[(ioff, off)] = data
        except ValueError:
            continue

    return meshes