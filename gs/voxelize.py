import numpy as np
import torch
from tqdm import tqdm

def build_voxel_grid(gs, grid_size=128, bounds=None):
    """
    Converts GS to a voxel grid of densities.

    Args:
        gs: dict from loader
        grid_size: number of voxels per axis (int or tuple)
        bounds: ((xmin, ymin, zmin), (xmax, ymax, zmax))

    Returns:
        voxel_grid: (X, Y, Z) array of densities
    """
    # 1. Define voxel centers
    if bounds is None:
        min_bound = np.min(gs['positions'], axis=0)
        max_bound = np.max(gs['positions'], axis=0)
    else:
        min_bound, max_bound = bounds

    X = np.linspace(min_bound[0], max_bound[0], grid_size)
    Y = np.linspace(min_bound[1], max_bound[1], grid_size)
    Z = np.linspace(min_bound[2], max_bound[2], grid_size)
    xx, yy, zz = np.meshgrid(X, Y, Z, indexing='ij')
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    # 2. Evaluate density at all voxel centers
    with torch.no_grad():
        D, _ = compute_density_and_grad(coords_tensor, gs)

    return D.reshape((grid_size, grid_size, grid_size)).numpy(), (min_bound, max_bound)
