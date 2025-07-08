import numpy as np
import open3d as o3d

def load_gs_from_ply(ply_path):
    """
    Load GS parameters from a .ply file.

    Returns:
        dict with keys: positions (N,3), scales (N,3), rotations (N,4), alphas (N,), sh_coeffs (N, 3, L)
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    data = np.asarray(pcd.points)  # placeholder
    
    # TODO: Read attributes (via open3d or plyfile), e.g., scale, rotation, alpha, SH
    # positions = ...
    # scales = ...
    # rotations = ...
    # alphas = ...
    # sh_coeffs = ...

    return {
        "positions": ...,       # (N, 3)
        "scales": ...,          # (N, 3)
        "rotations": ...,       # (N, 4) quaternion
        "alphas": ...,          # (N,)
        "sh_coeffs": ...        # (N, 3, L)
    }
