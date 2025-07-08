import torch

def compute_density(x, gs):
    """
    x: (B, 3) tensor of query points
    gs: dict containing GS parameters from loader

    Returns:
        D: (B,) density values
    """
    mus = torch.tensor(gs['positions'])      # (N, 3)
    scales = torch.tensor(gs['scales'])      # (N, 3)
    rots = torch.tensor(gs['rotations'])     # (N, 4)
    alphas = torch.tensor(gs['alphas'])      # (N,)

    # Build covariance matrices from scales & rotations
    covs = build_covariance(rots, scales)    # (N, 3, 3)
    
    return gaussian_density_and_grad(x, mus, covs, alphas)[0]

def compute_density_grad(x, gs):
    mus = torch.tensor(gs['positions'])
    scales = torch.tensor(gs['scales'])
    rots = torch.tensor(gs['rotations'])
    alphas = torch.tensor(gs['alphas'])

    covs = build_covariance(rots, scales)
    return gaussian_density_and_grad(x, mus, covs, alphas)[1]

def build_covariance(rot_q, scale_vec):
    """
    Converts quaternion rotation and scale to full covariance matrix
    Returns: (N, 3, 3) covariance tensors
    """
    # TODO: Implement quaternion to rotation matrix
    # TODO: Compute Σ = R diag(s²) Rᵀ
    pass

def gaussian_density_and_grad(x, mus, covs, alphas):
    """
    Same implementation as previously provided in PyTorch.
    """
    pass
