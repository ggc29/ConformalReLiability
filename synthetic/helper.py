"""Shared utilities for the synthetic CReL experiments.

Utilities for reproducible random sampling, directional quantile regression,
and grid construction for approximate prediction-set evaluation.
"""

import copy
import os
import random

import numpy as np
import torch
from scipy.stats import ortho_group


device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

_cached_u_list = None
_cached_gamma = None
USE_SAME_DIRECTIONS = True
MAX_DIRECTIONS = 2048


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, map_location=None):
    """Load PyTorch files across old and new PyTorch versions.

    Pass ``weights_only=False`` when supported; otherwise use the compatible
    loading signature without that argument.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def create_folder_if_it_doesnt_exist(path):
    os.makedirs(path, exist_ok=True)


def generate_direction_aux(direction_shape, num_vectors, tau):
    """Sample random directions used to define halfspaces in latent space."""
    if direction_shape == 1:
        assert num_vectors % 2 == 0
        u_list = torch.tensor([tau] * (num_vectors // 2) + [-tau] * (num_vectors // 2)).unsqueeze(1).to(device)
        gamma = torch.zeros(num_vectors).to(device)
        idx = np.random.permutation(len(u_list))
        return u_list[idx], gamma

    unitaries = ortho_group.rvs(dim=direction_shape, size=num_vectors)
    unitaries = torch.tensor(unitaries, dtype=torch.float32).to(device)
    u_list = unitaries[:, :, -1]
    gamma = unitaries[:, :, :-1]
    return u_list.to(device), gamma.to(device)


def generate_directions(direction_shape, num_vectors, tau, return_idx=False):
    """Select and shuffle directions from a cached sampling pool.

    Use the first ``num_vectors`` directions from the pool and shuffle only
    their order. Requesting the full pool preserves its order.
    """
    if not USE_SAME_DIRECTIONS:
        return generate_direction_aux(direction_shape, num_vectors, tau)

    global _cached_u_list, _cached_gamma
    if _cached_u_list is None:
        _cached_u_list, _cached_gamma = generate_direction_aux(direction_shape, MAX_DIRECTIONS, tau)

    if num_vectors > MAX_DIRECTIONS:
        raise ValueError(f"num_vectors must be <= {MAX_DIRECTIONS}.")

    if num_vectors == MAX_DIRECTIONS:
        return copy.deepcopy(_cached_u_list), copy.deepcopy(_cached_gamma)

    idx = np.random.permutation(num_vectors)
    if return_idx:
        return copy.deepcopy(_cached_u_list[idx]), copy.deepcopy(_cached_gamma[idx]), idx
    return copy.deepcopy(_cached_u_list[idx]), copy.deepcopy(_cached_gamma[idx])


def get_min_distance(y, points, ignore_zero_distance=False, y_batch_size=50, points_batch_size=10000):
    """Compute nearest-neighbor distances from each row of ``y`` to ``points``.

    The batching keeps memory use bounded when CReL sets are represented by many
    sampled/grid points.
    """
    min_dists_from_points = []

    for i in range(0, y.shape[0], y_batch_size):
        yi = y[i:min(i + y_batch_size, y.shape[0])]
        yi_min_dists_from_points = []
        for j in range(0, points.shape[0], points_batch_size):
            pts = points[j:min(j + points_batch_size, points.shape[0])]
            dist_from_pts = (yi - pts.unsqueeze(1).repeat(1, yi.shape[0], 1)).norm(dim=-1)
            if ignore_zero_distance:
                dist_from_pts[dist_from_pts == 0] = np.inf
            yi_min_dists_from_points.append(dist_from_pts.min(dim=0)[0])

        if yi_min_dists_from_points:
            min_dists_from_points.append(torch.stack(yi_min_dists_from_points))

    if not min_dists_from_points:
        return torch.tensor([np.inf], device=y.device).repeat(len(y))
    return torch.cat(min_dists_from_points, dim=1).min(dim=0)[0]


def get_grid_borders_and_stride(y_train, grid_size, pad=0.0):
    """Build robust grid borders from central training quantiles."""
    q = 0.01
    border_min = y_train.quantile(q, dim=0)
    border_max = y_train.quantile(1 - q, dim=0)
    border_max += pad
    border_min -= pad
    stride = (border_max - border_min) / grid_size
    return border_max, border_min, stride


def get_grid_from_borders(border_max, border_min, stride, target_device):
    """Materialize a Cartesian product grid from per-dimension ranges."""
    shifts = [
        torch.arange(border_min[i], border_max[i], step=stride[i], dtype=torch.float32, device=target_device)
        for i in range(border_max.shape[0])
    ]
    return torch.cartesian_prod(*shifts)


def get_grid(y_train, grid_size, grid_shape, pad=0.0, get_grid_area=False):
    """Create an evaluation grid covering the empirical support of training data."""
    border_max, border_min, stride = get_grid_borders_and_stride(y_train, grid_size, pad)
    grid = get_grid_from_borders(border_max, border_min, stride, y_train.device)
    if len(grid.shape) == 1:
        grid = grid.unsqueeze(1)

    if get_grid_area:
        grid_area = (border_max - border_min).prod().item()
        return grid, grid_area
    return grid
