"""Calibrate and test CReL for the synthetic experiment.

By default it reports latent-space calibration and test coverage. Passing
``--compute-y-space`` additionally decodes calibrated latent sets and estimates
their coverage/size in output space using grids.
"""

import argparse
import json
import os
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets import get_split_data
from distance import if_in_region, quadratic_programming_to_find_distance
from experiment import file_digest, resolve_checkpoint, training_signature, validate_synthetic_config
from helper import generate_directions, get_grid, get_min_distance, safe_torch_load, set_seeds
from losses import predict_y
from quantile_model import MultivariateQuantileModel
from vae import VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate and test CReL on synthetic regression.")
    parser.add_argument("--alpha", type=float, default=0.02, help="Miscoverage level for conformal calibration.")
    parser.add_argument("--num-test", type=int, default=4000, help="Number of test samples for optional Y-space CReL.")
    parser.add_argument("--z-grid-total", type=float, default=2e4, help="Approximate number of latent grid points.")
    parser.add_argument("--y-eval-grid-total", type=float, default=2e4, help="Approximate number of Y-grid points.")
    parser.add_argument("--compute-y-space", action="store_true", help="Decode calibrated sets and save Y-space stats.")
    parser.add_argument("--recompute-distances", action="store_true", help="Recompute distances even if a matching cache exists.")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def data_name(data_args):
    nonlinear_tag = "nonlinear" if data_args["is_nonlinear"] else "linear"
    return (
        f"n-{data_args['num']}_xdim-{data_args['x_dim']}_ydim-{data_args['y_dim']}"
        f"_std-{data_args['std']}_{nonlinear_tag}"
    )


def get_vae_root(data_args, vae_args):
    return os.path.join(
        "./results",
        data_name(data_args),
        f"VAE-{vae_args['loss_name']}-zdim_{vae_args['z_dim']}-klmult_{vae_args['kl_mult']}",
    )


def get_dqr_root(vae_root, dqr_args):
    return os.path.join(
        vae_root,
        (
            f"dqr-tau_{dqr_args['conformalization_tau']}-numu_{dqr_args['num_u']}"
            f"-hd_{dqr_args['hidden_dimensions']}-wait_{dqr_args['patience']}"
            f"-dropout_{dqr_args['dropout']}-lr-{dqr_args['learning_rate']}"
            f"-wd_{dqr_args['weight_decay']}"
        ),
    )


def calculate_distances(loader, model, train_args, z_dim, device, desc):
    """Compute distances from encoded samples to predicted DQR regions."""
    distances = []
    for x_batch, z_batch in tqdm(loader, desc=desc):
        xi = x_batch.float().to(device)
        zi = z_batch.float().to(device).view(-1, z_dim)

        u_list, gamma = generate_directions(z_dim, train_args["num_u"], train_args["tau_list"][0])
        train_args["gamma"] = gamma
        b = predict_y(model, xi, u_list, get_unflattened=True).float()

        # Convert the halfspace representation to numpy for the QP solver.
        w_np = zi.squeeze().cpu().detach().numpy()
        b_np = b.squeeze().cpu().detach().numpy()
        u_np = u_list.cpu().detach().numpy()

        if if_in_region(w_np, u_np, b_np):
            distance = 0.0
        else:
            try:
                _, distance = quadratic_programming_to_find_distance(w_np, u_np, b_np)
            except ValueError as err:
                if "OSQP did not find an optimal solution" in str(err):
                    # Assign zero distance on solver failure, as specified by
                    # the experimental protocol; region membership is unverified.
                    distance = 0.0
                else:
                    raise

        distances.append(float(distance))
    return distances


def calibrate_radius(distances, alpha):
    """Compute the conformal radius using NumPy's interpolated quantile."""
    n = len(distances)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    q = min(q, 1.0)
    return float(np.quantile(distances, q=q))


def coverage_ratio(distances, radius):
    """Compute coverage using distance < radius, including when radius is zero."""
    coverage = np.asarray(distances) < radius
    return float(np.sum(coverage != 0) / len(distances))


def point_in_y_region(yi, y_in_region):
    """Approximate membership in a decoded set using nearest-neighbor distance."""
    if len(y_in_region) == 0:
        return False
    in_region_distances = get_min_distance(
        y_in_region, y_in_region, ignore_zero_distance=True, y_batch_size=10000
    )
    in_region_threshold = torch.quantile(in_region_distances, q=0.9).item()
    distance = get_min_distance(yi, y_in_region)
    return bool((distance <= in_region_threshold).item())


def evaluate_y_space(model, vae, data, train_args, z_train, radius, args, dqr_root, device):
    """Decode latent CReL sets and estimate output-space coverage and size."""
    y_dim = data["y_test"].shape[1]
    x_test = data["x_test"].to(device)
    y_test = data["y_test"].to(device)
    y_train = data["y_train"].to(device)
    test_loader = DataLoader(TensorDataset(x_test[:args.num_test], y_test[:args.num_test]), shuffle=False, batch_size=1)

    save_dir_name = (
        f"crel-alpha_{args.alpha}-numtest_{args.num_test}"
        f"-z_grid_{args.z_grid_total}-y_grid_{args.y_eval_grid_total}"
    )
    save_root = os.path.join(dqr_root, save_dir_name)
    os.makedirs(save_root, exist_ok=True)

    z_grid_size = args.z_grid_total ** (1 / z_train.shape[1])
    # Grid points approximate the latent set because exact set enumeration is
    # not available for a neural DQR boundary.
    z_grid = get_grid(z_train, z_grid_size, z_train.shape[1], pad=1).to(device)
    y_eval_grid_size = args.y_eval_grid_total ** (1 / y_train.shape[1])
    y_eval_grid = get_grid(y_train, y_eval_grid_size, y_train.shape[1], pad=0.5).to(device)
    u_list, _ = generate_directions(z_train.shape[1], train_args["num_u"], train_args["tau_list"][0])
    u_list = u_list.to(device)

    coverage_ori_all = []
    coverage_cal_all = []
    region_sizes_all = []

    for x_batch, y_batch in tqdm(test_loader, desc="Testing decoded CReL sets"):
        with torch.no_grad():
            b = predict_y(model, x_batch.to(device), u_list, get_unflattened=True).float().squeeze()
            dot_products = torch.matmul(z_grid, u_list.T)
            in_ori_region_mask = torch.all(dot_products >= b, dim=1)
            z_in_ori_region = z_grid[in_ori_region_mask]
            z_outside_region = z_grid[~in_ori_region_mask]

            # Calibration expands the original region by radius in latent space.
            if len(z_in_ori_region) > 0 and len(z_outside_region) > 0:
                z_outside_distance = get_min_distance(z_outside_region, z_in_ori_region)
                z_to_add = z_outside_region[z_outside_distance <= radius]
                z_in_cal_region = torch.cat([z_in_ori_region, z_to_add], dim=0)
            else:
                z_in_cal_region = z_in_ori_region

            y_in_ori_region = vae.decode(z_in_ori_region) if len(z_in_ori_region) > 0 else torch.empty(0, y_dim).to(device)
            y_in_cal_region = vae.decode(z_in_cal_region) if len(z_in_cal_region) > 0 else torch.empty(0, y_dim).to(device)

            coverage_ori_all.append(1 if point_in_y_region(y_batch.to(device), y_in_ori_region) else 0)
            coverage_cal_all.append(1 if point_in_y_region(y_batch.to(device), y_in_cal_region) else 0)

            if len(y_in_cal_region) > 0:
                in_region_distances = get_min_distance(
                    y_in_cal_region, y_in_cal_region, ignore_zero_distance=True, y_batch_size=10000
                )
                in_region_threshold = torch.quantile(in_region_distances, q=0.9).item()
                distances_from_grid = get_min_distance(y_eval_grid, y_in_cal_region, y_batch_size=10000)
                region_sizes_all.append(int(torch.sum(distances_from_grid <= in_region_threshold).item()))
            else:
                region_sizes_all.append(0)

    results = {
        "alpha": args.alpha,
        "radius": radius,
        "num_test": len(test_loader.dataset),
        "z_grid_points": len(z_grid),
        "y_grid_points": len(y_eval_grid),
        "area_unit": "grid points",
        "coverage_ori_all": coverage_ori_all,
        "coverage_cal_all": coverage_cal_all,
        "region_sizes_all": region_sizes_all,
        "statistics": {
            "uncalibrated_coverage_ratio": float(np.mean(coverage_ori_all)) if coverage_ori_all else 0.0,
            "calibrated_coverage_ratio": float(np.mean(coverage_cal_all)) if coverage_cal_all else 0.0,
            "avg_region_size": float(np.mean(region_sizes_all)) if region_sizes_all else 0.0,
            "std_region_size": float(np.std(region_sizes_all)) if region_sizes_all else 0.0,
        },
    }
    output_path = os.path.join(save_root, "coverage_and_size_statistics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Y-space CReL statistics saved to {output_path}")


def main():
    args = parse_args()
    set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config("./configs/train.yaml")
    validate_synthetic_config(config)
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if args.num_test <= 0 or args.z_grid_total <= 0 or args.y_eval_grid_total <= 0:
        raise ValueError("Test sample and grid sizes must be positive.")
    data_args = config["dataset"]
    vae_args = config["vae"]
    dqr_args = deepcopy(config["dqr"])

    x_path = os.path.join("./generated_data", f"{data_name(data_args)}_X.pt")
    y_path = os.path.join("./generated_data", f"{data_name(data_args)}_Y.pt")
    x = safe_torch_load(x_path)
    y = safe_torch_load(y_path)
    data = get_split_data(X=x, Y=y, task="dqr", scale=data_args["scale"])

    vae = VAE(
        y_dim=data_args["y_dim"],
        z_dim=vae_args["z_dim"],
        device=device,
        train_size=len(data["x_train"]),
        mode=vae_args["loss_name"],
        dropout=vae_args["dropout"],
        lr=vae_args["lr"],
        wd=vae_args["wd"],
        batch_norm=False,
        kl_mult=vae_args["kl_mult"],
    )
    vae_root = get_vae_root(data_args, vae_args)
    vae_checkpoint = resolve_checkpoint(vae_root, training_signature(config, [x_path, y_path]))
    vae.load_state_dict(safe_torch_load(vae_checkpoint, map_location=device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    dqr_args["conformalization_tau"] = dqr_args["tau"]
    dqr_args["tau"] = dqr_args["tau"] - 0.03
    dqr_args["tau_list"] = torch.tensor([dqr_args["tau"]], device=device)
    dqr_root = get_dqr_root(vae_root, dqr_args)

    model_ens = MultivariateQuantileModel(
        input_size=data_args["x_dim"],
        y_size=vae_args["z_dim"],
        hidden_dimensions=dqr_args["hidden_dimensions"],
        dropout=dqr_args["dropout"],
        lr=dqr_args["learning_rate"],
        wd=dqr_args["weight_decay"],
        num_ens=dqr_args["num_ens"],
        device=device,
    )
    signature = training_signature(config, [x_path, y_path], vae_checkpoint)
    dqr_checkpoint = resolve_checkpoint(dqr_root, signature)
    model_ens.model[0].load_state_dict(safe_torch_load(dqr_checkpoint, map_location=device))
    model = model_ens.model[0]
    model.eval()

    with torch.no_grad():
        y_train = data["y_train"].to(device)
        y_val = data["y_val"].to(device)
        y_test = data["y_test"].to(device)
        z_train = vae.encode(y_train)
        z_val = vae.encode(y_val)
        z_test = vae.encode(y_test)

    dist_file_path = os.path.join(dqr_root, "dist_list.pt")
    # Alpha does not affect distances: both nominal levels can reuse one cache.
    # Model/data/code changes do affect distances and must invalidate it.
    cache_signature = {
        "training": signature,
        "dqr_sha256": file_digest(dqr_checkpoint),
        "code_sha256": {
            name: file_digest(Path(__file__).parent / name)
            for name in [Path(__file__).name, "helper.py", "distance.py", "datasets.py", "vae.py", "losses.py", "quantile_model.py"]
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scipy_version": version("scipy"),
        "sklearn_version": version("scikit-learn"),
        "osqp_version": version("osqp"),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    dist_all = None
    if os.path.exists(dist_file_path) and not args.recompute_distances:
        cached = safe_torch_load(dist_file_path, map_location="cpu")
        if isinstance(cached, dict) and cached.get("signature") == cache_signature:
            dist_all = cached["distances"]
            print(f"Loaded matching latent distance cache from {dist_file_path}")
        else:
            print("Distance cache does not match this run; recomputing.")
    if dist_all is None:
        val_loader = DataLoader(TensorDataset(data["x_val"], z_val.cpu()), shuffle=False, batch_size=1)
        test_loader = DataLoader(TensorDataset(data["x_test"], z_test.cpu()), shuffle=False, batch_size=1)
        dist_all = {
            "val": calculate_distances(val_loader, model, dqr_args, vae_args["z_dim"], device, "Validation distances"),
            "test": calculate_distances(test_loader, model, dqr_args, vae_args["z_dim"], device, "Test distances"),
        }
        temporary_cache = dist_file_path + ".tmp"
        torch.save({"signature": cache_signature, "distances": dist_all}, temporary_cache)
        os.replace(temporary_cache, dist_file_path)
        print(f"Saved latent distance cache to {dist_file_path}")

    radius = calibrate_radius(dist_all["val"], args.alpha)
    # The radius is calibrated only on validation distances and then reused on
    # test distances to estimate held-out coverage.
    val_coverage = coverage_ratio(dist_all["val"], radius)
    test_coverage = coverage_ratio(dist_all["test"], radius)

    summary = {
        "alpha": args.alpha,
        "radius": radius,
        "validation_coverage": val_coverage,
        "test_coverage": test_coverage,
        "num_calibration": len(dist_all["val"]),
        "num_test": len(dist_all["test"]),
        "vae_checkpoint": vae_checkpoint,
        "dqr_checkpoint": dqr_checkpoint,
    }
    summary_path = os.path.join(dqr_root, f"crel_calibration_test_alpha_{args.alpha}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Calibrated radius: {radius:.6f}")
    print(f"Validation coverage after calibration: {val_coverage:.4f}")
    print(f"Test coverage after calibration: {test_coverage:.4f}")
    print(f"Summary saved to {summary_path}")

    if args.compute_y_space:
        evaluate_y_space(model, vae, data, dqr_args, z_train, radius, args, dqr_root, device)


if __name__ == "__main__":
    main()
