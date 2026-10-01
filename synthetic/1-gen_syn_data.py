"""Generate regression data for the synthetic CReL experiment.

Output filenames encode the data settings in ``configs/train.yaml`` and are
resolved from the same configuration by downstream scripts.
"""

import os

import numpy as np
import torch
import yaml

from helper import set_seeds


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_to_folder(folder_path, n, k, d, std, x, y, is_nonlinear):
    """Save X/Y tensors with a filename that encodes the data configuration."""
    os.makedirs(folder_path, exist_ok=True)
    nonlinear_tag = "nonlinear" if is_nonlinear else "linear"
    x_file_name = os.path.join(folder_path, f"n-{n}_xdim-{k}_ydim-{d}_std-{std}_{nonlinear_tag}_X.pt")
    y_file_name = os.path.join(folder_path, f"n-{n}_xdim-{k}_ydim-{d}_std-{std}_{nonlinear_tag}_Y.pt")
    torch.save(x, x_file_name)
    torch.save(y, y_file_name)
    print(f"Saved X to {x_file_name}")
    print(f"Saved Y to {y_file_name}")


def generate_linear_data(n_samples, p_features, d_response, noise_std=0.1):
    """Generate Y = X A + noise."""
    x = np.random.uniform(0.8, 3.2, size=(n_samples, p_features))
    a = np.random.normal(0, 1, size=(p_features, d_response))
    y = x @ a
    noise = np.random.normal(0, noise_std, size=(n_samples, d_response))
    return x, y + noise


def generate_polynomial_data(n_samples, p_features, d_response, noise_std=0.1, degree=2):
    """Generate a nonlinear polynomial response with additive Gaussian noise."""
    x = np.random.uniform(0.8, 3.2, size=(n_samples, p_features))
    a = np.random.normal(0, 1, size=(p_features, d_response))
    b = np.random.normal(0, 1, size=(p_features, d_response))
    y = x @ a + (x ** degree) @ b
    noise = np.random.normal(0, noise_std, size=(n_samples, d_response))
    return x, y + noise


def generate_and_save_data(n, k, d, std, folder_path, is_nonlinear=False):
    if is_nonlinear:
        x, y = generate_polynomial_data(n_samples=n, p_features=k, d_response=d, noise_std=std)
    else:
        x, y = generate_linear_data(n_samples=n, p_features=k, d_response=d, noise_std=std)
    save_to_folder(folder_path, n, k, d, std, x, y, is_nonlinear)


def main():
    set_seeds(42)
    config = load_config("./configs/train.yaml")
    data_args = config["dataset"]
    # All downstream scripts expect this exact folder and filename convention.
    generate_and_save_data(
        n=data_args["num"],
        k=data_args["x_dim"],
        d=data_args["y_dim"],
        std=data_args["std"],
        folder_path="./generated_data",
        is_nonlinear=data_args["is_nonlinear"],
    )


if __name__ == "__main__":
    main()
