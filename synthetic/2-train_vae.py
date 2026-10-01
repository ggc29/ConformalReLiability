"""Train the VAE used to embed synthetic responses into latent space.

The DQR/CReL model is trained in latent space, so this script must be run before
``3-train_dqr.py``. The VAE is trained only on the response variable Y.
"""

import math
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets import get_split_data
from experiment import record_checkpoint, training_signature, validate_synthetic_config
from helper import safe_torch_load, set_seeds
from vae import VAE


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_data_paths(data_args):
    """Recover generated data paths from the YAML dataset settings."""
    nonlinear_tag = "nonlinear" if data_args["is_nonlinear"] else "linear"
    data_name = (
        f"n-{data_args['num']}_xdim-{data_args['x_dim']}_ydim-{data_args['y_dim']}"
        f"_std-{data_args['std']}_{nonlinear_tag}"
    )
    return os.path.join("./generated_data", f"{data_name}_X.pt"), os.path.join("./generated_data", f"{data_name}_Y.pt")


def get_vae_root(data_args, vae_args):
    """Create the checkpoint folder name used by later scripts."""
    nonlinear_tag = "nonlinear" if data_args["is_nonlinear"] else "linear"
    experiment_root = (
        f"./results/n-{data_args['num']}_xdim-{data_args['x_dim']}_ydim-{data_args['y_dim']}"
        f"_std-{data_args['std']}_{nonlinear_tag}"
    )
    return os.path.join(
        experiment_root,
        f"VAE-{vae_args['loss_name']}-zdim_{vae_args['z_dim']}-klmult_{vae_args['kl_mult']}",
    )


def train_vae_model(vae, y_train, y_val, batch_size, wait, num_ep):
    """Train with early stopping on validation reconstruction/KL loss."""
    loader = DataLoader(TensorDataset(y_train), shuffle=True, batch_size=batch_size)
    vae = best_vae_model = deepcopy(vae)

    val_losses = []
    train_losses = []
    kl_losses = []
    rec_losses = []
    best_va_loss = None
    best_epoch = 0

    pbar = tqdm(range(num_ep), desc="Training VAE")
    for ep in pbar:
        ep_train_loss = []
        for (yi,) in loader:
            # The synthetic VAE is unconditional: it reconstructs Y without X.
            loss, *_ = vae.loss(yi, take_vae_grad_step=True, take_adversary_grad_step=False)
            ep_train_loss.append(loss.detach().cpu().item())

        ep_tr_loss = np.nanmean(np.stack(ep_train_loss, axis=0), axis=0).item()
        train_losses.append(ep_tr_loss)

        with torch.no_grad():
            ep_va_loss, data_loss, kl_loss, _, _ = vae.loss(y_val)
            ep_va_loss = ep_va_loss.item()
            data_loss = data_loss.item()
            kl_loss = kl_loss.item()
            assert not math.isnan(ep_va_loss)

        val_losses.append(ep_va_loss)
        kl_losses.append(kl_loss)
        rec_losses.append(data_loss)
        pbar.set_postfix(val_loss=f"{ep_va_loss:.4f}")

        if best_va_loss is None or ep_va_loss < best_va_loss:
            best_epoch = ep
            best_va_loss = ep_va_loss
            best_vae_model = deepcopy(vae)
        elif ep - best_epoch > wait:
            break

    return best_epoch, best_va_loss, best_vae_model, val_losses, train_losses, kl_losses, rec_losses


def plot_losses(save_root, best_epoch, val_losses, train_losses, kl_losses, rec_losses, kl_mult):
    """Save diagnostic curves for the VAE training run."""
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label=f"Train Loss {train_losses[best_epoch]:.4f}")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label=f"Val Loss {val_losses[best_epoch]:.4f}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_root, "loss_curves.png"))

    plt.figure()
    plt.plot(range(1, len(rec_losses) + 1), rec_losses, label=f"Reconstruction {rec_losses[best_epoch]:.4f}")
    plt.plot(
        range(1, len(kl_losses) + 1),
        [kl / kl_mult for kl in kl_losses],
        label=f"KL raw {kl_losses[best_epoch]:.4f}",
        linestyle="--",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Components")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_root, "loss_curves2_raw_KL.png"))
    plt.close("all")


def main():
    set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config("./configs/train.yaml")
    validate_synthetic_config(config)
    data_args = config["dataset"]
    vae_args = config["vae"]

    x_path, y_path = get_data_paths(data_args)
    x = safe_torch_load(x_path)
    y = safe_torch_load(y_path)
    data = get_split_data(X=x, Y=y, task="vae", scale=data_args["scale"])
    # Scaling is fitted on the VAE training split inside get_split_data.
    y_train = data["y_train"].to(device)
    y_val = data["y_val"].to(device)

    vae = VAE(
        y_dim=data_args["y_dim"],
        z_dim=vae_args["z_dim"],
        device=device,
        train_size=len(y_train),
        mode=vae_args["loss_name"],
        dropout=vae_args["dropout"],
        lr=vae_args["lr"],
        wd=vae_args["wd"],
        batch_norm=False,
        kl_mult=vae_args["kl_mult"],
    )

    save_root = get_vae_root(data_args, vae_args)
    os.makedirs(save_root, exist_ok=True)

    # The best checkpoint is selected by validation loss and reused by DQR.
    best_epoch, best_va_loss, best_vae_model, val_losses, train_losses, kl_losses, rec_losses = train_vae_model(
        vae,
        y_train,
        y_val,
        batch_size=vae_args["batch_size"],
        wait=vae_args["wait"],
        num_ep=vae_args["num_ep"],
    )

    best_model_path = os.path.join(save_root, f"epoch_{best_epoch}-loss_{best_va_loss:.4f}.pth")
    torch.save(best_vae_model.state_dict(), best_model_path)
    record_checkpoint(
        best_model_path, training_signature(config, [x_path, y_path]),
        best_epoch, best_va_loss,
    )
    plot_losses(save_root, best_epoch, val_losses, train_losses, kl_losses, rec_losses, vae_args["kl_mult"])
    print(f"Best VAE checkpoint saved to {best_model_path}")


if __name__ == "__main__":
    main()
