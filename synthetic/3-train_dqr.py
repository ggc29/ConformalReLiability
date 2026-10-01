"""Train DQR to construct CReL prediction sets in VAE latent space."""

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets import get_split_data
from experiment import record_checkpoint, resolve_checkpoint, training_signature, validate_synthetic_config
from helper import generate_directions, safe_torch_load, set_seeds
from losses import multivariate_qr_loss
from quantile_model import MultivariateQuantileModel
from vae import VAE


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def data_name(data_args):
    """Return the shared experiment name derived from synthetic data settings."""
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
    """Create the DQR checkpoint folder name from calibration/training settings."""
    return os.path.join(
        vae_root,
        (
            f"dqr-tau_{dqr_args['conformalization_tau']}-numu_{dqr_args['num_u']}"
            f"-hd_{dqr_args['hidden_dimensions']}-wait_{dqr_args['patience']}"
            f"-dropout_{dqr_args['dropout']}-lr-{dqr_args['learning_rate']}"
            f"-wd_{dqr_args['weight_decay']}"
        ),
    )


def plot_loss_curves(train_losses, val_losses, save_root):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss Curves")
    plt.legend()
    plt.savefig(os.path.join(save_root, "loss_curves.png"))
    plt.close()


def train_dqr_model(model_ens, train_loader, val_loader, train_args, z_dim, device, signature):
    """Train the directional quantile regressor on (X, Z) pairs."""
    tau_list = train_args["tau_list"]
    assert len(tau_list) == 1, "Only a single tau value is supported."

    best_va_loss = None
    best_epoch = 0
    best_model = deepcopy(model_ens.model)
    train_losses = []
    val_losses = []

    for ep in tqdm(range(train_args["num_epochs"]), desc="Training DQR"):
        if model_ens.done_training:
            break

        ep_train_loss = []
        for x_batch, z_batch in tqdm(train_loader, desc=f"Epoch {ep + 1}", leave=False):
            x_batch = x_batch.to(device)
            z_batch = z_batch.to(device)
            # A random set of directions converts the multivariate region
            # problem into many projected one-dimensional quantile losses.
            u_list, gamma = generate_directions(z_dim, train_args["num_u"], tau_list[0])
            train_args["gamma"] = gamma
            loss_values = model_ens.loss(
                multivariate_qr_loss,
                x_batch,
                z_batch.view(-1, z_dim),
                u_list,
                batch_q=True,
                take_step=True,
                args=train_args,
            )
            ep_train_loss.append(float(np.nanmean(loss_values)))

        ep_val_loss = []
        with torch.no_grad():
            for x_batch, z_batch in val_loader:
                x_batch = x_batch.to(device)
                z_batch = z_batch.to(device)
                u_list, gamma = generate_directions(z_dim, train_args["num_u"], tau_list[0])
                train_args["gamma"] = gamma
                loss_values = model_ens.loss(
                    multivariate_qr_loss,
                    x_batch,
                    z_batch.view(-1, z_dim),
                    u_list,
                    batch_q=True,
                    take_step=False,
                    args=train_args,
                )
                ep_val_loss.append(float(np.nanmean(loss_values)))

        avg_train_loss = float(np.mean(ep_train_loss))
        avg_val_loss = float(np.mean(ep_val_loss))
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if best_va_loss is None or avg_val_loss < best_va_loss:
            best_va_loss = avg_val_loss
            best_epoch = ep
            best_model = deepcopy(model_ens.model)
        elif ep - best_epoch > train_args["patience"]:
            print(f"Early stopping at epoch {ep}. Best epoch: {best_epoch}, val loss: {best_va_loss:.4f}")
            model_ens.done_training = True
            break

        print(f"Epoch {ep + 1}: train loss {avg_train_loss:.4f}, val loss {avg_val_loss:.4f}")

    save_root = train_args["save_root"]
    os.makedirs(save_root, exist_ok=True)
    best_model_path = os.path.join(save_root, f"best_DQR_model_epoch_{best_epoch}.pth")
    torch.save(best_model[0].state_dict(), best_model_path)
    record_checkpoint(best_model_path, signature, best_epoch, best_va_loss)
    plot_loss_curves(train_losses, val_losses, save_root)
    print(f"Best DQR checkpoint saved to {best_model_path}")
    return best_model


def main():
    set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config("./configs/train.yaml")
    validate_synthetic_config(config)
    data_args = config["dataset"]
    vae_args = config["vae"]
    dqr_args = deepcopy(config["dqr"])

    x_path = os.path.join("./generated_data", f"{data_name(data_args)}_X.pt")
    y_path = os.path.join("./generated_data", f"{data_name(data_args)}_Y.pt")
    x = safe_torch_load(x_path)
    y = safe_torch_load(y_path)

    data = get_split_data(X=x, Y=y, task="dqr", scale=data_args["scale"])
    x_train = data["x_train"].to(device)
    y_train = data["y_train"].to(device)
    x_val = data["x_val"].to(device)
    y_val = data["y_val"].to(device)

    vae = VAE(
        y_dim=data_args["y_dim"],
        z_dim=vae_args["z_dim"],
        device=device,
        train_size=len(x_train),
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
    print(f"VAE checkpoint loaded from {vae_root}")

    with torch.no_grad():
        # DQR sees the VAE latent representation of Y, not the raw response.
        z_train = vae.encode(y_train)
        z_val = vae.encode(y_val)

    dqr_args["conformalization_tau"] = dqr_args["tau"]
    # Train the base region for slightly higher coverage. Calibration then
    # expands it to the requested conformal level.
    dqr_args["tau"] = dqr_args["tau"] - 0.03
    dqr_args["tau_list"] = torch.tensor([dqr_args["tau"]], device=device)

    train_loader = DataLoader(TensorDataset(x_train, z_train), shuffle=True, batch_size=dqr_args["batch_size"])
    val_loader = DataLoader(TensorDataset(x_val, z_val), shuffle=False, batch_size=dqr_args["batch_size"])

    save_root = get_dqr_root(vae_root, dqr_args)
    os.makedirs(save_root, exist_ok=True)
    dqr_args["save_root"] = save_root

    model_ens = MultivariateQuantileModel(
        # The model input is [x, direction], and the output is one threshold.
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
    train_dqr_model(model_ens, train_loader, val_loader, dqr_args, vae_args["z_dim"], device, signature)


if __name__ == "__main__":
    main()
