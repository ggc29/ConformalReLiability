"""Train DQR/CReL on real-world feature latents.

This script first obtains ``Z_hat = VAE.encode(y_hat)`` for the VAE validation
split, then trains a directional quantile regressor from condition features
``x`` to generated-output latents ``z_hat``.
"""

import argparse
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import common
from dataset import FeatureTripletDataset
from helper import generate_directions, set_seeds
from losses import multivariate_qr_loss
from quantile_model import MultivariateQuantileModel
from feature_vae import VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQR on VAE latents of generated target features.")
    common.add_arguments(parser)
    return parser.parse_args()


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


def build_or_load_z_hat(data_args, vae_args, vae_root, device, checkpoint, signature):
    """Create or reuse cached VAE latents for generated target features."""
    z_hat_path = os.path.join(vae_root, "Z_hat.pt")
    cached = common.load_cache(z_hat_path, signature)
    if cached is not None and cached.get("tensor_sha256") == common.tensor_digest(cached["value"]):
        common.restore_rng(cached["rng"])
        print(f"Loaded matching Z_hat cache from {z_hat_path}")
        return cached["value"]

    val_dataset = FeatureTripletDataset(**common.dataset_kwargs(data_args, task="vae", split="val"))
    val_loader = DataLoader(val_dataset, batch_size=vae_args["batch_size"], shuffle=False)
    _, y_dim = val_dataset.feature_dims()

    vae = VAE(
        y_dim=y_dim,
        z_dim=vae_args["z_dim"],
        device=device,
        train_size=len(val_dataset),
        mode=data_args.get("metric_name", "feature"),
        dropout=vae_args["dropout"],
        lr=vae_args["lr"],
        wd=vae_args["wd"],
        batch_norm=False,
        kl_mult=vae_args["kl_mult"],
    )
    vae.load_state_dict(common.safe_torch_load(checkpoint, map_location=device))
    vae.to(device)
    vae.eval()

    z_hat = []
    with torch.no_grad():
        for batch in val_loader:
            # DQR models the generated output distribution in latent space.
            y_hat = batch["y_hat"].to(device)
            z_hat.append(vae.encode(y_hat).cpu())
    z_hat = torch.cat(z_hat, dim=0)
    common.save_cache(z_hat_path, signature, z_hat, rng=common.rng_state(),
                      tensor_sha256=common.tensor_digest(z_hat))
    print(f"Embedded Z_hat saved to {z_hat_path}")
    return z_hat


def train_dqr_model(model_ens, train_loader, val_loader, train_args, z_dim, device, signature):
    """Train the directional quantile regressor with early stopping."""
    tau_list = train_args["tau_list"]
    best_va_loss = None
    best_epoch = 0
    best_model = deepcopy(model_ens.model)
    train_losses = []
    val_losses = []

    for ep in tqdm(range(train_args["num_epochs"]), desc="Training DQR"):
        if model_ens.done_training:
            break

        ep_train_loss = []
        for batch in tqdm(train_loader, desc=f"Epoch {ep + 1}", leave=False):
            x = batch["x"].to(device)
            z = batch["z_hat"].to(device)
            # Random directions define the halfspaces that make up the CReL set.
            u_list, gamma = generate_directions(z_dim, train_args["num_u"], tau_list[0])
            train_args["gamma"] = gamma
            loss_values = model_ens.loss(
                multivariate_qr_loss,
                x,
                z.view(-1, z_dim),
                u_list,
                batch_q=True,
                take_step=True,
                args=train_args,
            )
            ep_train_loss.append(float(np.nanmean(loss_values)))

        ep_val_loss = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                z = batch["z_hat"].to(device)
                u_list, gamma = generate_directions(z_dim, train_args["num_u"], tau_list[0])
                train_args["gamma"] = gamma
                loss_values = model_ens.loss(
                    multivariate_qr_loss,
                    x,
                    z.view(-1, z_dim),
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
    common.record_checkpoint(best_model_path, signature, best_epoch, best_va_loss)
    plot_loss_curves(train_losses, val_losses, save_root)
    print(f"Best DQR checkpoint saved to {best_model_path}")


def main():
    args = parse_args()
    set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = common.load_config(args.config, args.data_root)
    data_args = config["dataset"]
    vae_args = config["vae"]
    dqr_args = deepcopy(config["dqr"])

    vae_root = common.get_vae_root(data_args, vae_args)
    signature = common.training_signature(config)
    checkpoint = common.resolve_checkpoint(vae_root, signature)
    latent_sig = common.latent_signature(signature, checkpoint)
    z_hat = build_or_load_z_hat(data_args, vae_args, vae_root, device, checkpoint, latent_sig)
    dqr_sig = common.dqr_signature(config, latent_sig, z_hat)

    train_dataset = FeatureTripletDataset(**common.dataset_kwargs(data_args, task="dqr", split="train", z_hat=z_hat))
    val_dataset = FeatureTripletDataset(**common.dataset_kwargs(data_args, task="dqr", split="val", z_hat=z_hat))
    x_dim, _ = train_dataset.feature_dims()
    z_dim = vae_args["z_dim"]
    print(f"Inferred x_dim={x_dim}, z_dim={z_dim}")

    train_loader = DataLoader(train_dataset, batch_size=dqr_args["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=dqr_args["batch_size"], shuffle=False)

    dqr_args["conformalization_tau"] = dqr_args["tau"]
    # Train for slightly higher nominal coverage; conformal calibration will
    # then expand the learned region using validation distances.
    dqr_args["tau"] = dqr_args["tau"] - 0.03
    dqr_args["tau_list"] = torch.tensor([dqr_args["tau"]], device=device)

    save_root = common.get_dqr_root(vae_root, dqr_args)
    os.makedirs(save_root, exist_ok=True)
    dqr_args["save_root"] = save_root

    model_ens = MultivariateQuantileModel(
        input_size=x_dim,
        y_size=z_dim,
        hidden_dimensions=dqr_args["hidden_dimensions"],
        dropout=dqr_args["dropout"],
        lr=dqr_args["learning_rate"],
        wd=dqr_args["weight_decay"],
        num_ens=dqr_args["num_ens"],
        device=device,
    )
    train_dqr_model(model_ens, train_loader, val_loader, dqr_args, z_dim, device, dqr_sig)


if __name__ == "__main__":
    main()
