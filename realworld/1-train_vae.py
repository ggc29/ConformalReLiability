"""Train the real-world VAE on generated target features.

For I2T/T2I, raw data is not consumed here. The script only sees feature
triplets and trains the VAE on ``y_hat`` so that generated outputs can be mapped
to a compact latent space before DQR/CReL training.
"""

import argparse
import math
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import common
from dataset import FeatureTripletDataset
from helper import set_seeds
from feature_vae import VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Train the VAE on generated target features.")
    common.add_arguments(parser)
    return parser.parse_args()


def train_vae_model(vae, train_loader, val_loader, wait, num_ep, device):
    """Train VAE with early stopping on validation loss."""
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
        for batch in train_loader:
            # The real-world latent model is learned from generated target
            # features, not from ground-truth target features.
            y_hat = batch["y_hat"].to(device)
            loss = vae.loss(y=y_hat, take_vae_grad_step=True, take_adversary_grad_step=False)[0]
            ep_train_loss.append(loss.detach().cpu().item())

        train_losses.append(np.nanmean(np.stack(ep_train_loss, axis=0), axis=0).item())

        # Validation retains training-mode dropout and BatchNorm behavior
        # according to the experimental protocol.
        with torch.no_grad():
            val_losses_epoch = []
            rec_losses_epoch = []
            kl_losses_epoch = []
            for batch in val_loader:
                y_hat = batch["y_hat"].to(device)
                loss, data_loss, kl_loss, _, _ = vae.loss(y=y_hat)
                val_losses_epoch.append(loss.item())
                rec_losses_epoch.append(data_loss.item())
                kl_losses_epoch.append(kl_loss.item())

            ep_va_loss = float(np.mean(val_losses_epoch))
            if math.isnan(ep_va_loss):
                raise ValueError("Validation loss is NaN.")
            data_loss = float(np.mean(rec_losses_epoch))
            kl_loss = float(np.mean(kl_losses_epoch))

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


def plot_losses(save_root, val_losses, train_losses, kl_losses, rec_losses, kl_mult):
    """Write lightweight diagnostics for the training run."""
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_root, "loss_curves.png"))

    plt.figure()
    plt.plot(range(1, len(rec_losses) + 1), rec_losses, label="Reconstruction Loss")
    plt.plot(range(1, len(kl_losses) + 1), [kl / kl_mult for kl in kl_losses], label="KL Loss (raw)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Components")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_root, "loss_curves2_raw_KL.png"))
    plt.close("all")


def main():
    args = parse_args()
    set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = common.load_config(args.config, args.data_root)
    signature = common.training_signature(config)
    data_args = config["dataset"]
    vae_args = config["vae"]

    train_dataset = FeatureTripletDataset(**common.dataset_kwargs(data_args, task="vae", split="train"))
    val_dataset = FeatureTripletDataset(**common.dataset_kwargs(data_args, task="vae", split="val"))
    # x_dim may differ from y_dim; only y_dim is needed by the VAE.
    x_dim, y_dim = train_dataset.feature_dims()
    print(f"Inferred feature dimensions: x_dim={x_dim}, y_dim={y_dim}")

    train_loader = DataLoader(train_dataset, batch_size=vae_args["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=vae_args["batch_size"], shuffle=False)

    vae = VAE(
        y_dim=y_dim,
        z_dim=vae_args["z_dim"],
        device=device,
        train_size=len(train_dataset),
        mode=data_args.get("metric_name", "feature"),
        dropout=vae_args["dropout"],
        lr=vae_args["lr"],
        wd=vae_args["wd"],
        batch_norm=False,
        kl_mult=vae_args["kl_mult"],
    )

    save_root = common.get_vae_root(data_args, vae_args)
    os.makedirs(save_root, exist_ok=True)

    # Downstream scripts discover the VAE checkpoint from this result folder.
    best_epoch, best_va_loss, best_vae_model, val_losses, train_losses, kl_losses, rec_losses = train_vae_model(
        vae,
        train_loader,
        val_loader,
        wait=vae_args["wait"],
        num_ep=vae_args["num_ep"],
        device=device,
    )

    best_model_path = os.path.join(save_root, f"epoch_{best_epoch}-loss_{best_va_loss:.4f}.pth")
    torch.save(best_vae_model.state_dict(), best_model_path)
    common.record_checkpoint(best_model_path, signature, best_epoch, best_va_loss)
    plot_losses(save_root, val_losses, train_losses, kl_losses, rec_losses, vae_args["kl_mult"])
    print(f"Best VAE checkpoint saved to {best_model_path}")


if __name__ == "__main__":
    main()
