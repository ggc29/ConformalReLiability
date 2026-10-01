"""VAE used to map generated outputs into a lower-dimensional latent space."""

import functools

import torch
import torch.nn as nn


def get_non_linearity(layer_type="relu"):
    if layer_type == "relu":
        return functools.partial(nn.ReLU, inplace=True)
    if layer_type == "lrelu":
        return functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    if layer_type == "elu":
        return functools.partial(nn.ELU, inplace=True)
    raise NotImplementedError(f"Unknown non-linearity: {layer_type}")


class BaseModel(nn.Module):
    """Configurable MLP block used by the VAE encoder and decoder."""

    def __init__(self, in_dim, out_dim, hidden_dims=None, n_layers=3, dropout=0.0,
                 bias=True, non_linearity="lrelu", batch_norm=False, last_layer=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64] * n_layers

        modules = [nn.Linear(in_dim, hidden_dims[0], bias=bias)]
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        if batch_norm:
            modules.append(nn.BatchNorm1d(hidden_dims[0]))
        modules.append(get_non_linearity(non_linearity)())

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias))
            if batch_norm:
                modules.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            modules.append(get_non_linearity(non_linearity)())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_dims[-1], out_dim, bias=bias))
        if last_layer is not None:
            modules.append(last_layer())

        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)


def optimizer_to(optim, device):
    """Move optimizer state tensors when a model is moved between devices."""
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class VAE(nn.Module):
    """Unconditional VAE for response embeddings with MSE and KL loss."""

    def __init__(self, y_dim, z_dim, device, train_size, mode, dropout=0.1,
                 lr=1e-3, wd=0.0, batch_norm=False, kl_mult=0.01):
        super().__init__()
        if y_dim <= 5:
            hidden_dims = [32, 64, 128, 256, 128, 64, 32]
        elif y_dim <= 8:
            hidden_dims = [64, 128, 256, 128, 64]
        elif y_dim <= 10:
            hidden_dims = [64, 128, 256, 512, 256, 128, 64]
        elif y_dim <= 25:
            hidden_dims = [64, 128, 256, 256, 128, 64]
        else:
            hidden_dims = [128, 256, 512, 512, 256, 128]

        self.kl_mult = kl_mult
        batch_norm = batch_norm or y_dim > 60
        encoded_dim = hidden_dims[-1]
        hidden_dims = hidden_dims[:-1]
        self.train_size = train_size

        self.features_encoder = BaseModel(
            y_dim, encoded_dim, hidden_dims, dropout=dropout, batch_norm=batch_norm
        ).to(device)
        self.features_decoder = BaseModel(
            encoded_dim, y_dim, hidden_dims, dropout=dropout, batch_norm=batch_norm
        ).to(device)
        self.mu_linear = torch.nn.Linear(encoded_dim, z_dim).to(device)
        self.sigma_linear = torch.nn.Linear(encoded_dim, z_dim).to(device)
        self.decoder = torch.nn.Linear(z_dim, encoded_dim).to(device)

        vae_params = []
        for model in [self.features_encoder, self.features_decoder, self.mu_linear, self.sigma_linear, self.decoder]:
            vae_params += list(model.parameters())

        self.optimizer_G = torch.optim.Adam(vae_params, lr=lr, weight_decay=wd)
        self.optimizers = [self.optimizer_G]
        self.z_dim = z_dim
        self.mode = mode

    def encode_and_reconstruct(self, y):
        z = self.encode(y)
        y_rec = self.decode(z)
        return y_rec, z

    def encode(self, y):
        z, _, _ = self.get_encode_parameters(y)
        return z

    def get_encode_parameters(self, y):
        """Return a sampled latent vector and posterior parameters."""
        encoded = self.features_encoder(y)
        mu = self.mu_linear(encoded)
        log_sigma2 = self.sigma_linear(encoded)
        std = torch.exp(0.5 * log_sigma2)
        z = mu + torch.randn_like(std) * std
        return z, mu, log_sigma2

    def decode(self, z):
        z_decoded = self.decoder(z)
        return self.features_decoder(z_decoded)

    def generator_losses(self, y):
        """Compute reconstruction plus weighted KL loss."""
        z, mu, log_sigma2 = self.get_encode_parameters(y=y)
        y_rec = self.decode(z=z)
        kl_losses = -0.5 * torch.sum(1 + log_sigma2 - mu ** 2 - log_sigma2.exp(), dim=1)
        kl_loss = torch.mean(kl_losses, dim=0) * self.kl_mult
        reconstruction_loss = ((y - y_rec) ** 2).mean(dim=1).mean()
        loss = reconstruction_loss + kl_loss
        zero = torch.tensor(0.0, device=y.device)
        return loss, zero, zero, kl_loss, reconstruction_loss, zero, zero

    def loss(self, y, take_vae_grad_step=False, take_adversary_grad_step=False,
             calc_vae_loss=True, calc_adversary_loss=True):
        discriminator_loss = classifier_loss = torch.tensor(0.0, device=y.device)

        if calc_vae_loss or take_vae_grad_step:
            generator_loss, _, _, kl_loss, reconstruction_loss, _, _ = self.generator_losses(y)
            if take_vae_grad_step:
                self.optimizer_G.zero_grad()
                generator_loss.backward()
                self.optimizer_G.step()
        else:
            generator_loss = kl_loss = reconstruction_loss = torch.tensor(0.0, device=y.device)

        return generator_loss, reconstruction_loss, kl_loss, discriminator_loss, classifier_loss

    def to(self, device):
        super().to(device)
        for optimizer in self.optimizers:
            optimizer_to(optimizer, device)
        return self
