"""Feed-forward models and optimizer management for directional quantile regression."""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    """MLP for predicting directional quantile thresholds."""

    def __init__(self, input_size, output_size, hidden_dimensions, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dimensions:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultivariateQuantileModel:
    """Manage DQR models, optimizers, and per-model loss evaluation."""

    def __init__(self, input_size, y_size, hidden_dimensions, dropout, lr, wd,
                 num_ens, device, output_size=1, nn_input_size=None):
        self.num_ens = num_ens
        self.device = device
        if nn_input_size is None:
            nn_input_size = input_size + y_size

        self.model = [
            FeedForwardNet(nn_input_size, output_size, hidden_dimensions, dropout).to(device)
            for _ in range(num_ens)
        ]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            for model in self.model
        ]
        self.keep_training = [True for _ in range(num_ens)]
        self.best_va_loss = [np.inf for _ in range(num_ens)]
        self.best_va_model = [deepcopy(model) for model in self.model]
        self.best_va_ep = [0 for _ in range(num_ens)]
        self.done_training = False

    def loss(self, loss_fn, x, y, q_list, batch_q, take_step, args):
        """Evaluate the loss for each ensemble member and optionally step optimizers."""
        ens_loss = []
        for idx in range(self.num_ens):
            self.optimizers[idx].zero_grad()
            if self.keep_training[idx]:
                if batch_q:
                    loss = loss_fn(self.model[idx], y, x, q_list, self.device, args)
                else:
                    losses = [loss_fn(self.model[idx], y, x, q, self.device, args) for q in q_list]
                    loss = torch.mean(torch.stack(losses))
                ens_loss.append(loss.detach().cpu().item())

                if take_step:
                    loss.backward()
                    self.optimizers[idx].step()
            else:
                ens_loss.append(np.nan)

        return np.asarray(ens_loss)
