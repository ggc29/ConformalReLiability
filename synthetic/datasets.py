"""Synthetic data splitting utilities.

The VAE validation split supplies the DQR training, calibration, and test folds.
The calibration fold is also used for DQR early stopping.
"""

from sklearn.preprocessing import StandardScaler
import torch


def _as_tensor(value):
    """Convert numpy arrays or tensors to float tensors."""
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.tensor(value, dtype=torch.float32)


def get_split_data(X, Y, task, scale=True):
    """
    Split X and Y for the synthetic experiments.

    The VAE is trained on the second half of the generated data. The DQR/CReL
    model uses the held-out part of that half, split into train/validation/test.
    """
    # Fit and apply StandardScaler in float64 before converting to float32.
    # Casting before scaling changes the numerical preprocessing results.
    data = {}
    n = X.shape[0]

    if task == "vae":
        # Use the second half of the generated data for the VAE train/val split.
        split_idx_a = n // 2
        x_part = X[split_idx_a:]
        y_part = Y[split_idx_a:]
        split_idx_train = int(x_part.shape[0] * 3 // 5)

        data["x_train"] = x_part[:split_idx_train]
        data["y_train"] = y_part[:split_idx_train]
        data["x_val"] = x_part[split_idx_train:]
        data["y_val"] = y_part[split_idx_train:]
        x_names = ["x_train", "x_val"]
        y_names = ["y_train", "y_val"]

    elif task == "dqr":
        # Partition VAE validation data into DQR train, calibration, and test folds.
        # The calibration fold is also used for DQR validation.
        split_idx_a = n // 2
        x_part = X[split_idx_a:]
        y_part = Y[split_idx_a:]

        split_idx_b = int(x_part.shape[0] * 3 // 5)
        x_part = x_part[split_idx_b:]
        y_part = y_part[split_idx_b:]

        split_idx_train = int(x_part.shape[0] * 3 // 5)
        split_idx_val = split_idx_train + int(x_part.shape[0] * 1 // 5)

        data["x_train"] = x_part[:split_idx_train]
        data["y_train"] = y_part[:split_idx_train]
        data["x_val"] = x_part[split_idx_train:split_idx_val]
        data["y_val"] = y_part[split_idx_train:split_idx_val]
        data["x_test"] = x_part[split_idx_val:]
        data["y_test"] = y_part[split_idx_val:]
        x_names = ["x_train", "x_val", "x_test"]
        y_names = ["y_train", "y_val", "y_test"]

    else:
        raise ValueError("task must be either 'vae' or 'dqr'.")

    if scale:
        # Fit scalers only on each task's training split to avoid leakage into
        # validation/test data.
        s_tr_x = StandardScaler().fit(data["x_train"])
        data["s_tr_x"] = s_tr_x
        for name in x_names:
            data[name] = torch.tensor(s_tr_x.transform(data[name]), dtype=torch.float32)

        s_tr_y = StandardScaler().fit(data["y_train"])
        data["s_tr_y"] = s_tr_y
        for name in y_names:
            data[name] = torch.tensor(s_tr_y.transform(data[name]), dtype=torch.float32)
    else:
        for name in x_names + y_names:
            data[name] = _as_tensor(data[name])

    return data
