"""Feature-triplet dataset for real-world CReL experiments.

Each sample contains precomputed features for the condition ``x``, reference
target ``y``, and generated target ``y_hat``. The loader uses the COCO feature
directory conventions and sorted sample order described in the README.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


def safe_torch_load(path, map_location=None):
    """Load tensors with compatibility across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def l2_normalize(tensor):
    """Apply L2 normalization in the stored dtype before conversion to float32."""
    norm = tensor.norm(p=2, dim=-1, keepdim=True)
    return tensor / norm


def feature_storage_device(storage, location):
    """Retain CPU storage and map CUDA storage to the first visible GPU.

    The experimental BERT features are CUDA-saved, while CLIP/DINOv2 features
    are CPU-saved. Retaining the device type preserves normalization behavior.
    When CUDA is unavailable, all tensors are loaded on CPU.
    """
    if location.startswith("cuda") and torch.cuda.is_available():
        return storage.cuda(0)
    return storage


def list_feature_files(folder):
    """Return sorted .pt feature files from a folder."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Feature folder does not exist: {folder}")
    files = sorted([path for path in folder.iterdir() if path.suffix == ".pt"])
    if not files:
        raise FileNotFoundError(f"No .pt feature files found in {folder}")
    return files


def align_files(x_files, y_files, y_hat_files, match="sorted"):
    """Align x/y/y_hat feature files either by stem or sorted order."""
    if match == "sorted":
        if not (len(x_files) == len(y_files) == len(y_hat_files)):
            raise ValueError("x, y, and y_hat folders must contain the same number of files.")
        return x_files, y_files, y_hat_files

    if match != "stem":
        raise ValueError("match must be either 'stem' or 'sorted'.")

    x_by_stem = {path.stem: path for path in x_files}
    y_by_stem = {path.stem: path for path in y_files}
    y_hat_by_stem = {path.stem: path for path in y_hat_files}
    stems = sorted(set(x_by_stem) & set(y_by_stem) & set(y_hat_by_stem))
    if not stems:
        raise ValueError("No matching stems found across x, y, and y_hat feature folders.")
    if len(stems) != len(x_files) or len(stems) != len(y_files) or len(stems) != len(y_hat_files):
        raise ValueError("Feature folders contain unmatched stems. Use match: sorted if sorted order is intended.")
    return [x_by_stem[s] for s in stems], [y_by_stem[s] for s in stems], [y_hat_by_stem[s] for s in stems]


def resolve_feature_roots(data_root, layout, img_res=None, metric_name=None, model=None, text_len=None):
    """Resolve COCO feature directories beneath the configured data root."""
    if layout == "i2t_coco":
        if text_len is None:
            raise ValueError("layout i2t_coco requires text_len.")
        return (
            os.path.join(data_root, f"val2014-{img_res}_clip"),
            os.path.join(data_root, f"val2014_text_{metric_name}"),
            os.path.join(data_root, f"val2014-{img_res}-{model}-len_{text_len}-{metric_name}"),
        )

    if layout == "t2i_coco":
        return (
            os.path.join(data_root, "val2014_text_clip"),
            os.path.join(data_root, f"val2014_{img_res}_{metric_name}"),
            os.path.join(data_root, f"{model}_{img_res}_{metric_name}"),
        )

    raise ValueError("layout must be i2t_coco or t2i_coco.")


class FeatureTripletDataset(Dataset):
    """Dataset returning feature triplets and optional VAE latents."""

    def __init__(self, data_root, layout="i2t_coco", img_res=None, metric_name=None,
                 model=None, text_len=None, task="vae", split="train",
                 l2_norm=True, z_hat=None, match="sorted"):
        self.task = task
        self.split = split
        self.l2_norm = l2_norm
        self.z_hat = z_hat

        x_root, y_root, y_hat_root = resolve_feature_roots(
            data_root=data_root,
            layout=layout,
            img_res=img_res,
            metric_name=metric_name,
            model=model,
            text_len=text_len,
        )
        x_files, y_files, y_hat_files = align_files(
            list_feature_files(x_root),
            list_feature_files(y_root),
            list_feature_files(y_hat_root),
            match=match,
        )

        total_samples = len(x_files)
        if task == "vae":
            # VAE learns the latent space for generated target features y_hat.
            train_size = int(0.75 * total_samples)
            if split == "train":
                indices = list(range(train_size))
            elif split == "val":
                indices = list(range(train_size, total_samples))
            else:
                raise ValueError("For task='vae', split must be 'train' or 'val'.")

        elif task == "dqr":
            # DQR is trained only on the held-out VAE validation quarter. That
            # quarter is split into train/val/test for CReL calibration.
            if z_hat is None:
                raise ValueError("task='dqr' requires z_hat.")
            vae_val_start = 3 * total_samples // 4
            vae_val_indices = list(range(vae_val_start, total_samples))
            dqr_train_size = int(0.6 * len(vae_val_indices))
            dqr_val_size = int(0.2 * len(vae_val_indices))
            dqr_train_indices = vae_val_indices[:dqr_train_size]
            dqr_val_indices = vae_val_indices[dqr_train_size:dqr_train_size + dqr_val_size]
            dqr_test_indices = vae_val_indices[dqr_train_size + dqr_val_size:]

            if split == "train":
                indices = dqr_train_indices
            elif split == "val":
                indices = dqr_val_indices
            elif split == "test":
                indices = dqr_test_indices
            else:
                raise ValueError("For task='dqr', split must be 'train', 'val', or 'test'.")
        else:
            raise ValueError("task must be either 'vae' or 'dqr'.")

        self.x_files = [x_files[i] for i in indices]
        self.y_files = [y_files[i] for i in indices]
        self.y_hat_files = [y_hat_files[i] for i in indices]
        self.x_data = [self._load_feature(path) for path in self.x_files]
        self.y_data = [self._load_feature(path) for path in self.y_files]
        self.y_hat_data = [self._load_feature(path) for path in self.y_hat_files]

        if self.task == "dqr":
            n = len(self.z_hat)
            train_end = n * 3 // 5
            val_end = train_end + n * 1 // 5
            if self.split == "train":
                self.z_hat_data = self.z_hat[:train_end]
            elif self.split == "val":
                self.z_hat_data = self.z_hat[train_end:val_end]
            else:
                self.z_hat_data = self.z_hat[val_end:]
            if not (len(self.z_hat_data) == len(self.x_data) == len(self.y_data) == len(self.y_hat_data)):
                raise ValueError("z_hat length does not match the DQR split length.")

    def _load_feature(self, path):
        """Load and flatten a single feature tensor."""
        # Stored features may be float16; casting first changes normalization.
        tensor = safe_torch_load(path, map_location=feature_storage_device)
        if self.l2_norm:
            tensor = l2_normalize(tensor)
        return tensor.to(torch.float32).view(-1)

    def feature_dims(self):
        """Return inferred condition and target feature dimensions."""
        return self.x_data[0].numel(), self.y_data[0].numel()

    def __getitem__(self, idx):
        item = {
            "x": self.x_data[idx],
            "y": self.y_data[idx],
            "y_hat": self.y_hat_data[idx],
            "filename": self.x_files[idx].stem,
        }
        if self.task == "dqr":
            item["z_hat"] = self.z_hat_data[idx].to(torch.float32).view(-1)
        return item

    def __len__(self):
        return len(self.x_data)
