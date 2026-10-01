"""Checkpoint manifests and input validation for CReL experiments.

Configuration and content hashes identify compatible models and input data
across training and evaluation stages.
"""

import hashlib
import json
from pathlib import Path

import torch


def file_digest(path):
    """Hash file contents, so replacing a file also invalidates its cache."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Replace a small manifest only after its complete contents are written."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def training_signature(config, data_paths, vae_checkpoint=None):
    """Describe the data, configuration, and upstream model for this stage."""
    sections = ["dataset", "vae"]
    if vae_checkpoint is not None:
        sections.append("dqr")
    signature = {
        "config": {key: config[key] for key in sections},
        "data_sha256": [file_digest(path) for path in data_paths],
        "seed": 42,
    }
    if vae_checkpoint is not None:
        signature["vae_sha256"] = file_digest(vae_checkpoint)
    return signature


def record_checkpoint(path, signature, epoch, validation_loss):
    """Point downstream stages to the checkpoint selected by this training run."""
    path = Path(path)
    write_json(path.parent / "best_checkpoint.json", {
        "checkpoint": path.name,
        "sha256": file_digest(path),
        "signature": signature,
        "epoch": int(epoch),
        "validation_loss": float(validation_loss),
        "torch_version": torch.__version__,
    })


def resolve_checkpoint(folder, expected_signature):
    """Resolve the selected checkpoint and validate its input and content hashes."""
    manifest_path = Path(folder) / "best_checkpoint.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run the corresponding release training "
            "script to create a checkpoint and its manifest."
        )
    manifest = json.loads(manifest_path.read_text())
    path = Path(folder) / manifest["checkpoint"]
    if manifest["signature"] != expected_signature:
        raise ValueError(f"Training inputs/configuration changed for {folder}; retrain this stage.")
    if file_digest(path) != manifest["sha256"]:
        raise ValueError(f"Checkpoint contents changed after training: {path}")
    return str(path)


def validate_synthetic_config(config):
    """Validate the supported synthetic VAE loss and DQR model count."""
    if config["vae"]["loss_name"] != "MSE":
        raise ValueError("Synthetic experiments support loss_name: MSE only.")
    if config["dqr"]["num_ens"] != 1:
        raise ValueError("The original experiment uses num_ens: 1; ensemble evaluation is not supported.")
