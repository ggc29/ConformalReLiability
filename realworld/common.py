"""Common helpers for real-world feature experiments."""

import os
import sys
import hashlib
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml


SYNTHETIC_DIR = Path(__file__).resolve().parents[1] / "synthetic"
if str(SYNTHETIC_DIR) not in sys.path:
    # Reuse DQR, distance, and checkpoint helpers across both experiment types.
    # The real-world VAE is defined in feature_vae.py.
    sys.path.insert(0, str(SYNTHETIC_DIR))

from helper import safe_torch_load  # noqa: E402
from experiment import file_digest, record_checkpoint, resolve_checkpoint, write_json  # noqa: E402


def add_arguments(parser):
    """Add command-line arguments for the experiment preset and feature root."""
    parser.add_argument("--config", default="./configs/i2t.yaml")
    parser.add_argument("--data-root", help="Override dataset.data_root in the YAML file.")


def load_config(config_path, data_root=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if data_root is not None:
        config["dataset"]["data_root"] = data_root
    data = config["dataset"]
    if data["layout"] not in {"i2t_coco", "t2i_coco"}:
        raise ValueError("Choose the i2t_coco or t2i_coco layout.")
    allowed = {"clip", "bert"} if data["layout"] == "i2t_coco" else {"clip", "dinov2"}
    if data["metric_name"] not in allowed:
        raise ValueError(f"Unsupported metric for {data['layout']}: {data['metric_name']}")
    # CLIP scores use the condition; BERT/DINOv2 scores use the reference target.
    data["score_reference"] = "condition" if data["metric_name"] == "clip" else "target"
    if config["dqr"]["num_ens"] != 1:
        raise ValueError("Only the source setting num_ens: 1 is supported.")
    return config


def dataset_kwargs(data_args, task, split, z_hat=None):
    """Build keyword arguments for FeatureTripletDataset from YAML config."""
    kwargs = {
        "data_root": data_args["data_root"],
        "layout": data_args["layout"],
        "img_res": data_args.get("img_res"),
        "metric_name": data_args.get("metric_name"),
        "model": data_args.get("model"),
        "text_len": data_args.get("text_len"),
        "task": task,
        "split": split,
        "l2_norm": data_args.get("l2_norm", True),
        "match": data_args.get("match", "sorted"),
    }
    if z_hat is not None:
        kwargs["z_hat"] = z_hat
    return kwargs


def experiment_name(data_args):
    """Return the folder name for this real-world run."""
    return data_args.get(
        "experiment_name",
        f"{data_args['layout']}-{data_args['img_res']}-{data_args['model']}"
        f"-len_{data_args.get('text_len', 'na')}-{data_args['metric_name']}",
    )


def get_vae_root(data_args, vae_args):
    """Checkpoint folder for the VAE trained on y_hat features."""
    return os.path.join(
        "./results",
        experiment_name(data_args),
        f"{data_args.get('metric_name', 'feature')}_cos-VAE-zdim_{vae_args['z_dim']}-klmult_{vae_args['kl_mult']}",
    )


def get_dqr_root(vae_root, dqr_args):
    """Checkpoint folder for DQR/CReL model outputs."""
    return os.path.join(
        vae_root,
        (
            f"dqr-tau_{dqr_args['conformalization_tau']}-numu_{dqr_args['num_u']}"
            f"-hd_{dqr_args['hidden_dimensions']}-wait_{dqr_args['patience']}"
            f"-dropout_{dqr_args['dropout']}-lr-{dqr_args['learning_rate']}"
            f"-wd_{dqr_args['weight_decay']}"
        ),
    )


def runtime_signature():
    return {
        "packages": {name: version(name) for name in ["torch", "numpy", "scipy", "osqp"]},
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def training_signature(config):
    """Fingerprint ordered feature contents, training settings, and implementation."""
    from dataset import resolve_feature_roots, list_feature_files, align_files
    data = dict(config["dataset"])
    data["data_root"] = str(Path(data["data_root"]).resolve())
    folders = resolve_feature_roots(**{key: data.get(key) for key in
        ["data_root", "layout", "img_res", "metric_name", "model", "text_len"]})
    aligned = align_files(*(list_feature_files(folder) for folder in folders), match=data.get("match", "sorted"))
    digest = hashlib.sha256()
    # Hash bytes rather than only filenames: replacing a tensor invalidates artifacts.
    for paths in aligned:
        for path in paths:
            digest.update(path.name.encode())
            digest.update(bytes.fromhex(file_digest(path)))
    local = Path(__file__).parent
    return {"dataset": data, "vae": config["vae"], "seed": 42,
            "feature_sha256": digest.hexdigest(), "num_samples": len(aligned[0]),
            "code": {name: file_digest(local / name) for name in
                     ["feature_vae.py", "dataset.py", "1-train_vae.py"]},
            "runtime": runtime_signature()}


def latent_signature(signature, checkpoint):
    return {"training": signature, "vae_sha256": file_digest(checkpoint),
            "encoder_code": file_digest(Path(__file__).parent / "2-train_dqr.py")}


def dqr_signature(config, latent_sig, z_hat):
    local = Path(__file__).parent
    return {"latent": latent_sig, "z_hat_sha256": tensor_digest(z_hat), "dqr": config["dqr"],
            "code": {name: file_digest(SYNTHETIC_DIR / name) for name in
                     ["helper.py", "losses.py", "quantile_model.py"]},
            "training_code": file_digest(local / "2-train_dqr.py")}


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_state():
    """A cache hit must leave downstream random draws identical to a cache miss."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(0) if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), 0)


def save_cache(path, signature, value, **extra):
    temporary = str(path) + ".tmp"
    torch.save({"signature": signature, "value": value, **extra}, temporary)
    os.replace(temporary, path)


def load_cache(path, signature):
    if os.path.exists(path):
        cached = safe_torch_load(path, map_location="cpu")
        if isinstance(cached, dict) and cached.get("signature") == signature:
            return cached
        print(f"Stale or legacy cache: {path}")
    return None


def save_json(path, data):
    write_json(path, data)


def cosine_score(a, b):
    """Cosine score used by the optional reliability optimization."""
    return torch.nn.functional.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=1)
