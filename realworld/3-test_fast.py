"""Calibrate/test real-world CReL and optionally optimize reliability scores."""

import argparse
import os
from copy import deepcopy
from itertools import islice

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import common
from dataset import FeatureTripletDataset
from distance import if_in_region, quadratic_programming_to_find_distance
from helper import generate_directions, set_seeds
from losses import predict_y
from quantile_model import MultivariateQuantileModel
from feature_vae import VAE


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate/test CReL and optionally optimize reliability scores.")
    common.add_arguments(parser)
    parser.add_argument("--recompute-distances", action="store_true")
    parser.add_argument("--coverage-only", action="store_true", help="Skip reliability optimization.")
    parser.add_argument("--num-test", type=int, help="Limit reliability evaluation; coverage still uses the full test split.")
    return parser.parse_args()


def calculate_distances(loader, model, train_args, z_dim, device, desc):
    """Distance from each generated latent to its predicted DQR region."""
    distances = []
    for batch in tqdm(loader, desc=desc):
        x = batch["x"].float().to(device)
        z = batch["z_hat"].float().to(device).view(-1, z_dim)
        u_list, gamma = generate_directions(z_dim, train_args["num_u"], train_args["tau_list"][0])
        train_args["gamma"] = gamma
        b = predict_y(model, x, u_list, get_unflattened=True).float()

        w_np = z.squeeze().cpu().detach().numpy()
        b_np = b.squeeze().cpu().detach().numpy()
        u_np = u_list.cpu().detach().numpy()

        if if_in_region(w_np, u_np, b_np):
            distance = 0.0
        else:
            # Propagate QP solver failures instead of assigning a distance.
            _, distance = quadratic_programming_to_find_distance(w_np, u_np, b_np)
        distances.append(float(distance))
    return distances


def calibrate_radius(distances, alpha):
    """Conformal radius from validation distances."""
    n = len(distances)
    q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(distances, q=q))


def coverage_ratio(distances, radius):
    coverage = np.asarray(distances) < radius
    return float(np.sum(coverage != 0) / len(distances))


def project_to_calibrated_set(z, radius, u, b, device):
    """Project a latent point back to the calibrated feasible set if needed."""
    w = z.squeeze().cpu().detach().numpy()
    _, distance = quadratic_programming_to_find_distance(w, u, b)
    if distance > radius:
        closest, _ = quadratic_programming_to_find_distance(w, u, b)
        z.data = torch.tensor(closest, device=device, dtype=torch.float32).view_as(z)
    return z


def optimize_z(radius, u, b, x, y, learning_rate, num_iterations, device,
               vae, z_dim, score_reference, patience, initial_z=None):
    """Minimize cosine similarity inside the calibrated prediction set.

    The current score is cosine similarity. ``score_reference`` chooses whether
    the decoded target feature is compared to the condition feature ``x`` or the
    ground-truth target feature ``y``.
    """
    if initial_z is None:
        z_0 = torch.randn(z_dim, device=device, dtype=torch.float32)
    else:
        z_0 = initial_z.clone().detach().to(device)
    z_0 = project_to_calibrated_set(z_0, radius, u, b, device)
    z_0.requires_grad = True
    optimizer = torch.optim.Adam([z_0], lr=learning_rate)

    reference = x if score_reference == "condition" else y
    best_loss = float("inf")
    best_z = None
    patience_counter = 0

    for _ in tqdm(range(num_iterations), desc="Optimizing z", leave=False):
        decoded = vae.decode(z_0.view(1, -1))
        if decoded.numel() != reference.numel():
            raise ValueError(
                "Decoded output and score reference have different dimensions. "
                "Use score_reference: target unless x and y_hat share the same feature dimension."
            )
        loss = common.cosine_score(decoded, reference).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        z_0 = project_to_calibrated_set(z_0, radius, u, b, device)

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            # The reported score is evaluated before the optimizer step;
            # the stored latent is captured after the step and projection.
            best_z = deepcopy(z_0.detach())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    return best_loss, best_z


def run_reliability_optimization(test_loader, model, vae, train_args, reliab_args,
                                 data_args, radius, save_root, z_dim, device):
    """Run the optional, slower latent optimization for reliability scoring."""
    learning_rate = reliab_args["learning_rate"]
    patience = reliab_args["patience"]
    num_iterations = reliab_args["num_iterations"]
    num_z0 = reliab_args["num_z0"]
    num_test = reliab_args.get("num_test", -1)
    score_reference = data_args.get("score_reference", "target")
    if score_reference not in {"condition", "target"}:
        raise ValueError("score_reference must be either 'condition' or 'target'.")

    reliability_dir = os.path.join(
        save_root,
        f"reliab-fast-numZ_{num_z0}-alpha_{reliab_args['alpha']}-radius_{radius:.3f}"
        f"-lr_{learning_rate}-wait_{patience}-iters_{num_iterations}",
    )
    if num_test is not None and num_test > 0:
        reliability_dir += f"-numTest_{num_test}"
    os.makedirs(reliability_dir, exist_ok=True)

    metric_values = {}
    best_losses = {}
    best_z_list = []

    iterator = tqdm(test_loader, desc="Reliability optimization")
    if num_test and num_test > 0:
        iterator = islice(iterator, num_test)

    for batch in iterator:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_hat = batch["y_hat"].to(device)
        file_name = batch["filename"][0]
        reference = x if score_reference == "condition" else y
        if y_hat.numel() != reference.numel():
            raise ValueError(
                "y_hat and score reference have different dimensions. "
                "Use score_reference: target unless x and y_hat share the same feature dimension."
            )
        metric_values[file_name] = common.cosine_score(y_hat, reference).item()

        u_list, gamma = generate_directions(z_dim, train_args["num_u"], train_args["tau_list"][0])
        train_args["gamma"] = gamma
        b = predict_y(model, x, u_list, get_unflattened=True).float().squeeze().cpu().detach().numpy()
        u = u_list.cpu().detach().numpy()

        # Select the projected random candidate with the lowest cosine score
        # as the starting point for a single optimization run.
        initial_candidates = []
        for _ in range(num_z0):
            z_candidate = torch.randn(z_dim, device=device, dtype=torch.float32)
            z_candidate = project_to_calibrated_set(z_candidate, radius, u, b, device)
            with torch.no_grad():
                decoded = vae.decode(z_candidate.view(1, -1))
                score = common.cosine_score(decoded, reference).item()
            initial_candidates.append((score, z_candidate.detach()))

        _, initial_z = min(initial_candidates, key=lambda item: item[0])
        best_loss, best_z = optimize_z(
            radius=radius,
            u=u,
            b=b,
            x=x,
            y=y,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            device=device,
            vae=vae,
            z_dim=z_dim,
            score_reference=score_reference,
            patience=patience,
            initial_z=initial_z,
        )
        best_losses[file_name] = best_loss
        best_z_list.append(best_z.cpu())

    # Report aggregate statistics and scores indexed by sample identifier.
    def statistics(values):
        return {"avg": float(np.mean(list(values.values()))),
                "std": float(np.std(list(values.values()))), "values": values}
    common.save_json(os.path.join(reliability_dir, "metric_value.json"), statistics(metric_values))
    common.save_json(os.path.join(reliability_dir, "reliability.json"), statistics(best_losses))
    if best_z_list:
        torch.save(torch.stack(best_z_list), os.path.join(reliability_dir, "best_z0s.pt"))
    print(f"Reliability outputs saved to {reliability_dir}")


def main():
    args = parse_args()
    set_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = common.load_config(args.config, args.data_root)
    data_args = config["dataset"]
    vae_args = config["vae"]
    dqr_args = deepcopy(config["dqr"])
    reliab_args = config["reliability"]
    if args.coverage_only:
        reliab_args["measure_reliability"] = False
    if args.num_test is not None:
        if args.num_test <= 0:
            raise ValueError("--num-test must be positive.")
        reliab_args["num_test"] = args.num_test
    if not 0 < reliab_args["alpha"] < 1:
        raise ValueError("alpha must be between 0 and 1.")

    vae_root = common.get_vae_root(data_args, vae_args)
    z_hat_path = os.path.join(vae_root, "Z_hat.pt")
    if not os.path.exists(z_hat_path):
        raise FileNotFoundError(f"{z_hat_path} does not exist. Run 2-train_dqr.py first.")
    signature = common.training_signature(config)
    vae_checkpoint = common.resolve_checkpoint(vae_root, signature)
    latent_sig = common.latent_signature(signature, vae_checkpoint)
    cached_latent = common.load_cache(z_hat_path, latent_sig)
    if cached_latent is None or cached_latent.get("tensor_sha256") != common.tensor_digest(cached_latent["value"]):
        raise ValueError("Z_hat cache is stale; run 2-train_dqr.py again.")
    z_hat = cached_latent["value"]
    dqr_sig = common.dqr_signature(config, latent_sig, z_hat)

    val_dataset = FeatureTripletDataset(**common.dataset_kwargs(data_args, task="dqr", split="val", z_hat=z_hat))
    test_dataset = FeatureTripletDataset(**common.dataset_kwargs(data_args, task="dqr", split="test", z_hat=z_hat))
    x_dim, y_dim = val_dataset.feature_dims()
    z_dim = vae_args["z_dim"]

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dqr_args["conformalization_tau"] = dqr_args["tau"]
    dqr_args["tau"] = dqr_args["tau"] - 0.03
    dqr_args["tau_list"] = torch.tensor([dqr_args["tau"]], device=device)
    dqr_root = common.get_dqr_root(vae_root, dqr_args)
    dqr_args["save_root"] = dqr_root

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
    dqr_checkpoint = common.resolve_checkpoint(dqr_root, dqr_sig)
    model_ens.model[0].load_state_dict(common.safe_torch_load(dqr_checkpoint, map_location=device))
    model = model_ens.model[0].to(device)
    model.eval()

    dist_file_path = os.path.join(dqr_root, "dist_list.pt")
    cache_sig = {"dqr": dqr_sig, "checkpoint": common.file_digest(dqr_checkpoint),
                 "runtime": common.runtime_signature(),
                 "evaluation_code": common.file_digest(__file__),
                 "distance_code": common.file_digest(common.SYNTHETIC_DIR / "distance.py")}
    cached = None if args.recompute_distances else common.load_cache(dist_file_path, cache_sig)
    if cached is not None:
        dist_all = cached["value"]
        common.restore_rng(cached["rng"])
        # Restore the sampled direction pool as well as its RNG state. Otherwise
        # the next reliability batch would create a different pool after a hit.
        import helper
        helper._cached_u_list = cached["directions"].to(device)
        helper._cached_gamma = cached["gamma"].to(device)
        print(f"Loaded matching distance cache from {dist_file_path}")
    else:
        dist_all = {
            "val": calculate_distances(val_loader, model, dqr_args, z_dim, device, "Validation distances"),
            "test": calculate_distances(test_loader, model, dqr_args, z_dim, device, "Test distances"),
        }
        import helper
        common.save_cache(dist_file_path, cache_sig, dist_all, rng=common.rng_state(),
                          directions=helper._cached_u_list.cpu(), gamma=helper._cached_gamma.cpu())
        print(f"Saved distance cache to {dist_file_path}")

    alpha = reliab_args["alpha"]
    radius = calibrate_radius(dist_all["val"], alpha)
    # Validation distances determine the conformal expansion; test distances
    # are used only for held-out coverage reporting.
    val_coverage = coverage_ratio(dist_all["val"], radius)
    test_coverage = coverage_ratio(dist_all["test"], radius)
    summary = {
        "alpha": alpha,
        "radius": radius,
        "validation_coverage": val_coverage,
        "test_coverage": test_coverage,
        "num_calibration": len(dist_all["val"]),
        "num_test": len(dist_all["test"]),
        "vae_checkpoint": vae_checkpoint,
        "dqr_checkpoint": dqr_checkpoint,
    }
    common.save_json(os.path.join(dqr_root, f"crel_calibration_test_alpha_{alpha}.json"), summary)
    print(f"Calibrated radius: {radius:.6f}")
    print(f"Validation coverage after calibration: {val_coverage:.4f}")
    print(f"Test coverage after calibration: {test_coverage:.4f}")

    if reliab_args.get("measure_reliability", False):
        vae = VAE(
            y_dim=y_dim,
            z_dim=z_dim,
            device=device,
            train_size=1,
            mode=data_args.get("metric_name", "feature"),
            dropout=vae_args["dropout"],
            lr=vae_args["lr"],
            wd=vae_args["wd"],
            batch_norm=False,
            kl_mult=vae_args["kl_mult"],
        )
        vae.load_state_dict(common.safe_torch_load(vae_checkpoint, map_location=device))
        vae.to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        run_reliability_optimization(test_loader, model, vae, dqr_args, reliab_args, data_args, radius, dqr_root, z_dim, device)


if __name__ == "__main__":
    main()
