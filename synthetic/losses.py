"""Loss and prediction helpers for directional quantile regression."""

import torch


def predict_y(model, x, u_list, get_unflattened=False):
    """Predict directional quantile thresholds for each x and direction u."""
    num_pts = len(x)
    num_u = u_list.shape[0]
    u_rep = u_list.repeat(num_pts, 1)

    if x is None:
        model_in = u_rep
    else:
        # The network consumes one row per (sample, direction) pair:
        # [condition feature, direction vector] -> directional threshold.
        x_stacked = x.unsqueeze(1).repeat(1, num_u, 1).flatten(0, 1)
        model_in = torch.cat([x_stacked, u_rep], dim=1)

    model_in.requires_grad = True
    pred = model(model_in)[:, 0]

    if get_unflattened:
        pred = torch.nn.Unflatten(0, (num_pts, num_u))(pred)
    return pred


def calc_y_u(u_list, y):
    """Project each y vector onto each sampled direction."""
    num_pts = len(y)
    if y.device != u_list.device:
        y = y.to(u_list.device)
    return torch.bmm(u_list.unsqueeze(0).repeat(num_pts, 1, 1), y.unsqueeze(-1)).squeeze(-1)


def multivariate_qr_loss(model, y, x, u_list, device, args):
    """Pinball loss over random directions.

    Each direction induces a one-dimensional quantile regression problem in the
    projected latent space. Averaging across directions approximates the
    multivariate quantile region objective.
    """
    tau_list = args["tau_list"]
    pred = predict_y(model=model, x=x, u_list=u_list, get_unflattened=True)
    y_u = calc_y_u(u_list, y)
    diff = y_u - pred
    flattened_diff = diff.flatten()
    mask = (tau_list - flattened_diff.le(0).float()).detach()
    return (mask * flattened_diff).mean()
