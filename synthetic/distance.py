"""Distance calculations for conformal calibration.

The DQR model represents a prediction set as an intersection of halfspaces in
latent space. Calibration requires the distance from a point to this set, which
is solved as a small quadratic program.
"""

import numpy as np
import osqp
import scipy.sparse as sp


def if_in_region(y, u, b):
    """Return whether y belongs to the intersection {z: u_i^T z >= b_i for all i}."""
    for i in range(len(b)):
        if np.dot(u[i], y) < b[i]:
            return False
    return True


def quadratic_programming_to_find_distance(w, u, b):
    """
    Compute the distance from w to the halfspace intersection u^T z >= b.
    """
    dim = w.shape[0]
    # Minimize ||z - w||_2^2 subject to -u z <= -b, equivalent to u z >= b.
    p = sp.csc_matrix(2 * np.eye(dim))
    q = -2 * w
    g = sp.csc_matrix(-u)
    h = -b

    prob = osqp.OSQP()
    prob.setup(P=p, q=q, A=g, l=None, u=h, verbose=False)
    res = prob.solve()

    if res.info.status != "solved":
        raise ValueError("OSQP did not find an optimal solution.")

    closest = res.x
    distance = np.linalg.norm(w - closest)
    return closest, distance
