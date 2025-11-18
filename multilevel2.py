import numpy as np
from numpy.polynomial.legendre import legval


def evaluate(points, comps):
    """
    Evaluate a tensor-train surrogate model at given input points using Legendre polynomials.

    Args:
        points (ndarray): shape (N, d)
        comps (list of ndarray): TT components, each of shape (r_{k-1}, P, r_k)

    Returns:
        ndarray: shape (N,)
    """
    points = np.asarray(points)

    # --- input validation ---
    if points.ndim != 2:
        raise ValueError(f"`points` must be 2D (N, d), got shape {points.shape}")

    d = len(comps)
    if d == 0:
        raise ValueError("`comps` must be a non-empty list of TT cores")

    if points.shape[1] != d:
        raise ValueError(
            f"points.shape[1] (= {points.shape[1]}) must equal len(comps) (= {d})"
        )

    P = comps[0].shape[1]  # polynomial degree + 1
    for j, A in enumerate(comps):
        if A.ndim != 3 or A.shape[1] != P:
            raise ValueError(
                f"TT component {j} must be 3D (r_{j-1}, {P}, r_{j}), got shape {A.shape}"
            )

    # --- evaluate ---
    leg_sups = np.sqrt(2 * np.arange(P) + 1)
    evaluated_legendre = legval(points, np.diag(leg_sups)).T  # shape (d, N, P)

    result = np.einsum("mi,hij->mj", evaluated_legendre[0], comps[0])

    for mode in range(1, d - 1):
        result = np.einsum(
            "mh,mi,hij->mj", result, evaluated_legendre[mode], comps[mode]
        )

    return np.einsum("mh,mi,hij->m", result, evaluated_legendre[-1], comps[-1])


class MultilevelSurrogate:
    def __init__(self, d, L):
        self.d = d
        self.L = L
        self.layers = [None] * self.L

    def set_layer(self, idx, ttcomponents):
        if not isinstance(ttcomponents, list) and len(ttcomponents) == self.d:
            raise ValueError(f"ttcomponents must be list of length {self.d}")

        self.layers[idx] = ttcomponents

    def __call__(self, points):
        if not all(isinstance(ttcomponents, list) for ttcomponents in self.layers):
            raise ValueError("Tensor train layers not set")

        res = evaluate(points, self.layers[0])
        for i in range(1, self.L):
            res += evaluate(points, self.layers[i])
        return res
