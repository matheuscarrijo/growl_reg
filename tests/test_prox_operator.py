"""Regression tests for the OWL/GrOWL proximal operators.

These tests guard against the single-pass PAV bug fixed in 2026-07: the
isotonic step of the OWL prox must re-merge pooled blocks *backward*
(pool adjacent violators), otherwise the output can violate the ordering
property of the OWL prox and miss the (unique) minimizer.

Run with: pytest tests/test_prox_operator.py
"""

import numpy as np

from growl.prox_operator import prox_owl, prox_growl


def owl_norm(x, w):
    return float(np.sum(w * np.sort(np.abs(x))[::-1]))


def prox_objective(x, v, w):
    return 0.5 * np.sum((x - v) ** 2) + owl_norm(x, w)


def brute_force_prox(v, w, n_starts=400, seed=0):
    """Nelder-Mead multistart reference for tiny instances."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    best_x, best_val = None, np.inf
    starts = [v, np.zeros_like(v)] + [
        v + rng.standard_normal(v.size) * s for s in rng.uniform(0.1, 1.5, n_starts)
    ]
    for x0 in starts:
        res = minimize(prox_objective, x0, args=(v, w), method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x
    return best_x, best_val


def test_backward_merge_case():
    """The crafted case that defeats a single forward PAV pass."""
    v = np.array([5.0, 4.9, 4.8])
    w = np.array([3.0, 1.0, 1.0])
    out = prox_owl(v.copy(), w.copy())
    expected = np.full(3, 9.7 / 3)  # pooled mean of z = [2, 3.9, 3.8]
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_order_preservation():
    """|prox(v)| must be ordered like |v| (OWL prox property)."""
    rng = np.random.default_rng(1)
    for _ in range(500):
        p = int(rng.integers(3, 12))
        v = rng.standard_normal(p) * rng.uniform(0.5, 3)
        w = np.sort(rng.uniform(0, 2, p))[::-1]
        out = np.abs(prox_owl(v.copy(), w.copy()))
        order = np.argsort(-np.abs(v))
        assert np.all(np.diff(out[order]) <= 1e-10)


def test_constant_weights_reduce_to_soft_threshold():
    """OWL prox with constant weights == elementwise soft-thresholding."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        p = int(rng.integers(2, 15))
        v = rng.standard_normal(p) * 2
        lam = rng.uniform(0.05, 1.5)
        out = prox_owl(v.copy(), np.full(p, lam))
        soft = np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)
        np.testing.assert_allclose(out, soft, atol=1e-12)


def test_optimality_against_brute_force():
    """On tiny instances, the prox must match a brute-force minimizer."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        p = 3
        v = rng.standard_normal(p) * rng.uniform(0.5, 2)
        w = np.sort(rng.uniform(0, 1.5, p))[::-1]
        out = prox_owl(v.copy(), w.copy())
        _, brute_val = brute_force_prox(v, w)
        assert prox_objective(out, v, w) <= brute_val + 1e-6


def test_prox_growl_rescales_rows():
    """GrOWL prox = OWL prox on row norms, rows rescaled proportionally."""
    rng = np.random.default_rng(4)
    V = rng.standard_normal((6, 4))
    w = np.sort(rng.uniform(0, 1, 6))[::-1]
    out = prox_growl(V, w)
    norms_in = np.linalg.norm(V, axis=1)
    norms_out = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms_out, prox_owl(norms_in, w), atol=1e-10)
    for i in range(6):
        if norms_out[i] > 1e-12:
            np.testing.assert_allclose(out[i] / norms_out[i], V[i] / norms_in[i], atol=1e-10)
