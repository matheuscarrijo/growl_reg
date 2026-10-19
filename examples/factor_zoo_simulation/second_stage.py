"""
Second stage: GrOWL mapping, hv-block CV and benchmark estimators.

Two exact computational devices keep the Monte Carlo affordable.

1. Low-rank reduction. The GrOWL target is the standardized common component
   Y (T x N), whose rank is the number of PCA factors k. Write Y = Y_r V' with
   V (N x k) orthonormal. For any B = C V' + B_perp with B_perp V = 0, the
   loss splits as ||Y_r - X C||^2 + ||X B_perp||^2 and each row norm splits
   as ||c_i||^2 + ||b_perp,i||^2, so the optimum has B_perp = 0 and
   B_hat = C_hat V', where C_hat solves the same GrOWL problem with the k
   columns of Y_r. Row norms, cosines between rows and the support of B_hat
   equal those of C_hat. The same holds for the multi-task Lasso and the
   multi-task Elastic Net penalties.

2. Gram-form FISTA. ``growl_fista_gram`` repeats the iterations of
   ``growl.fista_solver.growl_fista`` with ``check_type='solution_diff'``
   (same weights, step 1/L with L = ||X||_2^2 / n, same prox, same momentum
   and same stopping rule), but it uses X'X / n and X'Y / n and a vectorized
   prox. ``tests/test_factor_zoo_simulation.py`` checks that it returns the
   package solution, and each replication re-solves sample cells with the
   package solver.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import isotonic_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet

CV_MAX_ITER = 5000
CV_TOL = 1e-4
HV_C = 0.25
HV_D = 0.5
N_FOLDS = 5


# ---------------------------------------------------------------------------
# GrOWL solver
# ---------------------------------------------------------------------------

def growl_weights(p, lambda_1, lambda_2, ramp_size):
    """Weight vector built exactly as in growl_fista."""
    ramp = int(np.ceil(ramp_size * p))
    ramp = max(0, min(p, ramp))
    i = np.arange(p)
    w = np.where(i < ramp, lambda_1 + lambda_2 * (ramp - i), lambda_1).astype(float)
    return np.sort(w)[::-1]


def prox_growl_fast(V, w):
    """GrOWL prox: OWL prox on the row norms, then row rescaling."""
    norms = np.sqrt(np.einsum("ij,ij->i", V, V))
    order = np.argsort(-norms, kind="stable")
    z = norms[order] - w
    if np.any(z[1:] > z[:-1]):
        z = isotonic_regression(z, increasing=False).x
    shrunk = np.empty_like(norms)
    shrunk[order] = np.maximum(z, 0.0)
    scale = np.zeros_like(norms)
    pos = norms > 0
    scale[pos] = shrunk[pos] / norms[pos]
    return V * scale[:, None]


class GramProblem:
    """Sufficient statistics of a least-squares problem in FISTA form."""

    def __init__(self, X, Y):
        n = X.shape[0]
        self.n = n
        self.XtX = X.T @ X / n
        self.XtY = X.T @ Y / n
        self.L = np.linalg.norm(X, 2) ** 2 / n


try:
    import numba
except ImportError:          # the NumPy path below is used instead
    numba = None

if numba is not None:
    @numba.njit(cache=True)
    def _fista_kernel(XtX, XtY, L, w, max_iter, tol):
        """Compiled copy of growl_fista (solution_diff); same PAV arithmetic."""
        p, r = XtY.shape
        B = np.zeros((p, r))
        Z = np.zeros((p, r))
        t = 1.0
        wl = w / L
        norms = np.empty(p)
        z = np.empty(p)
        shrunk = np.empty(p)
        means = np.empty(p)
        sizes = np.empty(p, dtype=np.int64)
        n_iter = max_iter
        for it in range(max_iter):
            V = Z - (XtX @ Z - XtY) / L
            for i in range(p):
                s = 0.0
                for c in range(r):
                    s += V[i, c] * V[i, c]
                norms[i] = np.sqrt(s)
            order = np.argsort(-norms)
            for i in range(p):
                z[i] = norms[order[i]] - wl[i]
            nb = 0
            for i in range(p):
                mean = z[i]
                size = 1
                while nb > 0 and means[nb - 1] < mean:
                    nb -= 1
                    mean = (mean * size + means[nb] * sizes[nb]) / (size + sizes[nb])
                    size += sizes[nb]
                means[nb] = mean
                sizes[nb] = size
                nb += 1
            pos = 0
            for b in range(nb):
                val = means[b] if means[b] > 0.0 else 0.0
                for _ in range(sizes[b]):
                    shrunk[order[pos]] = val
                    pos += 1
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            mom = (t - 1.0) / t_new
            diff = 0.0
            for i in range(p):
                scale = shrunk[i] / norms[i] if norms[i] > 0.0 else 0.0
                for c in range(r):
                    b_new = scale * V[i, c]
                    d = b_new - B[i, c]
                    diff += d * d
                    B[i, c] = b_new
                    Z[i, c] = b_new + mom * d
            t = t_new
            if it > 0 and np.sqrt(diff) < tol:
                n_iter = it + 1
                break
        return B, n_iter


def growl_fista_gram(prob: GramProblem, w, max_iter=CV_MAX_ITER, tol=CV_TOL,
                     use_numba=True):
    """FISTA for GrOWL, iteration by iteration equal to growl_fista."""
    if use_numba and numba is not None:
        return _fista_kernel(np.ascontiguousarray(prob.XtX), np.ascontiguousarray(prob.XtY),
                             float(prob.L), np.ascontiguousarray(w, dtype=float),
                             int(max_iter), float(tol))
    p, r = prob.XtY.shape
    B = np.zeros((p, r))
    Z = B.copy()
    t = 1.0
    wl = w / prob.L
    n_iter = max_iter
    for it in range(max_iter):
        B_old = B
        V = Z - (prob.XtX @ Z - prob.XtY) / prob.L
        B = prox_growl_fast(V, wl)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = B + ((t - 1.0) / t_new) * (B - B_old)
        t = t_new
        if it > 0 and np.linalg.norm(B - B_old) < tol:
            n_iter = it + 1
            break
    return B, n_iter


# ---------------------------------------------------------------------------
# Target reduction and standardization
# ---------------------------------------------------------------------------

def standardizer(A):
    mean = A.mean(axis=0)
    scale = A.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def reduce_target(Y, rel_tol=1e-9):
    """Exact rank-k factorization Y = Y_r V' of a low-rank target."""
    gram = Y @ Y.T
    vals, vecs = np.linalg.eigh(gram)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    r = int(np.sum(vals > rel_tol * vals[0]))
    s = np.sqrt(vals[:r])
    V = (Y.T @ vecs[:, :r]) / s
    return vecs[:, :r] * s, V


# ---------------------------------------------------------------------------
# hv-block cross-validation (Racine, 2000), as in the research pipeline
# ---------------------------------------------------------------------------

def hv_block_splits(n, n_folds=N_FOLDS, c=HV_C, d=HV_D):
    h = int(np.floor(c * n))
    v = int(np.floor(n ** d))
    lo, hi = h + v, n - h - v - 1
    if lo > hi:
        raise ValueError("hv-block parameters leave no valid center range.")
    centers = np.linspace(lo, hi, n_folds).astype(int)
    splits = []
    for center in centers:
        val = np.arange(center - v, center + v + 1)
        excl_start = max(0, center - v - h)
        excl_end = min(n, center + v + h + 1)
        train = np.concatenate([np.arange(0, excl_start), np.arange(excl_end, n)])
        splits.append((train, val))
    return splits


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

LAMBDA1_GRID = np.logspace(-2, 2, 9)
LAMBDA2_GRID = np.array([1e-3, 1e-2, 1e-1, 1.0])
RAMP_GRID = [0.25, 0.75, 1.0]

ENET_ALPHA_GRID = np.logspace(-2, 2.5, 10)
ENET_L1_GRID = [0.1, 0.5, 0.9]


def growl_cells():
    """117 cells: the multi-task Lasso path (rho = 0) and 108 GrOWL cells."""
    cells = [(l1, 0.0, 0.0) for l1 in LAMBDA1_GRID]
    cells += [(l1, l2, rs) for l1 in LAMBDA1_GRID for l2 in LAMBDA2_GRID for rs in RAMP_GRID]
    return cells


def enet_cells():
    return [(a, l1) for a in ENET_ALPHA_GRID for l1 in ENET_L1_GRID]


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def fit_growl_cells(prob: GramProblem, cells, max_iter=CV_MAX_ITER, tol=CV_TOL):
    p = prob.XtY.shape[0]
    coefs, iters = [], []
    for l1, l2, rs in cells:
        B, n_iter = growl_fista_gram(prob, growl_weights(p, l1, l2, rs), max_iter, tol)
        coefs.append(B)
        iters.append(n_iter)
    return coefs, np.array(iters)


def fit_enet_cells(X, Y, cells):
    coefs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for alpha, l1_ratio in cells:
            model = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False,
                                        max_iter=10000, tol=1e-6)
            model.fit(X, Y)
            coefs.append(model.coef_.T.copy())
    return coefs


def validation_mse(coefs, X_val, Y_val_r, resid_ss, n_targets):
    """Mean squared error per cell, in the units of the full T x N target."""
    n_val = X_val.shape[0]
    return np.array([
        (np.sum((Y_val_r - X_val @ B) ** 2) + resid_ss) / (n_val * n_targets)
        for B in coefs
    ])
