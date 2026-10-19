"""
First stage: Stock-Watson EM-PCA with Bai-Ng factor selection.

This is a NumPy port of the latent-factor step of the companion research
pipeline (``factors_em`` with ``DEMEAN = 2``, information criterion PC_p2 and
``kmax = 10``), including its outlier filter, its convergence rule
(relative squared change of the common component below 1e-6, at most 500
iterations) and the projection of out-of-fold dates onto the in-fold loading
space. The only change is numerical: principal components come from the
T x T Gram matrix instead of the N x N one, which gives the same common
component when T < N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

EM_TOL = 1e-6
EM_MAXIT = 500
KMAX = 10
IC = 2                   # Bai-Ng criterion PC_p2
OUTLIER_IQR_MULT = 10.0


def bai_ng_select(top_eigvals, total_ss, n_dates, n_series, kmax=KMAX, jj=IC):
    """Number of factors that minimizes the Bai and Ng (2002) criterion."""
    nt, nt1 = n_series * n_dates, n_series + n_dates
    ii = np.arange(1, kmax + 1)
    if jj == 1:
        penalty = np.log(nt / nt1) * ii * nt1 / nt
    elif jj == 2:
        penalty = np.log(min(n_series, n_dates)) * ii * nt1 / nt
    elif jj == 3:
        gct = min(n_series, n_dates)
        penalty = np.log(gct) / gct * ii
    else:
        raise ValueError("jj must be 1, 2 or 3.")
    sigma = (total_ss - np.cumsum(top_eigvals[:kmax])) / nt
    ic = np.append(np.log(sigma) + penalty, np.log(total_ss / nt))
    pos = int(np.argmin(ic))
    return pos + 1 if pos < kmax else 0


def _top_components(x3, kmax):
    """Leading left singular vectors and eigenvalues of x3 x3'."""
    n_dates, n_series = x3.shape
    if n_dates <= n_series:
        gram = x3 @ x3.T
        vals, vecs = eigh(gram, subset_by_index=[n_dates - kmax, n_dates - 1])
        vals, vecs = vals[::-1], vecs[:, ::-1]
        total = float(np.trace(gram))
    else:
        gram = x3.T @ x3
        vals, v = eigh(gram, subset_by_index=[n_series - kmax, n_series - 1])
        vals, v = vals[::-1], v[:, ::-1]
        vecs = x3 @ v / np.sqrt(np.maximum(vals, 1e-300))
        total = float(np.trace(gram))
    return np.maximum(vals, 0.0), vecs, total


def _standardize(x2):
    mean = x2.mean(axis=0)
    std = x2.std(axis=0, ddof=1)
    return (x2 - mean) / std, mean, std


@dataclass
class EMPCAModel:
    pred: np.ndarray            # (T, N_kept) common component, return units
    fhat: np.ndarray            # (T, k)
    lamhat: np.ndarray          # (N_kept, k), scaled by sqrt(N_kept)
    k: int
    n_iter: int
    columns: np.ndarray         # indices of the kept series
    mean: np.ndarray            # moments of the final EM-filled panel
    std: np.ndarray
    out_median: np.ndarray
    out_iqr: np.ndarray

    def project(self, returns):
        """Project new dates onto the in-sample loading space (out-of-fold)."""
        x = returns[:, self.columns]
        x = np.where(np.abs(x - self.out_median) > OUTLIER_IQR_MULT * self.out_iqr, np.nan, x)
        filled = np.where(np.isnan(x), self.mean, x)
        z = (filled - self.mean) / self.std
        n_series = self.lamhat.shape[0]
        factors = z @ self.lamhat / n_series
        chat = factors @ self.lamhat.T
        return chat * self.std + self.mean

    def project_ols(self, returns):
        """
        Alternative projection: for each date, regress the standardized
        returns of the observed series on their loadings. This is the
        E-step of the EM algorithm with the loadings held fixed. Unlike
        ``project``, it does not shrink the factors toward zero when many
        series are unobserved.
        """
        x = returns[:, self.columns]
        x = np.where(np.abs(x - self.out_median) > OUTLIER_IQR_MULT * self.out_iqr, np.nan, x)
        z = (x - self.mean) / self.std
        factors = np.zeros((x.shape[0], self.k))
        for t in range(x.shape[0]):
            o = ~np.isnan(z[t])
            if o.sum() >= self.k:
                factors[t] = np.linalg.lstsq(self.lamhat[o], z[t, o], rcond=None)[0]
        chat = factors @ self.lamhat.T
        return chat * self.std + self.mean


def fit_em_pca(returns, kmax=KMAX, jj=IC, tol=EM_TOL, maxit=EM_MAXIT):
    """EM-PCA of a (T, N) return panel with NaN for unobserved cells."""
    obs = ~np.isnan(returns)
    keep = obs.sum(axis=0) > 0
    x = returns[:, keep]

    med = np.nanmedian(x, axis=0)
    iqr = np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
    x = np.where(np.abs(x - med) > OUTLIER_IQR_MULT * iqr, np.nan, x)
    obs = ~np.isnan(x)
    counts = obs.sum(axis=0)
    std_obs = np.nanstd(x, axis=0, ddof=1) if x.shape[0] > 1 else np.zeros(x.shape[1])
    usable = (counts > 1) & (np.nan_to_num(std_obs) > 1e-12)
    columns = np.flatnonzero(keep)[usable]
    x, obs, med, iqr = x[:, usable], obs[:, usable], med[usable], iqr[usable]
    if np.any(obs.sum(axis=1) == 0):
        raise ValueError("A date has no observed series.")

    n_dates, n_series = x.shape
    x2 = np.where(obs, x, np.nanmean(x, axis=0))
    x3, mean, std = _standardize(x2)
    vals, vecs, total = _top_components(x3, kmax)
    k = bai_ng_select(vals, total, n_dates, n_series, kmax, jj)
    chat = vecs[:, :k] @ (vecs[:, :k].T @ x3)
    chat0 = chat
    err, it = np.inf, 0
    while err > tol and it < maxit:
        it += 1
        x2 = np.where(obs, x, chat * std + mean)
        x3, mean, std = _standardize(x2)
        vals, vecs, total = _top_components(x3, kmax)
        k = bai_ng_select(vals, total, n_dates, n_series, kmax, jj)
        u = vecs[:, :k]
        chat = u @ (u.T @ x3)
        err = float(np.sum((chat - chat0) ** 2) / np.sum(chat0 ** 2))
        chat0 = chat

    u = vecs[:, :k]
    s = np.sqrt(vals[:k])
    v = (x3.T @ u) / s                       # right singular vectors
    lamhat = v * np.sqrt(n_series)
    fhat = x3 @ lamhat / n_series
    return EMPCAModel(
        pred=chat * std + mean,
        fhat=fhat,
        lamhat=lamhat,
        k=k,
        n_iter=it,
        columns=columns,
        mean=mean,
        std=std,
        out_median=med,
        out_iqr=iqr,
    )
