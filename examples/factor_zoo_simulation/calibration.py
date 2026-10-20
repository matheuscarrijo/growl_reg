"""
Empirical moments that calibrate the factor-zoo Monte Carlo.

Every number in this module is a summary statistic. The numbers were computed
once from the data of the companion research project (CRSP monthly stock
returns, 1973-03 to 2023-12, and the Jensen-Kelly-Pedersen Global Factor Data,
US, value-weighted capped portfolios) with ``compute_calibration_moments``
below. No data file is shipped with this example; the function is kept only to
document how each number was obtained.

Notation
--------
Theta : the 14 "theme" return series, in this order: the market excess return
        (``mkt``) followed by the 13 JKP themes in ``THEMES``.
The first seven entries of Theta (``mkt`` plus the first six themes) are the
return factors of the data-generating processes in ``dgp.py``.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Sample dimensions of the empirical application
# ---------------------------------------------------------------------------

T_EMP = 610          # months with both returns and observed factors
N_EMP = 1965         # firms in the cleaned CRSP panel
K_EMP = 154          # observed factors (153 JKP factors + mkt-rf)
MISSING_SHARE_EMP = 0.683

# ---------------------------------------------------------------------------
# Factor zoo: themes, members and theme-level moments
# ---------------------------------------------------------------------------

# JKP theme order. The first six themes are the style factors that drive
# returns in the simulation ("relevant" families); the last seven are
# "irrelevant" families that are correlated with the relevant ones but do not
# enter the return-generating process.
THEMES = [
    "size", "value", "profitability", "investment", "momentum", "low_risk",
    "accruals", "debt_issuance", "low_leverage", "profit_growth", "quality",
    "seasonality", "short_term_reversal",
]
N_RELEVANT_THEMES = 6
RETURN_FACTORS = ["mkt"] + THEMES[:N_RELEVANT_THEMES]

# Correlation of each JKP factor with its own theme return, sorted. Each
# factor is assigned to the theme with which it has the highest correlation.
# Theme sizes sum to 153; with mkt-rf the zoo has 154 columns.
MEMBER_THEME_CORR = {
    "size": [0.918, 0.885, 0.881, 0.787, 0.61],
    "value": [0.957, 0.953, 0.952, 0.944, 0.937, 0.934, 0.93, 0.927, 0.922,
              0.91, 0.9, 0.898, 0.893, 0.893, 0.858, 0.856, 0.817],
    "profitability": [0.956, 0.934, 0.921, 0.913, 0.906, 0.863, 0.856, 0.842,
                      0.798, 0.787, 0.765, 0.707, 0.704],
    "investment": [0.963, 0.944, 0.94, 0.936, 0.926, 0.925, 0.914, 0.911,
                   0.908, 0.905, 0.901, 0.893, 0.889, 0.888, 0.876, 0.868,
                   0.856, 0.852, 0.832, 0.813, 0.759, 0.592, 0.487],
    "momentum": [0.952, 0.929, 0.926, 0.921, 0.802, 0.794, 0.783, 0.713],
    "low_risk": [0.988, 0.976, 0.976, 0.971, 0.969, 0.969, 0.967, 0.949,
                 0.938, 0.935, 0.933, 0.929, 0.922, 0.909, 0.831, 0.814,
                 0.697, 0.615],
    "accruals": [0.827, 0.824, 0.774, 0.705, 0.664, 0.501],
    "debt_issuance": [0.729, 0.702, 0.65, 0.626, 0.597, 0.568, 0.337],
    "low_leverage": [0.967, 0.944, 0.937, 0.91, 0.899, 0.872, 0.86, 0.815,
                     0.8, 0.698, 0.644, 0.236],
    "profit_growth": [0.828, 0.796, 0.761, 0.758, 0.749, 0.719, 0.607, 0.536,
                      0.534, 0.495, 0.485, 0.337, 0.262],
    "quality": [0.902, 0.901, 0.874, 0.86, 0.84, 0.814, 0.76, 0.75, 0.706,
                0.698, 0.623, 0.622, 0.603, 0.512, 0.499, 0.491, 0.273],
    "seasonality": [0.513, 0.484, 0.389, 0.348, 0.346, 0.319, 0.315, 0.258],
    "short_term_reversal": [0.788, 0.759, 0.735, 0.726, 0.699, 0.605],
}

# Monthly means and volatilities of Theta (mkt first, then THEMES).
THETA_MEAN = np.array([
    0.00609, 0.00096, 0.00401, 0.00243, 0.0031, 0.00322, 0.00154,
    0.00248, 0.0022, -0.00046, 0.0016, 0.0027, 0.00129, 0.0012,
])
THETA_VOL = np.array([
    0.04635, 0.02332, 0.03438, 0.02275, 0.02217, 0.03351, 0.03901,
    0.01159, 0.00735, 0.03176, 0.0114, 0.01611, 0.00583, 0.01266,
])
THETA_CORR = np.array([
    [1.0, 0.197, -0.305, -0.229, -0.405, -0.251, -0.621, -0.171, -0.017, 0.354, 0.094, -0.037, -0.25, 0.038],
    [0.197, 1.0, -0.028, -0.55, 0.009, -0.396, -0.404, 0.087, 0.202, 0.24, -0.328, -0.315, 0.061, 0.029],
    [-0.305, -0.028, 1.0, 0.548, 0.83, -0.267, 0.743, 0.023, -0.22, -0.927, -0.484, -0.219, 0.143, 0.304],
    [-0.229, -0.55, 0.548, 1.0, 0.281, 0.087, 0.665, -0.377, -0.256, -0.637, 0.099, 0.526, 0.12, 0.249],
    [-0.405, 0.009, 0.83, 0.281, 1.0, -0.01, 0.701, 0.215, 0.122, -0.789, -0.496, -0.365, 0.129, 0.124],
    [-0.251, -0.396, -0.267, 0.087, -0.01, 1.0, 0.164, -0.019, 0.238, 0.039, 0.445, 0.22, 0.214, -0.27],
    [-0.621, -0.404, 0.743, 0.665, 0.701, 0.164, 1.0, -0.056, -0.226, -0.84, -0.223, 0.055, 0.186, 0.254],
    [-0.171, 0.087, 0.023, -0.377, 0.215, -0.019, -0.056, 1.0, 0.047, 0.036, -0.154, -0.39, 0.075, -0.094],
    [-0.017, 0.202, -0.22, -0.256, 0.122, 0.238, -0.226, 0.047, 1.0, 0.322, 0.151, 0.086, -0.051, -0.204],
    [0.354, 0.24, -0.927, -0.637, -0.789, 0.039, -0.84, 0.036, 0.322, 1.0, 0.32, 0.175, -0.234, -0.263],
    [0.094, -0.328, -0.484, 0.099, -0.496, 0.445, -0.223, -0.154, 0.151, 0.32, 1.0, 0.38, 0.094, -0.145],
    [-0.037, -0.315, -0.219, 0.526, -0.365, 0.22, 0.055, -0.39, 0.086, 0.175, 0.38, 1.0, 0.073, 0.027],
    [-0.25, 0.061, 0.143, 0.12, 0.129, 0.214, 0.186, 0.075, -0.051, -0.234, 0.094, 0.073, 1.0, -0.031],
    [0.038, 0.029, 0.304, 0.249, 0.124, -0.27, 0.254, -0.094, -0.204, -0.263, -0.145, 0.027, -0.031, 1.0],
])

# ---------------------------------------------------------------------------
# Firm-level return process
# ---------------------------------------------------------------------------

# Cross-sectional distribution of firm betas on RETURN_FACTORS, from
# firm-by-firm OLS of excess returns on mkt-rf and the six relevant theme
# returns (firms with at least 120 monthly observations, 1,148 firms). The
# standard deviations and correlations are deflated: the average sampling
# covariance of the OLS betas is subtracted from their cross-sectional
# covariance.
BETA_MEAN = np.array([0.977, 0.436, 0.261, 0.082, -0.088, 0.100, -0.187])
BETA_SD = np.array([0.188, 0.678, 0.982, 0.906, 0.755, 0.315, 0.747])
BETA_CORR = np.array([
    [1.0, 0.15, -0.09, 0.18, 0.16, -0.24, 0.17],
    [0.15, 1.0, -0.27, 0.47, 0.02, 0.07, 0.13],
    [-0.09, -0.27, 1.0, -0.53, -0.69, 0.71, -0.30],
    [0.18, 0.47, -0.53, 1.0, 0.58, -0.34, -0.19],
    [0.16, 0.02, -0.69, 0.58, 1.0, -0.68, -0.07],
    [-0.24, 0.07, 0.71, -0.34, -0.68, 1.0, -0.29],
    [0.17, 0.13, -0.30, -0.19, -0.07, -0.29, 1.0],
])

# log of the monthly idiosyncratic volatility is normal with these moments.
# The OLS residual moments are -2.491 (mean) and 0.331 (sd). With the betas
# above they give a mean firm R^2 of 0.356 (7 factors) and 0.248 (market),
# against 0.318 and 0.230 in the data. The mean is shifted up by 0.085 so
# that the simulated mean firm R^2 matches the data; this also brings the
# market-only R^2 to the empirical value.
IDIO_LOGVOL_MEAN = -2.491 + 0.085
IDIO_LOGVOL_SD = 0.331

# Listing spells. Firms are observed on one contiguous spell (only 2% of the
# CRSP firms have internal gaps). 27.7% of the firms are listed in the first
# sample month. The quantile vectors are at probabilities 0, 0.05, ..., 1.
P_START_AT_FIRST_MONTH = 0.277
SPELL_LENGTH_Q_START0 = [58, 75, 96, 109, 127, 143, 155, 166, 182, 197, 259,
                         295, 318, 330, 390, 436, 531, 610, 610, 610, 610]
SPELL_START_Q_LATER = [2, 168, 208, 227, 242, 254, 272, 284, 302, 323, 350,
                       379, 399, 424, 456, 475, 492, 505, 520, 537, 558]
SPELL_LENGTH_Q_LATER = [52, 63, 67, 73, 78, 84, 89, 95, 102, 110, 119, 129,
                        140, 157, 174, 201, 230, 265, 310, 355, 549]

# ---------------------------------------------------------------------------
# Empirical pipeline outputs, used as targets for the model check
# ---------------------------------------------------------------------------

EMPIRICAL_TARGETS = {
    # Factor zoo (154 columns).
    "zoo_eig_share": [0.357, 0.144, 0.077, 0.051, 0.037, 0.032, 0.025,
                      0.022, 0.018, 0.015],
    "zoo_corr_q05_q25_q50_q75_q95": [-0.573, -0.155, 0.053, 0.339, 0.766],
    "zoo_share_abs_corr_gt_0.8": 0.048,
    # First stage (EM-PCA, Bai-Ng PC_p2, kmax = 10).
    "bai_ng_factors": 7,
    "mean_firm_r2_7_factors": 0.318,
    "mean_firm_r2_mkt": 0.230,
    # Standardized common component Y (the GrOWL target).
    "Y_sv2_share": [0.406, 0.140, 0.114, 0.106, 0.099, 0.070, 0.065],
    "Y_r2_on_mkt": 0.357,
    "Y_r2_on_all_X_in_sample": 0.857,
    "cancorr_Fhat_X": [0.990, 0.954, 0.939, 0.911, 0.864, 0.840, 0.800],
    # Inside-loop PCA hv-block CV (594-cell grid, 5 folds), with the study's
    # original out-of-fold projection (z·Λ/N; superseded).
    "cv_raw_min_cell": (3.162, 0.1, 0.75),
    "cv_promoted_cell": (10.0, 0.1, 0.25),
    "cv_promoted_support": ["mkt-rf"],
    # Same CV with the OLS out-of-fold projection (2026-10).
    "cv_olsproj_min_cell": (3.162, 0.0316, 0.5),
    # Split-half statistics of montecarlo._split_half on the full-sample
    # target at cv_olsproj_min_cell (halves 1973-03..1998-07 and
    # 1998-08..2023-12).
    "split_half": {
        "split_within_r2": 0.596, "split_across_r2": 0.296,
        "split_mkt_within_r2": 0.389, "split_mkt_across_r2": 0.270,
        "split_support_jaccard": 0.489, "split_row_cosine": 0.343,
        "split_n_selected": 97.5,
    },
}


def compute_calibration_moments(tese_root):
    """
    Recompute the moments above from a local checkout of the research project.

    This function is documentation of provenance. It needs the processed data
    files of the research project, which are not part of this package:
    ``data/processed/obs_F.csv``, ``data/processed/ret_pivot.csv`` and
    ``data/raw/[usa]_[all_themes]_[monthly]_[vw_cap].csv``.
    """
    from pathlib import Path

    import pandas as pd

    root = Path(tese_root)
    themes = pd.read_csv(root / "data/raw/[usa]_[all_themes]_[monthly]_[vw_cap].csv")
    themes = themes.pivot(index="date", columns="name", values="ret")
    themes.index = pd.to_datetime(themes.index)
    obs_f = pd.read_csv(root / "data/processed/obs_F.csv", index_col=0)
    obs_f.index = pd.to_datetime(obs_f.index)
    ret = pd.read_csv(root / "data/processed/ret_pivot.csv", index_col=0)
    ret.index = pd.to_datetime(ret.index)
    dates = obs_f.index.intersection(themes.index).intersection(ret.index)
    themes, obs_f, ret = themes.loc[dates, THEMES], obs_f.loc[dates], ret.loc[dates]

    theta = pd.concat([obs_f[["mkt-rf"]], themes], axis=1)
    out = {
        "THETA_MEAN": theta.mean().to_numpy(),
        "THETA_VOL": theta.std().to_numpy(),
        "THETA_CORR": theta.corr().to_numpy(),
    }

    members = obs_f.drop(columns="mkt-rf")
    corr = pd.concat([members, themes], axis=1).corr().loc[members.columns, THEMES]
    theme_of = corr.idxmax(axis=1)
    own = corr.max(axis=1)
    out["MEMBER_THEME_CORR"] = {
        t: sorted(own[theme_of == t].round(3).tolist(), reverse=True) for t in THEMES
    }

    factors = theta.iloc[:, : len(RETURN_FACTORS)].to_numpy()
    betas, sampling_cov, idio_var = [], [], []
    for col in ret.columns:
        y = ret[col].to_numpy()
        m = ~np.isnan(y)
        if m.sum() < 120:
            continue
        a = np.column_stack([np.ones(m.sum()), factors[m]])
        b, *_ = np.linalg.lstsq(a, y[m], rcond=None)
        e = y[m] - a @ b
        s2 = e.var(ddof=a.shape[1])
        betas.append(b[1:])
        idio_var.append(s2)
        sampling_cov.append(s2 * np.linalg.inv(a.T @ a)[1:, 1:])
    betas = np.array(betas)
    cov = np.cov(betas.T) - np.mean(sampling_cov, axis=0)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    out["BETA_MEAN"] = betas.mean(axis=0)
    out["BETA_SD"] = sd
    out["BETA_CORR"] = cov / np.outer(sd, sd)
    log_vol = np.log(np.sqrt(idio_var))
    out["IDIO_LOGVOL_MEAN"] = log_vol.mean()
    out["IDIO_LOGVOL_SD"] = log_vol.std()

    obs = ret.notna().to_numpy()
    n_dates = obs.shape[0]
    first = obs.argmax(axis=0)
    last = n_dates - 1 - obs[::-1].argmax(axis=0)
    length = last - first + 1
    q = np.linspace(0.0, 1.0, 21)
    out["P_START_AT_FIRST_MONTH"] = float(np.mean(first == 0))
    out["SPELL_LENGTH_Q_START0"] = np.quantile(length[first == 0], q).round()
    out["SPELL_START_Q_LATER"] = np.quantile(first[first > 0], q).round()
    out["SPELL_LENGTH_Q_LATER"] = np.quantile(length[first > 0], q).round()
    return out
