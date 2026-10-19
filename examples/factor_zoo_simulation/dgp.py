"""
Data-generating processes for the factor-zoo Monte Carlo.

Both processes are linear approximate factor models of the APT type (Ross,
1976; Chamberlain and Rothschild, 1983) with zero pricing errors:

    r_it = beta_i' f_t + e_it,    f_t ~ N(mu_f, Sigma_f),    e_it ~ N(0, s_i^2),

where f_t holds the market excess return and six style factors (size, value,
profitability, investment, momentum, low risk). The difference between the
two processes is the relation between f_t and the observed factor zoo X:

``sparse`` (D1, exact sparse spanning, as in Feng, Giglio and Xiu, 2020)
    Each true factor is one column of the zoo. The other members of the six
    relevant themes are redundant noisy copies, and the members of seven
    further themes are correlated with the true factors but do not enter the
    return equation. The population mapping B has exactly seven nonzero rows.

``proxy`` (D2, latent factors with noisy observed proxies, as in Giglio and
Xiu, 2021)
    Only the market is observed exactly. Each style factor is latent and the
    zoo contains noisy proxies of it, x_k = lambda_k f_c + noise. The
    population mapping is dense inside the relevant themes and close to zero
    outside them, so the truth is a family-level support.

All moments come from ``calibration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import calibration as cal

N_THETA = 1 + len(cal.THEMES)
N_STYLE = cal.N_RELEVANT_THEMES


@dataclass(frozen=True)
class Scenario:
    """One Monte Carlo design."""

    name: str
    dgp: str = "sparse"            # "sparse" (D1) or "proxy" (D2)
    n_dates: int = cal.T_EMP
    n_firms: int = cal.N_EMP
    unbalanced: bool = True        # CRSP-like listing spells
    style_scale: float = 1.0       # multiplies the style betas
    market_only: bool = False      # only the market drives returns
    n_unspanned: int = 0           # latent factors with no proxy in the zoo
    unspanned_vol: float = 0.0     # their monthly volatility
    description: str = ""

    def __post_init__(self):
        if self.dgp not in {"sparse", "proxy"}:
            raise ValueError(f"Unknown dgp {self.dgp!r}.")

    @property
    def n_return_factors(self) -> int:
        return 1 if self.market_only else 1 + N_STYLE


@dataclass
class Structure:
    """Parameters that stay fixed within one replication."""

    scenario: Scenario
    names: list
    family: np.ndarray            # (K,) 0 = market, c + 1 = THEMES[c]
    loading: np.ndarray           # (K,) correlation with own theme factor
    true_columns: np.ndarray      # (K,) bool, nonzero rows of the population B
    relevant_family: np.ndarray   # (14,) bool
    betas: np.ndarray             # (N, J)
    idio_vol: np.ndarray          # (N,)
    spell: np.ndarray             # (N, 2) first and last observed month
    theta_chol: np.ndarray = field(repr=False, default=None)
    gamma: np.ndarray = None      # (N, n_unspanned) loadings on unspanned factors

    @property
    def n_factors(self) -> int:
        return len(self.names)


@dataclass
class Sample:
    X: np.ndarray                 # (T, K) observed factor zoo
    F: np.ndarray                 # (T, J) true return factors
    C: np.ndarray                 # (T, N) true common component beta_i' f_t
    R: np.ndarray                 # (T, N) returns, NaN outside listing spells
    G: np.ndarray = None          # (T, n_unspanned) unspanned factors

    @property
    def all_factors(self):
        return self.F if self.G is None else np.hstack([self.F, self.G])


def _quantile_draw(rng, quantiles, size):
    """Inverse-CDF draw from a piecewise-linear quantile function."""
    probs = np.linspace(0.0, 1.0, len(quantiles))
    return np.interp(rng.uniform(size=size), probs, quantiles)


def draw_spells(rng, n_firms, n_dates):
    """Contiguous listing spells calibrated to the CRSP panel."""
    scale = n_dates / cal.T_EMP
    start = np.zeros(n_firms, dtype=int)
    length = np.zeros(n_firms, dtype=int)
    at_first = rng.uniform(size=n_firms) < cal.P_START_AT_FIRST_MONTH
    n0 = int(at_first.sum())
    length[at_first] = np.round(scale * _quantile_draw(rng, cal.SPELL_LENGTH_Q_START0, n0))
    n1 = n_firms - n0
    start[~at_first] = np.round(scale * _quantile_draw(rng, cal.SPELL_START_Q_LATER, n1))
    length[~at_first] = np.round(scale * _quantile_draw(rng, cal.SPELL_LENGTH_Q_LATER, n1))
    length = np.clip(length, 2, n_dates - start)
    return np.column_stack([start, start + length - 1])


def build_structure(scenario: Scenario, rng: np.random.Generator) -> Structure:
    """Draw the zoo loadings, the firm betas, idiosyncratic risk and spells."""
    names, family, loading = ["mkt"], [0], [1.0]
    true_columns = [True]
    for c, theme in enumerate(cal.THEMES):
        relevant = c < N_STYLE and not scenario.market_only
        for j, lam in enumerate(cal.MEMBER_THEME_CORR[theme]):
            names.append(f"{theme}_{j + 1:02d}")
            family.append(c + 1)
            # D1: the first (most representative) member of each relevant
            # theme is the true factor itself.
            exact = scenario.dgp == "sparse" and c < N_STYLE and j == 0
            loading.append(1.0 if exact else lam)
            true_columns.append(relevant and exact if scenario.dgp == "sparse" else relevant)
    family = np.array(family)
    relevant_family = np.zeros(N_THETA, dtype=bool)
    relevant_family[0] = True
    if not scenario.market_only:
        relevant_family[1:1 + N_STYLE] = True

    n_firms = scenario.n_firms
    j_factors = scenario.n_return_factors
    sd = cal.BETA_SD[:j_factors]
    cov = cal.BETA_CORR[:j_factors, :j_factors] * np.outer(sd, sd)
    betas = rng.multivariate_normal(cal.BETA_MEAN[:j_factors], cov, size=n_firms)
    if j_factors > 1:
        betas[:, 1:] *= scenario.style_scale
    idio_vol = np.exp(cal.IDIO_LOGVOL_MEAN + cal.IDIO_LOGVOL_SD * rng.standard_normal(n_firms))

    # Spells are always drawn, so that balanced and unbalanced designs share
    # the same later random draws (common random numbers across scenarios).
    spell = draw_spells(rng, n_firms, scenario.n_dates)
    if not scenario.unbalanced:
        spell = np.column_stack([np.zeros(n_firms, int), np.full(n_firms, scenario.n_dates - 1)])

    # Drawn last, so that the other draws do not change (common random numbers).
    gamma = rng.standard_normal((n_firms, scenario.n_unspanned)) if scenario.n_unspanned else None

    return Structure(
        scenario=scenario,
        gamma=gamma,
        names=names,
        family=family,
        loading=np.array(loading),
        true_columns=np.array(true_columns),
        relevant_family=relevant_family,
        betas=betas,
        idio_vol=idio_vol,
        spell=spell,
        theta_chol=np.linalg.cholesky(cal.THETA_CORR),
    )


def draw_sample(struct: Structure, n_dates: int, rng: np.random.Generator,
                with_returns: bool = True, with_missing: bool = True) -> Sample:
    """Draw factors, the zoo and (optionally) the return panel."""
    z = rng.standard_normal((n_dates, N_THETA)) @ struct.theta_chol.T
    theta = cal.THETA_MEAN + cal.THETA_VOL * z

    k = struct.n_factors
    fam = struct.family
    lam = struct.loading
    noise = rng.standard_normal((n_dates, k))
    x_std = lam * z[:, fam] + np.sqrt(np.clip(1.0 - lam ** 2, 0.0, None)) * noise
    X = cal.THETA_MEAN[fam] + cal.THETA_VOL[fam] * x_std

    j_factors = struct.betas.shape[1]
    F = theta[:, :j_factors]
    C = F @ struct.betas.T
    R = None
    if with_returns:
        R = C + struct.idio_vol * rng.standard_normal(C.shape)
    G = None
    if struct.gamma is not None:
        # Drawn last, so that the other draws do not change.
        G = struct.scenario.unspanned_vol * rng.standard_normal((n_dates, struct.gamma.shape[1]))
        C = C + G @ struct.gamma.T
        if R is not None:
            R = R + G @ struct.gamma.T
    if R is not None and with_missing:
        t = np.arange(n_dates)[:, None]
        observed = (t >= struct.spell[:, 0]) & (t <= struct.spell[:, 1])
        R = np.where(observed, R, np.nan)
    return Sample(X=X, F=F, C=C, R=R, G=G)


def population_quantities(struct: Structure) -> dict:
    """
    Population spanning error and population mapping in standardized units.

    With standardized zoo columns x and standardized factors z_F, the
    population regression of the standardized common component of firm i on
    x has coefficient Sigma_xx^{-1} Cov(x, z_F) D_F beta_i / sd(c_i). The
    spanning error of firm i is the share of Var(c_i) that the whole zoo
    cannot reproduce.
    """
    j_factors = struct.betas.shape[1]
    corr = cal.THETA_CORR
    fam, lam = struct.family, struct.loading
    sxx = np.outer(lam, lam) * corr[np.ix_(fam, fam)]
    np.fill_diagonal(sxx, 1.0)
    g = lam[:, None] * corr[np.ix_(fam, np.arange(j_factors))]       # Cov(x, z_F)
    cff = corr[:j_factors, :j_factors]
    proj = np.linalg.solve(sxx, g)                                      # (K, J)
    resid = cff - g.T @ proj

    d = cal.THETA_VOL[:j_factors]
    b = struct.betas * d                                                 # betas on z_F
    var_c = np.einsum("ij,jk,ik->i", b, cff, b)
    var_res = np.einsum("ij,jk,ik->i", b, resid, b)
    if struct.gamma is not None:
        var_g = struct.scenario.unspanned_vol ** 2 * np.sum(struct.gamma ** 2, axis=1)
        var_c, var_res = var_c + var_g, var_res + var_g
    B_pop = proj @ (b / np.sqrt(var_c)[:, None]).T                       # (K, N)

    row_sq = np.sum(B_pop ** 2, axis=1)
    relevant_rows = struct.relevant_family[fam]
    return {
        "pop_span_error": float(np.mean(var_res / var_c)),
        "pop_relevant_norm_share": float(row_sq[relevant_rows].sum() / row_sq.sum()),
        "pop_row_norm": np.sqrt(row_sq),
    }
