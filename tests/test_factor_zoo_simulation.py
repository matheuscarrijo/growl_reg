"""Checks for the factor-zoo Monte Carlo in examples/factor_zoo_simulation.

The Monte Carlo uses a Gram-form copy of growl_fista and an exact low-rank
reduction of the multi-task target. These tests check that both give the
package solution.

Run with: pytest tests/test_factor_zoo_simulation.py
"""

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

SIM_DIR = Path(__file__).resolve().parents[1] / "examples" / "factor_zoo_simulation"
sys.path.insert(0, str(SIM_DIR))

import dgp  # noqa: E402
import second_stage as st  # noqa: E402
from growl.fista_solver import growl_fista  # noqa: E402
from growl.prox_operator import prox_growl  # noqa: E402


def _package_fit(X, Y, l1, l2, rs, max_iter=5000, tol=1e-4):
    with contextlib.redirect_stdout(io.StringIO()):
        B, hist = growl_fista(X, Y, lambda_1=l1, lambda_2=l2, ramp_size=rs,
                              max_iter=max_iter, tol=tol, check_type="solution_diff",
                              scale_objective=True)
    return B, len(hist)


@pytest.fixture(scope="module")
def small_problem():
    rng = np.random.default_rng(0)
    scen = dgp.Scenario("test", dgp="proxy", n_dates=240, n_firms=150)
    struct = dgp.build_structure(scen, rng)
    sample = dgp.draw_sample(struct, scen.n_dates, rng, with_missing=False)
    mx, sx = st.standardizer(sample.X)
    X = (sample.X - mx) / sx
    my, sy = st.standardizer(sample.C)
    Y = (sample.C - my) / sy
    return X, Y


def test_prox_matches_package():
    rng = np.random.default_rng(1)
    for _ in range(300):
        V = rng.standard_normal((40, 5)) * rng.uniform(0.1, 3.0)
        w = np.sort(rng.uniform(0.0, 3.0, 40))[::-1]
        np.testing.assert_allclose(st.prox_growl_fast(V, w), prox_growl(V, w), atol=1e-13)


def test_weights_match_package_rule():
    for l1, l2, rs in [(1.0, 0.0, 0.0), (0.5, 0.1, 0.25), (2.0, 1.0, 1.0)]:
        w = st.growl_weights(10, l1, l2, rs)
        ramp = int(np.ceil(rs * 10))
        expected = np.array([l1 + l2 * (ramp - i) if i < ramp else l1 for i in range(10)])
        np.testing.assert_allclose(w, np.sort(expected)[::-1])


def test_low_rank_reduction_is_exact(small_problem):
    X, Y = small_problem
    Yr, V = st.reduce_target(Y)
    assert Yr.shape[1] == 7
    np.testing.assert_allclose(V.T @ V, np.eye(7), atol=1e-10)
    np.testing.assert_allclose(Yr @ V.T, Y, atol=1e-9)
    for l1, l2, rs in [(0.3, 0.0, 0.0), (0.1, 0.01, 0.75)]:
        B_full, n_full = _package_fit(X, Y, l1, l2, rs, tol=1e-8)
        C, n_red = _package_fit(X, Yr, l1, l2, rs, tol=1e-8)
        assert n_full == n_red
        np.testing.assert_allclose(C @ V.T, B_full, atol=1e-8)


USE_NUMBA = [False] + ([True] if st.numba is not None else [])
CELLS = [(0.01, 0.0, 0.0), (0.1, 0.0, 0.0), (1.0, 0.0, 0.0), (0.03, 0.01, 0.25),
         (0.1, 0.1, 0.75), (0.3, 0.001, 1.0), (3.0, 0.1, 0.25)]


@pytest.mark.parametrize("use_numba", USE_NUMBA)
@pytest.mark.parametrize("cell", CELLS)
def test_gram_solver_iterates_equal_growl_fista(small_problem, cell, use_numba):
    """
    With the stopping rule switched off, both solvers give the same iterates.

    Only the first iterations are compared: the two gradients differ in
    rounding, and on ill-conditioned designs FISTA amplifies such rounding
    differences along the path (about 1e-14 after 50 iterations, 1e-7
    after 200), although both paths go to the same minimizer.
    """
    X, Y = small_problem
    Yr, _ = st.reduce_target(Y)
    l1, l2, rs = cell
    B_pkg, n_pkg = _package_fit(X, Yr, l1, l2, rs, max_iter=50, tol=0.0)
    prob = st.GramProblem(X, Yr)
    w = st.growl_weights(X.shape[1], l1, l2, rs)
    B_fast, n_fast = st.growl_fista_gram(prob, w, max_iter=50, tol=0.0, use_numba=use_numba)
    assert n_pkg == n_fast == 50
    np.testing.assert_allclose(B_fast, B_pkg, atol=1e-11 * max(1.0, np.abs(B_pkg).max()))


@pytest.mark.parametrize("use_numba", USE_NUMBA)
@pytest.mark.parametrize("cell", CELLS[3:6])
def test_gram_solver_and_growl_fista_reach_the_same_minimizer(small_problem, cell, use_numba):
    X, Y = small_problem
    Yr, _ = st.reduce_target(Y)
    l1, l2, rs = cell
    B_pkg, _ = _package_fit(X, Yr, l1, l2, rs, max_iter=50000, tol=1e-10)
    prob = st.GramProblem(X, Yr)
    w = st.growl_weights(X.shape[1], l1, l2, rs)
    B_fast, _ = st.growl_fista_gram(prob, w, max_iter=50000, tol=1e-10, use_numba=use_numba)
    np.testing.assert_allclose(B_fast, B_pkg, atol=1e-6 * max(1.0, np.abs(B_pkg).max()))


@pytest.mark.parametrize("use_numba", USE_NUMBA)
@pytest.mark.parametrize("cell", CELLS)
def test_gram_solver_matches_growl_fista_at_paper_settings(small_problem, cell, use_numba):
    """
    Under tol = 1e-4 (solution_diff) the two solvers stop at the same
    iteration or one apart: the gradient is X'X Z - X'Y in one and
    X'(X Z - Y) in the other, and the rounding difference can move the
    iteration where ||B_k - B_{k-1}||_F crosses the tolerance.
    """
    X, Y = small_problem
    Yr, _ = st.reduce_target(Y)
    l1, l2, rs = cell
    B_pkg, n_pkg = _package_fit(X, Yr, l1, l2, rs)
    prob = st.GramProblem(X, Yr)
    w = st.growl_weights(X.shape[1], l1, l2, rs)
    B_fast, n_fast = st.growl_fista_gram(prob, w, use_numba=use_numba)
    assert abs(n_fast - n_pkg) <= 1
    assert np.abs(B_fast - B_pkg).max() <= 10 * st.CV_TOL
    supp_pkg = np.linalg.norm(B_pkg, axis=1) > 0
    supp_fast = np.linalg.norm(B_fast, axis=1) > 0
    assert np.array_equal(supp_pkg, supp_fast)


def test_hv_block_splits_match_paper_design():
    splits = st.hv_block_splits(610)
    assert len(splits) == 5
    for train, val in splits:
        assert val.size == 2 * 24 + 1
        assert train.size == 610 - val.size - 2 * 152
        gap = np.min(np.abs(train[:, None] - val[None, :]))
        assert gap == 153


def test_population_mapping_is_sparse_in_d1():
    rng = np.random.default_rng(2)
    struct = dgp.build_structure(dgp.Scenario("d1", dgp="sparse", n_firms=200), rng)
    pop = dgp.population_quantities(struct)
    assert pop["pop_span_error"] < 1e-12
    nonzero = pop["pop_row_norm"] > 1e-8 * pop["pop_row_norm"].max()
    assert np.array_equal(nonzero, struct.true_columns)


def test_rho_zero_equals_sklearn_multitask_lasso(small_problem):
    """With ramp_size = 0 the GrOWL penalty is the multi-task LASSO penalty."""
    from sklearn.linear_model import MultiTaskLasso

    X, Y = small_problem
    Yr, _ = st.reduce_target(Y)
    for alpha in (0.05, 0.3):
        B, _ = _package_fit(X, Yr, alpha, 0.0, 0.0, max_iter=50000, tol=1e-10)
        mtl = MultiTaskLasso(alpha=alpha, fit_intercept=False, tol=1e-10, max_iter=100000)
        mtl.fit(X, Yr)
        np.testing.assert_allclose(B, mtl.coef_.T, atol=1e-6)


def test_regime_change_keeps_the_first_regime_of_d2():
    """With common random numbers, the first regime of D2_break is D2_proxy."""
    base = dgp.Scenario("base", dgp="proxy", n_dates=120, n_firms=60)
    brk = dgp.Scenario("brk", dgp="proxy", n_dates=120, n_firms=60, break_at=0.5,
                       break_new_themes=3)
    s0 = dgp.build_structure(base, np.random.default_rng(3))
    s1 = dgp.build_structure(brk, np.random.default_rng(3))
    j = s0.betas.shape[1]
    np.testing.assert_array_equal(s1.betas[:, :j], s0.betas)
    np.testing.assert_array_equal(s1.loadings[0], s0.loading)
    # Loadings are permuted inside each theme only.
    for c in np.unique(s1.family):
        rows = s1.family == c
        np.testing.assert_allclose(np.sort(s1.loadings[1][rows]), np.sort(s1.loadings[0][rows]))
    # Three new themes become relevant.
    assert s1.relevant_family.sum() == s0.relevant_family.sum() + 3

    sample = dgp.draw_sample(s1, brk.n_dates, np.random.default_rng(4), with_missing=False)
    regime = s1.regime(brk.n_dates)
    assert regime[59] == 0 and regime[60] == 1
    # Columns of a regime are zero in the other regime.
    assert np.all(sample.F[regime == 1][:, 1:j] == 0)
    assert np.all(sample.F[regime == 0][:, j:] == 0)
    np.testing.assert_allclose(sample.C, sample.F @ s1.betas.T)
    assert sample.span_factors.shape[1] == 1 + 6 + 3


def test_regime_change_is_only_for_the_proxy_process():
    with pytest.raises(ValueError):
        dgp.Scenario("bad", dgp="sparse", break_at=0.5)
    with pytest.raises(ValueError):
        dgp.Scenario("bad", dgp="proxy", break_at=1.5)
