"""
One Monte Carlo replication of the two-stage latent + observed factor pipeline.

Steps of a replication
----------------------
1. Draw the structure (zoo loadings, firm betas, idiosyncratic risk, listing
   spells), an estimation sample of T months and an independent test sample
   of T months (``dgp.py``).
2. First stage on the full sample: EM-PCA with Bai-Ng selection on the
   unbalanced return panel (``first_stage.py``).
3. Second stage for two targets:
   - ``pipeline``: the standardized EM-PCA common component (the paper's
     target);
   - ``oracle``: the standardized true common component (no first-stage
     error), which separates first-stage error from second-stage behavior.
4. Inside-loop hv-block CV (5 folds, h = floor(0.25 T), v = floor(sqrt(T))):
   for the pipeline target the EM-PCA is refit on each training block and
   the validation dates are projected onto the training loading space.
   GrOWL (117 cells, which include the multi-task Lasso path rho = 0) and the
   multi-task Elastic Net (50 cells) are scored on the same folds.
5. Full-sample refit of every cell, selection rules (CV minimum, CV minimum
   among cells that converged in all folds, one-standard-error rule, and the
   infeasible oracle cell), and recovery metrics against the known truth.
6. Split-half stability: GrOWL at the CV-selected cell is fitted on each
   half of the sample, and each half's fit is scored on the other half
   (``split_*`` diagnostics). In the regime-change scenarios the break is at
   the midpoint, so the halves are the two regimes.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np

import dgp
import first_stage as fs
import metrics as mt
import second_stage as st

BASE_SEED = 20261019

SCENARIOS = {
    "D1_sparse": dgp.Scenario(
        "D1_sparse", dgp="sparse",
        description="Exact sparse spanning: mkt and one member of each of six style "
                    "themes are the true factors; unbalanced CRSP-like panel."),
    "D2_proxy": dgp.Scenario(
        "D2_proxy", dgp="proxy",
        description="Latent style factors with noisy zoo proxies (mkt observed); "
                    "unbalanced CRSP-like panel."),
    "D0_market_only": dgp.Scenario(
        "D0_market_only", dgp="sparse", market_only=True,
        description="Only the market drives returns; unbalanced CRSP-like panel."),
    "D2_proxy_balanced": dgp.Scenario(
        "D2_proxy_balanced", dgp="proxy", unbalanced=False,
        description="As D2_proxy, but every firm is observed in every month."),
    "D2_proxy_weak": dgp.Scenario(
        "D2_proxy_weak", dgp="proxy", style_scale=0.5,
        description="As D2_proxy, with style betas halved (weaker non-market factors)."),
    "D2_proxy_strong": dgp.Scenario(
        "D2_proxy_strong", dgp="proxy", style_scale=1.5,
        description="As D2_proxy, with style betas x1.5; matches the PCA moments of the "
                    "data (market share of the common component) instead of the firm R^2."),
    "D1_sparse_balanced": dgp.Scenario(
        "D1_sparse_balanced", dgp="sparse", unbalanced=False,
        description="As D1_sparse, but every firm is observed in every month."),
    "D2_break": dgp.Scenario(
        "D2_break", dgp="proxy", break_at=0.5,
        description="As D2_proxy, with a regime change at mid-sample: style betas drawn "
                    "again and proxy loadings permuted inside each theme."),
    "D2_break_themes": dgp.Scenario(
        "D2_break_themes", dgp="proxy", break_at=0.5, break_new_themes=3,
        description="As D2_break, and at the break three style themes (investment, momentum, "
                    "low risk) are replaced by three other themes (accruals, debt issuance, "
                    "low leverage)."),
}




def _select(mean_mse, std_mse, hit_any, support, oos_rel, n_folds):
    """Indices chosen by each selection rule."""
    i_min = int(np.argmin(mean_mse))
    clean = np.flatnonzero(~hit_any)
    i_clean = int(clean[np.argmin(mean_mse[clean])]) if clean.size else i_min
    threshold = mean_mse[i_min] + std_mse[i_min] / np.sqrt(n_folds)
    cand = np.flatnonzero(mean_mse <= threshold)
    i_1se = int(cand[np.lexsort((mean_mse[cand], support[cand]))[0]])
    return {
        "cv_min": i_min,
        "cv_min_clean": i_clean,
        "cv_1se": i_1se,
        "oracle": int(np.argmin(oos_rel)),
    }


def _r2_reduced(Yr, x):
    b, *_ = np.linalg.lstsq(x, Yr, rcond=None)
    return float(1.0 - np.sum((Yr - x @ b) ** 2) / np.sum(Yr ** 2))


def _firm_r2(R, F, min_obs=120):
    obs = ~np.isnan(R)
    r2_all, r2_mkt = [], []
    for i in np.flatnonzero(obs.sum(axis=0) >= min_obs):
        m = obs[:, i]
        y = R[m, i]
        a = np.column_stack([np.ones(m.sum()), F[m]])
        for cols, out in ((slice(None), r2_all), (slice(0, 2), r2_mkt)):
            b, *_ = np.linalg.lstsq(a[:, cols], y, rcond=None)
            out.append(1.0 - np.var(y - a[:, cols] @ b) / np.var(y))
    return float(np.mean(r2_all)), float(np.mean(r2_mkt))


def _solver_check(Xs, Yr, cells, coefs, iters, rng, n_cells):
    """Re-solve random cells with the package solver growl_fista."""
    from growl.fista_solver import growl_fista

    out = []
    for i in rng.choice(len(cells), size=n_cells, replace=False):
        l1, l2, rs = cells[i]
        with contextlib.redirect_stdout(io.StringIO()):
            B, hist = growl_fista(Xs, Yr, lambda_1=l1, lambda_2=l2, ramp_size=rs,
                                  max_iter=st.CV_MAX_ITER, tol=st.CV_TOL,
                                  check_type="solution_diff", scale_objective=True)
        scale = max(np.abs(B).max(), 1e-12)
        out.append({
            "cell": int(i),
            "max_rel_diff": float(np.abs(B - coefs[i]).max() / scale),
            "same_n_iter": bool(len(hist) == iters[i]),
            "same_support": bool(np.array_equal(np.linalg.norm(B, axis=1) > 0,
                                                np.linalg.norm(coefs[i], axis=1) > 0)),
        })
    return out




def _split_half(Xs, Yr, cell):
    """
    Fit GrOWL at `cell` on each half of the sample and score it on both halves.

    Each half is centered on its own means, so that only the slopes transfer.
    Returns the mean within-half R^2, the mean across-half R^2, the Jaccard
    index of the two supports and the mean cosine between the rows that both
    halves select. Market-only OLS gives the same two R^2 for reference.
    """
    n = Xs.shape[0]
    halves = [np.arange(n // 2), np.arange(n // 2, n)]
    data = [(Xs[h] - Xs[h].mean(axis=0), Yr[h] - Yr[h].mean(axis=0)) for h in halves]
    w = st.growl_weights(Xs.shape[1], *cell)
    fits = [st.growl_fista_gram(st.GramProblem(x, y), w)[0] for x, y in data]
    mkt = [np.linalg.lstsq(x[:, :1], y, rcond=None)[0] for x, y in data]

    def r2(d, b, cols=slice(None)):
        x, y = d
        return 1.0 - np.sum((y - x[:, cols] @ b) ** 2) / np.sum(y ** 2)

    s0, s1 = (np.linalg.norm(B, axis=1) > 0 for B in fits)
    both = np.flatnonzero(s0 & s1)
    cos = [fits[0][j] @ fits[1][j] / (np.linalg.norm(fits[0][j]) * np.linalg.norm(fits[1][j]))
           for j in both]
    union = int(np.sum(s0 | s1))
    return {
        "split_within_r2": float(np.mean([r2(data[a], fits[a]) for a in (0, 1)])),
        "split_across_r2": float(np.mean([r2(data[1 - a], fits[a]) for a in (0, 1)])),
        "split_support_jaccard": float(len(both) / union) if union else np.nan,
        "split_row_cosine": float(np.mean(cos)) if cos else np.nan,
        "split_n_selected": float(np.mean([s0.sum(), s1.sum()])),
        "split_mkt_within_r2": float(np.mean([r2(data[a], mkt[a], slice(0, 1)) for a in (0, 1)])),
        "split_mkt_across_r2": float(np.mean([r2(data[1 - a], mkt[a], slice(0, 1)) for a in (0, 1)])),
    }


def _split_half_diagnostics(Xs, targets, pipeline_cell, oracle_cell):
    out = {}
    for prefix, tname, cell in (("", "pipeline", pipeline_cell), ("oracle_", "oracle", oracle_cell)):
        for k, v in _split_half(Xs, targets[tname]["Yr"], cell).items():
            out[prefix + k] = v
    return out


def split_half_for_stored(scenario: dgp.Scenario, result: dict, base_seed: int = BASE_SEED) -> dict:
    """
    Split-half diagnostics for a replication stored without them.

    The draws, the EM-PCA target and the selected cells are those of the
    stored replication, so the values equal those of a new run.
    """
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, result["rep"]]))
    struct = dgp.build_structure(scenario, rng)
    sample = dgp.draw_sample(struct, scenario.n_dates, rng)
    pca = fs.fit_em_pca(sample.R)
    mx, sx = st.standardizer(sample.X)
    targets = {}
    for name, level in (("pipeline", pca.pred), ("oracle", sample.C[:, pca.columns])):
        my, sy = st.standardizer(level)
        targets[name] = {"Yr": st.reduce_target((level - my) / sy)[0]}

    def cell(variant):
        row = next(r for r in result["rows"] if r["target"] == variant and r["method"] == "growl"
                   and r["rule"] == "cv_min")
        return row["lambda_1"], row["lambda_2"], row["ramp_size"]

    variant = "pipeline_olsproj" if any(r["target"] == "pipeline_olsproj" for r in result["rows"]) \
        else "pipeline"
    return _split_half_diagnostics((sample.X - mx) / sx, targets, cell(variant), cell("oracle"))


def _tight_check(prob, cells, i, B, max_iter=200_000, tol=1e-9):
    """Re-solve one cell to a tight tolerance and compare with the CV solution."""
    l1, l2, rs = cells[i]
    B_tight, n_iter = st.growl_fista_gram(prob, st.growl_weights(B.shape[0], l1, l2, rs),
                                          max_iter=max_iter, tol=tol)
    s0 = np.linalg.norm(B, axis=1) > 0
    s1 = np.linalg.norm(B_tight, axis=1) > 0
    union = np.sum(s0 | s1)
    denom = np.linalg.norm(B_tight)
    return {
        "tight_n_selected": int(s1.sum()),
        "tight_support_jaccard": float(np.sum(s0 & s1) / union) if union else 1.0,
        "tight_rel_diff": float(np.linalg.norm(B - B_tight) / denom) if denom > 0 else 0.0,
        "tight_converged": bool(n_iter < max_iter),
    }


def run_replication(scenario: dgp.Scenario, rep: int, base_seed: int = BASE_SEED,
                    solver_check_cells: int = 2, ols_projection: bool = True,
                    tight_check: bool = True) -> dict:
    t_start = time.time()
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, rep]))
    struct = dgp.build_structure(scenario, rng)
    sample = dgp.draw_sample(struct, scenario.n_dates, rng)
    test = dgp.draw_sample(struct, scenario.n_dates, rng, with_returns=False)
    pop = dgp.population_quantities(struct)

    cells = st.growl_cells()
    ecells = st.enet_cells()
    mtl_idx = np.array([i for i, c in enumerate(cells) if c[2] == 0.0])
    n_dates = scenario.n_dates

    # CV variant -> full-sample target whose refits it selects among.
    variants = {"pipeline": "pipeline"}
    if ols_projection:
        variants["pipeline_olsproj"] = "pipeline"
    variants["oracle"] = "oracle"

    # ---------------- full-sample first stage ----------------
    t0 = time.time()
    pca = fs.fit_em_pca(sample.R)
    t_pca = time.time() - t0
    kept = pca.columns
    X = sample.X
    mx, sx = st.standardizer(X)
    Xs = (X - mx) / sx
    loadings = struct.betas if struct.gamma is None else np.hstack([struct.betas, struct.gamma])
    oos = mt.OOSEvaluator((test.X - mx) / sx, test.all_factors, loadings[kept])

    targets = {}
    for name, level in (("pipeline", pca.pred), ("oracle", sample.C[:, kept])):
        my, sy = st.standardizer(level)
        Yr, V = st.reduce_target((level - my) / sy)
        targets[name] = {"Yr": Yr, "V": V, "sy": sy}

    # ---------------- inside-loop hv-block CV ----------------
    splits = st.hv_block_splits(n_dates)
    n_folds = len(splits)
    cv = {v: {"g_mse": np.zeros((n_folds, len(cells))),
              "g_iter": np.zeros((n_folds, len(cells)), dtype=int),
              "e_mse": np.zeros((n_folds, len(ecells))),
              "val_var": np.zeros(n_folds)} for v in variants}
    fold_k, fold_em_iter, fold_n_firms = [], [], []
    for f, (tr, va) in enumerate(splits):
        mxf, sxf = st.standardizer(X[tr])
        Xtr, Xva = (X[tr] - mxf) / sxf, (X[va] - mxf) / sxf
        fpca = fs.fit_em_pca(sample.R[tr])
        fold_k.append(fpca.k)
        fold_em_iter.append(fpca.n_iter)
        fold_n_firms.append(len(fpca.columns))
        fold_targets = {"pipeline": ("pca", fpca.pred, fpca.project(sample.R[va]))}
        if ols_projection:
            fold_targets["pipeline_olsproj"] = ("pca", fpca.pred, fpca.project_ols(sample.R[va]))
        fold_targets["oracle"] = ("true", sample.C[tr][:, kept], sample.C[va][:, kept])
        trained = {}
        for name, (train_key, ytr_level, yva_level) in fold_targets.items():
            myf, syf = st.standardizer(ytr_level)
            Yva = (yva_level - myf) / syf
            if train_key not in trained:
                Ytr_r, Vf = st.reduce_target((ytr_level - myf) / syf)
                prob = st.GramProblem(Xtr, Ytr_r)
                coefs, iters = st.fit_growl_cells(prob, cells)
                ecoefs = st.fit_enet_cells(Xtr, Ytr_r, ecells)
                trained[train_key] = (Vf, coefs, iters, ecoefs)
            Vf, coefs, iters, ecoefs = trained[train_key]
            Yva_r = Yva @ Vf
            resid = max(float(np.sum(Yva ** 2) - np.sum(Yva_r ** 2)), 0.0)
            n_targets = Yva.shape[1]
            cv[name]["g_mse"][f] = st.validation_mse(coefs, Xva, Yva_r, resid, n_targets)
            cv[name]["g_iter"][f] = iters
            cv[name]["e_mse"][f] = st.validation_mse(ecoefs, Xva, Yva_r, resid, n_targets)
            cv[name]["val_var"][f] = float(np.mean(Yva.var(axis=0)))

    # ---------------- full-sample refits ----------------
    full = {}
    solver_check = []
    for name, tg in targets.items():
        prob = st.GramProblem(Xs, tg["Yr"])
        coefs, iters = st.fit_growl_cells(prob, cells)
        ecoefs = st.fit_enet_cells(Xs, tg["Yr"], ecells)
        if name == "pipeline" and solver_check_cells:
            solver_check = _solver_check(Xs, tg["Yr"], cells, coefs, iters, rng, solver_check_cells)
        full[name] = {
            "prob": prob, "coefs": coefs, "iters": iters, "ecoefs": ecoefs,
            "g_oos": np.array([oos(B, tg["V"], tg["sy"]) for B in coefs]),
            "e_oos": np.array([oos(B, tg["V"], tg["sy"]) for B in ecoefs]),
            "g_supp": np.array([int(np.sum(np.linalg.norm(B, axis=1) > 0)) for B in coefs]),
            "e_supp": np.array([int(np.sum(np.linalg.norm(B, axis=1) > 0)) for B in ecoefs]),
            "tight": {},
        }

    # ---------------- selection and metrics ----------------
    rows, norms_out, landscape = [], {}, {}
    true_cols = np.flatnonzero(struct.true_columns)

    def add_row(variant, tname, meth, rule, B, oos_pair, cv_mse, extra):
        row_norms = np.linalg.norm(B, axis=1)
        row = {"target": variant, "fit_target": tname, "method": meth, "rule": rule,
               "cv_mse": float(cv_mse), "oos_rel_mse": float(oos_pair[0]),
               "oos_span_err": float(oos_pair[1]),
               "subspace_r2": mt.subspace_r2(X[:, row_norms > 0], sample.span_factors)}
        row.update(extra)
        row.update(mt.support_metrics(B, struct))
        rows.append(row)
        norms_out[(variant, meth, rule)] = row_norms.astype(np.float32)

    chosen = {}
    for variant, tname in variants.items():
        fu = full[tname]
        g_mean = cv[variant]["g_mse"].mean(axis=0)
        g_std = cv[variant]["g_mse"].std(axis=0, ddof=1)
        g_hit = (cv[variant]["g_iter"] >= st.CV_MAX_ITER).any(axis=0)
        e_mean = cv[variant]["e_mse"].mean(axis=0)
        e_std = cv[variant]["e_mse"].std(axis=0, ddof=1)
        landscape[variant] = {
            "g_mean": g_mean, "g_std": g_std, "g_hit_any": g_hit,
            "g_fold_mse": cv[variant]["g_mse"], "g_full_iter": fu["iters"],
            "g_support": fu["g_supp"], "g_oos_rel": fu["g_oos"][:, 0],
            "g_oos_span": fu["g_oos"][:, 1],
            "e_mean": e_mean, "e_std": e_std, "e_support": fu["e_supp"],
            "e_oos_rel": fu["e_oos"][:, 0], "e_oos_span": fu["e_oos"][:, 1],
            "val_target_var": cv[variant]["val_var"],
        }

        for rule, i in _select(g_mean, g_std, g_hit, fu["g_supp"], fu["g_oos"][:, 0],
                               n_folds).items():
            chosen[(variant, rule)] = i
            l1, l2, rs = cells[i]
            extra = {"lambda_1": l1, "lambda_2": l2, "ramp_size": rs,
                     "hit_max_iter_any_fold": bool(g_hit[i]), "full_n_iter": int(fu["iters"][i])}
            if tight_check and rule in ("cv_min", "cv_min_clean", "oracle"):
                if i not in fu["tight"]:
                    fu["tight"][i] = _tight_check(fu["prob"], cells, i, fu["coefs"][i])
                extra.update(fu["tight"][i])
            add_row(variant, tname, "growl", rule, fu["coefs"][i], fu["g_oos"][i], g_mean[i], extra)

        sel = _select(g_mean[mtl_idx], g_std[mtl_idx], g_hit[mtl_idx], fu["g_supp"][mtl_idx],
                      fu["g_oos"][mtl_idx, 0], n_folds)
        for rule, j in sel.items():
            i = mtl_idx[j]
            add_row(variant, tname, "mtl", rule, fu["coefs"][i], fu["g_oos"][i], g_mean[i],
                    {"lambda_1": cells[i][0], "hit_max_iter_any_fold": bool(g_hit[i]),
                     "full_n_iter": int(fu["iters"][i])})

        no_hit = np.zeros(len(ecells), dtype=bool)
        for rule, i in _select(e_mean, e_std, no_hit, fu["e_supp"], fu["e_oos"][:, 0],
                               n_folds).items():
            if rule == "cv_min_clean":
                continue
            alpha, l1r = ecells[i]
            add_row(variant, tname, "enet", rule, fu["ecoefs"][i], fu["e_oos"][i], e_mean[i],
                    {"alpha": alpha, "l1_ratio": l1r})

    for tname, tg in targets.items():
        Yr, V, sy = tg["Yr"], tg["V"], tg["sy"]
        ols_all = np.linalg.lstsq(Xs, Yr, rcond=None)[0]
        ols_mkt = np.zeros_like(ols_all)
        ols_mkt[0] = np.linalg.lstsq(Xs[:, :1], Yr, rcond=None)[0]
        ols_true = np.zeros_like(ols_all)
        ols_true[true_cols] = np.linalg.lstsq(Xs[:, true_cols], Yr, rcond=None)[0]
        for meth, B in (("ols_all", ols_all), ("ols_mkt", ols_mkt), ("ols_true", ols_true)):
            add_row(tname, tname, meth, "none", B, oos(B, V, sy), np.nan, {})

    # ---------------- first-stage diagnostics ----------------
    Yr_p = targets["pipeline"]["Yr"]
    sv2 = np.sum(Yr_p ** 2, axis=0)
    obs = ~np.isnan(sample.R[:, kept])
    pred, truth = pca.pred, sample.C[:, kept]
    pc = pred - pred.mean(axis=0)
    tc = truth - truth.mean(axis=0)
    corr2 = np.sum(pc * tc, axis=0) ** 2 / (np.sum(pc ** 2, axis=0) * np.sum(tc ** 2, axis=0))
    firm_r2_all, firm_r2_mkt = _firm_r2(sample.R, sample.F)
    diag = {
        "k_full": pca.k,
        "em_iter_full": pca.n_iter,
        "k_folds": fold_k,
        "em_iter_folds": fold_em_iter,
        "n_firms_folds": fold_n_firms,
        "missing_share": float(1.0 - obs.mean()),
        "firm_r2_true_factors": firm_r2_all,
        "firm_r2_true_mkt": firm_r2_mkt,
        "common_component_corr2_mean": float(np.mean(corr2)),
        "cc_Fhat_F": mt.canonical_correlations(pca.fhat, sample.F) if pca.k else np.array([]),
        "cc_Fhat_X": mt.canonical_correlations(pca.fhat, X) if pca.k else np.array([]),
        "Y_sv2_share": sv2 / sv2.sum(),
        "Y_r2_on_mkt": _r2_reduced(Yr_p, Xs[:, :1]),
        "Y_r2_on_all_X": _r2_reduced(Yr_p, Xs),
        "Yoracle_r2_on_mkt": _r2_reduced(targets["oracle"]["Yr"], Xs[:, :1]),
        "Yoracle_r2_on_all_X": _r2_reduced(targets["oracle"]["Yr"], Xs),
        "pop_span_error": pop["pop_span_error"],
        "pop_relevant_norm_share": pop["pop_relevant_norm_share"],
        "t_pca_full": t_pca,
    }
    split_variant = "pipeline_olsproj" if ols_projection else "pipeline"
    diag.update(_split_half_diagnostics(Xs, targets, cells[chosen[(split_variant, "cv_min")]],
                                        cells[chosen[("oracle", "cv_min")]]))
    diag["t_total"] = time.time() - t_start
    return {
        "scenario": scenario.name,
        "rep": rep,
        "rows": rows,
        "diag": diag,
        "landscape": landscape,
        "norms": norms_out,
        "pop_row_norm": pop["pop_row_norm"].astype(np.float32),
        "solver_check": solver_check,
        "factor_names": struct.names,
        "family": struct.family,
        "true_columns": struct.true_columns,
        "cells": cells,
        "enet_cells": ecells,
    }
