"""
Aggregate the Monte Carlo replications into tables and figures.

Reads results/raw/<scenario>/rep_*.pkl and writes to results/:
  replication_metrics.csv.gz     one row per replication x CV variant x method x rule
  first_stage_diagnostics.csv    one row per replication
  summary_by_method.csv          means over replications
  family_selection_frequency.csv share of replications that select each family
  cv_landscape_diagnostics.csv   rank agreement between CV loss and true loss
  solver_checks.csv              package-solver spot checks
  summary_tables.md              the main tables in Markdown
  fig_*.png                      figures

Usage:  python summarize.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration as cal  # noqa: E402
import metrics as mt  # noqa: E402

RAW = HERE / "results" / "raw"
OUT = HERE / "results"

SCENARIO_ORDER = ["D1_sparse", "D2_proxy", "D2_proxy_strong", "D2_proxy_weak",
                  "D2_proxy_balanced", "D0_market_only", "D1_sparse_balanced"]
SCENARIO_TICK = {
    "D1_sparse": "D1\nsparse", "D2_proxy": "D2\nproxy", "D2_proxy_strong": "D2 proxy\nstrong styles",
    "D2_proxy_weak": "D2 proxy\nweak styles", "D2_proxy_balanced": "D2 proxy\nno missing",
    "D0_market_only": "D0\nmarket only", "D1_sparse_balanced": "D1 sparse\nno missing",
}
VARIANT_LABEL = {
    "pipeline": "Paper CV (projection z·Λ/N)",
    "pipeline_olsproj": "CV with OLS projection",
    "oracle": "CV on true common component",
}
# Categorical slots 1-3 of the validated reference palette, light mode.
VARIANT_COLOR = {"pipeline": "#2a78d6", "pipeline_olsproj": "#eb6834", "oracle": "#1baf7a"}
INK, INK_2, GRID, BENCH = "#0b0b0b", "#52514e", "#e4e3df", "#8d8c88"
FAMILIES = ["mkt"] + cal.THEMES

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
})


def load(scenario):
    files = sorted((RAW / scenario).glob("rep_*.pkl"))
    out = []
    for f in files:
        with f.open("rb") as fh:
            out.append(pickle.load(fh))
    return out


def _tie_purity(norms, fam):
    """Same-family share of the pairs inside tie groups, and its chance level."""
    groups = mt._tie_groups(norms.astype(np.float32))
    within = total = 0
    for g in groups:
        f = fam[np.array(g)]
        same = f[:, None] == f[None, :]
        iu = np.triu_indices(len(g), 1)
        within += int(same[iu].sum())
        total += iu[0].size
    sel = np.flatnonzero(norms > 0)
    f = fam[sel]
    iu = np.triu_indices(sel.size, 1)
    base = float(np.mean((f[:, None] == f[None, :])[iu])) if sel.size > 1 else np.nan
    return {"tie_pairs": total, "tie_pairs_same_family": within, "selected_pairs_same_family": base}


def tables(results):
    rows, diags, fam_rows, land_rows, solver_rows = [], [], [], [], []
    for scen, reps in results.items():
        for res in reps:
            for r in res["rows"]:
                rows.append({"scenario": scen, "rep": res["rep"], **r})
            d = {"scenario": scen, "rep": res["rep"]}
            for k, v in res["diag"].items():
                if isinstance(v, (list, np.ndarray)):
                    v = np.asarray(v, dtype=float)
                    d[k] = " ".join(f"{x:.4g}" for x in v)
                    if k in ("k_folds", "em_iter_folds"):
                        d[k + "_mean"] = float(np.mean(v))
                    if k in ("Y_sv2_share",) and v.size:
                        d["Y_sv2_share_1"] = float(v[0])
                    if k in ("cc_Fhat_X", "cc_Fhat_F") and v.size:
                        d[k + "_min_top7"] = float(np.min(v[:7]))
                else:
                    d[k] = v
            diags.append(d)
            fam = np.asarray(res["family"])
            for (variant, meth, rule), norms in res["norms"].items():
                norms = np.asarray(norms, dtype=float)
                sel = norms > 0
                rec = {"scenario": scen, "rep": res["rep"], "target": variant,
                       "method": meth, "rule": rule}
                for f, name in enumerate(FAMILIES):
                    rec[name] = bool(np.any(sel[fam == f]))
                rec.update(_tie_purity(norms, fam))
                fam_rows.append(rec)
            for variant, L in res["landscape"].items():
                rho, _ = spearmanr(L["g_mean"], L["g_oos_rel"])
                best = int(np.argmin(L["g_oos_rel"]))
                chosen = int(np.argmin(L["g_mean"]))
                land_rows.append({
                    "scenario": scen, "rep": res["rep"], "target": variant,
                    "spearman_cv_vs_oos": float(rho),
                    "regret_oos_rel_mse": float(L["g_oos_rel"][chosen] - L["g_oos_rel"][best]),
                    "cv_rank_of_oracle_cell": int(np.sum(L["g_mean"] < L["g_mean"][best]) + 1),
                    "n_cells_within_1se": int(np.sum(L["g_mean"] <= L["g_mean"][chosen]
                                                     + L["g_std"][chosen] / np.sqrt(5))),
                    "val_target_var_mean": float(np.mean(L["val_target_var"])),
                    "share_cells_hit_max_iter": float(np.mean(L["g_hit_any"])),
                })
            for c in res["solver_check"]:
                solver_rows.append({"scenario": scen, "rep": res["rep"], **c})
    return (pd.DataFrame(rows), pd.DataFrame(diags), pd.DataFrame(fam_rows),
            pd.DataFrame(land_rows), pd.DataFrame(solver_rows))


AGG = {
    "oos_rel_mse": "mean", "oos_span_err": "mean", "n_selected": "mean",
    "market_only": "mean", "empty": "mean", "mkt_selected": "mean",
    "fam_tpr": "mean", "fam_fpr": "mean", "fam_exact": "mean", "tpr": "mean", "fdr": "mean",
    "exact_support": "mean", "subspace_r2": "mean", "relevant_norm_share": "mean",
    "mkt_norm_share": "mean", "within_family_cosine": "mean",
    "members_per_selected_style_family": "mean", "n_tied_factors": "mean",
    "tie_within_family_share": "mean",
}


def summarize_rows(df):
    cols = {k: v for k, v in AGG.items() if k in df}
    g = df.groupby(["scenario", "target", "method", "rule"], sort=False)
    out = g.agg(cols)
    out["n_reps"] = g.size()
    out["oos_rel_mse_se"] = g["oos_rel_mse"].std() / np.sqrt(out["n_reps"])
    for k in ("lambda_1", "tight_support_jaccard", "hit_max_iter_any_fold"):
        if k in df:
            out[k + ("_median" if k == "lambda_1" else "_mean")] = (
                g[k].median() if k == "lambda_1" else g[k].mean())
    return out.reset_index()


KEY_ROWS = [
    ("pipeline", "growl", "cv_min", "GrOWL, paper CV (min)"),
    ("pipeline", "growl", "cv_min_clean", "GrOWL, paper CV (min, converged)"),
    ("pipeline_olsproj", "growl", "cv_min", "GrOWL, CV with OLS projection"),
    ("pipeline_olsproj", "growl", "cv_1se", "GrOWL, CV with OLS projection, 1-SE"),
    ("pipeline", "growl", "oracle", "GrOWL, best grid cell (infeasible)"),
    ("pipeline", "mtl", "cv_min", "MT-LASSO, paper CV"),
    ("pipeline_olsproj", "mtl", "cv_min", "MT-LASSO, CV with OLS projection"),
    ("pipeline", "enet", "cv_min", "MT-ENet, paper CV"),
    ("pipeline_olsproj", "enet", "cv_min", "MT-ENet, CV with OLS projection"),
    ("pipeline", "ols_mkt", "none", "OLS on mkt only"),
    ("pipeline", "ols_all", "none", "OLS on all 154"),
    ("pipeline", "ols_true", "none", "OLS on true support"),
    ("oracle", "growl", "cv_min", "GrOWL, true target, CV (min)"),
    ("oracle", "mtl", "cv_min", "MT-LASSO, true target, CV (min)"),
    ("oracle", "enet", "cv_min", "MT-ENet, true target, CV (min)"),
]


def markdown_tables(summary, diag, land, solver, fam):
    lines = []
    fmt = lambda x, d=3: "" if pd.isna(x) else f"{x:.{d}f}"  # noqa: E731
    pct = lambda x: "" if pd.isna(x) else f"{100 * x:.0f}%"  # noqa: E731
    for scen in [s for s in SCENARIO_ORDER if s in summary["scenario"].unique()]:
        S = summary[summary["scenario"] == scen].set_index(["target", "method", "rule"])
        n = int(S["n_reps"].max())
        lines += [f"### {scen} ({n} replications)", "",
                  "| Estimator and tuning | OOS rel. MSE | OOS 1-corr² | # selected | mkt only | "
                  "family TPR | family FPR | factor TPR | factor FDR | median λ₁ |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for key in KEY_ROWS:
            if key[:3] not in S.index:
                continue
            r = S.loc[key[:3]]
            lines.append(
                f"| {key[3]} | {fmt(r.oos_rel_mse)} ± {fmt(1.96 * r.oos_rel_mse_se)} | "
                f"{fmt(r.oos_span_err)} | {fmt(r.n_selected, 1)} | {pct(r.market_only)} | "
                f"{fmt(r.fam_tpr, 2)} | {fmt(r.fam_fpr, 2)} | {fmt(r.tpr, 2)} | {fmt(r.fdr, 2)} | "
                f"{fmt(r.get('lambda_1_median', np.nan), 3)} |")
        lines.append("")

    lines += ["### First stage (means over replications)", "",
              "| Scenario | Bai-Ng k (full) | Bai-Ng k (folds) | EM iter. (full) | missing | "
              "firm R² (7 f.) | firm R² (mkt) | sv² share PC1 of Y | R²(Y on mkt) | "
              "R²(Y on all X) | min cancorr(F̂, X) top 7 | corr²(ĉ, c) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    t = cal.EMPIRICAL_TARGETS
    lines.append(f"| data (CRSP / GFD) | {t['bai_ng_factors']} | 3, 4, 5, 7, 7 | 390 | "
                 f"{cal.MISSING_SHARE_EMP:.2f} | {t['mean_firm_r2_7_factors']:.3f} | "
                 f"{t['mean_firm_r2_mkt']:.3f} | {t['Y_sv2_share'][0]:.3f} | {t['Y_r2_on_mkt']:.3f} | "
                 f"{t['Y_r2_on_all_X_in_sample']:.3f} | {min(t['cancorr_Fhat_X']):.3f} | n/a |")
    for scen in [s for s in SCENARIO_ORDER if s in diag["scenario"].unique()]:
        D = diag[diag["scenario"] == scen]
        lines.append(
            f"| {scen} | {D.k_full.mean():.2f} | {D.k_folds_mean.mean():.2f} | "
            f"{D.em_iter_full.mean():.0f} | {D.missing_share.mean():.2f} | "
            f"{D.firm_r2_true_factors.mean():.3f} | {D.firm_r2_true_mkt.mean():.3f} | "
            f"{D.Y_sv2_share_1.mean():.3f} | {D.Y_r2_on_mkt.mean():.3f} | {D.Y_r2_on_all_X.mean():.3f} | "
            f"{D.cc_Fhat_X_min_top7.mean():.3f} | {D.common_component_corr2_mean.mean():.3f} |")
    lines.append("")

    lines += ["### CV landscape: does the CV loss rank cells like the true loss?", "",
              "| Scenario | CV variant | validation-target variance | Spearman(CV MSE, OOS MSE) | "
              "regret of CV choice | CV rank of best cell | cells within 1 SE |",
              "|---|---|---|---|---|---|---|"]
    for scen in [s for s in SCENARIO_ORDER if s in land["scenario"].unique()]:
        for variant in VARIANT_LABEL:
            L = land[(land.scenario == scen) & (land.target == variant)]
            if L.empty:
                continue
            lines.append(f"| {scen} | {VARIANT_LABEL[variant]} | {L.val_target_var_mean.mean():.3f} | "
                         f"{L.spearman_cv_vs_oos.mean():.2f} | {L.regret_oos_rel_mse.mean():.3f} | "
                         f"{L.cv_rank_of_oracle_cell.median():.0f} | {L.n_cells_within_1se.median():.0f} |")
    lines.append("")
    lines += ["### Grouping inside the six style families", "",
              "Members per selected style family, mean cosine between the selected rows of one "
              "family, and number of factors in exact row-norm ties.", "",
              "| Scenario | Tuning | Method | # selected | members / selected family | "
              "within-family cosine | tied factors | tie pairs in same family | "
              "same-family share of all selected pairs (chance) |",
              "|---|---|---|---|---|---|---|---|---|"]
    group_keys = [("pipeline_olsproj", "cv_min", "CV, OLS projection"),
                  ("pipeline", "oracle", "best grid cell"),
                  ("oracle", "cv_min", "true target, CV")]
    for scen in [s for s in SCENARIO_ORDER if s in summary["scenario"].unique()]:
        if scen.startswith("D0"):
            continue
        S = summary[summary["scenario"] == scen].set_index(["target", "method", "rule"])
        for target, rule, label in group_keys:
            for meth in ("growl", "mtl", "enet"):
                if (target, meth, rule) not in S.index:
                    continue
                r = S.loc[(target, meth, rule)]
                F = fam[(fam.scenario == scen) & (fam.target == target) & (fam.method == meth)
                        & (fam.rule == rule)]
                tie_share = (F.tie_pairs_same_family.sum() / F.tie_pairs.sum()
                             if F.tie_pairs.sum() else np.nan)
                lines.append(f"| {scen} | {label} | {meth} | {fmt(r.n_selected, 1)} | "
                             f"{fmt(r.members_per_selected_style_family, 2)} | "
                             f"{fmt(r.within_family_cosine, 2)} | {fmt(r.n_tied_factors, 1)} | "
                             f"{pct(tie_share)} | {pct(F.selected_pairs_same_family.mean())} |")
    lines.append("")

    if "tight_support_jaccard_mean" in summary:
        T = summary[(summary.method == "growl") & summary.tight_support_jaccard_mean.notna()]
        lines += ["### Accuracy of the tol = 1e-4 solutions at the selected cells", "",
                  "Each selected GrOWL cell is solved again with tol = 1e-9.", "",
                  "| Scenario | CV variant | rule | Jaccard(support, tight support) | "
                  "share of cells that hit max_iter in a fold |",
                  "|---|---|---|---|---|"]
        for _, r in T.iterrows():
            lines.append(f"| {r.scenario} | {r.target} | {r.rule} | "
                         f"{fmt(r.tight_support_jaccard_mean, 3)} | "
                         f"{pct(r.get('hit_max_iter_any_fold_mean', np.nan))} |")
        lines.append("")

    if not solver.empty:
        lines += ["### Package-solver spot checks", "",
                  f"{len(solver)} cells re-solved with `growl.fista_solver.growl_fista`: "
                  f"max relative difference {solver.max_rel_diff.max():.1e}; "
                  f"same iteration count in {100 * solver.same_n_iter.mean():.1f}% of cells; "
                  f"same support in {100 * solver.same_support.mean():.1f}% of cells.", ""]
    return "\n".join(lines)


def _save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def fig_oos(summary):
    scen = [s for s in SCENARIO_ORDER if s in summary["scenario"].unique()]
    S = summary.set_index(["scenario", "target", "method", "rule"])
    series = [("pipeline", "growl", "cv_min"), ("pipeline_olsproj", "growl", "cv_min"),
              ("oracle", "growl", "cv_min")]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    width = 0.24
    x = np.arange(len(scen))
    for j, key in enumerate(series):
        vals = [S.loc[(s, *key)].oos_rel_mse if (s, *key) in S.index else np.nan for s in scen]
        err = [1.96 * S.loc[(s, *key)].oos_rel_mse_se if (s, *key) in S.index else 0 for s in scen]
        ax.bar(x + (j - 1) * width, vals, width - 0.02, yerr=err, color=VARIANT_COLOR[key[0]],
               label=f"GrOWL, {VARIANT_LABEL[key[0]]}", error_kw={"ecolor": INK_2, "lw": 1})
    best = [S.loc[(s, "pipeline", "growl", "oracle")].oos_rel_mse
            if (s, "pipeline", "growl", "oracle") in S.index else np.nan for s in scen]
    mkt = [S.loc[(s, "pipeline", "ols_mkt", "none")].oos_rel_mse
           if (s, "pipeline", "ols_mkt", "none") in S.index else np.nan for s in scen]
    ax.scatter(x, best, marker="D", s=36, color=INK, zorder=3,
               label="GrOWL, best grid cell, EM-PCA target (infeasible)")
    ax.scatter(x, mkt, marker="_", s=400, color=BENCH, linewidths=2.5, zorder=3,
               label="OLS on mkt only, EM-PCA target")
    ax.set_xticks(x, [SCENARIO_TICK.get(s, s) for s in scen])
    ax.set_ylabel("Out-of-sample relative MSE\nof the common component")
    ax.set_title("Recovery of the true common component on new months (lower is better)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, "fig_oos_error.png")


def fig_support(rows):
    scen = [s for s in SCENARIO_ORDER if s in rows["scenario"].unique()]
    fig, axes = plt.subplots(1, len(scen), figsize=(2.4 * len(scen), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, scen):
        data, colors = [], []
        for variant in VARIANT_LABEL:
            r = rows[(rows.scenario == s) & (rows.target == variant) & (rows.method == "growl")
                     & (rows.rule == "cv_min")]
            data.append(r.n_selected.to_numpy())
            colors.append(VARIANT_COLOR[variant])
        parts = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                           medianprops={"color": INK, "lw": 1.5},
                           flierprops={"marker": "o", "ms": 3, "mec": INK_2})
        for patch, c in zip(parts["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_edgecolor(INK_2)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks([1, 2, 3], ["paper", "OLS proj.", "true Y"], fontsize=8)
        ax.set_title(s, fontsize=9, color=INK)
    axes[0].set_ylabel("Number of selected factors\n(GrOWL, CV minimum)")
    _save(fig, "fig_support_size.png")


def fig_family(fam):
    scen = [s for s in SCENARIO_ORDER if s in fam["scenario"].unique()]
    keys = [("pipeline", "growl", "cv_min", "paper CV"),
            ("pipeline_olsproj", "growl", "cv_min", "OLS-proj. CV"),
            ("pipeline", "growl", "oracle", "best cell"),
            ("oracle", "growl", "cv_min", "true Y, CV")]
    fig, axes = plt.subplots(1, len(scen), figsize=(2.6 * len(scen), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, scen):
        mat = []
        for t, m, r, _ in keys:
            sub = fam[(fam.scenario == s) & (fam.target == t) & (fam.method == m) & (fam.rule == r)]
            mat.append(sub[FAMILIES].mean().to_numpy() if len(sub) else np.full(len(FAMILIES), np.nan))
        mat = np.array(mat).T
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if mat[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(keys)), [k[3] for k in keys], rotation=45, ha="right", fontsize=7.5)
        ax.set_title(s, fontsize=9, color=INK)
        ax.grid(False)
        ax.axhline(6.5, color=INK, lw=1)
    axes[0].set_yticks(range(len(FAMILIES)), FAMILIES, fontsize=8)
    fig.suptitle("Share of replications that select at least one member of each family "
                 "(rows above the line drive returns)", fontsize=10, x=0.02, ha="left")
    _save(fig, "fig_family_frequency.png")


def fig_lambda(rows):
    scen = [s for s in SCENARIO_ORDER if s in rows["scenario"].unique()]
    grid = np.logspace(-2, 2, 9)
    fig, axes = plt.subplots(1, len(scen), figsize=(2.5 * len(scen), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.27
    for ax, s in zip(axes, scen):
        for j, variant in enumerate(VARIANT_LABEL):
            r = rows[(rows.scenario == s) & (rows.target == variant) & (rows.method == "growl")
                     & (rows.rule == "cv_min")]
            share = [np.mean(np.isclose(r.lambda_1, g)) for g in grid]
            ax.bar(np.arange(9) + (j - 1) * width, share, width - 0.02, color=VARIANT_COLOR[variant],
                   label=VARIANT_LABEL[variant])
        ax.set_xticks(np.arange(9)[::2], [f"{g:g}" for g in grid[::2]], fontsize=8)
        ax.set_xlabel("λ₁ chosen by CV")
        ax.set_title(s, fontsize=9, color=INK)
    axes[0].set_ylabel("Share of replications")
    axes[-1].legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, "fig_selected_lambda.png")


def main():
    results = {s: load(s) for s in SCENARIO_ORDER if (RAW / s).exists()}
    results = {s: r for s, r in results.items() if r}
    if not results:
        sys.exit("No replications found under results/raw/.")
    rows, diag, fam, land, solver = tables(results)
    summary = summarize_rows(rows)
    rows.to_csv(OUT / "replication_metrics.csv.gz", index=False)
    diag.to_csv(OUT / "first_stage_diagnostics.csv", index=False)
    summary.to_csv(OUT / "summary_by_method.csv", index=False)
    fam.groupby(["scenario", "target", "method", "rule"])[FAMILIES + ["selected_pairs_same_family"]].mean().reset_index().to_csv(
        OUT / "family_selection_frequency.csv", index=False)
    land.to_csv(OUT / "cv_landscape_diagnostics.csv", index=False)
    solver.to_csv(OUT / "solver_checks.csv", index=False)
    md = markdown_tables(summary, diag, land, solver, fam)
    (OUT / "summary_tables.md").write_text(md, encoding="utf-8")
    fig_oos(summary)
    fig_support(rows)
    fig_family(fam)
    fig_lambda(rows)
    print(md)


if __name__ == "__main__":
    main()
