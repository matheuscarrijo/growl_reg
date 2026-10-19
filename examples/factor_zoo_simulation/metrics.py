"""
Recovery metrics for one estimated mapping.

A mapping is given in reduced form: C (K x k) with B_hat = C V'. Row norms,
supports and cosines between rows of C equal those of B_hat.
"""

from __future__ import annotations

import numpy as np

TIE_RTOL = 1e-9


def canonical_correlations(A, B):
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def subspace_r2(X_sel, F):
    """tr(P_X P_F) / J: share of the true factor space spanned by X_sel."""
    if X_sel.shape[1] == 0:
        return 0.0
    cc = canonical_correlations(X_sel, F)
    return float(np.sum(cc ** 2) / F.shape[1])


class OOSEvaluator:
    """
    Out-of-sample errors of an implied common component on fresh months.

    The prediction for firm i is c_hat_i = s_i * Xs_test C v_i (s_i is the
    scale of the standardized target, v_i the i-th row of V) and the truth is
    c_i = F_test beta_i. Both are of low rank, so the sums over months are
    computed from small cross-product matrices.

    rel_mse  : mean over firms of Var(c_hat - c) / Var(c)
    span_err : mean over firms of 1 - corr(c_hat, c)^2 (scale-free)
    """

    def __init__(self, Xs_test, F_test, betas):
        self.Xc = Xs_test - Xs_test.mean(axis=0)
        self.G = F_test - F_test.mean(axis=0)
        self.betas = betas
        self.tt = np.einsum("ij,jk,ik->i", betas, self.G.T @ self.G, betas)

    def __call__(self, C, V, y_scale):
        P = self.Xc @ C
        pp = np.einsum("ir,rs,is->i", V, P.T @ P, V) * y_scale ** 2
        pt = np.einsum("ir,rj,ij->i", V, P.T @ self.G, self.betas) * y_scale
        err = (pp - 2.0 * pt + self.tt) / self.tt
        corr2 = np.where(pp > 0, pt ** 2 / np.where(pp > 0, pp, 1.0) / self.tt, 0.0)
        return float(np.mean(err)), float(np.mean(1.0 - corr2))


def _tie_groups(norms):
    """Groups of selected rows with equal norms (GrOWL pooled blocks)."""
    idx = np.flatnonzero(norms > 0)
    if idx.size < 2:
        return []
    order = idx[np.argsort(-norms[idx])]
    groups, current = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if abs(norms[a] - norms[b]) <= TIE_RTOL * norms[a]:
            current.append(b)
        else:
            if len(current) > 1:
                groups.append(current)
            current = [b]
    if len(current) > 1:
        groups.append(current)
    return groups


def support_metrics(C, struct):
    """Factor-level, family-level and grouping metrics of one mapping."""
    norms = np.linalg.norm(C, axis=1)
    sel = norms > 0
    fam = struct.family
    truth = struct.true_columns
    relevant_fam = struct.relevant_family
    relevant_rows = relevant_fam[fam]
    n_rel_fam = int(relevant_fam.sum())
    n_irr_fam = int((~relevant_fam).sum())

    fams_sel = np.unique(fam[sel])
    fam_tp = int(np.sum(relevant_fam[fams_sel]))
    fam_fp = int(np.sum(~relevant_fam[fams_sel]))
    tp = int(np.sum(sel & truth))
    fp = int(np.sum(sel & ~truth))
    sq = norms ** 2
    total = sq.sum()

    # Grouping inside the relevant style themes (families 1..6).
    cos_list, members_list = [], []
    for f in np.flatnonzero(relevant_fam):
        if f == 0:
            continue
        rows = np.flatnonzero(sel & (fam == f))
        if rows.size == 0:
            continue
        members_list.append(rows.size)
        if rows.size >= 2:
            unit = C[rows] / norms[rows, None]
            g = unit @ unit.T
            cos_list.append(float(np.mean(g[np.triu_indices(rows.size, 1)])))

    groups = _tie_groups(norms)
    tied = sum(len(g) for g in groups)
    pairs_within = pairs_total = 0
    for g in groups:
        g = np.array(g)
        same = fam[g][:, None] == fam[g][None, :]
        iu = np.triu_indices(g.size, 1)
        pairs_within += int(same[iu].sum())
        pairs_total += iu[0].size

    return {
        "n_selected": int(sel.sum()),
        "mkt_selected": bool(sel[0]),
        "market_only": bool(sel[0] and sel.sum() == 1),
        "empty": bool(sel.sum() == 0),
        "tp": tp,
        "fp": fp,
        "tpr": tp / max(int(truth.sum()), 1),
        "fdr": fp / max(int(sel.sum()), 1),
        "exact_support": bool(np.array_equal(sel, truth)),
        "fam_tp": fam_tp,
        "fam_fp": fam_fp,
        "fam_tpr": fam_tp / n_rel_fam,
        "fam_fpr": fam_fp / max(n_irr_fam, 1),
        "fam_exact": bool(fam_tp == n_rel_fam and fam_fp == 0),
        "relevant_norm_share": float(sq[relevant_rows].sum() / total) if total > 0 else np.nan,
        "mkt_norm_share": float(sq[0] / total) if total > 0 else np.nan,
        "members_per_selected_style_family": float(np.mean(members_list)) if members_list else np.nan,
        "within_family_cosine": float(np.mean(cos_list)) if cos_list else np.nan,
        "n_tie_groups": len(groups),
        "n_tied_factors": tied,
        "tie_within_family_share": pairs_within / pairs_total if pairs_total else np.nan,
    }
