"""COMPONENT 1 -- residual as (half) a PPCA (Tipping & Bishop) negative log-likelihood.

Fit PPCA (q latent dims) to normal cells via the closed form: sigma^2 = mean of the discarded
eigenvalues, C = W W^T + sigma^2 I with W W^T = U_q diag(lambda_i - sigma^2) U_q^T. Then
  E(x) = -log p(x) = const + 0.5*M_in(x) + 0.5*r(x)^2/sigma^2
  M_in(x) = sum_i (U_i^T(x-mean))^2 / lambda_i   (in-subspace Mahalanobis distance)
  r(x)    = the existing projection residual ||x - UU^T x||
This asks whether malignant cells' extra "energy" (negative log-likelihood under the normal
tissue's PPCA density) is driven mainly by the residual term (out-of-subspace) or the in-subspace
Mahalanobis term (an extreme-but-typical-shape point within the normal manifold).

R1 robustness: every donor is analyzed in BOTH (i) the cohort 50-D PCA and (ii) a normal-only
space (top-2000 HVGs + 50-D PCA fit on that donor's normal cells only; malignant cells projected
with the same transform). See scripts/geom_common.py.

Normal cells' own E/r/M_in are computed 5-fold held-out (same scheme as
scripts/ricci_curvature_pilot.py's residuals_heldout) so normal cells don't get an in-sample
advantage; malignant cells are scored against the PPCA fit on ALL of that donor's normal cells.

DECISION RULE (B1), stated before the numbers:
"residual is the dominant component of the normal-tissue potential" is SUPPORTED if, in a
MAJORITY of donors (>=4/6) AND IN BOTH SPACES:
  - the residual term explains >50% of the malignant-vs-normal median delta-E
    (share = delta_E_resid / (delta_E_resid + delta_E_Min)), AND
  - AUROC(E) >= AUROC(r) - 0.02.
Otherwise: the potential is driven mainly by M_in (extreme points within the normal subspace),
reported as such. q=20 is the primary/decision value; q=10,30 are swept for robustness (1c-style).

Output: results/geom_ppca_potential.txt
"""
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", message=".*slogdet.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402

Q_LIST = [10, 20, 30]
Q_PRIMARY = 20
SEED = 0


def ppca_energy(pts, U, mean, eigvals_top, sigma2, d):
    """E(x) = const + 0.5*M_in + 0.5*r^2/sigma2, computed both directly (via C^-1) and via the
    decomposition, for the verification step."""
    Xc = pts - mean
    proj_coef = Xc @ U  # (n, q)
    M_in = np.sum(proj_coef ** 2 / eigvals_top[np.newaxis, :], axis=1)
    resid_vec = Xc - proj_coef @ U.T
    r = np.linalg.norm(resid_vec, axis=1)
    q = len(eigvals_top)
    logdetC = np.sum(np.log(eigvals_top)) + (d - q) * np.log(sigma2)
    const = 0.5 * (d * np.log(2 * np.pi) + logdetC)
    E_decomp = const + 0.5 * M_in + 0.5 * (r ** 2) / sigma2

    # direct verification via literal C^-1 (only computed once per donor/q/space to check the algebra)
    W = U * np.sqrt(np.maximum(eigvals_top - sigma2, 0))[np.newaxis, :]
    C = W @ W.T + sigma2 * np.eye(d)
    sign, logdetC_direct = np.linalg.slogdet(C)
    Cinv = np.linalg.inv(C)
    quad = np.einsum("ij,jk,ik->i", Xc, Cinv, Xc)
    E_direct = 0.5 * (d * np.log(2 * np.pi) + logdetC_direct + quad)
    max_abs_diff = float(np.max(np.abs(E_direct - E_decomp)))

    return E_decomp, r, M_in, max_abs_diff


def fit_and_score(pts_n, pts_m, ngc_n, ngc_m, q, rng, buf, tag):
    d = pts_n.shape[1]
    n = len(pts_n)
    folds = rng.permutation(n) % 5

    E_n = np.empty(n); r_n = np.empty(n); Min_n = np.empty(n)
    max_diffs = []
    for f in range(5):
        train = pts_n[folds != f]
        U, mean, eig_full = gc.svd_subspace(train, q)
        eig_top, sigma2 = eig_full[:q], eig_full[q:].mean()
        Ef, rf, Mf, mdiff = ppca_energy(pts_n[folds == f], U, mean, eig_top, sigma2, d)
        E_n[folds == f], r_n[folds == f], Min_n[folds == f] = Ef, rf, Mf
        max_diffs.append(mdiff)

    U_all, mean_all, eig_full_all = gc.svd_subspace(pts_n, q)
    eig_top_all, sigma2_all = eig_full_all[:q], eig_full_all[q:].mean()
    E_m, r_m, Min_m, mdiff_m = ppca_energy(pts_m, U_all, mean_all, eig_top_all, sigma2_all, d)
    max_diffs.append(mdiff_m)

    buf.log(f"  [{tag}, q={q}] PPCA decomposition check: max|E_direct - E_decomp| = "
            f"{max(max_diffs):.2e} (should be ~0)")

    dE_total = np.median(E_m) - np.median(E_n)
    dE_r = 0.5 * np.median(r_m ** 2 / sigma2_all) - 0.5 * np.median(r_n ** 2 / sigma2_all)
    dE_Min = 0.5 * np.median(Min_m) - 0.5 * np.median(Min_n)
    share_r = dE_r / (dE_r + dE_Min) if (dE_r + dE_Min) != 0 else np.nan
    buf.log(f"  [{tag}, q={q}] median E: normal={np.median(E_n):.3f} malignant={np.median(E_m):.3f} "
            f"delta_E={dE_total:+.3f}")
    buf.log(f"  [{tag}, q={q}] delta_E decomposition: from_residual={dE_r:+.3f} "
            f"from_M_in={dE_Min:+.3f}  share_residual={share_r:.3f}  "
            f"(sum={dE_r+dE_Min:+.3f} vs total {dE_total:+.3f}, small median-nonlinearity gap expected)")

    y = np.r_[np.zeros(len(E_n)), np.ones(len(E_m))]
    scores = {"E": np.r_[E_n, E_m], "r": np.r_[r_n, r_m], "M_in": np.r_[Min_n, Min_m],
              "n_genes": np.r_[ngc_n, ngc_m]}
    aurocs = {name: roc_auc_score(y, s) for name, s in scores.items()}
    buf.log(f"  [{tag}, q={q}] AUROC: E={aurocs['E']:.3f}  r={aurocs['r']:.3f}  "
            f"M_in={aurocs['M_in']:.3f}  n_genes={aurocs['n_genes']:.3f}")

    return {"share_r": share_r, "auroc_E": aurocs["E"], "auroc_r": aurocs["r"],
            "auroc_Min": aurocs["M_in"], "auroc_ngenes": aurocs["n_genes"], "dE": dE_total}


def main():
    buf = gc.Buf()
    buf.log("COMPONENT 1 -- residual as PPCA potential (Tipping & Bishop closed form)")
    buf.log(f"q values: {Q_LIST} (primary/decision q={Q_PRIMARY}); normal cells 5-fold held-out; seed={SEED}\n")
    buf.log("DECISION RULE (B1), stated before the numbers: 'residual is the dominant component' is")
    buf.log("SUPPORTED if, in >=4/6 donors AND IN BOTH SPACES (cohort PCA, normal-only): residual term")
    buf.log("explains >50% of the malignant-vs-normal median delta-E, AND AUROC(E) >= AUROC(r) - 0.02.")
    buf.log("Otherwise the potential is driven mainly by M_in (extremeness within the normal subspace).\n")

    pcs, ng, donor, disease, ctype = gc.load_cohort_pca()
    adata = gc.load_adata()

    results = {sp: {dn: {} for dn in gc.DONORS} for sp in gc.SPACES}
    for dn in gc.DONORS:
        buf.log("=" * 96)
        buf.log(dn)
        buf.log("=" * 96)
        rng = np.random.default_rng(SEED)

        mask_n_c, mask_m_c = gc.donor_masks_cohort(donor, disease, ctype, dn)
        pts_n_c, pts_m_c = pcs[mask_n_c], pcs[mask_m_c]
        ngc_n_c, ngc_m_c = ng[mask_n_c], ng[mask_m_c]

        sp_norm = gc.build_normal_only_space(adata, dn)
        pts_n_no, pts_m_no = sp_norm["pts_normal"], sp_norm["pts_malignant"]
        mask_n_f, mask_m_f = gc.donor_masks_full(adata, dn)
        ngc_n_no = adata.obs["n_genes_by_counts"].values[mask_n_f] if "n_genes_by_counts" in adata.obs \
            else ngc_n_c
        ngc_m_no = adata.obs["n_genes_by_counts"].values[mask_m_f] if "n_genes_by_counts" in adata.obs \
            else ngc_m_c

        buf.log(f"n_normal={mask_n_c.sum()} n_malignant={mask_m_c.sum()}")
        for q in Q_LIST:
            results["cohort_pca"][dn][q] = fit_and_score(pts_n_c, pts_m_c, ngc_n_c, ngc_m_c, q,
                                                          np.random.default_rng(SEED), buf, "cohort_pca")
            results["normal_only"][dn][q] = fit_and_score(pts_n_no, pts_m_no, ngc_n_no, ngc_m_no, q,
                                                           np.random.default_rng(SEED), buf, "normal_only")

    buf.log("\n" + "=" * 96)
    buf.log(f"SUMMARY (q={Q_PRIMARY}, primary)")
    buf.log("=" * 96)
    hdr = f"{'donor':<11}{'space':<14}{'share_resid':>12}{'AUROC_E':>9}{'AUROC_r':>9}{'AUROC_Min':>10}{'AUROC_ng':>9}"
    buf.log(hdr)
    pass_count = {sp: 0 for sp in gc.SPACES}
    for dn in gc.DONORS:
        for sp in gc.SPACES:
            r = results[sp][dn][Q_PRIMARY]
            cond = r["share_r"] > 0.5 and r["auroc_E"] >= r["auroc_r"] - 0.02
            if cond:
                pass_count[sp] += 1
            buf.log(f"{dn:<11}{sp:<14}{r['share_r']:>12.3f}{r['auroc_E']:>9.3f}{r['auroc_r']:>9.3f}"
                    f"{r['auroc_Min']:>10.3f}{r['auroc_ngenes']:>9.3f}  {'PASS' if cond else 'FAIL'}")

    buf.log(f"\nDonors passing B1 criteria: cohort_pca={pass_count['cohort_pca']}/6, "
            f"normal_only={pass_count['normal_only']}/6")
    b1_supported = pass_count["cohort_pca"] >= 4 and pass_count["normal_only"] >= 4
    buf.log(f"VERDICT (B1): {'SUPPORTED' if b1_supported else 'NOT SUPPORTED'} "
            f"(need >=4/6 in BOTH spaces)")
    if not b1_supported:
        buf.log("-> potential is driven substantially by M_in (in-subspace extremeness) in at least "
                "one space, not solely by the out-of-subspace residual.")
    if pass_count["cohort_pca"] != pass_count["normal_only"]:
        buf.log("NOTE: pass counts differ between the two spaces -- the two spaces give a DIFFERENT "
                "answer to B1 and this is flagged explicitly, per the R1 instruction.")

    buf.log("\n" + "=" * 96)
    buf.log(f"q-ROBUSTNESS (all q in {Q_LIST}, both spaces)")
    buf.log("=" * 96)
    hdr2 = f"{'donor':<11}{'space':<14}{'q':>4}{'share_resid':>12}{'AUROC_E':>9}{'AUROC_r':>9}"
    buf.log(hdr2)
    for dn in gc.DONORS:
        for sp in gc.SPACES:
            for q in Q_LIST:
                r = results[sp][dn][q]
                buf.log(f"{dn:<11}{sp:<14}{q:>4}{r['share_r']:>12.3f}{r['auroc_E']:>9.3f}{r['auroc_r']:>9.3f}")

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- E, r, and M_in are all derived from the same normal-cell PPCA fit and the same PCA")
    buf.log("  coordinates as the residual itself; this is an internal decomposition, not an")
    buf.log("  independent measurement.")
    buf.log("- The malignant-vs-normal separation is an easy problem in general (both terms and")
    buf.log("  n_genes alone give high AUROC in most donors); the comparison of AUROCs to each other,")
    buf.log("  not their absolute level, is what the decision rule uses.")
    buf.log("- HTA8_6002 has only 60 malignant cells and 264 normal cells -- its per-fold held-out")
    buf.log("  normal groups (~53 cells/fold) are small, and its estimates are the least stable.")
    buf.log("- The normal-only space's HVG selection and PCA are refit independently for this")
    buf.log("  script; they are not guaranteed to match the exact cohort PCA subspace used elsewhere")
    buf.log("  in the repo even for the same donor, by construction (that is the point of space ii).")

    gc.RESULTS_DIR
    buf.save(os.path.join(gc.RESULTS_DIR, "geom_ppca_potential.txt"))


if __name__ == "__main__":
    main()
