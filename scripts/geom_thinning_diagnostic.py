"""DIAGNOSTIC -- why did B1'(b) collapse after binomial thinning in 4/6 donors
(results/geom_r1_fullgene.txt)? Depth artefact, or noise dilution in a 3000-D space?

This does NOT reopen B1' (still NOT SUPPORTED, 2/6). It asks a narrower question about the
MECHANISM of the collapse, using the SAME binomially-thinned counts (target = 10th percentile of
included cells' total counts) as scripts/geom_r1_fullgene.py.

D1 (low-dimensional check): on the thinned counts, select a FRESH label-blind top-3000 HVG set
  (re-selected on the thinned+renormalized data, not reusing geom_r1_fullgene's original G1 list),
  fit an unsupervised 50-D PCA on ALL included cells (label-blind), then a normal-cell q=20
  subspace inside that 50-D space (5-fold held-out for normal residual, full-fit for malignant),
  and rerun Test A (imported, unmodified, from scripts/confound_and_detector.py). If the
  malignant>normal signal survives at 50-D but not at 3000-D on the SAME thinned counts, the
  3000-D collapse is more consistent with noise dilution in a high-dimensional space than with a
  pure depth artefact (which should collapse the signal regardless of dimensionality).

D2 (noise-normalized effect size, 3000-D thinned residual from B1'): reports, within n_genes
  quintiles (same stratification as Test A), (i) mean(r_mal^2 - r_norm^2) / Var(held-out normal
  r^2) [divided by VARIANCE, as specified -- not a standard Cohen's d, which divides by SD; noted
  explicitly to avoid mislabeling] and (ii) Cliff's delta (malignant vs normal r^2).

D3 (noise floor): held-out normal r^2, vs. the Poisson counting-noise contribution EXPECTED from
  the donor's own thinned counts, propagated into log-normalized space by the delta method.
  Reports the noise floor as a share of the observed held-out normal r^2.

Reuses (imported, not copied): thin_to_depth (scripts/geom_dnb.py); test_a_sample
(scripts/confound_and_detector.py); fit_subspace, residual (scripts/ricci_curvature_pilot.py);
select_g1, held_out_and_malignant, Q (scripts/geom_r1_fullgene.py); svd_subspace
(scripts/geom_common.py).

INTERPRETATION RULE, stated before the numbers:
  - D1's Test A rule holding in >=4/6 donors -> "B1'(b)'s collapse is most likely noise dilution
    in the high-dimensional space; the effect survives at low dimension after the same depth
    equalization."
  - D1 holding in <=2/6 donors -> "the residual difference is substantially driven by
    depth/gene-count, and the main claim is seriously weakened."
  - 3/6 -> inconclusive.
  - If D3's noise floor share exceeds 70% for a donor, that is additionally flagged as evidence
    the 3000-D test lacked power for that donor specifically.

Output: results/geom_thinning_diagnostic.txt
"""
import os
import sys
import warnings

import numpy as np
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402
from confound_and_detector import test_a_sample, quintile_ratio_test  # noqa: E402
from geom_dnb import thin_to_depth  # noqa: E402
from geom_r1_fullgene import select_g1, held_out_and_malignant, Q  # noqa: E402
from ricci_curvature_pilot import fit_subspace, residual  # noqa: E402

warnings.filterwarnings("ignore", message=".*slogdet.*")

SEED = 0
N_HVG = 3000
Q_PCA = 50
Q_SUB = Q  # 20, imported from geom_r1_fullgene
MIN_QUINTILE_N = 20


def thin_donor(adata, dn, mask_n, mask_m, rng):
    """Same thinning as geom_r1_fullgene.py's run_test_a_g1: target = 10th percentile of the
    donor's included (normal+malignant) cells' raw total counts."""
    mask_incl = mask_n | mask_m
    raw_X = adata.raw.X.tocsr()
    incl_idx = np.where(mask_incl)[0]
    total_incl = np.asarray(raw_X[incl_idx].sum(axis=1)).ravel()
    target = np.percentile(total_incl, 10)
    keep_local = total_incl >= target
    keep_idx = incl_idx[keep_local]
    thinned = thin_to_depth(raw_X, keep_idx, target, rng)
    is_norm_kept = mask_n[keep_idx]
    is_mal_kept = mask_m[keep_idx]
    ng_thin = np.asarray((thinned > 0).sum(axis=1)).ravel()
    return thinned, is_norm_kept, is_mal_kept, ng_thin, target, keep_idx


def cliffs_delta(x, y):
    """Cliff's delta = P(X>Y) - P(X<Y), computed in O((n+m) log(n+m)) via ranks."""
    x, y = np.asarray(x), np.asarray(y)
    n, m = len(x), len(y)
    all_vals = np.concatenate([x, y])
    order = np.argsort(all_vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_vals) + 1)
    # average ranks for ties
    sorted_vals = all_vals[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rank_x_sum = ranks[:n].sum()
    # Mann-Whitney U for x
    U_x = rank_x_sum - n * (n + 1) / 2
    delta = (2 * U_x) / (n * m) - 1
    return delta


def d1_lowdim(adata, dn, mask_n, mask_m, thinned, is_norm_kept, is_mal_kept, ng_thin, buf):
    buf.log(f"\n  D1 -- low-dimensional (50D) check on the SAME thinned counts")
    ad_thin = sc.AnnData(X=thinned, var=adata.var.copy())
    sc.pp.normalize_total(ad_thin, target_sum=1e4)
    sc.pp.log1p(ad_thin)
    sc.pp.highly_variable_genes(ad_thin, n_top_genes=N_HVG, flavor="seurat")
    hvg_mask = ad_thin.var["highly_variable"].values
    buf.log(f"    fresh thinned-data label-blind HVG: {int(hvg_mask.sum())} genes selected")

    X_all = np.asarray(ad_thin[:, hvg_mask].X.todense())
    V50, mean50, _ = gc.svd_subspace(X_all, Q_PCA)
    pts_all = (X_all - mean50) @ V50  # (n_kept, 50)

    pts_n, pts_m = pts_all[is_norm_kept], pts_all[is_mal_kept]
    rng = np.random.default_rng(SEED)
    n = len(pts_n)
    folds = rng.permutation(n) % 5
    r_n = np.empty(n)
    for f in range(5):
        Uf, meanf = fit_subspace(pts_n[folds != f])
        r_n[folds == f] = residual(pts_n[folds == f], Uf, meanf)
    U_all, mean_all = fit_subspace(pts_n)
    r_m = residual(pts_m, U_all, mean_all)

    ng_n, ng_m = ng_thin[is_norm_kept], ng_thin[is_mal_kept]
    res_d1 = test_a_sample(f"{dn}_D1_thinned_50D", r_n, ng_n, r_m, ng_m, buf)
    return res_d1


def d2_effect_size(res_thin_g1, ng_n, ng_m, buf):
    buf.log(f"\n  D2 -- noise-normalized effect size (3000D thinned residual, same as B1'(b))")
    r_n, r_m = res_thin_g1["r_n"], res_thin_g1["r_m"]
    r2_n, r2_m = r_n ** 2, r_m ** 2
    var_r2_n = np.var(r2_n, ddof=1)
    all_ng = np.concatenate([ng_n, ng_m])
    import pandas as pd
    try:
        bins = pd.qcut(all_ng, 5, labels=False, duplicates="drop")
    except ValueError:
        bins = np.zeros(len(all_ng), dtype=int)
    is_mal = np.concatenate([np.zeros(len(ng_n), dtype=bool), np.ones(len(ng_m), dtype=bool)])
    r2_all = np.concatenate([r2_n, r2_m])

    buf.log(f"    overall: mean(r2_mal-r2_norm)/Var(r2_norm_heldout) = "
            f"{(r2_m.mean() - r2_n.mean()) / max(var_r2_n, 1e-12):.4f}   "
            f"Cliff's delta(mal,norm) = {cliffs_delta(r2_m, r2_n):+.4f}")

    cliffs_hdr = "Cliff's delta"
    buf.log(f"    {'quintile':<10}{'n_norm':<8}{'n_mal':<8}{'noise-norm effect':<20}{cliffs_hdr}")
    results_by_q = []
    for b in sorted(pd.unique(bins)):
        m = bins == b
        m_n, m_m = m & ~is_mal, m & is_mal
        nn, nm = int(m_n.sum()), int(m_m.sum())
        if nn < MIN_QUINTILE_N or nm < MIN_QUINTILE_N:
            buf.log(f"    {b:<10}{nn:<8}{nm:<8}{'skipped (<' + str(MIN_QUINTILE_N) + '/group)'}")
            continue
        var_q = np.var(r2_all[m_n], ddof=1)
        eff = (r2_all[m_m].mean() - r2_all[m_n].mean()) / max(var_q, 1e-12)
        delta = cliffs_delta(r2_all[m_m], r2_all[m_n])
        buf.log(f"    {b:<10}{nn:<8}{nm:<8}{eff:<20.4f}{delta:+.4f}")
        results_by_q.append((int(b), eff, delta))
    return results_by_q


def d3_noise_floor(adata, thinned, is_norm_kept, gidx_g1, target, res_thin_g1, buf):
    buf.log(f"\n  D3 -- noise floor (Poisson counting noise, delta-method, G1 genes, normal cells)")
    thinned_g1 = thinned[:, gidx_g1]
    thinned_g1_norm = np.asarray(thinned_g1[is_norm_kept].todense())
    p_g = thinned_g1_norm.mean(axis=0) / target  # mean thinned proportion per gene
    target_sum = 1e4
    deriv = target_sum / (1 + target_sum * p_g)
    var_g = (deriv ** 2) * (p_g / target)
    expected_noise_r2 = float(np.sum(var_g))
    observed_r2 = float(np.mean(res_thin_g1["r_n"] ** 2))
    share = expected_noise_r2 / max(observed_r2, 1e-12)
    buf.log(f"    sum_g Poisson noise variance (delta-method, log-normalized space) = {expected_noise_r2:.4f}")
    buf.log(f"    observed held-out normal mean(r^2) = {observed_r2:.4f}")
    buf.log(f"    noise floor share = {share:.3f} ({100*share:.1f}%)")
    if share > 1.0:
        buf.log(f"    NOTE: share exceeds 100% -- the delta-method linearization is known to overstate")
        buf.log(f"    noise for very sparsely-expressed genes (p_g near 0, where log1p is strongly")
        buf.log(f"    concave and the linear approximation breaks down); read as 'noise floor at least")
        buf.log(f"    comparable to the observed signal for this donor', not literally >100% real noise.")
    return expected_noise_r2, observed_r2, share


def main():
    buf = gc.Buf()
    buf.log("DIAGNOSTIC -- mechanism of B1'(b)'s post-thinning collapse (depth artefact vs. "
            "high-dimensional noise dilution). Does NOT reopen B1' (still NOT SUPPORTED, 2/6).\n")
    buf.log("INTERPRETATION RULE, stated before the numbers:")
    buf.log("  D1 Test A rule holds in >=4/6 donors -> collapse is most likely noise dilution in the")
    buf.log("  high-dimensional space; the effect survives at low dimension after the same depth")
    buf.log("  equalization.")
    buf.log("  D1 holds in <=2/6 donors -> the residual difference is substantially driven by")
    buf.log("  depth/gene-count; the main claim is seriously weakened.")
    buf.log("  3/6 -> inconclusive.")
    buf.log("  D3 noise-floor share > 70% for a donor is flagged as evidence the 3000-D test lacked")
    buf.log("  power for that donor specifically.\n")

    adata = gc.load_adata()
    if "n_genes_by_counts" not in adata.obs.columns:
        raw_counts = adata.raw.to_adata()
        raw_counts.var["mt"] = raw_counts.var["feature_name"].astype(str).str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(raw_counts, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
        adata.obs["n_genes_by_counts"] = raw_counts.obs["n_genes_by_counts"].values

    d1_results, d3_results = {}, {}
    for dn in gc.DONORS:
        buf.log("=" * 96)
        buf.log(dn)
        buf.log("=" * 96)
        rng = np.random.default_rng(SEED)
        mask_n, mask_m = gc.donor_masks_full(adata, dn)
        mask_incl = mask_n | mask_m

        gidx_g1 = select_g1(adata, mask_incl)
        thinned, is_norm_kept, is_mal_kept, ng_thin, target, keep_idx = thin_donor(
            adata, dn, mask_n, mask_m, rng)
        buf.log(f"  depth target (10th pct) = {target:.0f}; kept {len(keep_idx)}/{mask_incl.sum()} cells")

        d1_res = d1_lowdim(adata, dn, mask_n, mask_m, thinned, is_norm_kept, is_mal_kept, ng_thin, buf)
        d1_results[dn] = d1_res

        Xn_thin_g1 = np.asarray(thinned[is_norm_kept][:, gidx_g1].todense())
        ad_norm_check = sc.AnnData(X=thinned[:, gidx_g1])
        sc.pp.normalize_total(ad_norm_check, target_sum=1e4)
        sc.pp.log1p(ad_norm_check)
        Xn_thin_g1_ln = np.asarray(ad_norm_check[is_norm_kept].X.todense())
        Xm_thin_g1_ln = np.asarray(ad_norm_check[is_mal_kept].X.todense())
        res_thin_g1 = held_out_and_malignant(Xn_thin_g1_ln, Xm_thin_g1_ln, Q_SUB, np.random.default_rng(SEED))

        d2_effect_size(res_thin_g1, ng_thin[is_norm_kept], ng_thin[is_mal_kept], buf)
        _, _, share = d3_noise_floor(adata, thinned, is_norm_kept, gidx_g1, target, res_thin_g1, buf)
        d3_results[dn] = share

    buf.log("\n" + "=" * 96)
    buf.log("SUMMARY")
    buf.log("=" * 96)
    hdr = f"{'donor':<11}{'D1 survives?':>14}{'D3 noise share':>16}"
    buf.log(hdr)
    n_d1_pass = 0
    for dn in gc.DONORS:
        survives = d1_results[dn]["survives"]
        if survives:
            n_d1_pass += 1
        flag = "  <-- >70%, low power" if d3_results[dn] > 0.70 else ""
        buf.log(f"{dn:<11}{('YES' if survives else 'NO'):>14}{100*d3_results[dn]:>15.1f}%{flag}")

    buf.log(f"\nD1 (low-dimensional Test A) survives in {n_d1_pass}/6 donors.")
    if n_d1_pass >= 4:
        verdict = ("B1'(b)'s collapse is most likely noise dilution in the high-dimensional space; "
                   "the effect survives at low dimension after the same depth equalization.")
    elif n_d1_pass <= 2:
        verdict = ("The residual difference is substantially driven by depth/gene-count; the main "
                   "claim is seriously weakened.")
    else:
        verdict = "Inconclusive (3/6)."
    buf.log(f"VERDICT: {verdict}")

    low_power_donors = [dn for dn in gc.DONORS if d3_results[dn] > 0.70]
    if low_power_donors:
        buf.log(f"D3 noise floor exceeds 70% in: {low_power_donors} -- flagged as evidence the 3000-D "
                f"test specifically lacked power for these donors.")

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- D2's 'noise-normalized effect size' divides by VARIANCE, exactly as specified, not by")
    buf.log("  standard deviation -- it is not a standard Cohen's d and should not be read as one.")
    buf.log("- D3's Poisson noise floor is a delta-method approximation assuming every gene's")
    buf.log("  Poisson noise projects fully into the residual (out-of-subspace) term, which")
    buf.log("  overstates the noise floor somewhat since a small fraction of any gene's noise does")
    buf.log("  fall inside the top-20 subspace. It uses each gene's OWN mean thinned proportion, not")
    buf.log("  a shared dispersion model, and ignores any non-Poisson (over-dispersed) variance.")
    buf.log("- All quantities are derived from the same expression matrix as the residual being")
    buf.log("  diagnosed; this is an internal mechanism check, not independent validation.")
    buf.log("- This diagnostic does not change the B1' verdict (NOT SUPPORTED, 2/6, see")
    buf.log("  results/geom_r1_fullgene.txt); it only characterizes why (b) collapsed where it did.")

    buf.save(os.path.join(gc.RESULTS_DIR, "geom_thinning_diagnostic.txt"))


if __name__ == "__main__":
    main()
