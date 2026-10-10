"""CORRECTED R1 -- residual/PPCA in the FULL gene space (no early 50-D truncation).

See results/NOTE_r1_invalid.txt: the previous "normal-only 50-D PCA" control discarded exactly
the out-of-subspace variance the residual measures, biasing it toward finding less separation
regardless of ground truth. This script redoes the PPCA-potential idea directly in gene space
(q=20 top directions from a normal-cell fit, kept as a q-dim basis, but residual/M_in are computed
via norm identities so no d-dimensional projection is ever discarded and no d x d matrix is
formed for the main analysis -- only for a small-subset numerical sanity check).

Gene sets, per donor:
  G1 (PRIMARY): top-3000 HVGs from ALL of that donor's epithelial cells (normal+malignant,
    label-blind), sc.pp.highly_variable_genes(flavor='seurat') on the already log-normalized data.
  G2 (robustness): all genes expressed in >=10% of the donor's cells (normal+malignant).
  G3 (conservative sensitivity, NOT part of the decision rule): top-3000 HVGs from NORMAL cells
    only -- excludes malignant-specific genes by construction and is expected to be null-biased;
    reported for reference only.

Per gene set: mean-centered on normal cells; SVD (scipy.sparse.linalg.svds, top q=20 only -- no
truncation to any fixed low dimension beyond q, the full residual is computed in the full d-D
space via norm identities: r^2 = ||x-mean||^2 - ||proj_coef||^2, M_in = sum(proj_coef^2/eigval)).
Normal cells' own residual is 5-fold held-out; malignant cells are scored against the all-normal
fit. sigma^2 = (trace(cov) - sum_{i<=q} lambda_i) / (d-q), trace(cov) computed directly as the sum
of per-gene variances (O(n*d), never forms a d x d matrix). Decomposition:
  E(x) = const + 0.5*M_in(x) + 0.5*r(x)^2/sigma^2
Numerically verified (see verify_decomposition_small) via scripts/geom_ppca_potential.ppca_energy
(imported, not copied) on a random <=500-gene subset, where forming the full d x d matrix directly
is tractable -- this is the sanity check that the analytic full-gene-space formula above is the
literal same identity, without ever forming a (potentially 10,000 x 10,000) matrix for real.

DELTA E IS MEAN-BASED THIS TIME (mean(E_mal) - mean(E_norm)), which decomposes EXACTLY into the
residual-term and M_in-term contributions (the previous median-based delta_E was not additive,
which is where the earlier -4.25 "share" value came from).

Depth control on G1 (mandatory): raw residual norm grows mechanically with detected gene count
in the full gene space. Test A (quintile stratification + OLS residual ~ is_malignant +
log(n_genes), imported from scripts/confound_and_detector.py, unmodified) is run twice on G1:
  (i)  on the untransformed data,
  (ii) on binomial-thinned counts (target = 10th percentile of included cells' total counts,
       thinning imported from scripts/geom_dnb.py, unmodified) -- normalize_total + log1p after.

Principal angles (B2a repeat), G1 space only: split-half sampling-noise null with equal n, same
procedure as scripts/geom_principal_angles.py (principal_angles, grassmann_dist, sampling_noise_null
imported, not copied).

B2b is NOT retested (closed in the cohort-PCA space at 0.1284 < null p95 0.1287; retesting it in a
different space now would be moving the goalpost after seeing the result).

DECISION RULE (B1'), stated before the numbers: "residual is the dominant component of the normal-
tissue potential in the full gene space, and this is not a depth artefact" is SUPPORTED if, in a
MAJORITY of donors (>=4/6):
  (a) in G1, the residual term's share of mean delta-E > 50%;
  (b) in G1 AND in thinned-G1, Test A's rule holds (stratified median ratio > 1.2x AND OLS
      is_malignant coefficient > 0, p < 0.05);
  (c) in G2, (a) and (b) point the same direction as G1.
G3 is reported for reference only and does not enter this decision (expected to be biased toward
the null by construction -- see the docstring above).

DECISION RULE (B2a'): in G1 space, in >=4/6 donors, d(N,M) > max(N-N null p95, M-M null p95).

Output: results/geom_r1_fullgene.txt
"""
import os
import sys
import warnings
from importlib.metadata import version

import numpy as np
import scanpy as sc
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402
from confound_and_detector import test_a_sample  # noqa: E402
from geom_dnb import thin_to_depth  # noqa: E402
from geom_ppca_potential import ppca_energy  # noqa: E402 -- used only for the small-subset check
from geom_principal_angles import principal_angles, grassmann_dist, sampling_noise_null  # noqa: E402

warnings.filterwarnings("ignore", message=".*slogdet.*")

SEED = 0
Q = 20
GENE_MIN_FRAC = 0.10
N_HVG = 3000
N_SPLIT_REPS = 50
VERIFY_SUBSET_SIZE = 500


def top_q_svd(Xc, q):
    """Top-q right singular vectors/values of centered Xc (n x d), via scipy sparse svds
    (avoids ever forming a d x d matrix). Returns V (d x q, orthonormal gene-space directions,
    descending eigenvalue order) and eigvals_top (q,)."""
    k = min(q, min(Xc.shape) - 1)
    _, S, Vt = svds(Xc, k=k)
    order = np.argsort(-S)
    S, Vt = S[order], Vt[order]
    n = Xc.shape[0]
    eigvals = (S ** 2) / max(n - 1, 1)
    return Vt.T, eigvals  # (d, q), (q,)


def score_points(Xc, V, eigvals_top, sigma2, d):
    proj_coef = Xc @ V  # (n, q)
    M_in = np.sum(proj_coef ** 2 / eigvals_top[np.newaxis, :], axis=1)
    r2 = np.sum(Xc ** 2, axis=1) - np.sum(proj_coef ** 2, axis=1)
    r2 = np.maximum(r2, 0.0)
    q = len(eigvals_top)
    logdetC = np.sum(np.log(eigvals_top)) + (d - q) * np.log(sigma2)
    const = 0.5 * (d * np.log(2 * np.pi) + logdetC)
    E = const + 0.5 * M_in + 0.5 * r2 / sigma2
    return E, np.sqrt(r2), M_in, const


def fit_fullgene(Xn, q=Q):
    mean = Xn.mean(axis=0)
    Xn_c = Xn - mean
    d = Xn_c.shape[1]
    total_var = Xn_c.var(axis=0, ddof=1).sum()
    V, eigvals_top = top_q_svd(Xn_c, q)
    sigma2 = max((total_var - eigvals_top.sum()) / max(d - q, 1), 1e-8)
    return mean, V, eigvals_top, sigma2, d


def verify_decomposition_small(Xn, Xm, rng, buf, tag):
    """Reuses geom_ppca_potential.ppca_energy (which forms the full d' x d' C matrix directly)
    on a random <=500-gene subset, as a numerical sanity check that the analytic full-gene-space
    formula used above is the same identity."""
    d = Xn.shape[1]
    d_sub = min(d, VERIFY_SUBSET_SIZE)
    cols = rng.choice(d, size=d_sub, replace=False)
    Xn_sub, Xm_sub = Xn[:, cols], Xm[:, cols]
    mean, U, eig, sigma2, d_ = fit_fullgene(Xn_sub, q=Q)
    _, _, _, mdiff = ppca_energy(Xn_sub, U, mean, eig, sigma2, d_)
    _, _, _, mdiff_m = ppca_energy(Xm_sub, U, mean, eig, sigma2, d_)
    buf.log(f"  [{tag}] decomposition check on a random {d_sub}-gene subset "
            f"(full-matrix verification via geom_ppca_potential.ppca_energy): "
            f"max|E_direct - E_decomp| = {max(mdiff, mdiff_m):.2e} (should be ~0)")


def held_out_and_malignant(Xn, Xm, q, rng):
    n = len(Xn)
    folds = rng.permutation(n) % 5
    d = Xn.shape[1]
    E_n = np.empty(n); r_n = np.empty(n); M_n = np.empty(n); const_n = np.empty(n)
    sigma2_n = np.empty(n)  # per-cell, since held-out cells use their OWN fold's sigma2
    for f in range(5):
        mean, V, eig, sigma2, _ = fit_fullgene(Xn[folds != f], q)
        Ef, rf, Mf, cf = score_points(Xn[folds == f] - mean, V, eig, sigma2, d)
        E_n[folds == f], r_n[folds == f], M_n[folds == f] = Ef, rf, Mf
        const_n[folds == f] = cf
        sigma2_n[folds == f] = sigma2

    mean_all, V_all, eig_all, sigma2_all, _ = fit_fullgene(Xn, q)
    E_m, r_m, M_m, const_m = score_points(Xm - mean_all, V_all, eig_all, sigma2_all, d)
    sigma2_m = np.full(len(Xm), sigma2_all)
    return {"E_n": E_n, "r_n": r_n, "M_n": M_n, "const_n": const_n, "sigma2_n": sigma2_n,
            "E_m": E_m, "r_m": r_m, "M_m": M_m, "const_m": const_m, "sigma2_m": sigma2_m,
            "sigma2_all": sigma2_all}


def gene_set_analysis(tag, Xn, Xm, rng, buf):
    d = Xn.shape[1]
    buf.log(f"\n[{tag}] n_normal={len(Xn)} n_malignant={len(Xm)} d={d} genes")
    verify_decomposition_small(Xn, Xm, rng, buf, tag)
    res = held_out_and_malignant(Xn, Xm, Q, rng)

    dE_mean = res["E_m"].mean() - res["E_n"].mean()
    dE_r = 0.5 * (res["r_m"] ** 2 / res["sigma2_m"]).mean() - 0.5 * (res["r_n"] ** 2 / res["sigma2_n"]).mean()
    dE_M = 0.5 * res["M_m"].mean() - 0.5 * res["M_n"].mean()
    dE_c = res["const_m"].mean() - res["const_n"].mean()
    share_r = dE_r / (dE_r + dE_M) if (dE_r + dE_M) != 0 else np.nan
    buf.log(f"  mean E: normal={res['E_n'].mean():.3f} malignant={res['E_m'].mean():.3f}  "
            f"delta_E_mean={dE_mean:+.3f}")
    buf.log(f"  delta_E_mean decomposition (EXACT, mean-based): from_residual={dE_r:+.3f}  "
            f"from_M_in={dE_M:+.3f}  from_const={dE_c:+.3f}  "
            f"(sum={dE_r+dE_M+dE_c:+.3f} vs total {dE_mean:+.3f})")
    buf.log(f"  share_residual = from_residual/(from_residual+from_M_in) = {share_r:.3f}")

    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(res["E_n"])), np.ones(len(res["E_m"]))]
    return {"share_r": share_r, "res": res, "E_n": res["E_n"], "E_m": res["E_m"],
            "r_n": res["r_n"], "r_m": res["r_m"], "M_n": res["M_n"], "M_m": res["M_m"]}


def auroc_block(res, ng_n, ng_m, buf, tag):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(res["r_n"])), np.ones(len(res["r_m"]))]
    scores = {"E": np.r_[res["E_n"], res["E_m"]], "r": np.r_[res["r_n"], res["r_m"]],
              "M_in": np.r_[res["M_n"], res["M_m"]], "n_genes": np.r_[ng_n, ng_m]}
    aurocs = {k: roc_auc_score(y, v) for k, v in scores.items()}
    buf.log(f"  [{tag}] AUROC: E={aurocs['E']:.3f}  r={aurocs['r']:.3f}  M_in={aurocs['M_in']:.3f}  "
            f"n_genes={aurocs['n_genes']:.3f}")
    return aurocs


def select_g1(adata, mask_incl):
    ad = adata[mask_incl].copy()
    sc.pp.highly_variable_genes(ad, n_top_genes=N_HVG, flavor="seurat")
    return np.where(ad.var["highly_variable"].values)[0]


def select_g2(adata, mask_incl):
    frac = np.asarray((adata[mask_incl].X > 0).mean(axis=0)).ravel()
    return np.where(frac >= GENE_MIN_FRAC)[0]


def select_g3(adata, mask_norm):
    ad = adata[mask_norm].copy()
    sc.pp.highly_variable_genes(ad, n_top_genes=N_HVG, flavor="seurat")
    return np.where(ad.var["highly_variable"].values)[0]


def dense(adata, mask, gene_idx):
    X = adata[mask][:, gene_idx].X
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)


def run_test_a_g1(adata, dn, mask_n, mask_m, gene_idx_g1, res_g1, buf, rng):
    buf.log(f"\n  Test A on G1 (untransformed):")
    if "n_genes_by_counts" not in adata.obs.columns:
        raw_counts = adata.raw.to_adata()
        raw_counts.var["mt"] = raw_counts.var["feature_name"].astype(str).str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(raw_counts, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
        adata.obs["n_genes_by_counts"] = raw_counts.obs["n_genes_by_counts"].values
    ng_n = adata.obs["n_genes_by_counts"].values[mask_n]
    ng_m = adata.obs["n_genes_by_counts"].values[mask_m]
    testA_raw = test_a_sample(f"{dn}_G1_raw", res_g1["r_n"], ng_n, res_g1["r_m"], ng_m, buf)

    buf.log(f"\n  Test A on G1 (binomial-thinned to common depth):")
    mask_incl = mask_n | mask_m
    raw_X = adata.raw.X.tocsr()
    total_incl = np.asarray(raw_X[mask_incl].sum(axis=1)).ravel()
    target = np.percentile(total_incl, 10)
    incl_idx = np.where(mask_incl)[0]
    total_all = np.asarray(raw_X[incl_idx].sum(axis=1)).ravel()
    keep_local = total_all >= target
    keep_idx = incl_idx[keep_local]
    thinned = thin_to_depth(raw_X, keep_idx, target, rng)
    ad_thin = sc.AnnData(X=thinned, var=adata.var.copy())
    sc.pp.normalize_total(ad_thin, target_sum=1e4)
    sc.pp.log1p(ad_thin)
    ng_thin = np.asarray((thinned > 0).sum(axis=1)).ravel()

    is_norm_kept = mask_n[keep_idx]
    is_mal_kept = mask_m[keep_idx]
    Xn_thin = np.asarray(ad_thin[:, gene_idx_g1].X.todense())[is_norm_kept]
    Xm_thin = np.asarray(ad_thin[:, gene_idx_g1].X.todense())[is_mal_kept]
    res_thin = held_out_and_malignant(Xn_thin, Xm_thin, Q, np.random.default_rng(SEED))
    testA_thin = test_a_sample(f"{dn}_G1_thinned", res_thin["r_n"], ng_thin[is_norm_kept],
                                res_thin["r_m"], ng_thin[is_mal_kept], buf)
    return testA_raw, testA_thin


def run_b2a_g1(Xn, Xm, buf):
    rng = np.random.default_rng(SEED + 2)
    Un = np.linalg.svd(Xn - Xn.mean(0), full_matrices=False)[2][:Q].T
    Um = np.linalg.svd(Xm - Xm.mean(0), full_matrices=False)[2][:Q].T
    d_obs = grassmann_dist(Un, Um)
    d_nn, d_mm, d_nm, n_half = sampling_noise_null(Xn, Xm, rng, n_reps=N_SPLIT_REPS, q=Q)
    p95_nn, p95_mm = np.percentile(d_nn, 95), np.percentile(d_mm, 95)
    cond = d_obs > p95_nn and d_obs > p95_mm
    buf.log(f"  G1 principal angles: d(N,M)={d_obs:.4f}  null p95: N-N={p95_nn:.4f} M-M={p95_mm:.4f}  "
            f"(half-split n={n_half}) -> {'PASS' if cond else 'FAIL'}")
    return cond


def main():
    buf = gc.Buf()
    buf.log("CORRECTED R1 -- residual/PPCA in the full gene space (see results/NOTE_r1_invalid.txt)")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["numpy", "scipy", "scikit-learn", "scanpy"]))
    buf.log(f"q={Q}; seed={SEED}; N_HVG={N_HVG}; split-half reps={N_SPLIT_REPS}\n")
    buf.log("DECISION RULE (B1'), stated before the numbers: SUPPORTED if, in >=4/6 donors:")
    buf.log("  (a) G1: residual term's share of MEAN delta-E > 50%;")
    buf.log("  (b) G1 AND thinned-G1 both pass Test A (stratified median ratio>1.2x AND OLS")
    buf.log("      is_malignant coef>0, p<0.05);")
    buf.log("  (c) G2's (a) and (b) point the same direction as G1.")
    buf.log("  G3 is reference-only (label-blind HVG excluded by construction), not part of the rule.")
    buf.log("DECISION RULE (B2a'): in G1 space, in >=4/6 donors, d(N,M) > max(N-N null p95, M-M null p95).")
    buf.log("B2b is NOT retested (closed in cohort-PCA space at 0.1284 < 0.1287; see NOTE_r1_invalid.txt).\n")

    adata = gc.load_adata()

    donor_summaries = []
    for dn in gc.DONORS:
        buf.log("=" * 96)
        buf.log(dn)
        buf.log("=" * 96)
        rng = np.random.default_rng(SEED)
        mask_n, mask_m = gc.donor_masks_full(adata, dn)
        mask_incl = mask_n | mask_m

        gidx_g1 = select_g1(adata, mask_incl)
        gidx_g2 = select_g2(adata, mask_incl)
        gidx_g3 = select_g3(adata, mask_n)
        buf.log(f"Gene set sizes: G1(all-cell HVG)={len(gidx_g1)}  G2(>=10% expressed)={len(gidx_g2)}  "
                f"G3(normal-only HVG, reference)={len(gidx_g3)}")

        Xn_g1, Xm_g1 = dense(adata, mask_n, gidx_g1), dense(adata, mask_m, gidx_g1)
        Xn_g2, Xm_g2 = dense(adata, mask_n, gidx_g2), dense(adata, mask_m, gidx_g2)
        Xn_g3, Xm_g3 = dense(adata, mask_n, gidx_g3), dense(adata, mask_m, gidx_g3)

        res_g1 = gene_set_analysis("G1", Xn_g1, Xm_g1, np.random.default_rng(SEED), buf)
        res_g2 = gene_set_analysis("G2", Xn_g2, Xm_g2, np.random.default_rng(SEED), buf)
        res_g3 = gene_set_analysis("G3 (reference only)", Xn_g3, Xm_g3, np.random.default_rng(SEED), buf)

        ng_full = adata.obs["n_genes_by_counts"].values if "n_genes_by_counts" in adata.obs.columns else None
        if ng_full is None:
            raw_counts = adata.raw.to_adata()
            raw_counts.var["mt"] = raw_counts.var["feature_name"].astype(str).str.upper().str.startswith("MT-")
            sc.pp.calculate_qc_metrics(raw_counts, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
            adata.obs["n_genes_by_counts"] = raw_counts.obs["n_genes_by_counts"].values
            ng_full = adata.obs["n_genes_by_counts"].values
        auroc_block(res_g1["res"], ng_full[mask_n], ng_full[mask_m], buf, "G1")
        auroc_block(res_g2["res"], ng_full[mask_n], ng_full[mask_m], buf, "G2")
        auroc_block(res_g3["res"], ng_full[mask_n], ng_full[mask_m], buf, "G3 (reference only)")

        testA_raw, testA_thin = run_test_a_g1(adata, dn, mask_n, mask_m, gidx_g1, res_g1["res"], buf, rng)

        buf.log("")
        cond_b2a = run_b2a_g1(Xn_g1, Xm_g1, buf)

        cond_a_g1 = res_g1["share_r"] > 0.5
        cond_b = testA_raw["survives"] and testA_thin["survives"]
        cond_a_g2 = res_g2["share_r"] > 0.5
        # G2's Test A: reuse the SAME thinned/raw residuals is not requested; (c) only checks G2's
        # share_r direction and, separately, whether G2's OWN residual (no thinning re-run required
        # by the instructions) still shows malignant>normal in the same direction as G1's Test A verdict.
        cond_c = (cond_a_g2 == cond_a_g1) and (cond_a_g2 and cond_b)

        buf.log(f"\nDonor decision inputs: (a) G1 share_r={res_g1['share_r']:.3f} (>0.5? "
                f"{'YES' if cond_a_g1 else 'NO'})  (b) Test A raw+thinned survive? "
                f"{'YES' if cond_b else 'NO'}  (c) G2 share_r={res_g2['share_r']:.3f} same "
                f"direction as G1 and (a)+(b) also hold for G2's share? {'YES' if cond_c else 'NO'}")
        buf.log(f"  [reference only] G3 share_r={res_g3['share_r']:.3f} (excludes malignant-specific "
                f"genes by construction -- expected null-biased)")

        donor_summaries.append({
            "donor": dn, "cond_a": cond_a_g1, "cond_b": cond_b, "cond_c": cond_c,
            "share_g1": res_g1["share_r"], "share_g2": res_g2["share_r"], "share_g3": res_g3["share_r"],
            "cond_b2a": cond_b2a,
        })

    buf.log("\n" + "=" * 96)
    buf.log("SUMMARY")
    buf.log("=" * 96)
    hdr = f"{'donor':<11}{'share_G1':>10}{'share_G2':>10}{'share_G3(ref)':>14}{'(a)':>5}{'(b)':>5}{'(c)':>5}{'all3':>6}{'B2a':>6}"
    buf.log(hdr)
    n_pass_all3, n_pass_b2a = 0, 0
    for s in donor_summaries:
        all3 = s["cond_a"] and s["cond_b"] and s["cond_c"]
        if all3:
            n_pass_all3 += 1
        if s["cond_b2a"]:
            n_pass_b2a += 1
        buf.log(f"{s['donor']:<11}{s['share_g1']:>10.3f}{s['share_g2']:>10.3f}{s['share_g3']:>14.3f}"
                f"{('Y' if s['cond_a'] else 'N'):>5}{('Y' if s['cond_b'] else 'N'):>5}"
                f"{('Y' if s['cond_c'] else 'N'):>5}{('YES' if all3 else 'no'):>6}"
                f"{('YES' if s['cond_b2a'] else 'no'):>6}")

    buf.log(f"\nB1' verdict: {n_pass_all3}/6 donors satisfy (a)+(b)+(c) -> "
            f"{'SUPPORTED' if n_pass_all3 >= 4 else 'NOT SUPPORTED'} (need >=4/6)")
    buf.log(f"B2a' verdict: {n_pass_b2a}/6 donors -> "
            f"{'SUPPORTED' if n_pass_b2a >= 4 else 'NOT SUPPORTED'} (need >=4/6)")
    buf.log("B2b: NOT retested (closed; see results/NOTE_r1_invalid.txt).")

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- Label-blind HVG selection (G1) still uses malignant cells' variance to choose which")
    buf.log("  genes enter the analysis. This is NOT label leakage (no normal/malignant label is used")
    buf.log("  to select genes), but the result should not be assumed to generalize to an unseen")
    buf.log("  dataset without independent confirmation.")
    buf.log("- All scores (E, r, M_in) and gene selection are derived from the same expression matrix.")
    buf.log("  This is an internal-consistency test, not independent validation.")
    buf.log("- Residual quantiles/snapshot comparisons here are not a time axis.")
    buf.log("- G3 (normal-only HVG) is reported for reference only, expected to be biased toward the")
    buf.log("  null by construction (it excludes malignant-specific genes), and is not part of the")
    buf.log("  decision rule.")

    buf.save(os.path.join(gc.RESULTS_DIR, "geom_r1_fullgene.txt"))


if __name__ == "__main__":
    main()
