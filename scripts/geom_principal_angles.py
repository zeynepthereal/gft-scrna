"""COMPONENT 2 -- covariance structure change: principal (Grassmann) angles.

Does the malignant population's 50-D covariance subspace differ from the normal population's
subspace by more than sampling noise, and if so, are the "new" directions (in malignant variation,
not explained by the normal subspace) consistent across donors?

R1 robustness: every donor is analyzed in BOTH (i) the cohort 50-D PCA and (ii) a normal-only
space (top-2000 HVGs + 50-D PCA fit on that donor's normal cells only; malignant cells projected
with the same transform). See scripts/geom_common.py.

Steps (q=20 for both U_n, U_m):
  1. Principal angles: SVD(U_n^T U_m) -> cos(theta_i). Grassmann distance = sqrt(sum theta_i^2).
  2. Sampling-noise null: n=floor(min(N_n,N_m)/2), 50 reps. Each rep draws non-overlapping halves
     of size n from normal (N1,N2) and from malignant (M1,M2, same n), fits q=20 subspaces on
     each half, and records d(N1,N2), d(M1,M2), d(N1,M1) -- all subspaces built with equal n.
  3. "New directions": the 3 columns of U_m @ Q (Q from SVD(U_n^T U_m)=P S Q^T) with the SMALLEST
     singular value (largest angle to the normal subspace) -- an orthonormal 3-D subspace N_d of
     malignant variation least explained by the normal subspace.
  4. Cross-donor consistency (cohort-PCA space ONLY, the one space with a shared gene/PC basis
     across donors): for each donor pair, SVD(N_d^T N_d') gives 3 principal angles between the two
     3-D subspaces; "average cos^2" = mean of cos^2 of those 3 angles. Averaged over all 15 donor
     pairs. Null: 1000 reps, each donor's N_d replaced by a random 3-D subspace drawn from within
     that donor's U_m, same statistic recomputed.
  5. Gene-level: consensus new direction = first left singular vector of the 50 x 18 matrix formed
     by stacking all 6 donors' N_d (cohort-PCA space only). Mapped to genes via the cohort PCA gene
     loadings (recomputed once via sc.pp.pca(n_comps=50, mask_var=None) on the full cohort, since
     the cached npz does not store loadings). Gene weights written to CSV. Preranked GSEA
     (MSigDB Hallmark + the two requested GO BP terms; unresolved terms are skipped and reported,
     never approximated).

DECISION RULE (B2), stated before the numbers:
  (a) "covariance structure changes beyond sampling noise": in >=4/6 donors, d(N,M) exceeds the
      95th percentile of BOTH the N-N null and the M-M null. Reported as a baseline (likely true).
  (b) "new directions are common across donors" (the ACTUAL test): the observed cross-donor
      consistency score exceeds the 95th percentile of its null.
  (c) enrichment is interpreted ONLY if (b) holds. If not, each donor's new directions are reported
      as idiosyncratic (donor-specific), and enrichment is still run and saved for completeness but
      not interpreted as a finding.

Output: results/geom_principal_angles.txt, results/geom_new_direction_gene_weights.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402
from ricci_exploratory_phase_genes import resolve_gene_sets, run_gsea  # noqa: E402

Q = 20
N_SPLIT_REPS = 50
N_NULL_REPS = 1000
SEED = 0


def principal_angles(U1, U2):
    """cos(theta_i) via SVD(U1^T U2); returns (cos_theta sorted descending, P, Q) with
    U1 @ P and U2 @ Q the aligned bases (Q's columns ordered to match cos_theta)."""
    M = U1.T @ U2
    P, s, Qt = np.linalg.svd(M)
    s = np.clip(s, -1.0, 1.0)
    return s, P, Qt.T


def grassmann_dist(U1, U2):
    cos_t, _, _ = principal_angles(U1, U2)
    theta = np.arccos(cos_t)
    return float(np.sqrt(np.sum(theta ** 2)))


def fit_q(pts, q=Q):
    U, mean, _ = gc.svd_subspace(pts, q)
    return U


def sampling_noise_null(pts_n, pts_m, rng, n_reps=N_SPLIT_REPS, q=Q):
    n = min(len(pts_n), len(pts_m)) // 2
    d_nn, d_mm, d_nm = [], [], []
    for _ in range(n_reps):
        idx_n = rng.permutation(len(pts_n))[: 2 * n]
        idx_m = rng.permutation(len(pts_m))[: 2 * n]
        Un1, Un2 = fit_q(pts_n[idx_n[:n]], q), fit_q(pts_n[idx_n[n:]], q)
        Um1, Um2 = fit_q(pts_m[idx_m[:n]], q), fit_q(pts_m[idx_m[n:]], q)
        d_nn.append(grassmann_dist(Un1, Un2))
        d_mm.append(grassmann_dist(Um1, Um2))
        d_nm.append(grassmann_dist(Un1, Um1))
    return np.array(d_nn), np.array(d_mm), np.array(d_nm), n


def new_directions(U_n, U_m, n_new=3):
    cos_t, P, Q = principal_angles(U_n, U_m)
    order = np.argsort(cos_t)  # ascending cos -> descending angle ("most new" first)
    basis = U_m @ Q  # columns: orthonormal, in span(U_m), ordered to match cos_t
    return basis[:, order[:n_new]], cos_t[order[:n_new]]


def cross_donor_consistency(Nd_dict):
    donors = list(Nd_dict.keys())
    vals = []
    for i in range(len(donors)):
        for j in range(i + 1, len(donors)):
            cos_t, _, _ = principal_angles(Nd_dict[donors[i]], Nd_dict[donors[j]])
            vals.append(np.mean(cos_t ** 2))
    return float(np.mean(vals))


def null_consistency(U_m_dict, rng, n_reps=N_NULL_REPS, n_new=3):
    donors = list(U_m_dict.keys())
    null_vals = np.empty(n_reps)
    for r in range(n_reps):
        rand_subspaces = {}
        for dn in donors:
            Um = U_m_dict[dn]
            G = rng.standard_normal((Um.shape[1], n_new))
            Q_, _ = np.linalg.qr(Um @ G)
            rand_subspaces[dn] = Q_[:, :n_new]
        null_vals[r] = cross_donor_consistency(rand_subspaces)
    return null_vals


def run_space(tag, donor_pts, buf):
    """donor_pts: {donor: (pts_normal, pts_malignant)}. Returns per-donor results dict."""
    buf.log("\n" + "=" * 96)
    buf.log(f"SPACE: {tag}")
    buf.log("=" * 96)
    rng = np.random.default_rng(SEED)
    res = {}
    U_n_dict, U_m_dict, Nd_dict = {}, {}, {}
    for dn in gc.DONORS:
        pts_n, pts_m = donor_pts[dn]
        U_n, U_m = fit_q(pts_n), fit_q(pts_m)
        U_n_dict[dn], U_m_dict[dn] = U_n, U_m
        d_obs = grassmann_dist(U_n, U_m)
        d_nn, d_mm, d_nm, n_half = sampling_noise_null(pts_n, pts_m, rng)
        p95_nn, p95_mm = np.percentile(d_nn, 95), np.percentile(d_mm, 95)
        cond_a = d_obs > p95_nn and d_obs > p95_mm

        Nd, angles_new = new_directions(U_n, U_m)
        Nd_dict[dn] = Nd
        buf.log(f"\n{dn}: n_normal={len(pts_n)} n_malignant={len(pts_m)} (half-split n={n_half})")
        buf.log(f"  Grassmann d(N,M) observed = {d_obs:.4f}")
        buf.log(f"  null (50 reps, n={n_half}/group): d(N1,N2) median={np.median(d_nn):.4f} "
                f"p95={p95_nn:.4f}  |  d(M1,M2) median={np.median(d_mm):.4f} p95={p95_mm:.4f}  "
                f"|  d(N1,M1) [cross, sanity check] median={np.median(d_nm):.4f}")
        buf.log(f"  (a) d(N,M) > both p95 nulls? {'YES' if cond_a else 'NO'}")
        buf.log(f"  'new direction' angles to normal subspace (degrees): "
                f"{[f'{np.degrees(np.arccos(c)):.1f}' for c in angles_new]}")
        res[dn] = {"d_obs": d_obs, "p95_nn": p95_nn, "p95_mm": p95_mm, "cond_a": cond_a}

    n_pass_a = sum(r["cond_a"] for r in res.values())
    buf.log(f"\n(a) donors passing (d(N,M) beyond both nulls): {n_pass_a}/6 -> "
            f"{'SUPPORTED' if n_pass_a >= 4 else 'NOT SUPPORTED'}")

    consistency_obs = None
    p95_null = None
    if tag == "cohort_pca":
        consistency_obs = cross_donor_consistency(Nd_dict)
        null_vals = null_consistency(U_m_dict, np.random.default_rng(SEED + 1))
        p95_null = np.percentile(null_vals, 95)
        cond_b = consistency_obs > p95_null
        buf.log(f"\nCross-donor consistency of 'new directions' (cohort PCA space only, the shared basis):")
        buf.log(f"  observed mean cos^2 (15 donor pairs) = {consistency_obs:.4f}")
        buf.log(f"  null (1000 reps, random 3D subspaces within each donor's U_m): "
                f"median={np.median(null_vals):.4f}  p95={p95_null:.4f}")
        buf.log(f"  (b) observed > null p95? {'YES' if cond_b else 'NO'} -> "
                f"{'SUPPORTED' if cond_b else 'NOT SUPPORTED'}")
        res["_consistency"] = {"obs": consistency_obs, "p95_null": p95_null, "cond_b": cond_b,
                                "null_median": float(np.median(null_vals))}
        res["_Nd_dict"] = Nd_dict

    return res, n_pass_a


def gene_level(Nd_dict, adata, buf):
    buf.log("\n" + "=" * 96)
    buf.log("GENE-LEVEL MAPPING (cohort PCA space)")
    buf.log("=" * 96)
    stacked = np.hstack([Nd_dict[dn] for dn in gc.DONORS])  # 50 x 18
    U_stack, S_stack, _ = np.linalg.svd(stacked, full_matrices=False)
    consensus_dir = U_stack[:, 0]
    buf.log(f"Consensus new direction = 1st left singular vector of the stacked 50x18 "
            f"new-direction matrix (singular values: {np.round(S_stack, 3).tolist()})")
    buf.log(f"Fraction of stacked variance in the 1st singular vector: "
            f"{S_stack[0]**2 / np.sum(S_stack**2):.3f}")

    buf.log("\nRecomputing cohort PCA loadings via sc.pp.pca(n_comps=50, mask_var=None) "
            "(not cached by extract_pca50.py) to map the consensus direction to genes...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.pca(adata, n_comps=50, mask_var=None)
    loadings = adata.varm["PCs"]  # genes x 50
    gene_weights = loadings @ consensus_dir  # genes-length vector
    gene_names = adata.var["feature_name"].astype(str).values

    df = pd.DataFrame({"gene": gene_names, "weight": gene_weights}).sort_values(
        "weight", ascending=False)
    df = df.drop_duplicates(subset="gene", keep="first")
    csv_path = os.path.join(gc.RESULTS_DIR, "geom_new_direction_gene_weights.csv")
    df.to_csv(csv_path, index=False)
    buf.log(f"Saved gene weights for the consensus new direction -> "
            f"{os.path.basename(csv_path)} ({len(df)} genes)")
    buf.log("Top 15 positive-weight genes: " + ", ".join(df.head(15)["gene"]))
    buf.log("Top 15 negative-weight genes: " + ", ".join(df.tail(15)["gene"][::-1]))

    return df.set_index("gene")["weight"]


def main():
    buf = gc.Buf()
    buf.log("COMPONENT 2 -- covariance structure change: principal (Grassmann) angles")
    buf.log(f"q={Q}; sampling-noise null reps={N_SPLIT_REPS}; cross-donor null reps={N_NULL_REPS}; seed={SEED}\n")
    buf.log("DECISION RULE (B2), stated before the numbers:")
    buf.log("  (a) 'covariance changes beyond sampling noise': in >=4/6 donors, d(N,M) > 95th")
    buf.log("      percentile of BOTH the N-N null and the M-M null. Reported as a baseline.")
    buf.log("  (b) 'new directions common across donors' (the ACTUAL test): cross-donor consistency")
    buf.log("      score > 95th percentile of its null (cohort PCA space only -- the shared basis).")
    buf.log("  (c) enrichment interpreted ONLY if (b) holds; otherwise each donor's new directions")
    buf.log("      are reported as idiosyncratic.\n")

    pcs, ng, donor, disease, ctype = gc.load_cohort_pca()
    adata = gc.load_adata()

    donor_pts_cohort = {}
    for dn in gc.DONORS:
        mask_n, mask_m = gc.donor_masks_cohort(donor, disease, ctype, dn)
        donor_pts_cohort[dn] = (pcs[mask_n], pcs[mask_m])

    donor_pts_normal_only = {}
    for dn in gc.DONORS:
        sp = gc.build_normal_only_space(adata, dn)
        donor_pts_normal_only[dn] = (sp["pts_normal"], sp["pts_malignant"])

    res_cohort, npass_cohort = run_space("cohort_pca", donor_pts_cohort, buf)
    res_normal, npass_normal = run_space("normal_only", donor_pts_normal_only, buf)

    buf.log("\n" + "=" * 96)
    buf.log("MAIN DECISION")
    buf.log("=" * 96)
    buf.log(f"(a) 'covariance changes beyond sampling noise': cohort_pca {npass_cohort}/6, "
            f"normal_only {npass_normal}/6 -> "
            f"{'SUPPORTED' if npass_cohort >= 4 and npass_normal >= 4 else 'NOT SUPPORTED'} in both spaces "
            f"(reported as baseline, not the main test)")
    if npass_cohort != npass_normal:
        buf.log("NOTE: (a) pass counts differ between spaces -- flagged per the R1 instruction.")

    cond_b = res_cohort["_consistency"]["cond_b"]
    buf.log(f"\n(b) 'new directions common across donors' (cohort PCA space, the actual test): "
            f"{'SUPPORTED' if cond_b else 'NOT SUPPORTED'}")
    buf.log(f"    observed={res_cohort['_consistency']['obs']:.4f}  "
            f"null_median={res_cohort['_consistency']['null_median']:.4f}  "
            f"null_p95={res_cohort['_consistency']['p95_null']:.4f}")

    gene_weights = gene_level(res_cohort["_Nd_dict"], adata, buf)

    buf.log("\n" + "=" * 96)
    buf.log("(c) Enrichment (GSEA, preranked by consensus new-direction gene weight)")
    buf.log("=" * 96)
    if cond_b:
        buf.log("(b) is SUPPORTED -- enrichment below is interpreted as a candidate gene program.")
    else:
        buf.log("(b) is NOT SUPPORTED -- new directions are donor-specific/idiosyncratic. Enrichment")
        buf.log("is still computed and saved below for completeness, but is NOT interpreted as a")
        buf.log("finding, per the decision rule.")
    gene_sets = resolve_gene_sets(buf)
    rnk = gene_weights.sort_values(ascending=False)
    run_gsea(rnk, gene_sets, "consensus_new_direction", buf)

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- The cross-donor consistency test (b) is only meaningful in the cohort PCA space, which")
    buf.log("  is the one shared basis across donors; the normal-only space uses a different,")
    buf.log("  donor-specific HVG set and PCA fit for every donor, so 'new direction' vectors from")
    buf.log("  different donors in that space are not directly comparable and are not tested for (b).")
    buf.log("- 'Average cos^2' between two donors' new-direction subspaces is computed as the mean")
    buf.log("  cos^2 of the principal angles BETWEEN THE TWO 3-D SUBSPACES (via SVD(Nd^T Nd')), not as")
    buf.log("  a naive column-by-column cosine, since individual N_d column vectors have no canonical")
    buf.log("  ordering beyond spanning the subspace.")
    buf.log("- Gene weights, the consensus direction, and any enrichment are derived from the same")
    buf.log("  expression matrix used to build the PCA subspaces; this is an internal-consistency")
    buf.log("  analysis, not independent validation.")
    buf.log("- HTA8_6002 (60 malignant, 264 normal cells) gives the noisiest per-donor subspace fits")
    buf.log("  of the six, most visibly in the sampling-noise null (smallest half-split n).")

    buf.save(os.path.join(gc.RESULTS_DIR, "geom_principal_angles.txt"))


if __name__ == "__main__":
    main()
