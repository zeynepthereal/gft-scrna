"""Follow-up to scripts/heldout_crossdonor_control.py: in the CRC cohort, cross-donor normal
ratios were large (median 2.32x, max 5.58x) and NO malignant/held-out-normal ratio exceeded the
cross-donor max. Tests two explanations: anatomical site, and batch effect.

TEST 1 -- anatomical site
  Each qualifying donor's normal cells come from exactly one tissue (verified: no donor mixes
  tissues among its normal cells). Cross-donor normal ratios (scripts/heldout_crossdonor_control.py
  logic, reused via subspace_svd/residual imported from paper_final_verification.py) are split into
  same-tissue pairs and different-tissue pairs. For each donor with at least one same-tissue
  partner among the qualifying donors, its malignant/held-out-normal ratio's percentile within the
  same-tissue cross-donor distribution is reported (donors with no same-tissue partner -- here,
  the caecum and hepatic-flexure donors, each unique in this cohort -- are reported as N/A).

TEST 2 -- batch correction (Harmony)
  A Harmony-corrected embedding (harmonypy via scanpy.external.pp.harmony_integrate, key=donor_id,
  basis=the same 50-D PCA used throughout this project) is built ONCE over the full cohort (all
  cells, all donors, normal and malignant together). Test 1 (held-out normal + malignant) and
  Test 2 (cross-donor normal) from scripts/heldout_crossdonor_control.py are rerun UNCHANGED
  (run_sample, cross_donor_test, held_out_normal_and_malignant imported, not copied) on this
  Harmony embedding, and reported side by side with the original PCA-space numbers.

DECISION FRAMEWORK, stated before the numbers:
  - If same-tissue cross-donor ratios are small and malignant ratios exceed them: anatomical site
    explains most of the cross-donor variation seen before.
  - If Harmony shrinks cross-donor normal ratios much more than malignant ratios: batch effect
    explains it, and the disease signal is separable from it.
  - If both shrink about equally: the residual cannot separate disease from individual variation
    in this cohort, and neither explanation rescues the earlier CRC finding.

Output: results/crossdonor_site_batch.txt
"""
import os
import sys
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402
from paper_final_verification import subspace_svd, residual as paper_residual  # noqa: E402
from heldout_crossdonor_control import (  # noqa: E402
    MIN_NORMAL_N, K_EIG, SEED, held_out_normal_and_malignant, run_sample, cross_donor_test,
)

warnings.filterwarnings("ignore", message=".*slogdet.*")


def load_crc_with_embeddings(buf):
    adata = sc.read_h5ad(gc.H5AD)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.pca(adata, n_comps=50, mask_var=None)
    buf.log(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes; PCA(50) computed on the full cohort.")

    buf.log("Building Harmony-corrected embedding (key=donor_id, basis=X_pca) over ALL cells "
            "(normal + malignant, all donors)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, ["donor_id"], random_state=0)
    Z = np.asarray(ho.Z_corr)
    adata.obsm["X_pca_harmony"] = Z.T if Z.shape[0] == 50 else Z
    buf.log(f"Harmony embedding shape: {adata.obsm['X_pca_harmony'].shape}\n")

    if adata.raw is not None:
        raw_counts = adata.raw.to_adata()
        raw_counts.var["mt"] = raw_counts.var["feature_name"].astype(str).str.upper().str.startswith("MT-")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.calculate_qc_metrics(raw_counts, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
        adata.obs["n_genes_by_counts"] = raw_counts.obs["n_genes_by_counts"].values
    else:
        adata.obs["n_genes_by_counts"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

    return adata


def qualifying_donors(adata):
    donor = adata.obs["donor_id"].values
    disease = adata.obs["disease"].values
    ctype = adata.obs["cell_type"].values
    tissue = adata.obs["tissue"].values
    donors = {}
    for dn in sorted(pd.unique(donor)):
        mask_n = (donor == dn) & (disease == "normal")
        mask_m = (donor == dn) & (ctype == "malignant cell")
        if int(mask_n.sum()) < MIN_NORMAL_N:
            continue
        tiss = pd.unique(tissue[mask_n])
        donors[dn] = {"mask_n": mask_n, "mask_m": mask_m,
                      "tissue": tiss[0] if len(tiss) == 1 else "mixed:" + ",".join(tiss)}
    return donors


def samples_for_embedding(adata, donors, embed_key, buf):
    pts_all = adata.obsm[embed_key][:, :50]
    ng_all = adata.obs["n_genes_by_counts"].values
    results = {}
    for dn, info in donors.items():
        pts_n, pts_m = pts_all[info["mask_n"]], pts_all[info["mask_m"]]
        ng_n, ng_m = ng_all[info["mask_n"]], ng_all[info["mask_m"]]
        if len(pts_m) == 0:
            buf.log(f"{dn}: 0 malignant cells on this embedding -- skipped.")
            continue
        results[dn] = run_sample(dn, pts_n, pts_m, ng_n, ng_m, buf)
    return results


def test1_site(adata, donors, sample_results_pca, buf):
    buf.log("\n" + "=" * 96)
    buf.log("TEST 1 -- anatomical site (PCA embedding)")
    buf.log("=" * 96)
    buf.log(f"\n{'donor':<12}{'tissue':<28}{'n_normal':<10}")
    for dn, info in donors.items():
        buf.log(f"{dn:<12}{info['tissue']:<28}{int(info['mask_n'].sum()):<10}")

    pts_all = adata.obsm["X_pca"][:, :50]
    names = list(donors.keys())
    same_tissue_ratios, diff_tissue_ratios = [], []
    pair_records = []
    for a in names:
        for b in names:
            if a == b:
                continue
            U, mean = subspace_svd(pts_all[donors[a]["mask_n"]], k_eig=K_EIG)
            r_b_on_a = paper_residual(pts_all[donors[b]["mask_n"]], U, mean)
            ratio = np.median(r_b_on_a) / sample_results_pca[a]["med_heldout"]
            same = donors[a]["tissue"] == donors[b]["tissue"]
            (same_tissue_ratios if same else diff_tissue_ratios).append(ratio)
            pair_records.append((a, b, donors[a]["tissue"], donors[b]["tissue"], same, ratio))

    buf.log(f"\n{'A':<12}{'B':<12}{'tissue_A':<16}{'tissue_B':<16}{'same?':<7}{'ratio'}")
    for a, b, ta, tb, same, ratio in pair_records:
        buf.log(f"{a:<12}{b:<12}{ta:<16}{tb:<16}{('YES' if same else 'no'):<7}{ratio:.3f}x")

    same_arr, diff_arr = np.array(same_tissue_ratios), np.array(diff_tissue_ratios)
    buf.log(f"\nSame-tissue cross-donor ratios (n={len(same_arr)}): "
            + (f"min={same_arr.min():.3f} median={np.median(same_arr):.3f} max={same_arr.max():.3f}"
               if len(same_arr) else "none available"))
    buf.log(f"Different-tissue cross-donor ratios (n={len(diff_arr)}): "
            + (f"min={diff_arr.min():.3f} median={np.median(diff_arr):.3f} max={diff_arr.max():.3f}"
               if len(diff_arr) else "none available"))

    buf.log(f"\n{'donor':<12}{'tissue':<16}{'mal/heldout':<13}{'pct_in_same_tissue_dist':<25}")
    for dn in names:
        mal_ratio = sample_results_pca[dn]["ratio_mal_heldout"]
        same_for_dn = [r for a, b, ta, tb, same, r in pair_records if a == dn and same]
        if len(same_for_dn) == 0:
            pct_str = "N/A (no same-tissue partner)"
        else:
            pct = 100 * np.mean(np.array(same_for_dn) <= mal_ratio)
            pct_str = f"{pct:.1f}th percentile"
        buf.log(f"{dn:<12}{donors[dn]['tissue']:<16}{mal_ratio:<13.3f}{pct_str}")

    return same_arr, diff_arr


def test2_batch(adata, donors, sample_results_pca, cross_pca, buf):
    buf.log("\n" + "=" * 96)
    buf.log("TEST 2 -- batch correction (Harmony)")
    buf.log("=" * 96)
    sample_results_harmony = samples_for_embedding(adata, donors, "X_pca_harmony", buf)
    cross_harmony = cross_donor_test("CRC_epithelial (Harmony)", sample_results_harmony, buf)

    buf.log("\n" + "-" * 96)
    buf.log("SIDE BY SIDE -- PCA vs Harmony")
    buf.log("-" * 96)
    cd_pca = cross_pca["all_ratios"]
    cd_harm = cross_harmony["all_ratios"]
    buf.log(f"\nCross-donor normal ratio distribution:")
    buf.log(f"  PCA:     min={cd_pca.min():.3f}  median={np.median(cd_pca):.3f}  max={cd_pca.max():.3f}")
    buf.log(f"  Harmony: min={cd_harm.min():.3f}  median={np.median(cd_harm):.3f}  max={cd_harm.max():.3f}")
    shrink_crossdonor = 1 - np.median(cd_harm) / np.median(cd_pca)
    buf.log(f"  Cross-donor median shrinkage (Harmony vs PCA): {100*shrink_crossdonor:+.1f}%")

    buf.log(f"\n{'donor':<12}{'mal/heldout PCA':<18}{'mal/heldout Harmony':<20}{'shrinkage'}")
    shrinkages = []
    for dn in donors:
        if dn not in sample_results_pca or dn not in sample_results_harmony:
            continue
        r_pca = sample_results_pca[dn]["ratio_mal_heldout"]
        r_harm = sample_results_harmony[dn]["ratio_mal_heldout"]
        shrink = 1 - r_harm / r_pca
        shrinkages.append(shrink)
        buf.log(f"{dn:<12}{r_pca:<18.3f}{r_harm:<20.3f}{100*shrink:+.1f}%")

    median_mal_shrink = 100 * np.median(shrinkages)
    buf.log(f"\nMedian malignant-ratio shrinkage (Harmony vs PCA): {median_mal_shrink:+.1f}%")
    buf.log(f"Cross-donor-ratio shrinkage (median, Harmony vs PCA): {100*shrink_crossdonor:+.1f}%")

    if shrink_crossdonor > 0 and (median_mal_shrink < 0.5 * (100 * shrink_crossdonor)):
        verdict = ("Harmony shrinks cross-donor ratios substantially more than malignant ratios "
                   "-> consistent with batch explaining cross-donor variation while the disease "
                   "signal remains separable.")
    elif abs(shrink_crossdonor * 100 - median_mal_shrink) < 15:
        verdict = ("Cross-donor and malignant ratios shrink by a similar amount -> the residual "
                   "does not cleanly separate disease from individual/batch variation in this "
                   "cohort even after Harmony correction.")
    else:
        verdict = ("Malignant ratios shrink as much as or more than cross-donor ratios -> Harmony "
                   "removes some of the malignant signal along with the batch effect; no clean "
                   "separation demonstrated.")
    buf.log(f"\nVERDICT (TEST 2): {verdict}")

    return sample_results_harmony, cross_harmony


def main():
    buf = gc.Buf()
    buf.log("Follow-up: anatomical site and batch-effect explanations for the CRC cohort's large "
            "cross-donor normal residual ratios (scripts/heldout_crossdonor_control.py)")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["numpy", "scipy", "scikit-learn", "scanpy", "harmonypy"]))
    buf.log(f"seed={SEED}  k_eig={K_EIG}  min_normal_n={MIN_NORMAL_N}\n")
    buf.log("DECISION FRAMEWORK, stated before the numbers:")
    buf.log("  - Same-tissue cross-donor ratios small AND malignant ratios exceed them -> anatomical")
    buf.log("    site explains most of the earlier cross-donor variation.")
    buf.log("  - Harmony shrinks cross-donor ratios much more than malignant ratios -> batch effect")
    buf.log("    explains it, disease signal separable.")
    buf.log("  - Both shrink about equally -> residual cannot separate disease from individual")
    buf.log("    variation in this cohort.\n")

    adata = load_crc_with_embeddings(buf)
    donors = qualifying_donors(adata)
    buf.log(f"Qualifying donors (>={MIN_NORMAL_N} normal cells): {list(donors.keys())}\n")

    buf.log("=" * 96)
    buf.log("Re-establishing PCA-space baseline (Test 1: held-out; same as "
            "heldout_crossdonor_control.py)")
    buf.log("=" * 96)
    sample_results_pca = samples_for_embedding(adata, donors, "X_pca", buf)
    cross_pca = cross_donor_test("CRC_epithelial (PCA, baseline)", sample_results_pca, buf)

    same_arr, diff_arr = test1_site(adata, donors, sample_results_pca, buf)

    site_supported = (len(same_arr) > 0 and np.median(same_arr) < np.median(cross_pca["all_ratios"])
                       and all(sample_results_pca[dn]["ratio_mal_heldout"] > np.median(same_arr)
                               for dn in donors if donors[dn]["tissue"] == "sigmoid colon"))
    site_label = ("SUPPORTED (same-tissue ratios clearly smaller, malignant ratios still exceed "
                  "them)" if site_supported else "NOT cleanly supported")
    buf.log(f"\nTEST 1 assessment: same-tissue median ({np.median(same_arr):.3f}x) vs "
            f"overall cross-donor median ({np.median(cross_pca['all_ratios']):.3f}x); "
            f"site explanation {site_label}.")

    sample_results_harmony, cross_harmony = test2_batch(adata, donors, sample_results_pca, cross_pca, buf)

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- Among the 7 qualifying donors, only 5 share the same tissue (sigmoid colon); the")
    buf.log("  other 2 (caecum, hepatic flexure) are each unique in this cohort and have no")
    buf.log("  same-tissue partner, so their site percentile is N/A -- the same-tissue distribution")
    buf.log("  is really a same-tissue-among-5-donors comparison, not a fully general one.")
    buf.log("- Harmony integration is run over the FULL cohort (all donors, normal+malignant) so its")
    buf.log("  correction is influenced by all cells, not just the 7 analyzed here.")
    buf.log("- Harmony can remove real biological (including disease-relevant) variation along with")
    buf.log("  batch effects; a shrinking malignant ratio after Harmony does not by itself mean the")
    buf.log("  disease signal was 'batch', only that Harmony's correction affected it too.")
    buf.log("- All numbers here are derived from the same expression matrix as the residual under")
    buf.log("  test -- internal consistency, not independent validation.")
    buf.log("- This does not revisit scripts/heldout_crossdonor_control.py's own results/verdict;")
    buf.log("  it is a follow-up mechanism check only.")

    buf.save(os.path.join(gc.RESULTS_DIR, "crossdonor_site_batch.txt"))


if __name__ == "__main__":
    main()
