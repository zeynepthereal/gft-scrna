"""REPLICATION (not a blind validation) -- the frozen residual pipeline, on GSE161277.

*** GSE161277 Patients 1-3 were used during the exploratory development of the residual idea
*** (scripts/cnv_clone_vs_residual.py, malignant_only_stemness.py, confound_and_detector.py, etc.
*** were all built and iterated on using this dataset). This run is therefore a REPLICATION of a
*** now-FROZEN pipeline on a second dataset, NOT a blind independent validation. Any agreement
*** should be read with that in mind.

FROZEN PIPELINE (from results/geom_thinning_diagnostic.txt, D1, validated 5/6 donors on the CRC
epithelial cohort -- parameters are NOT retuned here):
  binomial thinning to a common depth (target = 10th percentile of included cells' total counts)
  -> unlabeled top-3000 HVG (flavor='seurat', on thinned + normalize_total(1e4) + log1p data)
  -> 50-D PCA fit on ALL included cells of the pair (no labels)
  -> normal subspace q=20 -> 5-fold held-out residual for normal cells, full-fit residual for the
     other group -> Test A (n_genes quintile stratification + OLS residual ~ is_malignant +
     log n_genes, imported unmodified from scripts/confound_and_detector.py).

IMPORTANT DATA NOTE: unlike crc_epithelial.h5ad, data/GSE161277_all.h5ad's adata.raw holds
LOG-NORMALIZED expression (see scripts/rebuild_gse161277_all.py: `adata.raw = adata` was set right
after normalize_total+log1p), not raw counts. Genuine raw UMI counts are instead read directly
from the original 10x matrices in data/GSE161277/ via rebuild_gse161277_all.load_sample (imported,
not copied), exactly as scripts/malignant_only_stemness.py already does for the same reason.

Patient pairs (from scripts/all_samples_residual.py):
  Patient1: normal GSM4904237, carcinoma GSM4904236
  Patient2: normal GSM4904240, carcinoma GSM4904239
  Patient3: normal GSM4904246, carcinoma GSM4904245
Patient0 has no matched normal sample and is excluded.
Cells: obs['cell_type'] == 'Epithelial/Tumor' only, from data/GSE161277_all.h5ad (read backed).

C1 (PRIMARY, conservative): normal = epithelial cells from the normal GSM; carcinoma = ALL
  epithelial cells from the carcinoma GSM (dilutes with non-malignant epithelium by construction).
C2 (SECONDARY): malignant = CNV-confirmed malignant-like cells (label_malignant_clones, imported
  unmodified from scripts/malignant_only_stemness.py, on top of the CNV pipeline imported from
  scripts/cnv_clone_vs_residual.py). Patient3 was previously excluded from the CNV pipeline
  ("known myeloid contamination") but is attempted here anyway, per instructions, and skipped with
  the specific reason if it fails or yields <50 malignant-like cells.

DECISION RULE (C1, 10th-percentile thinning), stated before the numbers:
  REPLICATED: Test A rule (median quintile ratio > 1.2x AND OLS is_malignant coef > 0, p < 0.05)
    met in >=2/3 patients, AND residual median(carcinoma) > median(normal) in 3/3 patients.
  STRONGLY REPLICATED: Test A rule met in 3/3 patients.
  NOT REPLICATED: otherwise.
  C2 uses the SAME rule over the patients where C2 could be run, reported separately.
Also reported per patient: whether residual's AUROC exceeds n_genes' AUROC (a known Patient1
weakness in earlier full-gene-space work).

Output: results/replicate_gse161277.txt
"""
import os
import sys
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402
from confound_and_detector import test_a_sample  # noqa: E402
from geom_dnb import thin_to_depth  # noqa: E402
from ricci_curvature_pilot import fit_subspace, residual  # noqa: E402
from rebuild_gse161277_all import load_sample  # noqa: E402
from cnv_clone_vs_residual import (  # noqa: E402
    REPO, PATIENTS as GSE_PATIENTS, setup_positions, build_patient_adata, run_patient,
)
from malignant_only_stemness import label_malignant_clones  # noqa: E402

warnings.filterwarnings("ignore", message=".*slogdet.*")

SEED = 0
N_HVG = 3000
Q_PCA = 50
Q_SUB = 20  # matches ricci_curvature_pilot's default K_EIG
MIN_QUINTILE_N = 20
MIN_C2_MALIGNANT = 50

H5AD = REPO / "data" / "GSE161277_all.h5ad"

PATIENT_GSMS = {
    "Patient1": {"normal": ("GSM4904237", "Patient1_normal"), "carcinoma": ("GSM4904236", "Patient1_carcinoma")},
    "Patient2": {"normal": ("GSM4904240", "Patient2_normal"), "carcinoma": ("GSM4904239", "Patient2_carcinoma")},
    "Patient3": {"normal": ("GSM4904246", "Patient3_normal"), "carcinoma": ("GSM4904245", "Patient3_carcinoma")},
}


def build_raw_group(gsm_id, label, epithelial_obs_names):
    """Genuine raw counts (from the original 10x matrix, not adata.raw) for one GSM, restricted
    to this GSM's epithelial, QC-passed barcodes."""
    ad_full = load_sample(gsm_id, label)
    keep = ad_full.obs_names.isin(epithelial_obs_names)
    return ad_full[keep].copy()


def frozen_pipeline(ad_n_raw, ad_m_raw, thin_pct, buf, tag, do_thin=True, rng_seed=SEED):
    """The frozen pipeline: [thinning ->] normalize_total+log1p -> unlabeled 3000-HVG -> 50D PCA
    (label-blind) -> q=20 normal subspace, 5-fold held-out residual. Returns r_n, r_m, ng_n, ng_m,
    pts_n/pts_m (50D, for the Euclidean-centroid baseline)."""
    rng = np.random.default_rng(rng_seed)
    var = ad_n_raw.var.copy()

    if do_thin:
        combined = sp.vstack([ad_n_raw.X.tocsr(), ad_m_raw.X.tocsr()]).tocsr()
        total = np.asarray(combined.sum(axis=1)).ravel()
        target = np.percentile(total, thin_pct)
        keep_local = total >= target
        keep_idx = np.where(keep_local)[0]
        thinned = thin_to_depth(combined, keep_idx, target, rng)
        is_n = np.r_[np.ones(ad_n_raw.n_obs, bool), np.zeros(ad_m_raw.n_obs, bool)][keep_idx]
        is_m = ~is_n
        ng = np.asarray((thinned > 0).sum(axis=1)).ravel()
        buf.log(f"    [{tag}] thinning target (pct={thin_pct})={target:.0f}; kept {len(keep_idx)}/"
                f"{combined.shape[0]} cells (n_normal={is_n.sum()}, n_group2={is_m.sum()})")
        X_counts = thinned
    else:
        X_counts = sp.vstack([ad_n_raw.X.tocsr(), ad_m_raw.X.tocsr()]).tocsr()
        is_n = np.r_[np.ones(ad_n_raw.n_obs, bool), np.zeros(ad_m_raw.n_obs, bool)]
        is_m = ~is_n
        ng = np.asarray((X_counts > 0).sum(axis=1)).ravel()
        buf.log(f"    [{tag}] no thinning (reference only): n_normal={is_n.sum()}, n_group2={is_m.sum()}")

    ad = sc.AnnData(X=X_counts, var=var)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=N_HVG, flavor="seurat")
    hvg = ad.var["highly_variable"].values
    X_all = np.asarray(ad[:, hvg].X.todense())

    V50, mean50, _ = gc.svd_subspace(X_all, Q_PCA)
    pts_all = (X_all - mean50) @ V50

    pts_n, pts_m = pts_all[is_n], pts_all[is_m]
    n = len(pts_n)
    folds = rng.permutation(n) % 5
    r_n = np.empty(n)
    for f in range(5):
        Uf, meanf = fit_subspace(pts_n[folds != f])
        r_n[folds == f] = residual(pts_n[folds == f], Uf, meanf)
    U_all, mean_all = fit_subspace(pts_n)
    r_m = residual(pts_m, U_all, mean_all)

    return r_n, r_m, ng[is_n], ng[is_m], pts_n, pts_m


def auroc_report(r_n, r_m, ng_n, ng_m, pts_n, pts_m, buf, tag):
    centroid = pts_n.mean(axis=0)
    dist_n = np.linalg.norm(pts_n - centroid, axis=1)
    dist_m = np.linalg.norm(pts_m - centroid, axis=1)
    y = np.r_[np.zeros(len(r_n)), np.ones(len(r_m))]
    aurocs = {
        "residual": roc_auc_score(y, np.r_[r_n, r_m]),
        "n_genes": roc_auc_score(y, np.r_[ng_n, ng_m]),
        "euclid_to_normal_centroid": roc_auc_score(y, np.r_[dist_n, dist_m]),
    }
    buf.log(f"    [{tag}] AUROC: residual={aurocs['residual']:.3f}  n_genes={aurocs['n_genes']:.3f}  "
            f"euclid_centroid={aurocs['euclid_to_normal_centroid']:.3f}  "
            f"-> residual > n_genes? {'YES' if aurocs['residual'] > aurocs['n_genes'] else 'NO'}")
    return aurocs


def run_comparison(patient, ad_n_raw, ad_m_raw, buf, tag):
    buf.log(f"\n  --- {tag}: PRIMARY (10th-percentile thinning) ---")
    r_n, r_m, ng_n, ng_m, pts_n, pts_m = frozen_pipeline(ad_n_raw, ad_m_raw, 10, buf, tag)
    buf.log(f"    n_normal={len(r_n)}  n_group2={len(r_m)}")
    buf.log(f"    median held-out residual: normal={np.median(r_n):.3f}  group2={np.median(r_m):.3f}  "
            f"ratio={np.median(r_m)/np.median(r_n):.2f}x")
    testA = test_a_sample(f"{patient}_{tag}_10pct", r_n, ng_n, r_m, ng_m, buf)
    aurocs = auroc_report(r_n, r_m, ng_n, ng_m, pts_n, pts_m, buf, tag)

    buf.log(f"\n  --- {tag}: reference only, NO thinning ---")
    r_n2, r_m2, ng_n2, ng_m2, pts_n2, pts_m2 = frozen_pipeline(ad_n_raw, ad_m_raw, 10, buf,
                                                                tag + "_nothin", do_thin=False)
    testA_nothin = test_a_sample(f"{patient}_{tag}_nothin", r_n2, ng_n2, r_m2, ng_m2, buf)

    buf.log(f"\n  --- {tag}: sensitivity, 25th-percentile thinning (reference only) ---")
    r_n3, r_m3, ng_n3, ng_m3, pts_n3, pts_m3 = frozen_pipeline(ad_n_raw, ad_m_raw, 25, buf, tag + "_25pct")
    testA_25 = test_a_sample(f"{patient}_{tag}_25pct", r_n3, ng_n3, r_m3, ng_m3, buf)

    return {"testA": testA, "aurocs": aurocs, "testA_nothin": testA_nothin, "testA_25": testA_25,
            "med_n": np.median(r_n), "med_m": np.median(r_m)}


def try_c2_malignant_cells(patient, samples, raw_pos, obs_all, raw_ref, pca_full, buf):
    adata_p = build_patient_adata(raw_pos, obs_all, patient, samples)
    adata_p.obsm["X_pca_full"] = pca_full.loc[adata_p.obs_names].values
    pipe_buf_lines = []

    class _QuietBuf:
        def log(self, *a):
            pipe_buf_lines.append(" ".join(str(x) for x in a))

    pipeline = run_patient(patient, adata_p, raw_ref, _QuietBuf())
    if pipeline is None:
        buf.log(f"  C2 SKIPPED for {patient}: no CNV structure detected by run_patient "
                f"(possibly MSI, or -- for Patient3 -- known myeloid contamination).")
        return None

    carc, adata_full = pipeline["carc"], pipeline["adata"]
    m_norm, m_carc = pipeline["m_norm"], pipeline["m_carc"]
    labels = label_malignant_clones(carc, adata_full, pipeline["gv"], pipeline["var"], m_norm, m_carc, _QuietBuf())
    malignant_clones = [cl for cl, lab in labels.items() if lab == "malignant-like"]
    mal_mask = carc.obs["cnv_clone"].isin(malignant_clones).values
    n_mal = int(mal_mask.sum())
    buf.log(f"  C2 CNV labeling for {patient}: malignant-like clones {malignant_clones}, "
            f"n_malignant-like={n_mal}/{len(carc)}")
    if n_mal < MIN_C2_MALIGNANT:
        buf.log(f"  C2 SKIPPED for {patient}: only {n_mal} malignant-like cells (<{MIN_C2_MALIGNANT}).")
        return None
    return carc.obs_names[mal_mask]


def main():
    buf = gc.Buf()
    buf.log("REPLICATION (not blind validation) of the frozen residual pipeline on GSE161277")
    buf.log("*** GSE161277 Patients 1-3 were used during exploratory development of the residual")
    buf.log("*** idea. This is a REPLICATION on a second dataset, not a blind independent test. ***\n")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["numpy", "scipy", "scikit-learn", "scanpy", "infercnvpy"]))
    buf.log(f"Frozen params: thinning target=10th pct (25th pct sensitivity), N_HVG={N_HVG}, "
            f"Q_PCA={Q_PCA}, Q_SUB={Q_SUB}, seed={SEED}. Not retuned after seeing results.\n")
    buf.log("DECISION RULE (C1), stated before the numbers:")
    buf.log("  REPLICATED: Test A rule (median quintile ratio>1.2x AND OLS coef>0,p<0.05) in >=2/3")
    buf.log("  patients, AND residual median(carcinoma)>median(normal) in 3/3 patients.")
    buf.log("  STRONGLY REPLICATED: Test A rule in 3/3 patients.  NOT REPLICATED: otherwise.")
    buf.log("  C2 uses the same rule over patients where C2 could be run, reported separately.")
    buf.log("Patient0 excluded (no matched normal sample).\n")

    full = sc.read_h5ad(H5AD, backed="r")
    obs = full.obs
    epi = obs["cell_type"] == "Epithelial/Tumor"

    c1_results = {}
    c2_malignant_names = {}
    for patient, gsms in PATIENT_GSMS.items():
        buf.log("=" * 96)
        buf.log(patient)
        buf.log("=" * 96)
        (n_gsm, n_label), (m_gsm, m_label) = gsms["normal"], gsms["carcinoma"]
        epi_n_names = obs.index[(obs["sample"] == n_label) & epi]
        epi_m_names = obs.index[(obs["sample"] == m_label) & epi]
        buf.log(f"  epithelial cells (pre-thinning): normal({n_label})={len(epi_n_names)}  "
                f"carcinoma({m_label})={len(epi_m_names)}")

        ad_n_raw = build_raw_group(n_gsm, n_label, epi_n_names)
        ad_m_raw = build_raw_group(m_gsm, m_label, epi_m_names)
        buf.log(f"  matched to raw 10x counts: normal={ad_n_raw.n_obs}  carcinoma={ad_m_raw.n_obs}")

        c1_results[patient] = run_comparison(patient, ad_n_raw, ad_m_raw, buf, "C1")
        c2_malignant_names[patient] = (ad_n_raw, ad_m_raw, epi_m_names)

    buf.log("\n" + "=" * 96)
    buf.log("C2 -- CNV-confirmed malignant-like cells")
    buf.log("=" * 96)
    buf.log("(loading GSE161277_all.h5ad fully in-memory for this section -- the CNV pipeline's")
    buf.log(" own raw.to_adata() needs actual in-memory data, not a backed dataset; C1 above used")
    buf.log(" backed mode throughout since it only touched obs and per-GSM raw 10x files)")
    setup_buf = gc.Buf()
    full_mem = sc.read_h5ad(H5AD)
    raw_all_full = full_mem.raw.to_adata()
    raw_all_full.obs = full_mem.obs.copy()
    raw_pos = setup_positions(raw_all_full, setup_buf)
    buf.log(setup_buf.text().rstrip())

    symbol_to_id = dict(zip(raw_all_full.var["gene_symbol"], raw_all_full.var_names))
    raw_ref = {"symbol_to_id": symbol_to_id, "var_names": raw_all_full.var_names, "X": raw_all_full.X,
               "row_idx": pd.Series(np.arange(raw_all_full.n_obs), index=raw_all_full.obs_names)}
    pca_full = pd.DataFrame(full_mem.obsm["X_pca"], index=full_mem.obs_names)

    c2_results = {}
    for patient, gsms in PATIENT_GSMS.items():
        samples = {"normal_sample": gsms["normal"][1], "carcinoma_sample": gsms["carcinoma"][1]}
        mal_names = try_c2_malignant_cells(patient, samples, raw_pos, obs, raw_ref, pca_full, buf)
        if mal_names is None:
            continue
        ad_n_raw, ad_m_raw, epi_m_names = c2_malignant_names[patient]
        ad_mal_raw = ad_m_raw[ad_m_raw.obs_names.isin(mal_names)].copy()
        buf.log(f"  {patient}: raw-count malignant-like subset = {ad_mal_raw.n_obs} cells "
                f"(of {ad_m_raw.n_obs} carcinoma-sample epithelial)")
        c2_results[patient] = run_comparison(patient, ad_n_raw, ad_mal_raw, buf, "C2")

    buf.log("\n" + "=" * 96)
    buf.log("SUMMARY -- C1")
    buf.log("=" * 96)
    hdr = f"{'patient':<10}{'med_ratio':>10}{'quint_ratio':>13}{'OLS_coef':>10}{'p':>11}{'survives?':>10}{'res_AUROC>ng?':>15}"
    buf.log(hdr)
    n_survive_c1, n_ratio_gt1_c1 = 0, 0
    for p, r in c1_results.items():
        t = r["testA"]
        if t["survives"]:
            n_survive_c1 += 1
        if r["med_m"] > r["med_n"]:
            n_ratio_gt1_c1 += 1
        gt = r["aurocs"]["residual"] > r["aurocs"]["n_genes"]
        buf.log(f"{p:<10}{r['med_m']/r['med_n']:>10.2f}{(t['median_quintile_ratio'] or float('nan')):>13.2f}"
                f"{t['ols_coef']:>10.4f}{t['ols_p']:>11.3e}{('YES' if t['survives'] else 'NO'):>10}"
                f"{('YES' if gt else 'NO'):>15}")
    buf.log(f"\nTest A rule met: {n_survive_c1}/3 patients. Residual median(carcinoma)>normal: "
            f"{n_ratio_gt1_c1}/3 patients.")
    if n_survive_c1 == 3:
        verdict_c1 = "STRONGLY REPLICATED"
    elif n_survive_c1 >= 2 and n_ratio_gt1_c1 == 3:
        verdict_c1 = "REPLICATED"
    else:
        verdict_c1 = "NOT REPLICATED"
    buf.log(f"C1 VERDICT: {verdict_c1}")

    buf.log("\n" + "=" * 96)
    buf.log("SUMMARY -- C2")
    buf.log("=" * 96)
    if not c2_results:
        buf.log("C2 could not be run for any patient (see skip reasons above).")
        buf.log("C2 VERDICT: NOT REPLICATED (no patients available)")
    else:
        buf.log(hdr)
        n_survive_c2, n_ratio_gt1_c2 = 0, 0
        for p, r in c2_results.items():
            t = r["testA"]
            if t["survives"]:
                n_survive_c2 += 1
            if r["med_m"] > r["med_n"]:
                n_ratio_gt1_c2 += 1
            gt = r["aurocs"]["residual"] > r["aurocs"]["n_genes"]
            buf.log(f"{p:<10}{r['med_m']/r['med_n']:>10.2f}{(t['median_quintile_ratio'] or float('nan')):>13.2f}"
                    f"{t['ols_coef']:>10.4f}{t['ols_p']:>11.3e}{('YES' if t['survives'] else 'NO'):>10}"
                    f"{('YES' if gt else 'NO'):>15}")
        n_patients_c2 = len(c2_results)
        buf.log(f"\nTest A rule met: {n_survive_c2}/{n_patients_c2} C2-eligible patients. "
                f"Residual median(malignant)>normal: {n_ratio_gt1_c2}/{n_patients_c2}.")
        if n_patients_c2 == 3 and n_survive_c2 == 3:
            verdict_c2 = "STRONGLY REPLICATED"
        elif n_survive_c2 / max(n_patients_c2, 1) >= 2 / 3 and n_ratio_gt1_c2 == n_patients_c2:
            verdict_c2 = "REPLICATED"
        else:
            verdict_c2 = "NOT REPLICATED"
        buf.log(f"C2 VERDICT (over {n_patients_c2}/3 eligible patients): {verdict_c2}")

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- NOT a blind validation: GSE161277 Patients 1-3 were used to build and iterate on the")
    buf.log("  residual pipeline itself (see top of this file).")
    buf.log("- Only 3 patients; Patient3 in particular has few epithelial cells and known myeloid")
    buf.log("  contamination noted in earlier CNV work.")
    buf.log("- C2's CNV clone labels and the residual are both derived from the same expression")
    buf.log("  matrix -- internal consistency, not an independent ground truth.")
    buf.log("- C1's carcinoma group contains non-malignant epithelium by construction (it dilutes")
    buf.log("  the malignant-vs-normal contrast toward the null, hence 'conservative').")

    buf.save(os.path.join(gc.RESULTS_DIR, "replicate_gse161277.txt"))


if __name__ == "__main__":
    main()
