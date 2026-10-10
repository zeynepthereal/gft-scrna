"""TEST A -- does the "malignant residual > normal residual" claim survive controlling for
   n_genes_by_counts (library complexity)?
TEST B -- how well does projection residual alone detect CNV-confirmed malignant-like cells
   (Patient1/Patient2), vs two simple baselines and an n_genes_by_counts confound check?

Reuses the CNV pipeline (run_patient) and malignant-clone labeling rule (label_malignant_clones)
from scripts/cnv_clone_vs_residual.py / scripts/malignant_only_stemness.py, and the CRC-cohort
loading (with the disease=='normal' fix) from scripts/malignant_only_stemness.py.

Output: results/confound_and_detector.txt
"""
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnv_clone_vs_residual import (  # noqa: E402
    REPO, RESULTS, H5AD, PATIENTS, Buf,
    setup_positions, build_patient_adata, run_patient, get_marker_expr,
)
from malignant_only_stemness import label_malignant_clones  # noqa: E402
from paper_final_verification import subspace_svd, residual as paper_residual  # noqa: E402

CRC_PATH = REPO / "data" / "crc_epithelial.h5ad"
CRC_MIN_N = 50
NORMAL_COLON_MARKERS = ["CA1", "CA2", "SLC26A3", "AQP8", "GUCA2A"]
N_QUANTILE_BINS = 5
MIN_PER_QUINTILE = 20


def quintile_ratio_test(res_norm, ng_norm, res_mal, ng_mal, buf, indent="  "):
    """Bin pooled (normal+malignant) cells into n_genes quintiles; within each quintile with
    >=MIN_PER_QUINTILE cells per group, report the malignant/normal residual median ratio."""
    ng_all = np.concatenate([ng_norm, ng_mal])
    is_mal = np.concatenate([np.zeros(len(ng_norm), dtype=bool), np.ones(len(ng_mal), dtype=bool)])
    res_all = np.concatenate([res_norm, res_mal])

    try:
        bins = pd.qcut(ng_all, N_QUANTILE_BINS, labels=False, duplicates="drop")
    except ValueError as e:
        buf.log(f"{indent}Could not form {N_QUANTILE_BINS} quintiles (n_genes too degenerate): {e}")
        return None

    ratios = []
    buf.log(f"{indent}{'quintile':<10}{'n_genes_range':<20}{'n_normal':<10}{'n_malignant':<12}"
            f"{'med_resid_norm':<16}{'med_resid_mal':<16}{'ratio'}")
    for b in sorted(pd.unique(bins)):
        m = bins == b
        m_norm_q = m & ~is_mal
        m_mal_q = m & is_mal
        n_n, n_m = int(m_norm_q.sum()), int(m_mal_q.sum())
        rng = f"[{ng_all[m].min():.0f}, {ng_all[m].max():.0f}]"
        if n_n < MIN_PER_QUINTILE or n_m < MIN_PER_QUINTILE:
            buf.log(f"{indent}{b:<10}{rng:<20}{n_n:<10}{n_m:<12}{'--':<16}{'--':<16}"
                    f"skipped (<{MIN_PER_QUINTILE}/group)")
            continue
        med_n = np.median(res_all[m_norm_q])
        med_m = np.median(res_all[m_mal_q])
        ratio = med_m / med_n if med_n > 0 else np.nan
        ratios.append(ratio)
        buf.log(f"{indent}{b:<10}{rng:<20}{n_n:<10}{n_m:<12}{med_n:<16.3f}{med_m:<16.3f}{ratio:.2f}")

    if not ratios:
        buf.log(f"{indent}No quintile had >={MIN_PER_QUINTILE} cells in both groups.")
        return None
    med_ratio = float(np.median(ratios))
    buf.log(f"{indent}Median of per-quintile ratios: {med_ratio:.2f}x  "
            f"(n_quintiles_used={len(ratios)}/{N_QUANTILE_BINS})")
    return med_ratio


def ols_regression(res_norm, ng_norm, res_mal, ng_mal, buf, indent="  "):
    """residual ~ is_malignant + log(n_genes_by_counts), OLS."""
    ng_all = np.concatenate([ng_norm, ng_mal]).astype(float)
    is_mal = np.concatenate([np.zeros(len(ng_norm)), np.ones(len(ng_mal))])
    res_all = np.concatenate([res_norm, res_mal]).astype(float)
    if np.any(ng_all <= 0):
        buf.log(f"{indent}WARNING: {int((ng_all <= 0).sum())} cells with n_genes<=0 dropped before log().")
        keep = ng_all > 0
        ng_all, is_mal, res_all = ng_all[keep], is_mal[keep], res_all[keep]

    X = sm.add_constant(np.column_stack([is_mal, np.log(ng_all)]))
    model = sm.OLS(res_all, X).fit()
    coef = model.params[1]
    ci_lo, ci_hi = model.conf_int()[1]
    p = model.pvalues[1]
    buf.log(f"{indent}OLS residual ~ is_malignant + log(n_genes_by_counts):")
    buf.log(f"{indent}  is_malignant coefficient = {coef:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  p={p:.3e}")
    buf.log(f"{indent}  log(n_genes) coefficient = {model.params[2]:.4f}  p={model.pvalues[2]:.3e}  "
            f"R^2={model.rsquared:.3f}  n={len(res_all)}")
    return {"coef": coef, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p}


def test_a_sample(sample_name, res_norm, ng_norm, res_mal, ng_mal, buf):
    buf.log(f"\n--- TEST A: {sample_name} ---")
    buf.log(f"  n_normal={len(res_norm)}  n_malignant={len(res_mal)}")
    buf.log(f"  median n_genes_by_counts: normal={np.median(ng_norm):.0f}  malignant={np.median(ng_mal):.0f}")
    raw_ratio = np.median(res_mal) / np.median(res_norm)
    buf.log(f"  raw (unstratified) residual median ratio malignant/normal = {raw_ratio:.2f}x")

    med_q_ratio = quintile_ratio_test(res_norm, ng_norm, res_mal, ng_mal, buf)
    ols = ols_regression(res_norm, ng_norm, res_mal, ng_mal, buf)

    survives = (med_q_ratio is not None and med_q_ratio > 1.2) and (ols["coef"] > 0 and ols["p"] < 0.05)
    buf.log(f"  Survives-per-sample verdict (stratified median ratio>1.2x AND OLS coef>0,p<0.05): "
            f"{'YES' if survives else 'NO'}")
    return {"sample": sample_name, "raw_ratio": raw_ratio, "median_quintile_ratio": med_q_ratio,
            "ols_coef": ols["coef"], "ols_ci_lo": ols["ci_lo"], "ols_ci_hi": ols["ci_hi"],
            "ols_p": ols["p"], "survives": survives}


def test_b_patient(patient, pipeline, raw_ref, full_obs, buf):
    buf.log(f"\n--- TEST B: {patient} (residual as malignant-cell detector) ---")
    carc, adata = pipeline["carc"], pipeline["adata"]
    m_norm, m_carc = pipeline["m_norm"], pipeline["m_carc"]

    labels = label_malignant_clones(carc, adata, pipeline["gv"], pipeline["var"], m_norm, m_carc, Buf())
    malignant_clones = [cl for cl, lab in labels.items() if lab == "malignant-like"]
    y_true = carc.obs["cnv_clone"].isin(malignant_clones).astype(int).values
    buf.log(f"  Ground truth: {int(y_true.sum())}/{len(y_true)} cells malignant-like "
            f"(CNV clones {malignant_clones}), {len(y_true) - int(y_true.sum())} normal-like.")
    buf.log("  Note: CNV clone labels and projection residual both derive from the same expression")
    buf.log("  data (same PCA/HVG pipeline upstream) -- this is a preliminary consistency check, not")
    buf.log("  an independent validation of the residual as a malignant-cell classifier.")

    residual = carc.obs["residual"].values
    pts_all = adata.obsm["X_pca_full"][:, :50]
    pts_norm, pts_carc = pts_all[m_norm], pts_all[m_carc]
    centroid = pts_norm.mean(axis=0)
    dist_to_centroid = np.linalg.norm(pts_carc - centroid, axis=1)

    marker_vals = np.column_stack([get_marker_expr(raw_ref, carc.obs_names, m) for m in NORMAL_COLON_MARKERS])
    marker_mean = np.nanmean(marker_vals, axis=1)
    marker_score_inverted = -marker_mean

    n_genes = full_obs.loc[carc.obs_names, "n_genes"].values

    scores = {
        "projection_residual": residual,
        "euclidean_dist_to_normal_centroid": dist_to_centroid,
        "normal_colon_marker_mean (inverted)": marker_score_inverted,
        "n_genes_by_counts (confound check)": n_genes,
    }
    buf.log(f"\n  {'score':<40}{'AUROC'}")
    aurocs = {}
    for name, s in scores.items():
        auc = roc_auc_score(y_true, s)
        aurocs[name] = auc
        buf.log(f"  {name:<40}{auc:.3f}")
    return aurocs


def load_crc_cohort():
    adata = sc.read_h5ad(CRC_PATH)
    symbol_to_id = None
    for col in ["feature_name", "gene_symbol", "gene"]:
        if col in adata.var.columns:
            cand = dict(zip(adata.var[col].astype(str), adata.var_names))
            if all(g in cand for g in ["ASCL2", "SOX4"]):
                symbol_to_id = cand
                break
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50, mask_var=None)
    if adata.raw is not None:
        raw_counts = adata.raw.to_adata()
        raw_counts.var["mt"] = raw_counts.var["feature_name"].astype(str).str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(raw_counts, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
        adata.obs["n_genes_by_counts"] = raw_counts.obs["n_genes_by_counts"].values
    return adata


def main():
    RESULTS.mkdir(exist_ok=True)
    buf = Buf()
    buf.log("Gene-count confound test (Test A) and residual-as-detector test (Test B)")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["scanpy", "infercnvpy", "numpy", "scipy", "statsmodels", "scikit-learn"]))
    buf.log("\nDecision rule for TEST A (stated before the numbers): the core claim -- malignant")
    buf.log("cells have higher residual than normal cells -- survives controlling for")
    buf.log("n_genes_by_counts if, in MOST samples, (a) the median-of-quintile-ratios stays clearly")
    buf.log("above 1 (>1.2x) after stratifying on n_genes, AND (b) the OLS is_malignant coefficient")
    buf.log("stays positive and significant (p<0.05) after adjusting for log(n_genes_by_counts).")

    full = sc.read_h5ad(H5AD)
    obs_all = full.obs
    raw_all = full.raw.to_adata()
    raw_all.obs = full.obs.copy()
    setup_buf = Buf()
    raw_pos = setup_positions(raw_all, setup_buf)
    buf.log("\n" + setup_buf.text().rstrip())

    symbol_to_id = dict(zip(raw_all.var["gene_symbol"], raw_all.var_names))
    raw_ref = {
        "symbol_to_id": symbol_to_id, "var_names": raw_all.var_names, "X": raw_all.X,
        "row_idx": pd.Series(np.arange(raw_all.n_obs), index=raw_all.obs_names),
    }
    pca_full = pd.DataFrame(full.obsm["X_pca"], index=full.obs_names)

    buf.log("\n" + "=" * 90)
    buf.log("TEST A -- GSE161277 patients")
    buf.log("=" * 90)

    a_rows = []
    pipelines = {}
    for patient, samples in PATIENTS.items():
        adata_p = build_patient_adata(raw_pos, obs_all, patient, samples)
        adata_p.obsm["X_pca_full"] = pca_full.loc[adata_p.obs_names].values
        pipe_buf = Buf()
        print(f"\n=== Re-running CNV pipeline for {patient} ===")
        pipeline = run_patient(patient, adata_p, raw_ref, pipe_buf)
        pipelines[patient] = pipeline

        carc, adata_full = pipeline["carc"], pipeline["adata"]
        m_norm, m_carc = pipeline["m_norm"], pipeline["m_carc"]
        labels = label_malignant_clones(carc, adata_full, pipeline["gv"], pipeline["var"],
                                         m_norm, m_carc, Buf())
        malignant_clones = [cl for cl, lab in labels.items() if lab == "malignant-like"]
        mal_mask = carc.obs["cnv_clone"].isin(malignant_clones).values

        pts_all = adata_full.obsm["X_pca_full"][:, :50]
        pts_norm = pts_all[m_norm]
        U, mean = subspace_svd(pts_norm, k_eig=20)
        res_norm = paper_residual(pts_norm, U, mean)
        res_mal = carc.obs["residual"].values[mal_mask]

        ng_norm = obs_all.loc[adata_full.obs_names[m_norm], "n_genes"].values
        ng_mal = obs_all.loc[carc.obs_names[mal_mask], "n_genes"].values

        a_rows.append(test_a_sample(patient, res_norm, ng_norm, res_mal, ng_mal, buf))

    buf.log("\n" + "=" * 90)
    buf.log("TEST A -- CRC epithelial cohort")
    buf.log("=" * 90)
    if not CRC_PATH.exists():
        buf.log(f"\ndata/crc_epithelial.h5ad not found -- skipping CRC donors for Test A.")
    else:
        crc_adata = load_crc_cohort()
        if "n_genes_by_counts" not in crc_adata.obs.columns:
            buf.log("Could not compute n_genes_by_counts for the CRC cohort -- skipping.")
        else:
            pts_all = crc_adata.obsm["X_pca"][:, :50]
            donor = crc_adata.obs["donor_id"].values
            disease = crc_adata.obs["disease"].values
            ctype = crc_adata.obs["cell_type"].values
            ng_all_crc = crc_adata.obs["n_genes_by_counts"].values

            for d in sorted(crc_adata.obs["donor_id"].unique()):
                mask_n = (donor == d) & (disease == "normal")
                mask_m = (donor == d) & (ctype == "malignant cell")
                n_normal, n_mal = int(mask_n.sum()), int(mask_m.sum())
                if n_normal < CRC_MIN_N or n_mal < CRC_MIN_N:
                    continue
                pts_norm, pts_mal = pts_all[mask_n], pts_all[mask_m]
                U, mean = subspace_svd(pts_norm, k_eig=20)
                res_norm = paper_residual(pts_norm, U, mean)
                res_mal = paper_residual(pts_mal, U, mean)
                ng_norm, ng_mal = ng_all_crc[mask_n], ng_all_crc[mask_m]
                a_rows.append(test_a_sample(d, res_norm, ng_norm, res_mal, ng_mal, buf))

    buf.log("\n" + "=" * 90)
    buf.log("TEST A -- summary across all samples")
    buf.log("=" * 90)
    hdr = f"{'sample':<14}{'raw_ratio':<11}{'med_quintile_ratio':<20}{'OLS_coef':<11}{'CI_lo':<9}{'CI_hi':<9}{'p':<11}{'survives?'}"
    buf.log(hdr)
    buf.log("-" * len(hdr))
    n_survive = 0
    for r in a_rows:
        mqr = r["median_quintile_ratio"]
        mqr_str = f"{mqr:.2f}" if mqr is not None else "n/a"
        buf.log(f"{r['sample']:<14}{r['raw_ratio']:<11.2f}{mqr_str:<20}{r['ols_coef']:<11.4f}"
                f"{r['ols_ci_lo']:<9.3f}{r['ols_ci_hi']:<9.3f}{r['ols_p']:<11.3e}"
                f"{'YES' if r['survives'] else 'NO'}")
        if r["survives"]:
            n_survive += 1
    buf.log(f"\n{n_survive}/{len(a_rows)} samples show the malignant effect surviving n_genes stratification "
            f"(stratified ratio>1.2x AND OLS coef>0 significant).")
    verdict_a = "SURVIVES" if n_survive > len(a_rows) / 2 else "DOES NOT CLEANLY SURVIVE"
    buf.log(f"By the stated decision rule: the core claim {verdict_a} controlling for n_genes_by_counts "
            f"in most samples.")

    buf.log("\n" + "=" * 90)
    buf.log("TEST B -- residual as a malignant-cell detector (Patient1, Patient2)")
    buf.log("=" * 90)
    for patient in PATIENTS:
        test_b_patient(patient, pipelines[patient], raw_ref, obs_all, buf)

    out_path = RESULTS / "confound_and_detector.txt"
    out_path.write_text(buf.text(), encoding="utf-8")
    print(f"\nSaved -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
