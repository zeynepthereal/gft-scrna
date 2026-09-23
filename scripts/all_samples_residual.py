"""Unified projection-residual analysis across ALL available samples, N>=50 threshold.

Method (identical across every sample):
  1. U_normal (50x20) via SVD of normal epithelial cells' 50D PCA coords.
  2. residual_i = ||x_i - U U^T x_i|| for cancer cells; self-projection for normal (control).
  3. Ratio = median(cancer residual) / median(normal residual).
  4. Mann-Whitney U test, cancer vs normal residual.
  5. Top-20 vs bottom-20 residual cancer cells (by individual cell): ASCL2, SOX4 mean expression.

Note: GSE161277 Patient0 (GSM4904234, carcinoma only) has NO matched normal sample in this
GEO series -- there is no GSM4904234-paired normal biopsy among the 13 deposited samples.
It is therefore excluded from this analysis (no normal reference subspace can be built for it).
"""
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
GSE_PATH = os.path.join(PROJECT_DIR, "data", "GSE161277_all.h5ad")
CRC_PATH = os.path.join(PROJECT_DIR, "data", "crc_epithelial.h5ad")
OUT_PATH = os.path.join(PROJECT_DIR, "results", "all_samples_residual.txt")

K_EIG = 20
MIN_N = 50
MARKERS = ["ASCL2", "SOX4"]

GSE_PATIENTS = {
    "Patient1": {"normal": "GSM4904237", "carcinoma": "GSM4904236"},
    "Patient2": {"normal": "GSM4904240", "carcinoma": "GSM4904239"},
    "Patient3": {"normal": "GSM4904246", "carcinoma": "GSM4904245"},
}


def subspace_svd(pts_normal, k_eig=K_EIG):
    mean = pts_normal.mean(axis=0)
    Xc = pts_normal - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k_eig].T, mean


def residual(pts, U, mean):
    X = pts - mean
    return np.linalg.norm(X - X @ U @ U.T, axis=1)


def top_bottom(expr_dict, res):
    order = np.argsort(res)
    bottom20, top20 = order[:20], order[-20:]
    out = {}
    for g, expr in expr_dict.items():
        out[g] = (expr[top20].mean(), expr[bottom20].mean())
    return out


def run_gse161277(log):
    print("=" * 100, file=log)
    print("GSE161277 -- Patient0 has no matched normal sample in this GEO series (carcinoma-only); excluded.", file=log)
    print("=" * 100, file=log)
    adata = sc.read_h5ad(GSE_PATH, backed="r")
    raw = adata.raw
    pts_all = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values
    cell_type = adata.obs["cell_type"].values
    symbol_to_id = dict(zip(raw.var["gene_symbol"][:], raw.var_names[:]))

    rows = []
    for patient, gsms in GSE_PATIENTS.items():
        mask_normal = (gsm == gsms["normal"]) & (cell_type == "Epithelial/Tumor")
        mask_carcinoma = (gsm == gsms["carcinoma"]) & (cell_type == "Epithelial/Tumor")
        pts_normal = pts_all[mask_normal]
        pts_carcinoma = pts_all[mask_carcinoma]
        n_normal, n_cancer = len(pts_normal), len(pts_carcinoma)

        if n_normal < MIN_N or n_cancer < MIN_N:
            print(f"SKIP {patient}: n_normal={n_normal}, n_cancer={n_cancer} (need >= {MIN_N})", file=log)
            continue

        U, mean = subspace_svd(pts_normal)
        res_cancer = residual(pts_carcinoma, U, mean)
        res_normal_self = residual(pts_normal, U, mean)
        med_n, med_c = np.median(res_normal_self), np.median(res_cancer)
        ratio = med_c / med_n
        u_stat, p_val = mannwhitneyu(res_cancer, res_normal_self, alternative="two-sided")

        carc_idx = np.where(mask_carcinoma)[0]
        expr_dict = {}
        for g in MARKERS:
            gid = symbol_to_id[g]
            col_idx = raw.var_names.get_loc(gid)
            e = raw.X[carc_idx][:, col_idx]
            e = np.asarray(e.todense()).flatten() if hasattr(e, "todense") else np.asarray(e).flatten()
            expr_dict[g] = e
        tb = top_bottom(expr_dict, res_cancer)

        print(f"\n{patient}: n_normal={n_normal}, n_cancer={n_cancer}", file=log)
        print(f"  normal_med={med_n:.4f}  cancer_med={med_c:.4f}  ratio={ratio:.2f}x  p={p_val:.3e}", file=log)
        for g, (hi, lo) in tb.items():
            print(f"  {g}: top20={hi:.4f}  bottom20={lo:.4f}  {'HIGH>low' if hi > lo else 'no'}", file=log)

        rows.append({"source": "GSE161277", "sample": patient, "n_normal": n_normal, "n_cancer": n_cancer,
                     "med_n": med_n, "med_c": med_c, "ratio": ratio, "p": p_val,
                     "ascl2_hi": tb["ASCL2"][0], "ascl2_lo": tb["ASCL2"][1],
                     "sox4_hi": tb["SOX4"][0], "sox4_lo": tb["SOX4"][1]})
    return rows


def run_crc(log):
    print("\n" + "=" * 100, file=log)
    print(f"CRC epithelial cohort -- donors with N_normal>={MIN_N} AND N_cancer>={MIN_N}", file=log)
    print("=" * 100, file=log)
    adata = sc.read_h5ad(CRC_PATH)
    sc.pp.pca(adata, n_comps=50, mask_var=None)
    pts_all = adata.obsm["X_pca"][:, :50]
    donor = adata.obs["donor_id"].values
    disease = adata.obs["disease"].values

    symbol_to_id = None
    for col in ["feature_name", "gene_symbol", "gene"]:
        if col in adata.var.columns:
            cand = dict(zip(adata.var[col].astype(str), adata.var_names))
            if all(g in cand for g in MARKERS):
                symbol_to_id = cand
                break
    assert symbol_to_id is not None

    donors = sorted(adata.obs["donor_id"].unique())
    rows = []
    for d in donors:
        mask_n = (donor == d) & (disease == "normal")
        mask_c = (donor == d) & (disease == "colorectal cancer")
        n_normal, n_cancer = int(mask_n.sum()), int(mask_c.sum())
        if n_normal < MIN_N or n_cancer < MIN_N:
            print(f"SKIP {d}: n_normal={n_normal}, n_cancer={n_cancer} (need >= {MIN_N})", file=log)
            continue

        pts_normal = pts_all[mask_n]
        pts_cancer = pts_all[mask_c]
        U, mean = subspace_svd(pts_normal)
        res_cancer = residual(pts_cancer, U, mean)
        res_normal_self = residual(pts_normal, U, mean)
        med_n, med_c = np.median(res_normal_self), np.median(res_cancer)
        ratio = med_c / med_n
        u_stat, p_val = mannwhitneyu(res_cancer, res_normal_self, alternative="two-sided")

        expr_dict = {}
        for g in MARKERS:
            gid = symbol_to_id[g]
            col_idx = adata.var_names.get_loc(gid)
            e = adata[mask_c, :].X[:, col_idx]
            e = np.asarray(e.todense()).flatten() if hasattr(e, "todense") else np.asarray(e).flatten()
            expr_dict[g] = e
        tb = top_bottom(expr_dict, res_cancer)

        print(f"\n{d}: n_normal={n_normal}, n_cancer={n_cancer}", file=log)
        print(f"  normal_med={med_n:.4f}  cancer_med={med_c:.4f}  ratio={ratio:.2f}x  p={p_val:.3e}", file=log)
        for g, (hi, lo) in tb.items():
            print(f"  {g}: top20={hi:.4f}  bottom20={lo:.4f}  {'HIGH>low' if hi > lo else 'no'}", file=log)

        rows.append({"source": "CRC_epith", "sample": d, "n_normal": n_normal, "n_cancer": n_cancer,
                     "med_n": med_n, "med_c": med_c, "ratio": ratio, "p": p_val,
                     "ascl2_hi": tb["ASCL2"][0], "ascl2_lo": tb["ASCL2"][1],
                     "sox4_hi": tb["SOX4"][0], "sox4_lo": tb["SOX4"][1]})
    return rows


def main():
    buf = io.StringIO()
    rows = run_gse161277(buf) + run_crc(buf)

    print("\n\n" + "=" * 130, file=buf)
    print("FINAL TABLE -- all samples, N>=50, unified SVD-based method", file=buf)
    print("=" * 130, file=buf)
    header = (f"{'Source':<12}{'Sample':<12}{'N_normal':<10}{'N_cancer':<10}{'Normal_med':<12}"
              f"{'Cancer_med':<12}{'Ratio':<8}{'p-value':<12}{'ASCL2_high':<12}{'ASCL2_low':<11}{'ASCL2_dir'}")
    print(header, file=buf)
    print("-" * 130, file=buf)
    for r in rows:
        direction = "high>low" if r["ascl2_hi"] > r["ascl2_lo"] else "low>=high"
        print(f"{r['source']:<12}{r['sample']:<12}{r['n_normal']:<10}{r['n_cancer']:<10}{r['med_n']:<12.4f}"
              f"{r['med_c']:<12.4f}{r['ratio']:<8.2f}{r['p']:<12.3e}{r['ascl2_hi']:<12.4f}{r['ascl2_lo']:<11.4f}{direction}",
              file=buf)

    n_total = len(rows)
    n_ratio_gt1 = sum(1 for r in rows if r["ratio"] > 1)
    n_ascl2_hi = sum(1 for r in rows if r["ascl2_hi"] > r["ascl2_lo"])
    n_sox4_hi = sum(1 for r in rows if r["sox4_hi"] > r["sox4_lo"])
    ratios = [r["ratio"] for r in rows]
    pvals = [r["p"] for r in rows]

    print(f"\nTotal samples analyzed: {n_total}", file=buf)
    print(f"cancer_residual > normal_residual: {n_ratio_gt1}/{n_total}", file=buf)
    print(f"ASCL2 higher in top20 vs bottom20: {n_ascl2_hi}/{n_total}", file=buf)
    print(f"SOX4 higher in top20 vs bottom20: {n_sox4_hi}/{n_total}", file=buf)
    print(f"Ratio: min={min(ratios):.2f}x  max={max(ratios):.2f}x  mean={np.mean(ratios):.2f}x  "
          f"median={np.median(ratios):.2f}x", file=buf)
    print(f"All Mann-Whitney p-values < 0.05: {all(p < 0.05 for p in pvals)}  "
          f"(max p-value across samples: {max(pvals):.3e})", file=buf)

    text = buf.getvalue()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
