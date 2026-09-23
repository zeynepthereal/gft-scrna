"""Unified, consistent verification pass for the paper: same method across all 8 samples.

Method (matches paper Section 2 exactly):
  1. U_normal (50x20) via SVD of normal epithelial cells' 50D PCA coords.
  2. residual_i = ||x_i - U U^T x_i|| for each cancer cell.
  3. Normal cells' self-projection residual as control.
  4. Top-20 vs bottom-20 residual cancer cells (by individual cell, not cluster) for
     ASCL2 / SOX4 / MKI67 mean expression.
"""
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
GSE_PATH = os.path.join(PROJECT_DIR, "data", "GSE161277_all.h5ad")
CRC_PATH = os.path.join(PROJECT_DIR, "data", "crc_epithelial.h5ad")
OUT_PATH = os.path.join(PROJECT_DIR, "results", "paper_final_verification.txt")

K_EIG = 20
MARKERS = ["ASCL2", "SOX4", "MKI67"]

GSE_PATIENTS = {
    "Patient1": {"normal": "GSM4904237", "carcinoma": "GSM4904236"},
    "Patient2": {"normal": "GSM4904240", "carcinoma": "GSM4904239"},
    "Patient3": {"normal": "GSM4904246", "carcinoma": "GSM4904245"},
}
CRC_MIN_N = 100


def subspace_svd(pts_normal, k_eig=K_EIG):
    mean = pts_normal.mean(axis=0)
    Xc = pts_normal - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k_eig].T, mean


def residual(pts, U, mean):
    X = pts - mean
    return np.linalg.norm(X - X @ U @ U.T, axis=1)


def top_bottom_marker_check(expr_dict, res, markers):
    """expr_dict: {gene: 1D array aligned with res}. Returns per-gene (mean_top20, mean_bottom20, higher_in_top)."""
    order = np.argsort(res)
    bottom20, top20 = order[:20], order[-20:]
    out = {}
    for g, expr in expr_dict.items():
        m_top, m_bot = expr[top20].mean(), expr[bottom20].mean()
        out[g] = (m_top, m_bot, m_top > m_bot)
    return out


def run_gse161277(log):
    print("=" * 90, file=log)
    print("GSE161277 (3 patients) -- unified method", file=log)
    print("=" * 90, file=log)
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

        U, mean = subspace_svd(pts_normal)
        res_cancer = residual(pts_carcinoma, U, mean)
        res_normal_self = residual(pts_normal, U, mean)
        med_n, med_c = np.median(res_normal_self), np.median(res_cancer)
        ratio_med = med_c / med_n
        ratio_mean = res_cancer.mean() / res_normal_self.mean()

        carc_idx = np.where(mask_carcinoma)[0]
        expr_dict = {}
        for g in MARKERS:
            gid = symbol_to_id[g]
            col_idx = list(raw.var_names[:]).index(gid) if not hasattr(raw.var_names, "get_loc") else raw.var_names.get_loc(gid)
            e = raw.X[carc_idx][:, col_idx]
            e = np.asarray(e.todense()).flatten() if hasattr(e, "todense") else np.asarray(e).flatten()
            expr_dict[g] = e
        marker_res = top_bottom_marker_check(expr_dict, res_cancer, MARKERS)

        from scipy.stats import mannwhitneyu
        u_stat, p_val = mannwhitneyu(res_cancer, res_normal_self, alternative="two-sided")

        print(f"\n{patient}: n_normal={n_normal}, n_cancer={n_cancer}", file=log)
        print(f"  normal_res_median={med_n:.4f}  cancer_res_median={med_c:.4f}  "
              f"ratio_median={ratio_med:.2f}x  ratio_mean={ratio_mean:.2f}x", file=log)
        print(f"  normal_res_std={res_normal_self.std():.4f}  cancer_res_std={res_cancer.std():.4f}", file=log)
        print(f"  Mann-Whitney U={u_stat:.1f}  p={p_val:.3e}", file=log)
        for g, (m_top, m_bot, higher) in marker_res.items():
            print(f"  {g}: top20={m_top:.4f}  bottom20={m_bot:.4f}  {'HIGH>low' if higher else 'no'}", file=log)

        rows.append({"source": "GSE161277", "sample": patient, "n_normal": n_normal, "n_cancer": n_cancer,
                     "med_n": med_n, "med_c": med_c, "ratio_med": ratio_med, "ratio_mean": ratio_mean,
                     "std_n": res_normal_self.std(), "std_c": res_cancer.std(),
                     "mw_p": p_val, "ascl2_higher": marker_res["ASCL2"][2],
                     "sox4_higher": marker_res["SOX4"][2], "mki67_higher": marker_res["MKI67"][2]})
    return rows


def run_crc(log):
    print("\n" + "=" * 90, file=log)
    print("CRC epithelial dataset (donors N>=100 both groups) -- unified method", file=log)
    print("=" * 90, file=log)
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
    assert symbol_to_id is not None, "could not resolve marker gene symbols"

    donors = sorted(adata.obs["donor_id"].unique())
    rows = []
    for d in donors:
        mask_n = (donor == d) & (disease == "normal")
        mask_c = (donor == d) & (disease == "colorectal cancer")
        n_normal, n_cancer = int(mask_n.sum()), int(mask_c.sum())
        if n_normal < CRC_MIN_N or n_cancer < CRC_MIN_N:
            continue

        pts_normal = pts_all[mask_n]
        pts_cancer = pts_all[mask_c]
        U, mean = subspace_svd(pts_normal)
        res_cancer = residual(pts_cancer, U, mean)
        res_normal_self = residual(pts_normal, U, mean)
        med_n, med_c = np.median(res_normal_self), np.median(res_cancer)
        ratio_med = med_c / med_n

        expr_dict = {}
        for g in MARKERS:
            gid = symbol_to_id[g]
            col_idx = adata.var_names.get_loc(gid)
            e = adata[mask_c, :].X[:, col_idx]
            e = np.asarray(e.todense()).flatten() if hasattr(e, "todense") else np.asarray(e).flatten()
            expr_dict[g] = e
        marker_res = top_bottom_marker_check(expr_dict, res_cancer, MARKERS)

        print(f"\n{d}: n_normal={n_normal}, n_cancer={n_cancer}", file=log)
        print(f"  normal_res_median={med_n:.4f}  cancer_res_median={med_c:.4f}  ratio_median={ratio_med:.2f}x", file=log)
        for g, (m_top, m_bot, higher) in marker_res.items():
            print(f"  {g}: top20={m_top:.4f}  bottom20={m_bot:.4f}  {'HIGH>low' if higher else 'no'}", file=log)

        rows.append({"source": "CRC_epith", "sample": d, "n_normal": n_normal, "n_cancer": n_cancer,
                     "med_n": med_n, "med_c": med_c, "ratio_med": ratio_med,
                     "ascl2_higher": marker_res["ASCL2"][2],
                     "sox4_higher": marker_res["SOX4"][2], "mki67_higher": marker_res["MKI67"][2]})
    return rows


def main():
    buf = io.StringIO()
    rows_gse = run_gse161277(buf)
    rows_crc = run_crc(buf)
    all_rows = rows_gse + rows_crc

    print("\n\n" + "=" * 100, file=buf)
    print("FINAL SUMMARY TABLE (all median-based, unified method)", file=buf)
    print("=" * 100, file=buf)
    header = f"{'Source':<12}{'Sample':<12}{'N_normal':<10}{'N_cancer':<10}{'med_n':<8}{'med_c':<8}{'ratio':<8}{'ASCL2':<7}{'SOX4':<6}{'MKI67'}"
    print(header, file=buf)
    print("-" * 100, file=buf)
    for r in all_rows:
        print(f"{r['source']:<12}{r['sample']:<12}{r['n_normal']:<10}{r['n_cancer']:<10}"
              f"{r['med_n']:<8.2f}{r['med_c']:<8.2f}{r['ratio_med']:<8.2f}"
              f"{'YES' if r['ascl2_higher'] else 'no':<7}{'YES' if r['sox4_higher'] else 'no':<6}"
              f"{'YES' if r['mki67_higher'] else 'no'}", file=buf)

    n_ratio_gt1 = sum(1 for r in all_rows if r["ratio_med"] > 1)
    n_ascl2 = sum(1 for r in all_rows if r["ascl2_higher"])
    n_sox4 = sum(1 for r in all_rows if r["sox4_higher"])
    print(f"\ncancer_residual > normal_residual: {n_ratio_gt1}/{len(all_rows)} samples", file=buf)
    print(f"ASCL2 higher in top20 vs bottom20: {n_ascl2}/{len(all_rows)} samples", file=buf)
    print(f"SOX4 higher in top20 vs bottom20: {n_sox4}/{len(all_rows)} samples", file=buf)
    print(f"ratio range: min={min(r['ratio_med'] for r in all_rows):.2f}x, "
          f"max={max(r['ratio_med'] for r in all_rows):.2f}x", file=buf)

    text = buf.getvalue()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
