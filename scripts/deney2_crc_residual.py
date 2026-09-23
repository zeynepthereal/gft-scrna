"""Deney 2: CRC epithelial dataset -- per-donor normal-subspace projection residual (N>=100 both groups)."""
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
PATH = os.path.join(PROJECT_DIR, "data", "crc_epithelial.h5ad")
OUT_PATH = os.path.join(PROJECT_DIR, "results", "deney2_crc_residual.txt")

N_COMPS = 50
K_EIG = 20
MIN_N = 100
SEED = 42


def subspace(pts_normal, k_eig=K_EIG):
    mean = pts_normal.mean(axis=0)
    Xc = pts_normal - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Ubasis = Vt[:k_eig].T
    return Ubasis, mean


def residual(pts, U, mean):
    X = pts - mean
    proj = X @ U @ U.T
    return np.linalg.norm(X - proj, axis=1)


def run(log):
    print("Loading data/crc_epithelial.h5ad...", file=log)
    adata = sc.read_h5ad(PATH)
    print(f"1) PCA (n_comps={N_COMPS}) on log-normalized X (no HVG mask)...", file=log)
    sc.pp.pca(adata, n_comps=N_COMPS, mask_var=None)
    pts_all = adata.obsm["X_pca"][:, :N_COMPS]

    donor = adata.obs["donor_id"].values
    disease = adata.obs["disease"].values
    symbol_to_id = dict(zip(adata.var["feature_name"].astype(str), adata.var_names)) \
        if "feature_name" in adata.var.columns else None
    if symbol_to_id is None or "ASCL2" not in symbol_to_id:
        # fall back: try var_names directly as symbols, or gene_symbol column
        for col in ["gene_symbol", "feature_name", "gene"]:
            if col in adata.var.columns:
                symbol_to_id = dict(zip(adata.var[col].astype(str), adata.var_names))
                if "ASCL2" in symbol_to_id:
                    break
    print(f"ASCL2 resolvable: {'ASCL2' in (symbol_to_id or {})}", file=log)

    donors = sorted(adata.obs["donor_id"].unique())
    rng = np.random.default_rng(SEED)

    rows = []
    ascl2_rows = []
    for d in donors:
        mask_n = (donor == d) & (disease == "normal")
        mask_c = (donor == d) & (disease == "colorectal cancer")
        n_normal, n_cancer = int(mask_n.sum()), int(mask_c.sum())

        if n_normal < MIN_N or n_cancer < MIN_N:
            print(f"SKIP {d}: normal={n_normal}, cancer={n_cancer} (need >= {MIN_N} in both)", file=log)
            continue

        pts_normal = pts_all[mask_n]
        pts_cancer = pts_all[mask_c]

        U, mean = subspace(pts_normal)
        res_cancer = residual(pts_cancer, U, mean)
        res_normal_self = residual(pts_normal, U, mean)

        med_normal = np.median(res_normal_self)
        med_cancer = np.median(res_cancer)
        ratio = med_cancer / med_normal if med_normal > 0 else float("inf")

        rows.append({"donor": d, "n_normal": n_normal, "n_cancer": n_cancer,
                     "med_normal": med_normal, "med_cancer": med_cancer, "ratio": ratio})
        print(f"{d}: n_normal={n_normal}, n_cancer={n_cancer}, "
              f"normal_res_median={med_normal:.4f}, cancer_res_median={med_cancer:.4f}, ratio={ratio:.2f}x", file=log)

        if symbol_to_id and "ASCL2" in symbol_to_id:
            gid = symbol_to_id["ASCL2"]
            col = adata.var_names.get_loc(gid)
            expr_cancer = adata[mask_c, :].X[:, col]
            expr_cancer = np.asarray(expr_cancer.todense()).flatten() if hasattr(expr_cancer, "todense") else np.asarray(expr_cancer).flatten()

            order_res = np.argsort(res_cancer)
            top20 = order_res[-20:]
            bottom20 = order_res[:20]
            mean_high = expr_cancer[top20].mean()
            mean_low = expr_cancer[bottom20].mean()
            ascl2_rows.append({"donor": d, "ASCL2_high20": mean_high, "ASCL2_low20": mean_low,
                               "higher_in_high20": mean_high > mean_low})
            print(f"    ASCL2: high20_residual mean={mean_high:.3f}, low20_residual mean={mean_low:.3f}, "
                  f"{'high>low' if mean_high > mean_low else 'high<=low'}", file=log)

    print("\n" + "=" * 100, file=log)
    print(f"{'Donor':<14}{'N_normal':<10}{'N_cancer':<10}{'normal_res_med':<16}{'cancer_res_med':<16}{'oran':<8}", file=log)
    print("-" * 100, file=log)
    for r in rows:
        print(f"{r['donor']:<14}{r['n_normal']:<10}{r['n_cancer']:<10}{r['med_normal']:<16.4f}"
              f"{r['med_cancer']:<16.4f}{r['ratio']:<8.2f}", file=log)
    print("=" * 100, file=log)

    n_higher = sum(1 for r in rows if r["ratio"] > 1)
    print(f"\ncancer residual > normal residual: {n_higher}/{len(rows)} donor", file=log)

    print("\n" + "=" * 70, file=log)
    print("ASCL2: yuksek-residual top-20 vs dusuk-residual top-20 (her donor)", file=log)
    print("=" * 70, file=log)
    print(f"{'Donor':<14}{'ASCL2_high20':<16}{'ASCL2_low20':<16}{'high>low'}", file=log)
    for r in ascl2_rows:
        print(f"{r['donor']:<14}{r['ASCL2_high20']:<16.4f}{r['ASCL2_low20']:<16.4f}"
              f"{'YES' if r['higher_in_high20'] else 'no'}", file=log)
    n_ascl2_higher = sum(1 for r in ascl2_rows if r["higher_in_high20"])
    print(f"\nASCL2 higher in high-residual group: {n_ascl2_higher}/{len(ascl2_rows)} donor", file=log)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    buf = io.StringIO()
    run(buf)
    text = buf.getvalue()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
