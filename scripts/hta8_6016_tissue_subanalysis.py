"""HTA8_6016 sub-analysis: primary (sigmoid colon) vs metastasis (liver) cancer cells,
projected onto the same normal (sigmoid colon) subspace."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "crc_epithelial.h5ad")
DONOR = "HTA8_6016"
K_EIG = 20
MIN_N = 30


def main():
    adata = sc.read_h5ad(PATH)
    sc.pp.pca(adata, n_comps=50, mask_var=None)
    pts_all = adata.obsm["X_pca"][:, :50]

    donor = adata.obs["donor_id"].values
    disease = adata.obs["disease"].values
    tissue = adata.obs["tissue"].values

    symbol_to_id = None
    for col in ["feature_name", "gene_symbol", "gene"]:
        if col in adata.var.columns:
            cand = dict(zip(adata.var[col].astype(str), adata.var_names))
            if "ASCL2" in cand:
                symbol_to_id = cand
                break

    mask_normal = (donor == DONOR) & (disease == "normal")  # all sigmoid colon (verified)
    mask_primary = (donor == DONOR) & (disease == "colorectal cancer") & (tissue == "sigmoid colon")
    mask_met = (donor == DONOR) & (disease == "colorectal cancer") & (tissue == "liver")

    n_normal, n_primary, n_met = mask_normal.sum(), mask_primary.sum(), mask_met.sum()
    print(f"{DONOR}: n_normal(sigmoid colon)={n_normal}, n_primary(sigmoid colon)={n_primary}, n_metastasis(liver)={n_met}")

    pts_normal = pts_all[mask_normal]
    mean = pts_normal.mean(axis=0)
    Xc = pts_normal - mean
    U_svd, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    U = Vt[:K_EIG].T
    print(f"U_normal: {U.shape} (from sigmoid colon normal cells only)")

    def residual(pts):
        X = pts - mean
        return np.linalg.norm(X - X @ U @ U.T, axis=1)

    res_normal_self = residual(pts_normal)
    med_normal = np.median(res_normal_self)
    print(f"\nNormal (self-projection): median={med_normal:.4f}")

    results = {}
    for label, mask, min_n in [("Primary (sigmoid colon)", mask_primary, MIN_N),
                                ("Metastasis (liver)", mask_met, MIN_N)]:
        n = mask.sum()
        if n < min_n:
            print(f"\n{label}: n={n} < {min_n}, SKIPPED")
            continue
        pts_group = pts_all[mask]
        res_group = residual(pts_group)
        med_group = np.median(res_group)
        ratio = med_group / med_normal
        print(f"\n{label}: n={n}")
        print(f"  residual median={med_group:.4f}  ratio vs normal={ratio:.2f}x")

        ascl2_top20 = ascl2_bottom20 = None
        if symbol_to_id is not None and "ASCL2" in symbol_to_id:
            gid = symbol_to_id["ASCL2"]
            col_idx = adata.var_names.get_loc(gid)
            expr = adata[mask, :].X[:, col_idx]
            expr = np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()
            order = np.argsort(res_group)
            top20, bottom20 = order[-20:], order[:20]
            ascl2_top20, ascl2_bottom20 = expr[top20].mean(), expr[bottom20].mean()
            print(f"  ASCL2: top20={ascl2_top20:.4f}  bottom20={ascl2_bottom20:.4f}  "
                  f"{'HIGH>low' if ascl2_top20 > ascl2_bottom20 else 'no'}")

        results[label] = {"n": n, "median": med_group, "ratio": ratio,
                          "ascl2_top20": ascl2_top20, "ascl2_bottom20": ascl2_bottom20}

    print("\n" + "=" * 80)
    print(f"{'Group':<28}{'N':<8}{'Residual_median':<18}{'Ratio_vs_normal'}")
    print("-" * 80)
    print(f"{'Normal (sigmoid colon)':<28}{n_normal:<8}{med_normal:<18.4f}{'1.00x (baseline)'}")
    for label, r in results.items():
        print(f"{label:<28}{r['n']:<8}{r['median']:<18.4f}{r['ratio']:.2f}x")
    print("=" * 80)

    if "Primary (sigmoid colon)" in results and "Metastasis (liver)" in results:
        pm_ratio = results["Metastasis (liver)"]["median"] / results["Primary (sigmoid colon)"]["median"]
        print(f"\nMetastasis/Primary residual ratio: {pm_ratio:.2f}x")


if __name__ == "__main__":
    main()
