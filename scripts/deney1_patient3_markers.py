"""Deney 1: Patient3 carcinoma epithelial cells -- normal-subspace projection residual, DEG, marker panel."""
import sys
import os
import io
import contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
PATH = os.path.join(PROJECT_DIR, "data", "GSE161277_all.h5ad")
OUT_PATH = os.path.join(PROJECT_DIR, "results", "deney1_patient3_markers.txt")

NORMAL_GSM = "GSM4904246"
CARCINOMA_GSM = "GSM4904245"
K_EIG = 20

MARKER_GENES = ["ASCL2", "SOX4", "MKI67", "HIF1A", "LDHA", "VEGFA", "TP53", "KRAS", "MYC", "CDX2"]


def run(log):
    print("Loading data/GSE161277_all.h5ad (backed='r' -- memory-constrained machine)...", file=log)
    adata = sc.read_h5ad(PATH, backed="r")
    raw = adata.raw

    pts_all = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values
    cell_type = adata.obs["cell_type"].values

    mask_normal = (gsm == NORMAL_GSM) & (cell_type == "Epithelial/Tumor")
    mask_carcinoma = (gsm == CARCINOMA_GSM) & (cell_type == "Epithelial/Tumor")
    pts_normal = pts_all[mask_normal]
    pts_carcinoma = pts_all[mask_carcinoma]
    print(f"Patient3 normal epithelial: n={len(pts_normal)}", file=log)
    print(f"Patient3 carcinoma epithelial: n={len(pts_carcinoma)}", file=log)

    print(f"\nNormal subspace: SVD of normal cells' 50D PCA coords, top-{K_EIG} components...", file=log)
    mean = pts_normal.mean(axis=0)
    Xc = pts_normal - mean
    Usvd, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    U = Vt[:K_EIG].T  # 50 x K_EIG, orthonormal (right singular vectors = PCA-of-PCA eigenvectors)
    explained = (S[:K_EIG] ** 2).sum() / (S ** 2).sum()
    print(f"U: {U.shape}, explained variance = {explained:.3f}", file=log)

    print("\nProjection residuals for carcinoma cells: residual_i = ||x_i - U U^T x_i||...", file=log)
    Xcarc = pts_carcinoma - mean
    proj = Xcarc @ U @ U.T
    res = np.linalg.norm(Xcarc - proj, axis=1)
    print(f"Carcinoma residual: median={np.median(res):.4f}  mean={res.mean():.4f}  std={res.std():.4f}", file=log)

    res_normal_self = np.linalg.norm(Xc - Xc @ U @ U.T, axis=1)
    print(f"Normal self-residual (control): median={np.median(res_normal_self):.4f}  "
          f"mean={res_normal_self.mean():.4f}  std={res_normal_self.std():.4f}", file=log)

    order_res = np.argsort(res)
    bottom20_local = order_res[:20]
    top20_local = order_res[-20:]
    print(f"\nTop-20 residual range: [{res[top20_local].min():.3f}, {res[top20_local].max():.3f}]", file=log)
    print(f"Bottom-20 residual range: [{res[bottom20_local].min():.3f}, {res[bottom20_local].max():.3f}]", file=log)

    carc_global_idx = np.where(mask_carcinoma)[0]
    top20_mask_local = np.zeros(len(pts_carcinoma), dtype=bool)
    top20_mask_local[top20_local] = True

    print("\nBuilding small AnnData from raw (lazy row slicing, memory-safe) for DEG + marker panel...", file=log)
    sub_X = raw.X[carc_global_idx]
    sub_obs = pd.DataFrame({
        "group": np.where(top20_mask_local, "high_residual", "rest")
    }, index=[f"cell_{i}" for i in range(len(carc_global_idx))])
    sub_obs["group"] = sub_obs["group"].astype("category")
    adata_sub = ad.AnnData(X=sub_X, obs=sub_obs, var=raw.var.to_pandas() if hasattr(raw.var, "to_pandas") else raw.var[:])
    adata_sub.var_names = raw.var_names[:] if not isinstance(raw.var_names, list) else raw.var_names
    print(f"n(high_residual, top20)={ (sub_obs['group']=='high_residual').sum() }, "
          f"n(rest)={ (sub_obs['group']=='rest').sum() }", file=log)

    print("\n=== rank_genes_groups (Wilcoxon, top-20 high-residual vs rest of carcinoma) ===", file=log)
    sc.tl.rank_genes_groups(adata_sub, groupby="group", groups=["high_residual"], reference="rest",
                             method="wilcoxon", use_raw=False)
    df = sc.get.rank_genes_groups_df(adata_sub, group="high_residual")
    id_to_symbol = dict(zip(raw.var_names[:], raw.var["gene_symbol"][:]))
    df["symbol"] = df["names"].map(id_to_symbol)
    top15 = df.head(15)
    print("\nTop 15 marker genes for high-residual (top-20) cells vs rest:", file=log)
    print(top15[["symbol", "names", "logfoldchanges", "pvals_adj", "scores"]].to_string(index=False), file=log)

    print("\n=== Marker panel: mean expression, high-residual (top-20) vs low-residual (bottom-20) ===", file=log)
    symbol_to_id = {sym: gid for gid, sym in id_to_symbol.items()}
    rows = []
    for gene in MARKER_GENES:
        if gene not in symbol_to_id:
            print(f"  WARNING: {gene} not found in gene set", file=log)
            continue
        gid = symbol_to_id[gene]
        col = adata_sub.var_names.get_loc(gid)
        expr = adata_sub.X[:, col]
        expr = np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()
        group_arr = adata_sub.obs["group"].values
        mean_high = expr[group_arr == "high_residual"].mean()
        mean_rest = expr[group_arr == "rest"].mean()
        # also explicit bottom-20 (lowest residual) subset for direct high-vs-low comparison
        bottom20_local_mask = np.zeros(len(pts_carcinoma), dtype=bool)
        bottom20_local_mask[bottom20_local] = True
        mean_low20 = expr[bottom20_local_mask].mean()
        pct_high = (expr[group_arr == "high_residual"] > 0).mean() * 100
        pct_low20 = (expr[bottom20_local_mask] > 0).mean() * 100
        rows.append({"gene": gene, "mean_high20": mean_high, "pct_high20": pct_high,
                     "mean_low20": mean_low20, "pct_low20": pct_low20, "mean_rest_all": mean_rest})

    header = f"{'Gene':<10}{'mean_high20':<14}{'%+_high20':<12}{'mean_low20':<14}{'%+_low20':<12}{'mean_rest_all':<14}"
    print(header, file=log)
    print("-" * len(header), file=log)
    for r in rows:
        print(f"{r['gene']:<10}{r['mean_high20']:<14.3f}{r['pct_high20']:<12.1f}"
              f"{r['mean_low20']:<14.3f}{r['pct_low20']:<12.1f}{r['mean_rest_all']:<14.3f}", file=log)


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
