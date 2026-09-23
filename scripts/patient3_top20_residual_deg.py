"""Patient3 carcinoma: top-20 residual cells vs rest -- rank_genes_groups DEG."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import h5py
from anndata._core.sparse_dataset import sparse_dataset

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
NORMAL_GSM = "GSM4904246"
CARCINOMA_GSM = "GSM4904245"
K_EIG = 20


def main():
    # backed='r' keeps raw.X as a lazy on-disk CSR dataset instead of eagerly loading the
    # full (50518 x 24111) sparse matrix into RAM -- this machine is memory-constrained
    # (~3GB free of 16GB) and the eager load kept failing with ArrayMemoryError.
    adata = sc.read_h5ad(PATH, backed="r")
    raw = adata.raw

    pts_all = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values
    cell_type = adata.obs["cell_type"].values

    mask_normal = (gsm == NORMAL_GSM) & (cell_type == "Epithelial/Tumor")
    mask_carcinoma = (gsm == CARCINOMA_GSM) & (cell_type == "Epithelial/Tumor")
    pts_normal = pts_all[mask_normal]
    pts_carcinoma = pts_all[mask_carcinoma]
    print(f"Normal epithelial: n={len(pts_normal)}, Carcinoma epithelial: n={len(pts_carcinoma)}")

    mean = pts_normal.mean(axis=0)
    Xc = pts_normal - mean
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    U = eigvecs[:, order][:, :K_EIG]

    Xcarc = pts_carcinoma - mean
    proj = Xcarc @ U @ U.T
    res = np.linalg.norm(Xcarc - proj, axis=1)

    order_res = np.argsort(res)
    top20_local = order_res[-20:]
    top20_mask_local = np.zeros(len(pts_carcinoma), dtype=bool)
    top20_mask_local[top20_local] = True
    print(f"Top-20 residual range: [{res[top20_local].min():.3f}, {res[top20_local].max():.3f}]  "
          f"(rest: median={np.median(res[~top20_mask_local]):.3f})")

    carc_global_idx = np.where(mask_carcinoma)[0]

    sub_X = raw.X[carc_global_idx]
    sub_obs = pd.DataFrame({
        "group": np.where(top20_mask_local, "top20_residual", "rest")
    }, index=[f"cell_{i}" for i in range(len(carc_global_idx))])
    sub_obs["group"] = sub_obs["group"].astype("category")
    adata_sub = ad.AnnData(X=sub_X, obs=sub_obs, var=raw.var.copy())
    adata_sub.var_names = raw.var_names
    print(f"n(top20_residual)={ (sub_obs['group']=='top20_residual').sum() }, "
          f"n(rest)={ (sub_obs['group']=='rest').sum() }")

    print("\nrank_genes_groups (Wilcoxon, top20_residual vs rest, raw log-normalized data)...")
    sc.tl.rank_genes_groups(adata_sub, groupby="group", groups=["top20_residual"], reference="rest",
                             method="wilcoxon", use_raw=False)

    df = sc.get.rank_genes_groups_df(adata_sub, group="top20_residual")
    id_to_symbol = dict(zip(raw.var_names, raw.var["gene_symbol"]))
    df["symbol"] = df["names"].map(id_to_symbol)

    print("\n=== Top 10 marker genes for TOP-20 residual cells (vs rest of carcinoma) ===")
    top10 = df.head(10)
    print(top10[["symbol", "names", "logfoldchanges", "pvals_adj", "scores"]].to_string(index=False))


if __name__ == "__main__":
    main()
