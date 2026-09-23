"""Patient1: normal-epithelial spectral/PCA subspace, carcinoma-cell projection residual.

Note on methodology (resolved ambiguity): the graph Laplacian eigenvectors (GFT, from
cosine-kNN on normal cells) are cell-indexed (n_normal x k_eig) and cannot be applied to
out-of-sample carcinoma cells via U.T @ x_i (dimension mismatch: x_i lives in 50D PCA
space, not in normal-cell index space). Per user's choice, the actual projection/residual
uses a PCA-of-PCA subspace instead: the top-20 eigenvectors of the covariance of normal
cells' 50D PCA coordinates -- a genuine (50x20) orthonormal basis directly applicable to
any point in PCA space. The cosine-kNN graph + Laplacian eigenvectors are still computed
and saved as requested in step 1, for reference, but are NOT used in the residual formula.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import save_npz

from gft import build_graph, gft_embed

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
K_NN = 15
K_EIG = 20

NORMAL_GSM = "GSM4904237"
CARCINOMA_GSM = "GSM4904236"


def main():
    adata = sc.read_h5ad(PATH)
    pts_all = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values
    cell_type = adata.obs["cell_type"].values
    leiden = adata.obs["leiden"].values

    mask_normal = (gsm == NORMAL_GSM) & (cell_type == "Epithelial/Tumor")
    mask_carcinoma = (gsm == CARCINOMA_GSM) & (cell_type == "Epithelial/Tumor")

    pts_normal = pts_all[mask_normal]
    pts_carcinoma = pts_all[mask_carcinoma]
    print(f"Normal epithelial (Patient1, {NORMAL_GSM}): n={len(pts_normal)}")
    print(f"Carcinoma epithelial (Patient1, {CARCINOMA_GSM}): n={len(pts_carcinoma)}")

    print(f"\n1) [reference] cosine kNN graph (k={K_NN}) on normal cells + Laplacian eigenvectors (k_eig={K_EIG})...")
    W_normal = build_graph(pts_normal, k_nn=K_NN, method="cosine")
    U_graph = gft_embed(W_normal, k_eig=K_EIG)  # cell-indexed, NOT used for projection (see module docstring)
    os.makedirs(OUT_DIR, exist_ok=True)
    save_npz(os.path.join(OUT_DIR, "patient1_normal_cosine_knn_graph.npz"), W_normal)
    np.save(os.path.join(OUT_DIR, "patient1_normal_U_graph_laplacian_eigvecs.npy"), U_graph)
    print(f"   W_normal: {W_normal.shape}, U_graph (cell-indexed, reference only): {U_graph.shape}")
    print(f"   Saved -> results/patient1_normal_cosine_knn_graph.npz, patient1_normal_U_graph_laplacian_eigvecs.npy")

    print(f"\n1b) U_normal = PCA-of-PCA subspace: top-{K_EIG} eigenvectors of normal cells' "
          f"50D-PCA covariance (used for the actual projection/residual, per resolved methodology)...")
    normal_mean = pts_normal.mean(axis=0)
    pts_normal_c = pts_normal - normal_mean
    cov = np.cov(pts_normal_c, rowvar=False)  # 50x50
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eigvals)[::-1]
    U_normal = eigvecs[:, order][:, :K_EIG]  # 50 x 20, orthonormal
    explained = eigvals[order][:K_EIG].sum() / eigvals.sum()
    np.save(os.path.join(OUT_DIR, "patient1_normal_U_pca_subspace.npy"), U_normal)
    print(f"   U_normal: {U_normal.shape}, explained variance (top {K_EIG}/50 PCs of normal cells) = {explained:.3f}")

    def residual(pts, center):
        X = pts - center
        proj = X @ U_normal @ U_normal.T
        return np.linalg.norm(X - proj, axis=1)

    print("\n2-3) Projection residuals...")
    res_carcinoma = residual(pts_carcinoma, normal_mean)
    res_normal_self = residual(pts_normal, normal_mean)  # control: normal cells projected onto their own subspace

    def stats(x, name):
        print(f"   {name}: n={len(x)}  median={np.median(x):.4f}  mean={x.mean():.4f}  std={x.std():.4f}")

    print("\n4) Residual distributions:")
    stats(res_normal_self, "Normal (self-projection, kontrol)")
    stats(res_carcinoma, "Carcinoma (out-of-sample)")

    from scipy.stats import mannwhitneyu
    u_stat, p_val = mannwhitneyu(res_carcinoma, res_normal_self, alternative="two-sided")
    fold = res_carcinoma.mean() / res_normal_self.mean()
    print(f"\n   Carcinoma/Normal mean residual ratio: {fold:.2f}x")
    print(f"   Mann-Whitney U test (carcinoma vs normal residual): U={u_stat:.1f}, p={p_val:.3e}")

    print("\n5) Top-20 highest-residual vs bottom-20 lowest-residual carcinoma cells...")
    carc_idx = np.where(mask_carcinoma)[0]
    carc_leiden = leiden[carc_idx]
    carc_cell_type = cell_type[carc_idx]  # all 'Epithelial/Tumor' by construction

    order_res = np.argsort(res_carcinoma)
    bottom20_idx = order_res[:20]
    top20_idx = order_res[-20:]

    print(f"   Bottom-20 (lowest residual) leiden clusters: {sorted(pd.Series(carc_leiden[bottom20_idx]).tolist())}")
    print(f"   Top-20 (highest residual) leiden clusters:   {sorted(pd.Series(carc_leiden[top20_idx]).tolist())}")

    print("\n   Bottom-20 leiden distribution:")
    print(pd.Series(carc_leiden[bottom20_idx]).value_counts())
    print("\n   Top-20 leiden distribution:")
    print(pd.Series(carc_leiden[top20_idx]).value_counts())

    print(f"\n   cell_type check -- bottom20 unique: {set(carc_cell_type[bottom20_idx])}, "
          f"top20 unique: {set(carc_cell_type[top20_idx])} (both subset to Epithelial/Tumor by construction)")

    print(f"\n   Bottom-20 residual range: [{res_carcinoma[bottom20_idx].min():.4f}, {res_carcinoma[bottom20_idx].max():.4f}]")
    print(f"   Top-20 residual range:    [{res_carcinoma[top20_idx].min():.4f}, {res_carcinoma[top20_idx].max():.4f}]")

    print("\n" + "=" * 70)
    print("OZET")
    print("=" * 70)
    print(f"Normal self-residual   : median={np.median(res_normal_self):.4f}  mean={res_normal_self.mean():.4f}  std={res_normal_self.std():.4f}")
    print(f"Carcinoma residual     : median={np.median(res_carcinoma):.4f}  mean={res_carcinoma.mean():.4f}  std={res_carcinoma.std():.4f}")
    print(f"Ratio (carcinoma/normal mean): {fold:.2f}x")
    print(f"Mann-Whitney p-value   : {p_val:.3e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
