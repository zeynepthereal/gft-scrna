"""Experiment B: low-heterogeneity negative control (lung epithelial dataset)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

from gft import build_graph, gft_embed
from gft import ari_multi_seed, heterogeneity_score

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lung_epithelial.h5ad")
SEED = 42
K_EIG = 5
K_NN = 20
N_SEEDS = 10


def main():
    # backed='r' + manual array extraction avoids AnnData.copy()'s full-object subset path
    # (this file carries extra dense/sparse layers that blow past available RAM on subset)
    adata = sc.read_h5ad(PATH, backed="r")
    cell_type_all = adata.obs["cell_type"].astype(str).values
    counts = adata.obs["cell_type"].value_counts()
    keep = set(counts[counts >= 10].index)
    mask = np.array([c in keep for c in cell_type_all])

    pts_full = adata.obsm["X_pca"][:, :50]
    pts = pts_full[mask]
    cell_type = cell_type_all[mask]

    le = LabelEncoder()
    y = le.fit_transform(cell_type)
    k = len(le.classes_)
    N = len(pts)
    print(f"Lung epithelial: N={N}  k={k}")
    print(f"classes: {list(le.classes_)}")

    het = heterogeneity_score(pts, y)
    print(f"heterogeneity_score={het:.2f}")

    W = build_graph(pts, k_nn=K_NN, method="cosine")
    emb = gft_embed(W, k_eig=K_EIG)

    pca_mean, pca_std = ari_multi_seed(pts, y, k, n_seeds=N_SEEDS, seed=SEED)
    gft_mean, gft_std = ari_multi_seed(emb, y, k, n_seeds=N_SEEDS, seed=SEED)
    ratio = gft_mean / max(pca_mean, 1e-6)

    print(f"\nPCA ARI: {pca_mean:.4f} +/- {pca_std:.4f}")
    print(f"GFT ARI: {gft_mean:.4f} +/- {gft_std:.4f}")
    print(f"Ratio (GFT/PCA): {ratio:.2f}x")
    print(f"\nGFT ratio < 1.0 (expected for het<7)? {'YES' if ratio < 1.0 else 'NO'}")
    print(f"het < 7? {'YES' if het < 7 else 'NO'} (het={het:.2f})")


if __name__ == "__main__":
    main()
