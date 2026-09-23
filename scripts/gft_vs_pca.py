"""GFT vs PCA comparison on GSE161277_all.h5ad using cell_type as ground truth labels."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import scanpy as sc
from sklearn.preprocessing import LabelEncoder

from gft import build_graph, gft_embed
from gft import ari_multi_seed, heterogeneity_score

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
SEED = 42
K_EIG = 5
K_NN = 20
METRIC = "cosine"
N_SEEDS = 10


def main():
    adata = sc.read_h5ad(PATH)
    pts = adata.obsm["X_pca"][:, :50]

    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"].astype(str).values)
    k = len(le.classes_)

    print(f"N={len(pts)} | k={k} | classes={list(le.classes_)}")

    het = heterogeneity_score(pts, y)
    print(f"Heterojenlik skoru (inter/intra PCA mesafesi): {het:.4f}")

    print(f"\nBuilding cosine kNN graph (k_nn={K_NN})...")
    W = build_graph(pts, k_nn=K_NN, method=METRIC)

    print(f"Computing GFT embedding (k_eig={K_EIG})...")
    emb = gft_embed(W, k_eig=K_EIG)

    print(f"\nRunning K-Means (k={k}) x {N_SEEDS} seeds...")
    pca_mean, pca_std = ari_multi_seed(pts, y, k, n_seeds=N_SEEDS, seed=SEED)
    gft_mean, gft_std = ari_multi_seed(emb, y, k, n_seeds=N_SEEDS, seed=SEED)

    ratio = gft_mean / max(pca_mean, 1e-6)

    print("\n" + "=" * 50)
    print(f"  PCA ARI          : {pca_mean:.4f} ± {pca_std:.4f}")
    print(f"  GFT ARI          : {gft_mean:.4f} ± {gft_std:.4f}")
    print(f"  GFT/PCA oranı    : {ratio:.2f}x")
    print(f"  Heterojenlik skoru: {het:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
