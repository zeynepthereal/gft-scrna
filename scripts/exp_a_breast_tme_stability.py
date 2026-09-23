"""Experiment A: Breast TME GFT stability check with n_seeds=20."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

from gft import build_graph, gft_embed
from gft import ari_multi_seed, heterogeneity_score

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "breast_tme.h5ad")
SEED = 42
K_EIG = 5
K_NN = 20
N_SEEDS = 20


def main():
    adata = sc.read_h5ad(PATH)
    counts = adata.obs["cell_type"].value_counts()
    keep = counts[counts >= 10].index
    mask = adata.obs["cell_type"].isin(keep).values
    adata = adata[mask].copy()

    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"].astype(str).values)
    pts = adata.obsm["X_pca"][:, :50]
    k = len(le.classes_)
    print(f"N={len(pts)}  k={k}")

    het = heterogeneity_score(pts, y)
    print(f"heterogeneity_score={het:.2f}")

    W = build_graph(pts, k_nn=K_NN, method="cosine")
    emb = gft_embed(W, k_eig=K_EIG)

    gft_mean, gft_std = ari_multi_seed(emb, y, k, n_seeds=N_SEEDS, seed=SEED)
    pca_mean, pca_std = ari_multi_seed(pts, y, k, n_seeds=N_SEEDS, seed=SEED)

    print(f"\nGFT ARI (n_seeds={N_SEEDS}, n_init=5 [ari_multi_seed default]): {gft_mean:.4f} +/- {gft_std:.4f}")
    print(f"PCA ARI (n_seeds={N_SEEDS}): {pca_mean:.4f} +/- {pca_std:.4f}")
    print(f"Ratio: {gft_mean/pca_mean:.2f}x")
    print(f"\nStd < 0.05? {'YES' if gft_std < 0.05 else 'NO'} (std={gft_std:.4f})")

    # Also explicitly with KMeans n_init=10 as requested (ari_multi_seed hardcodes n_init=5)
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    aris = []
    for s in range(SEED, SEED + N_SEEDS):
        pred = KMeans(k, random_state=s, n_init=10).fit_predict(emb)
        aris.append(adjusted_rand_score(y, pred))
    aris = np.array(aris)
    print(f"\n[explicit n_init=10] GFT ARI: {aris.mean():.4f} +/- {aris.std():.4f}")
    print(f"[explicit n_init=10] Std < 0.05? {'YES' if aris.std() < 0.05 else 'NO'}")


if __name__ == "__main__":
    main()
