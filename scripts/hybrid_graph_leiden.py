"""Hybrid graph experiment: W_hybrid = alpha * W_pca + (1-alpha) * W_gft, Leiden per alpha."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score

from gft import build_graph, gft_embed

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
K_NN = 20
K_EIG = 20
METRIC = "cosine"
RESOLUTION = 0.3
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main():
    adata = sc.read_h5ad(PATH)
    pts = adata.obsm["X_pca"][:, :50]

    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"].astype(str).values)
    print(f"N={adata.n_obs} | k_true={len(le.classes_)}")

    print(f"\n1) W_pca: cosine kNN (k={K_NN}) on PCA (50D)...")
    W_pca = build_graph(pts, k_nn=K_NN, method=METRIC)
    print(f"   W_pca: nnz={W_pca.nnz}, max={W_pca.max():.4f}")

    print(f"\n2) GFT embedding (k_eig={K_EIG}) from W_pca, then cosine kNN (k={K_NN}) on it...")
    emb = gft_embed(W_pca, k_eig=K_EIG)
    W_gft = build_graph(emb, k_nn=K_NN, method=METRIC)
    print(f"   W_gft: nnz={W_gft.nnz}, max={W_gft.max():.4f}")

    print("\n3) Normalizing both graphs (max=1)...")
    W_pca_n = W_pca / W_pca.max()
    W_gft_n = W_gft / W_gft.max()

    print(f"\n4-6) Hybrid Leiden sweep (resolution={RESOLUTION})...")
    rows = []
    for alpha in ALPHAS:
        W_hybrid = (alpha * W_pca_n + (1 - alpha) * W_gft_n).tocsr()
        key = f"leiden_hybrid_a{alpha}"
        sc.tl.leiden(adata, resolution=RESOLUTION, key_added=key, adjacency=W_hybrid,
                     flavor="igraph", n_iterations=2, directed=False)
        ari = adjusted_rand_score(y, adata.obs[key].values)
        n_clusters = adata.obs[key].nunique()
        rows.append({"alpha": alpha, "ari": ari, "n_clusters": n_clusters})
        print(f"   alpha={alpha}: ARI={ari:.4f}, n_clusters={n_clusters}")

    print("\n" + "=" * 40)
    print(f"{'alpha':<10}{'ARI':<10}{'n_clusters':<12}")
    print("-" * 40)
    for r in rows:
        print(f"{r['alpha']:<10}{r['ari']:<10.4f}{r['n_clusters']:<12}")
    print("=" * 40)

    best = max(rows, key=lambda r: r["ari"])
    print(f"\nEn iyi alpha: {best['alpha']} (ARI={best['ari']:.4f}, n_clusters={best['n_clusters']})")

    adata.write_h5ad(PATH)
    print(f"\nSaved leiden_hybrid_a* columns to {PATH}")


if __name__ == "__main__":
    main()
