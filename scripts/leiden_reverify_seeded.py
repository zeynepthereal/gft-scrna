"""Re-verify GFT-Leiden over-fragmentation with a deterministically seeded eigsh
(v0=ones(N)/sqrt(N), random_state=42), across k_eig in [5,10,20,50], vs a PCA-Leiden baseline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score

from gft import build_graph

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
K_NN = 20
N_NEIGHBORS = 15
RESOLUTION = 0.3
K_EIG_VALUES = [5, 10, 20, 50]
SEED = 42


def normalized_laplacian(W):
    N = W.shape[0]
    d = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = diags(1.0 / np.sqrt(d + 1e-10))
    return eye(N, format="csr") - D_inv_sqrt @ W @ D_inv_sqrt


def gft_embed_seeded(W, k_eig, random_state=SEED, tol=1e-4, maxiter=10000):
    N = W.shape[0]
    L = normalized_laplacian(W)
    v0 = np.ones(N) / np.sqrt(N)  # fixed deterministic v0 makes eigsh's internal rng moot
    _, vecs = eigsh(L, k=k_eig + 1, which="SM", tol=tol, maxiter=maxiter, v0=v0)
    return vecs[:, 1:]


def leiden_and_ari(adata, y, use_rep=None, n_pcs=None, key="leiden"):
    if use_rep is not None:
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=N_NEIGHBORS)
    else:
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=N_NEIGHBORS)
    sc.tl.leiden(adata, resolution=RESOLUTION, key_added=key, random_state=SEED,
                 flavor="igraph", n_iterations=2, directed=False)
    ari = adjusted_rand_score(y, adata.obs[key].values)
    n_clusters = adata.obs[key].nunique()
    return ari, n_clusters


def main():
    adata = sc.read_h5ad(PATH)
    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"].astype(str).values)
    k_true = len(le.classes_)
    print(f"N={adata.n_obs}  k_true={k_true}")

    pts = adata.obsm["X_pca"][:, :50]

    print("\n=== Baseline: PCA -> Leiden ===")
    ari_pca, ncl_pca = leiden_and_ari(adata, y, use_rep=None, n_pcs=50, key="leiden_pca")
    print(f"PCA -> Leiden: ARI={ari_pca:.4f}  n_clusters={ncl_pca}")

    print(f"\nBuilding cosine kNN graph (k={K_NN}) once (shared across k_eig)...")
    W = build_graph(pts, k_nn=K_NN, method="cosine")

    rows = [{"method": "PCA -> Leiden", "k_eig": "--", "ari": ari_pca, "n_clusters": ncl_pca}]
    for k_eig in K_EIG_VALUES:
        print(f"\n=== GFT (seeded, k_eig={k_eig}) -> Leiden ===")
        emb = gft_embed_seeded(W, k_eig)
        adata.obsm["X_gft"] = emb
        ari, ncl = leiden_and_ari(adata, y, use_rep="X_gft", key=f"leiden_gft_{k_eig}")
        print(f"GFT(k_eig={k_eig}) -> Leiden: ARI={ari:.4f}  n_clusters={ncl}")
        rows.append({"method": "GFT -> Leiden", "k_eig": k_eig, "ari": ari, "n_clusters": ncl})

    print("\n" + "=" * 60)
    print(f"{'Method':<16}{'k_eig':<8}{'ARI':<10}{'N clusters'}")
    print("-" * 60)
    for r in rows:
        print(f"{r['method']:<16}{str(r['k_eig']):<8}{r['ari']:<10.4f}{r['n_clusters']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
