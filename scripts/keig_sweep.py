"""GFT k_eig sweep: how does GFT-embedding + Leiden ARI change with k_eig?"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score

from gft import build_graph, gft_embed

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
K_NN = 20
METRIC = "cosine"
N_NEIGHBORS = 15
RESOLUTION = 0.3
K_EIG_VALUES = [5, 10, 15, 20, 30, 50]
PCA_BASELINE_ARI = 0.3544


def main():
    adata = sc.read_h5ad(PATH)
    pts = adata.obsm["X_pca"][:, :50]

    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"].astype(str).values)
    print(f"N={len(pts)} | k_true={len(le.classes_)}")

    print(f"\nBuilding cosine kNN graph (k_nn={K_NN}) once (shared across k_eig)...")
    W = build_graph(pts, k_nn=K_NN, method=METRIC)

    rows = []
    for k_eig in K_EIG_VALUES:
        t0 = time.time()
        emb = gft_embed(W, k_eig=k_eig)
        adata.obsm["X_gft"] = emb

        sc.pp.neighbors(adata, use_rep="X_gft", n_neighbors=N_NEIGHBORS)
        key = f"leiden_gft_keig{k_eig}"
        sc.tl.leiden(adata, resolution=RESOLUTION, key_added=key,
                     flavor="igraph", n_iterations=2, directed=False)

        ari = adjusted_rand_score(y, adata.obs[key].values)
        n_clusters = adata.obs[key].nunique()
        elapsed = time.time() - t0

        rows.append({"k_eig": k_eig, "ari": ari, "n_clusters": n_clusters, "elapsed": elapsed})
        print(f"  k_eig={k_eig:>3}: ARI={ari:.4f}, n_clusters={n_clusters}, süre={elapsed:.1f}sn")

    print("\n" + "=" * 60)
    print(f"{'k_eig':<8}{'ARI':<10}{'n_clusters':<12}{'süre (sn)':<10}")
    print("-" * 60)
    print(f"{'PCA':<8}{PCA_BASELINE_ARI:<10.4f}{'21':<12}{'128.0':<10}  (baseline, resolution=0.3)")
    for r in rows:
        print(f"{r['k_eig']:<8}{r['ari']:<10.4f}{r['n_clusters']:<12}{r['elapsed']:<10.1f}")
    print("=" * 60)

    best = max(rows, key=lambda r: r["ari"])
    print(f"\nEn iyi GFT k_eig: {best['k_eig']} (ARI={best['ari']:.4f}, n_clusters={best['n_clusters']})")

    adata.write_h5ad(PATH)
    print(f"\nSaved adata (X_gft from last k_eig={K_EIG_VALUES[-1]}, all leiden_gft_keig* columns) to {PATH}")


if __name__ == "__main__":
    main()
