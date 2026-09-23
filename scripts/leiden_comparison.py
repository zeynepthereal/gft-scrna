"""Leiden clustering comparison: raw PCA vs GFT embedding vs GFT-denoised, resolution sweep."""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score

from gft import build_graph, gft_embed, gft_denoise

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
K_EIG = 5
K_NN = 20
METRIC = "cosine"
N_NEIGHBORS = 15
N_PCS = 50
RESOLUTIONS = [0.3, 0.5, 0.7, 1.0]


def sweep(adata, y, use_rep=None, n_pcs=None, key_prefix="leiden"):
    t0 = time.time()
    if use_rep is not None:
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=N_NEIGHBORS)
    else:
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=N_NEIGHBORS)

    results = []
    for res in RESOLUTIONS:
        key = f"{key_prefix}_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key, flavor="igraph", n_iterations=2, directed=False)
        ari = adjusted_rand_score(y, adata.obs[key].values)
        n_clusters = adata.obs[key].nunique()
        results.append({"resolution": res, "ari": ari, "n_clusters": n_clusters})
        print(f"    resolution={res}: ARI={ari:.4f}, n_clusters={n_clusters}")

    elapsed = time.time() - t0
    best = max(results, key=lambda r: r["ari"])
    return {"results": results, "best": best, "elapsed": elapsed}


def main():
    adata = sc.read_h5ad(PATH)
    pts = adata.obsm["X_pca"][:, :50]

    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"].astype(str).values)
    print(f"N={len(pts)} | k_true={len(le.classes_)} | classes={list(le.classes_)}")

    print("\nBuilding cosine kNN graph (k_nn=20) + GFT embedding (k_eig=5)...")
    W = build_graph(pts, k_nn=K_NN, method=METRIC)
    emb = gft_embed(W, k_eig=K_EIG)
    Xhat = gft_denoise(pts, emb)

    adata.obsm["X_gft"] = emb
    adata.obsm["X_gft_denoised"] = Xhat

    print("\n=== 1) Standard pipeline: raw PCA -> neighbors(n_pcs=50) -> leiden ===")
    r1 = sweep(adata, y, use_rep=None, n_pcs=N_PCS, key_prefix="leiden_pca")

    print("\n=== 2) GFT embedding -> neighbors(use_rep='X_gft') -> leiden ===")
    r2 = sweep(adata, y, use_rep="X_gft", key_prefix="leiden_gft")

    print("\n=== 3) Denoised matrix (X_hat = U U^T X) -> neighbors -> leiden ===")
    r3 = sweep(adata, y, use_rep="X_gft_denoised", key_prefix="leiden_denoised")

    print("\n" + "=" * 70)
    for name, r in [("1) Standart (raw PCA)", r1), ("2) GFT embedding", r2), ("3) Denoised (UU^T X)", r3)]:
        print(f"  {name}")
        print(f"     En iyi ARI       : {r['best']['ari']:.4f} (resolution={r['best']['resolution']})")
        print(f"     Küme sayısı      : {r['best']['n_clusters']}")
        print(f"     Süre             : {r['elapsed']:.1f} sn")
    print("=" * 70)

    adata.write_h5ad(PATH)
    print(f"\nSaved adata with X_gft, X_gft_denoised obsm and leiden_* columns to {PATH}")


if __name__ == "__main__":
    main()
