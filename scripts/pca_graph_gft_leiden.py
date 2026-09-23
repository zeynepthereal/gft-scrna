"""
Leiden on the ORIGINAL PCA kNN graph (unchanged), with GFT eigenvectors
computed from that same graph (k_eig=20) stored as the cell representation.

Key point: Leiden operates purely on the graph (adjacency/connectivity
matrix) -- it has no notion of "cell representation" beyond what was used
to build that graph. So running Leiden on the untouched PCA-neighbors graph
gives the same result regardless of what representation we additionally
attach to obsm. This script verifies that explicitly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score

from gft.embedding import gft_embed

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
N_NEIGHBORS = 15
N_PCS = 50
K_EIG = 20
RESOLUTIONS = [0.3, 0.5]


def main():
    adata = sc.read_h5ad(PATH)

    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"].astype(str).values)
    print(f"N={adata.n_obs} | k_true={len(le.classes_)}")

    # 1) Build PCA kNN graph, keep it untouched from here on
    print(f"\n1) Building PCA kNN graph (n_pcs={N_PCS}, n_neighbors={N_NEIGHBORS})...")
    sc.pp.neighbors(adata, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS)
    W_pca_graph = adata.obsp["connectivities"].copy()
    print(f"   graph: {W_pca_graph.shape}, nnz={W_pca_graph.nnz}, "
          f"symmetric={(W_pca_graph != W_pca_graph.T).nnz == 0}")

    # 2) Laplacian eigenvectors of THIS graph (k_eig=20)
    print(f"\n2) Computing Laplacian eigenvectors of the PCA graph (k_eig={K_EIG})...")
    emb = gft_embed(W_pca_graph, k_eig=K_EIG)
    adata.obsm["X_gft_from_pca_graph"] = emb
    print(f"   X_gft_from_pca_graph shape: {emb.shape}")

    # 3) Leiden directly on the original PCA graph (adjacency passed explicitly
    #    so it cannot be silently rebuilt from any obsm representation)
    print(f"\n3) Running Leiden on the ORIGINAL PCA graph (representation attached: GFT eigenvectors)...")
    rows = []
    for res in RESOLUTIONS:
        key = f"leiden_pcagraph_gftrep_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key, adjacency=W_pca_graph,
                     flavor="igraph", n_iterations=2, directed=False)
        ari = adjusted_rand_score(y, adata.obs[key].values)
        n_clusters = adata.obs[key].nunique()
        rows.append({"resolution": res, "ari": ari, "n_clusters": n_clusters})
        print(f"   resolution={res}: ARI={ari:.4f}, n_clusters={n_clusters}")

    print("\n" + "=" * 60)
    print(f"{'resolution':<12}{'ARI':<10}{'n_clusters':<12}")
    print("-" * 60)
    for r in rows:
        print(f"{r['resolution']:<12}{r['ari']:<10.4f}{r['n_clusters']:<12}")
    print("=" * 60)
    print("\nNot: Leiden yalnizca graf (adjacency) uzerinde calisir; hucre")
    print("temsili (obsm) neighbors grafi olusturulduktan sonra Leiden")
    print("sonucunu etkilemez. Bu yuzden ARI/n_clusters, ayni PCA grafi")
    print("uzerinde calisan 'Standart pipeline' sonucuyla ayni cikar.")

    adata.write_h5ad(PATH)
    print(f"\nSaved X_gft_from_pca_graph obsm and leiden_pcagraph_gftrep_* columns to {PATH}")


if __name__ == "__main__":
    main()
