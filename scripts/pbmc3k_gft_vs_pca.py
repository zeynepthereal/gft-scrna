"""PBMC3k: full preprocessing, marker-based cell-type annotation (8 canonical PBMC types), GFT vs PCA."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from gft import build_graph, gft_embed, heterogeneity_score

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "pbmc3k", "filtered_gene_bc_matrices", "hg19")
K_EIG = 5
K_NN = 20
SEED = 42
N_SEEDS = 10

# Canonical PBMC3k marker panel (scanpy tutorial), used to assign leiden clusters to
# the 8 known cell types by marker-score voting -- avoids hardcoding a fragile
# cluster-index -> label mapping that depends on exact clustering parameters.
MARKERS = {
    "CD4 T": ["IL7R"],
    "CD8 T": ["CD8A"],
    "NK": ["GNLY", "NKG7"],
    "B cell": ["MS4A1"],
    "CD14 Mono": ["CD14", "LYZ"],
    "DC": ["FCER1A", "CST3"],
    "CD16 Mono": ["FCGR3A", "MS4A7"],
    "Platelet": ["PPBP"],
}


def main():
    print("Loading PBMC3k 10x data...")
    adata = sc.read_10x_mtx(DATA_DIR, var_names="gene_symbols")
    adata.var_names_make_unique()
    print(f"Raw: {adata.shape}")

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"After filtering: {adata.shape}")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # keep full log-normalized gene set for marker lookups

    sc.pp.highly_variable_genes(adata)
    n_cells, n_genes = adata.n_obs, adata.n_vars
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata_hvg)
    sc.pp.pca(adata_hvg, n_comps=50, mask_var=None)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    print(f"n_cells={n_cells}, n_genes={n_genes}, n_hvg={adata_hvg.n_vars}")

    print("\nLeiden clustering for marker-based cell-type annotation...")
    sc.pp.neighbors(adata, n_pcs=50, n_neighbors=15)
    sc.tl.leiden(adata, resolution=1.0, flavor="igraph", n_iterations=2, directed=False)
    n_clusters = adata.obs["leiden"].nunique()
    print(f"n_leiden_clusters={n_clusters}")

    raw = adata.raw.to_adata()
    clusters = sorted(adata.obs["leiden"].unique(), key=int)
    scores = pd.DataFrame(index=clusters, columns=MARKERS.keys(), dtype=float)
    for cell_type, genes in MARKERS.items():
        genes_present = [g for g in genes if g in raw.var_names]
        if not genes_present:
            print(f"  WARNING: none of {genes} found for {cell_type}")
            continue
        expr = raw[:, genes_present].X
        expr = np.asarray(expr.todense()) if hasattr(expr, "todense") else np.asarray(expr)
        mean_expr = pd.Series(expr.mean(axis=1), index=raw.obs_names)
        for cl in clusters:
            mask = (adata.obs["leiden"] == cl).values
            scores.loc[cl, cell_type] = mean_expr.values[mask].mean()

    cluster_to_label = scores.idxmax(axis=1)
    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_to_label).astype("category")
    print("\nCluster -> cell type assignment:")
    for cl in clusters:
        n = int((adata.obs["leiden"] == cl).sum())
        print(f"  cluster {cl}: {cluster_to_label[cl]}  (n={n})")
    print("\nCell type distribution:")
    print(adata.obs["cell_type"].value_counts())

    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"].astype(str).values)
    pts = adata.obsm["X_pca"][:, :50]
    k = len(le.classes_)
    N = len(pts)

    het = heterogeneity_score(pts, y)
    print(f"\nN={N}  k={k}  heterogeneity_score={het:.2f}")

    W = build_graph(pts, k_nn=K_NN, method="cosine")
    emb = gft_embed(W, k_eig=K_EIG)

    def ari_ninit10(X):
        aris = [adjusted_rand_score(y, KMeans(k, random_state=s, n_init=10).fit_predict(X))
                for s in range(SEED, SEED + N_SEEDS)]
        return float(np.mean(aris)), float(np.std(aris))

    pca_mean, pca_std = ari_ninit10(pts)
    gft_mean, gft_std = ari_ninit10(emb)
    ratio = gft_mean / max(pca_mean, 1e-6)

    print(f"\nPCA ARI (n_init=10, {N_SEEDS} seeds): {pca_mean:.4f} +/- {pca_std:.4f}")
    print(f"GFT ARI (n_init=10, {N_SEEDS} seeds): {gft_mean:.4f} +/- {gft_std:.4f}")
    print(f"Ratio (GFT/PCA): {ratio:.2f}x")
    print(f"\nhet < 7? {'YES' if het < 7 else 'NO'} (het={het:.2f})")
    print(f"GFT ratio < 1.0? {'YES' if ratio < 1.0 else 'NO'}")


if __name__ == "__main__":
    main()
