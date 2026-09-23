"""Leiden clustering + marker-based cell-type annotation for GSE161277_all.h5ad."""
import scanpy as sc
import pandas as pd
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "GSE161277_all.h5ad"

MARKERS = {
    "Epithelial/Tumor": ["EPCAM", "CDH1", "KRT18", "KRT19", "KRT20"],
    "T cell": ["CD3D", "CD3E", "CD8A", "CD4"],
    "B cell": ["CD19", "MS4A1", "CD79A"],
    "Macrophage/Monocyte": ["CD68", "CD14", "LYZ", "CSF1R"],
    "Fibroblast/CAF": ["COL1A1", "COL1A2", "FAP", "ACTA2"],
    "Endothelial": ["PECAM1", "VWF", "CDH5"],
    "NK cell": ["GNLY", "NKG7", "NCAM1"],
}


def main():
    adata = sc.read_h5ad(PATH)

    sc.pp.neighbors(adata, n_pcs=50)
    sc.tl.leiden(adata, resolution=0.5)
    n_clusters = adata.obs["leiden"].nunique()
    print(f"n_leiden_clusters={n_clusters}")
    print(adata.obs["leiden"].value_counts())

    raw = adata.raw.to_adata()
    symbol_to_id = dict(zip(raw.var["gene_symbol"], raw.var_names))

    clusters = sorted(adata.obs["leiden"].unique(), key=int)
    cluster_scores = pd.DataFrame(index=clusters, columns=MARKERS.keys(), dtype=float)

    for cell_type, symbols in MARKERS.items():
        gene_ids = [symbol_to_id[s] for s in symbols if s in symbol_to_id]
        missing = [s for s in symbols if s not in symbol_to_id]
        if missing:
            print(f"WARNING: markers not found for {cell_type}: {missing}")
        expr = raw[:, gene_ids].X
        expr = expr.toarray() if hasattr(expr, "toarray") else expr
        mean_expr = pd.Series(expr.mean(axis=1), index=raw.obs_names)
        for cl in clusters:
            mask = adata.obs["leiden"] == cl
            cluster_scores.loc[cl, cell_type] = mean_expr[mask.values].mean()

    print("\n=== Mean marker-set expression per cluster (log-normalized) ===")
    print(cluster_scores.round(3))

    cluster_to_label = cluster_scores.idxmax(axis=1)
    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_to_label).astype("category")

    print("\n=== Cluster -> assigned label ===")
    for cl in clusters:
        print(f"  cluster {cl}: {cluster_to_label[cl]}  (n={sum(adata.obs['leiden'] == cl)})")

    print("\n=== Cell type label distribution ===")
    counts = adata.obs["cell_type"].value_counts()
    print(counts)
    print(f"\nn_cell_types_present={counts[counts > 0].shape[0]}")

    adata.write_h5ad(PATH)
    print(f"\nSaved updated adata with 'leiden' and 'cell_type' columns to {PATH}")


if __name__ == "__main__":
    main()
