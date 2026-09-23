"""Inspect GSE161277_carcinoma.h5ad: obs columns, cell-type info, clustering, obsm."""
import scanpy as sc
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "GSE161277_carcinoma.h5ad"

adata = sc.read_h5ad(PATH)

print("=== adata ===")
print(adata)

print("\n=== obs columns ===")
print(list(adata.obs.columns))

cell_type_cols = [c for c in adata.obs.columns if any(k in c.lower() for k in ["cell_type", "celltype", "cluster", "leiden", "louvain"])]
print(f"\nCell-type-like columns found: {cell_type_cols if cell_type_cols else 'NONE'}")

print("\n=== obsm keys (before) ===")
print(list(adata.obsm.keys()))

if not cell_type_cols:
    print("\nNo cell type info found -> running neighbors + leiden (resolution=0.5)")
    sc.pp.neighbors(adata, n_pcs=50)
    sc.tl.leiden(adata, resolution=0.5)
    n_clusters = adata.obs["leiden"].nunique()
    print(f"\nn_leiden_clusters={n_clusters}")
    print(adata.obs["leiden"].value_counts())
    adata.write_h5ad(PATH)
    print(f"\nSaved leiden labels back to {PATH}")

print("\n=== obsm keys (after) ===")
print(list(adata.obsm.keys()))
