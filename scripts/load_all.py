"""Load all GSE161277 samples (carcinoma, normal, adenoma, para-cancer, blood), concatenate, and preprocess."""
import scanpy as sc
import anndata as ad
import pandas as pd
import scipy.io
import scipy.sparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "GSE161277"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "GSE161277_all.h5ad"


def discover_samples() -> dict[str, str]:
    samples = {}
    for f in sorted(DATA_DIR.glob("GSM*_matrix.mtx.gz")):
        prefix = f.name.replace("_matrix.mtx.gz", "")
        gsm_id, label = prefix.split("_", 1)
        samples[gsm_id] = label
    return samples


def load_sample(gsm_id: str, label: str) -> ad.AnnData:
    prefix = f"{gsm_id}_{label}"
    matrix = scipy.io.mmread(DATA_DIR / f"{prefix}_matrix.mtx.gz").T.tocsr()
    barcodes = pd.read_csv(DATA_DIR / f"{prefix}_barcodes.tsv.gz", header=None, sep="\t")[0].values
    features = pd.read_csv(DATA_DIR / f"{prefix}_features.tsv.gz", header=None, sep="\t")

    adata = ad.AnnData(X=scipy.sparse.csr_matrix(matrix, dtype="float32"))
    adata.obs_names = [f"{gsm_id}_{bc}" for bc in barcodes]
    adata.var_names = features[0].values
    adata.var["gene_symbol"] = features[1].values
    adata.obs["sample"] = label
    adata.obs["gsm_id"] = gsm_id
    adata.var_names_make_unique()
    return adata


def main():
    samples = discover_samples()
    print(f"Found {len(samples)} samples: {samples}")

    adatas = [load_sample(gsm_id, label) for gsm_id, label in samples.items()]
    adata = ad.concat(adatas, join="outer", index_unique=None)
    adata.var = adatas[0].var.reindex(adata.var_names)

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    n_cells, n_genes = adata.n_obs, adata.n_vars

    adata.raw = adata  # keep full log-normalized gene set for marker lookups
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=50, mask_var=None)  # already subsetted to HVG; avoid redundant masked copy

    adata.write_h5ad(OUT_PATH)

    print(f"n_cells={n_cells}")
    print(f"n_genes={n_genes}")
    print(f"n_hvg_used_for_scale_pca={adata.n_vars}")
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
