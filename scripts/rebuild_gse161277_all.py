"""Rebuild data/GSE161277_all.h5ad from data/GSE161277/ (extracted GSE161277_RAW.tar).

Faithful merge of the two original scripts, run back to back:
  scripts/load_all.py            -> load 13 samples, filter, normalize, log1p, HVG, raw, scale, PCA
  scripts/cluster_annotate_all.py -> neighbors, Leiden(res=0.5), marker-based cell_type

Paths are relative to the repo root. Leiden random_state defaults to the value the original
script used implicitly (scanpy default, 0); override with --leiden-seed.
"""
import argparse
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "GSE161277"
OUT_PATH = REPO / "data" / "GSE161277_all.h5ad"

EXPECTED_SAMPLES = {
    "GSM4904234": "Patient0_carcinoma",
    "GSM4904235": "Patient1_adenoma",
    "GSM4904236": "Patient1_carcinoma",
    "GSM4904237": "Patient1_normal",
    "GSM4904238": "Patient2_adenoma",
    "GSM4904239": "Patient2_carcinoma",
    "GSM4904240": "Patient2_normal",
    "GSM4904241": "Patient2_para-cancer",
    "GSM4904242": "Patient3_adenoma_1",
    "GSM4904243": "Patient3_adenoma_2",
    "GSM4904244": "Patient3_blood",
    "GSM4904245": "Patient3_carcinoma",
    "GSM4904246": "Patient3_normal",
}

MARKERS = {
    "Epithelial/Tumor": ["EPCAM", "CDH1", "KRT18", "KRT19", "KRT20"],
    "T cell": ["CD3D", "CD3E", "CD8A", "CD4"],
    "B cell": ["CD19", "MS4A1", "CD79A"],
    "Macrophage/Monocyte": ["CD68", "CD14", "LYZ", "CSF1R"],
    "Fibroblast/CAF": ["COL1A1", "COL1A2", "FAP", "ACTA2"],
    "Endothelial": ["PECAM1", "VWF", "CDH5"],
    "NK cell": ["GNLY", "NKG7", "NCAM1"],
}


def discover_samples() -> dict[str, str]:
    samples = {}
    for f in sorted(DATA_DIR.glob("GSM*_matrix.mtx.gz")):
        prefix = f.name.replace("_matrix.mtx.gz", "")
        gsm_id, label = prefix.split("_", 1)
        samples[gsm_id] = label
    assert samples == EXPECTED_SAMPLES, f"sample set mismatch: {samples}"
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


def preprocess(adata: ad.AnnData) -> ad.AnnData:
    """Same steps and order as scripts/load_all.py."""
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    print(f"n_cells={adata.n_obs}  n_genes={adata.n_vars}")

    adata.raw = adata  # keep full log-normalized gene set for marker lookups
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=50, mask_var=None)
    print(f"n_hvg_used_for_scale_pca={adata.n_vars}")
    return adata


def annotate(adata: ad.AnnData, key: str = "leiden") -> pd.Series:
    """Same scoring as scripts/cluster_annotate_all.py: per-cluster mean of per-cell mean marker
    expression (adata.raw), argmax -> label. Returns the cluster -> label map."""
    raw = adata.raw.to_adata()
    symbol_to_id = dict(zip(raw.var["gene_symbol"], raw.var_names))
    clusters = sorted(adata.obs[key].unique(), key=int)
    scores = pd.DataFrame(index=clusters, columns=MARKERS.keys(), dtype=float)
    for cell_type, symbols in MARKERS.items():
        gene_ids = [symbol_to_id[s] for s in symbols if s in symbol_to_id]
        missing = [s for s in symbols if s not in symbol_to_id]
        if missing:
            print(f"WARNING: markers not found for {cell_type}: {missing}")
        expr = raw[:, gene_ids].X
        expr = expr.toarray() if hasattr(expr, "toarray") else expr
        mean_expr = pd.Series(expr.mean(axis=1), index=raw.obs_names)
        for cl in clusters:
            scores.loc[cl, cell_type] = mean_expr[(adata.obs[key] == cl).values].mean()
    print("\n=== Mean marker-set expression per cluster (log-normalized) ===")
    print(scores.round(3))
    return scores.idxmax(axis=1)


def cluster_and_annotate(adata: ad.AnnData, leiden_seed: int) -> ad.AnnData:
    sc.pp.neighbors(adata, n_pcs=50)
    sc.tl.leiden(adata, resolution=0.5, random_state=leiden_seed)
    print(f"n_leiden_clusters={adata.obs['leiden'].nunique()}")

    cluster_to_label = annotate(adata)
    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_to_label).astype("category")
    print("\n=== Cluster -> assigned label ===")
    for cl, lab in cluster_to_label.items():
        print(f"  cluster {cl}: {lab}  (n={(adata.obs['leiden'] == cl).sum()})")
    print("\n=== Cell type label distribution ===")
    print(adata.obs["cell_type"].value_counts())
    return adata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leiden-seed", type=int, default=0,
                    help="Leiden random_state (original script used scanpy default = 0)")
    args = ap.parse_args()

    samples = discover_samples()
    print(f"Found {len(samples)} samples")
    adatas = [load_sample(g, lab) for g, lab in samples.items()]
    adata = ad.concat(adatas, join="outer", index_unique=None)
    adata.var = adatas[0].var.reindex(adata.var_names)

    adata = preprocess(adata)
    adata = cluster_and_annotate(adata, args.leiden_seed)
    adata.uns["rebuild_leiden_seed"] = args.leiden_seed

    adata.write_h5ad(OUT_PATH)
    print(f"\nsaved to {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
