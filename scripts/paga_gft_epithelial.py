"""PAGA on Patient1 normal vs carcinoma epithelial cells, then GFT on the PAGA connectivity graph."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix, save_npz

from gft.embedding import get_eigenvalues

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
N_PCS = 50
N_NEIGHBORS = 15
K_EIG = 10
MIN_CLUSTERS = K_EIG + 2  # need enough PAGA nodes for k_eig=10 eigenvectors

GROUPS = {
    "normal": "GSM4904237",
    "carcinoma": "GSM4904236",
}


def leiden_enough_clusters(adata, min_clusters=MIN_CLUSTERS, start_res=1.0, max_tries=6):
    res = start_res
    for _ in range(max_tries):
        sc.tl.leiden(adata, resolution=res, key_added="leiden",
                     flavor="igraph", n_iterations=2, directed=False)
        n = adata.obs["leiden"].nunique()
        print(f"    resolution={res:.2f} -> n_clusters={n}")
        if n >= min_clusters:
            return n
        res += 0.5
    raise RuntimeError(f"Could not reach {min_clusters} clusters after {max_tries} tries")


def analyze_group(adata_all, gsm_id, name):
    print(f"\n=== {name} ({gsm_id}) ===")
    mask = (adata_all.obs["gsm_id"] == gsm_id) & (adata_all.obs["cell_type"] == "Epithelial/Tumor")
    adata = adata_all[mask].copy()
    print(f"  n_cells={adata.n_obs}")

    print("  Building neighbors + Leiden (enough clusters for k_eig=10 PAGA-GFT)...")
    sc.pp.neighbors(adata, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS)
    n_clusters = leiden_enough_clusters(adata)

    print("  Running PAGA...")
    sc.tl.paga(adata, groups="leiden")
    W = csr_matrix(adata.uns["paga"]["connectivities"])
    print(f"  PAGA connectivity matrix: {W.shape}, nnz={W.nnz}")

    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, f"paga_connectivity_{name}.npz")
    save_npz(npz_path, W)
    print(f"  Saved PAGA connectivity matrix -> {npz_path}")

    eigs = get_eigenvalues(W, k_max=K_EIG)
    lambda2, lambda3 = eigs[0], eigs[1]
    spectral_gap = float(lambda3 - lambda2)
    total_energy = eigs.sum()
    e5 = float(eigs[:5].sum() / total_energy)
    e10 = float(eigs[:10].sum() / total_energy)

    return {
        "name": name, "n_cells": adata.n_obs, "n_clusters": n_clusters,
        "eigenvalues_10": eigs[:10].tolist(), "lambda2": float(lambda2),
        "spectral_gap": spectral_gap, "E5": e5, "E10": e10,
    }


def main():
    adata_all = sc.read_h5ad(PATH)
    adata_all.raw = None  # not needed here; avoids costly raw copy on each subset
    results = {}
    for name, gsm_id in GROUPS.items():
        r = analyze_group(adata_all, gsm_id, name)
        results[name] = r
        print(f"  first10 eigenvalues = {[round(v,5) for v in r['eigenvalues_10']]}")
        print(f"  lambda2={r['lambda2']:.5f}  gap={r['spectral_gap']:.5f}  E@5={r['E5']:.3f}  E@10={r['E10']:.3f}")

    print("\n" + "=" * 100)
    print(f"{'Group':<12}{'n_cells':<10}{'n_clusters':<12}{'lambda2':<10}{'gap':<10}{'E@5':<8}{'E@10':<8}")
    print("-" * 100)
    for name, r in results.items():
        print(f"{name:<12}{r['n_cells']:<10}{r['n_clusters']:<12}{r['lambda2']:<10.5f}"
              f"{r['spectral_gap']:<10.5f}{r['E5']:<8.3f}{r['E10']:<8.3f}")
    print("=" * 100)

    print("\nFirst 10 PAGA-connectivity Laplacian eigenvalues:")
    for name, r in results.items():
        print(f"  {name:<12}: {[round(v,4) for v in r['eigenvalues_10']]}")


if __name__ == "__main__":
    main()
