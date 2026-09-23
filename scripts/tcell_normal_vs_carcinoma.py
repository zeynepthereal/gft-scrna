"""Patient1: T cell spectral comparison, normal tissue vs carcinoma tissue (same cell type, same patient)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import scanpy as sc

from gft import build_graph
from gft.embedding import get_eigenvalues

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
K_NN = 15
K_MAX = 20
METRIC = "cosine"

GROUPS = {
    "Patient1 normal T cell": "GSM4904237",
    "Patient1 carcinoma T cell": "GSM4904236",
}


def analyze(pts):
    W = build_graph(pts, k_nn=K_NN, method=METRIC)
    eigs = get_eigenvalues(W, k_max=K_MAX)
    lambda2, lambda3 = eigs[0], eigs[1]
    spectral_gap = lambda3 - lambda2
    total_energy = eigs.sum()
    e5 = float(eigs[:5].sum() / total_energy)
    e10 = float(eigs[:10].sum() / total_energy)
    return {
        "n_cells": len(pts), "lambda2": float(lambda2), "lambda3": float(lambda3),
        "spectral_gap": float(spectral_gap), "E5": e5, "E10": e10,
        "first10": eigs[:10].tolist(),
    }


def main():
    adata = sc.read_h5ad(PATH)
    pts_all = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values
    cell_type = adata.obs["cell_type"].values

    results = {}
    for name, gsm_id in GROUPS.items():
        mask = (gsm == gsm_id) & (cell_type == "T cell")
        n = int(mask.sum())
        print(f"\n=== {name} ({gsm_id}) — n_cells={n} ===")
        r = analyze(pts_all[mask])
        results[name] = r
        print(f"  lambda2={r['lambda2']:.5f}  lambda3={r['lambda3']:.5f}  gap={r['spectral_gap']:.5f}")
        print(f"  E@5={r['E5']:.3f}  E@10={r['E10']:.3f}")
        print(f"  first10 eigenvalues = {[round(v,5) for v in r['first10']]}")

    print("\n" + "=" * 80)
    print(f"{'Group':<28}{'n_cells':<10}{'lambda2':<10}{'gap':<10}{'E@5':<8}{'E@10':<8}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<28}{r['n_cells']:<10}{r['lambda2']:<10.5f}{r['spectral_gap']:<10.5f}"
              f"{r['E5']:<8.3f}{r['E10']:<8.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
