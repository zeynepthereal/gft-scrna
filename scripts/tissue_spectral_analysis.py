"""Per-tissue-type spectral analysis: cosine kNN graph -> normalized Laplacian -> eigenvalue spectrum."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc

from gft import build_graph
from gft.embedding import get_eigenvalues

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
K_NN = 20
METRIC = "cosine"
K_MAX = 50

TISSUE_GROUPS = {
    "Normal": ["GSM4904237", "GSM4904240", "GSM4904246"],
    "Adenoma": ["GSM4904235", "GSM4904238", "GSM4904242", "GSM4904243"],
    "Carcinoma": ["GSM4904234", "GSM4904236", "GSM4904239", "GSM4904245"],
    "Para-cancer": ["GSM4904241"],
    "Blood": ["GSM4904244"],
}


def analyze_group(pts: np.ndarray) -> dict:
    W = build_graph(pts, k_nn=K_NN, method=METRIC)
    eigs = get_eigenvalues(W, k_max=K_MAX)  # sorted, 50 values, excludes trivial lambda_1=0

    lambda2 = eigs[0]
    spectral_gap = eigs[1] - eigs[0]
    total_energy = eigs.sum()
    energy = {
        n: float(eigs[:n].sum() / total_energy) for n in (5, 10, 20)
    }

    return {
        "n_cells": len(pts),
        "first10": eigs[:10].tolist(),
        "lambda2": float(lambda2),
        "spectral_gap": float(spectral_gap),
        "energy": energy,
    }


def main():
    adata = sc.read_h5ad(PATH)
    pts_all = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values

    results = {}
    for tissue, gsm_ids in TISSUE_GROUPS.items():
        mask = np.isin(gsm, gsm_ids)
        n = mask.sum()
        print(f"\n=== {tissue} (GSM: {gsm_ids}) — n_cells={n} ===")
        pts = pts_all[mask]
        r = analyze_group(pts)
        results[tissue] = r
        print(f"  lambda2 (alg. connectivity) = {r['lambda2']:.5f}")
        print(f"  spectral gap (l3-l2)        = {r['spectral_gap']:.5f}")
        print(f"  energy@5/10/20              = {r['energy'][5]:.3f} / {r['energy'][10]:.3f} / {r['energy'][20]:.3f}")
        print(f"  first10 eigenvalues          = {[round(v,5) for v in r['first10']]}")

    print("\n" + "=" * 100)
    print(f"{'Tissue':<14}{'n_cells':<10}{'lambda2':<10}{'gap(l3-l2)':<12}{'E@5':<8}{'E@10':<8}{'E@20':<8}")
    print("-" * 100)
    for tissue, r in results.items():
        print(f"{tissue:<14}{r['n_cells']:<10}{r['lambda2']:<10.5f}{r['spectral_gap']:<12.5f}"
              f"{r['energy'][5]:<8.3f}{r['energy'][10]:<8.3f}{r['energy'][20]:<8.3f}")
    print("=" * 100)

    print("\nFirst 10 eigenvalues per tissue:")
    for tissue, r in results.items():
        print(f"  {tissue:<14}: {[round(v,4) for v in r['first10']]}")


if __name__ == "__main__":
    main()
