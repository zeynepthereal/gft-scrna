"""Per-patient, per-tissue spectral analysis (controls for batch effect: single GSM sample per cell)."""
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
K_MAX = 30

SAMPLES = {
    ("Patient1", "normal"): "GSM4904237",
    ("Patient1", "adenoma"): "GSM4904235",
    ("Patient1", "carcinoma"): "GSM4904236",
    ("Patient2", "normal"): "GSM4904240",
    ("Patient2", "adenoma"): "GSM4904238",
    ("Patient2", "carcinoma"): "GSM4904239",
    ("Patient3", "normal"): "GSM4904246",
    ("Patient3", "adenoma"): "GSM4904242",  # adenoma_1
    ("Patient3", "carcinoma"): "GSM4904245",
}
TISSUE_ORDER = ["normal", "adenoma", "carcinoma"]


def analyze(pts: np.ndarray) -> dict:
    W = build_graph(pts, k_nn=K_NN, method=METRIC)
    eigs = get_eigenvalues(W, k_max=K_MAX)  # sorted, excludes trivial lambda_1=0

    lambda2 = eigs[0]
    lambda3 = eigs[1]
    spectral_gap = lambda3 - lambda2
    total_energy = eigs.sum()
    energy = {n: float(eigs[:n].sum() / total_energy) for n in (10, 20)}

    return {
        "n_cells": len(pts),
        "lambda2": float(lambda2),
        "lambda3": float(lambda3),
        "spectral_gap": float(spectral_gap),
        "energy": energy,
        "first10": eigs[:10].tolist(),
    }


def main():
    adata = sc.read_h5ad(PATH)
    pts_all = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values

    results = {}
    for (patient, tissue), gsm_id in SAMPLES.items():
        mask = gsm == gsm_id
        n = int(mask.sum())
        print(f"\n=== {patient} / {tissue} ({gsm_id}) — n_cells={n} ===")
        r = analyze(pts_all[mask])
        results[(patient, tissue)] = r
        print(f"  lambda2={r['lambda2']:.5f}  lambda3={r['lambda3']:.5f}  gap={r['spectral_gap']:.5f}")
        print(f"  E@10={r['energy'][10]:.3f}  E@20={r['energy'][20]:.3f}")

    print("\n" + "=" * 100)
    print(f"{'Patient':<10}{'Tissue':<12}{'n_cells':<10}{'lambda2':<10}{'gap(l3-l2)':<12}{'E@10':<8}{'E@20':<8}")
    print("-" * 100)
    for patient in ["Patient1", "Patient2", "Patient3"]:
        for tissue in TISSUE_ORDER:
            r = results[(patient, tissue)]
            print(f"{patient:<10}{tissue:<12}{r['n_cells']:<10}{r['lambda2']:<10.5f}"
                  f"{r['spectral_gap']:<12.5f}{r['energy'][10]:<8.3f}{r['energy'][20]:<8.3f}")
    print("=" * 100)

    print("\nMonotonic check (normal -> adenoma -> carcinoma, spectral gap):")
    for patient in ["Patient1", "Patient2", "Patient3"]:
        gaps = [results[(patient, t)]["spectral_gap"] for t in TISSUE_ORDER]
        is_monotonic = gaps[0] < gaps[1] < gaps[2]
        print(f"  {patient}: normal={gaps[0]:.5f} -> adenoma={gaps[1]:.5f} -> carcinoma={gaps[2]:.5f}  "
              f"{'MONOTONIC INCREASE' if is_monotonic else 'NOT monotonic'}")


if __name__ == "__main__":
    main()
