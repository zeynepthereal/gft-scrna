"""Control for sample-size effect: subsample Patient1 normal T cells to N=396, compare vs full N and vs carcinoma."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc

from gft import build_graph
from gft.embedding import get_eigenvalues

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
K_NN = 15
K_MAX = 20
METRIC = "cosine"
SEED = 42

NORMAL_GSM = "GSM4904237"
CARCINOMA_GSM = "GSM4904236"


def analyze(pts):
    W = build_graph(pts, k_nn=K_NN, method=METRIC)
    eigs = get_eigenvalues(W, k_max=K_MAX)
    lambda2, lambda3 = eigs[0], eigs[1]
    spectral_gap = lambda3 - lambda2
    total_energy = eigs.sum()
    e5 = float(eigs[:5].sum() / total_energy)
    e10 = float(eigs[:10].sum() / total_energy)
    return {
        "n_cells": len(pts), "lambda2": float(lambda2), "spectral_gap": float(spectral_gap),
        "E5": e5, "E10": e10,
    }


def main():
    adata = sc.read_h5ad(PATH)
    pts_all = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values
    cell_type = adata.obs["cell_type"].values

    normal_mask = (gsm == NORMAL_GSM) & (cell_type == "T cell")
    carcinoma_mask = (gsm == CARCINOMA_GSM) & (cell_type == "T cell")

    pts_normal_full = pts_all[normal_mask]
    pts_carcinoma = pts_all[carcinoma_mask]

    n_sub = pts_carcinoma.shape[0]
    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(pts_normal_full.shape[0], size=n_sub, replace=False)
    pts_normal_sub = pts_normal_full[sub_idx]

    print(f"Normal T cell full: N={pts_normal_full.shape[0]}")
    print(f"Normal T cell subsample: N={pts_normal_sub.shape[0]} (seed={SEED})")
    print(f"Carcinoma T cell: N={pts_carcinoma.shape[0]}")

    rows = {
        "Normal T cell (N=2234, full)": analyze(pts_normal_full),
        "Normal T cell (N=396, subsample)": analyze(pts_normal_sub),
        "Carcinoma T cell (N=396)": analyze(pts_carcinoma),
    }

    print("\n" + "=" * 90)
    print(f"{'Group':<34}{'N':<8}{'lambda2':<10}{'gap':<10}{'E@5':<8}{'E@10':<8}")
    print("-" * 90)
    for name, r in rows.items():
        print(f"{name:<34}{r['n_cells']:<8}{r['lambda2']:<10.5f}{r['spectral_gap']:<10.5f}"
              f"{r['E5']:<8.3f}{r['E10']:<8.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
