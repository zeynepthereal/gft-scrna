"""PAGA-GFT analysis for Patient2 and Patient3: normal vs carcinoma epithelial cells."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import scanpy as sc
from scipy.sparse import csr_matrix, save_npz

from gft.embedding import get_eigenvalues

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
N_PCS = 50
N_NEIGHBORS = 15
K_EIG = 10
MIN_CLUSTERS = K_EIG + 2

SAMPLES = {
    ("Patient2", "normal"): "GSM4904240",
    ("Patient2", "carcinoma"): "GSM4904239",
    ("Patient3", "normal"): "GSM4904246",
    ("Patient3", "carcinoma"): "GSM4904245",
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


def analyze_group(adata_all, gsm_id, patient, tissue):
    name = f"{patient}_{tissue}"
    print(f"\n=== {name} ({gsm_id}) ===")
    mask = (adata_all.obs["gsm_id"] == gsm_id) & (adata_all.obs["cell_type"] == "Epithelial/Tumor")
    adata = adata_all[mask].copy()
    print(f"  n_cells={adata.n_obs}")

    sc.pp.neighbors(adata, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS)
    n_clusters = leiden_enough_clusters(adata)

    sc.tl.paga(adata, groups="leiden")
    W = csr_matrix(adata.uns["paga"]["connectivities"])
    print(f"  PAGA connectivity matrix: {W.shape}, nnz={W.nnz}")

    os.makedirs(OUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUT_DIR, f"paga_connectivity_{name}.npz")
    save_npz(npz_path, W)
    print(f"  Saved -> {npz_path}")

    eigs = get_eigenvalues(W, k_max=K_EIG)
    lambda2, lambda3 = eigs[0], eigs[1]
    spectral_gap = float(lambda3 - lambda2)
    total_energy = eigs.sum()
    e5 = float(eigs[:5].sum() / total_energy)

    print(f"  first10 eigenvalues = {[round(v,5) for v in eigs[:10].tolist()]}")
    print(f"  lambda2={lambda2:.5f}  gap={spectral_gap:.5f}  E@5={e5:.3f}")

    return {"patient": patient, "tissue": tissue, "n_cells": adata.n_obs,
            "n_clusters": n_clusters, "lambda2": float(lambda2),
            "spectral_gap": spectral_gap, "E5": e5}


def main():
    adata_all = sc.read_h5ad(PATH)
    adata_all.raw = None

    rows = []
    for (patient, tissue), gsm_id in SAMPLES.items():
        r = analyze_group(adata_all, gsm_id, patient, tissue)
        rows.append(r)

    print("\n" + "=" * 90)
    print(f"{'Hasta':<10}{'Grup':<12}{'Kume sayisi':<14}{'spectral gap':<14}{'lambda2':<10}{'E@5':<8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['patient']:<10}{r['tissue']:<12}{r['n_clusters']:<14}{r['spectral_gap']:<14.5f}"
              f"{r['lambda2']:<10.5f}{r['E5']:<8.3f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
