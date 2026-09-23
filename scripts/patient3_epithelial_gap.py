"""Patient3: Epithelial/Tumor spectral gap, normal vs carcinoma, N-matched."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc

from gft import build_graph
from gft.embedding import get_eigenvalues

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
K_NN = 15
K_MAX = 10
METRIC = "cosine"
SEED = 42

NORMAL_GSM = "GSM4904246"
CARCINOMA_GSM = "GSM4904245"
CELL_TYPE = "Epithelial/Tumor"


def spectral_gap(pts):
    W = build_graph(pts, k_nn=K_NN, method=METRIC)
    eigs = get_eigenvalues(W, k_max=K_MAX)
    return float(eigs[1] - eigs[0])


def main():
    adata = sc.read_h5ad(PATH)
    pts_all = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values
    cell_type = adata.obs["cell_type"].values
    rng = np.random.default_rng(SEED)

    mask_n = (gsm == NORMAL_GSM) & (cell_type == CELL_TYPE)
    mask_c = (gsm == CARCINOMA_GSM) & (cell_type == CELL_TYPE)
    pts_n, pts_c = pts_all[mask_n], pts_all[mask_c]
    print(f"Normal Epithelial/Tumor: N={len(pts_n)}")
    print(f"Carcinoma Epithelial/Tumor: N={len(pts_c)}")

    N = min(len(pts_n), len(pts_c))
    idx_n = rng.choice(len(pts_n), size=N, replace=False)
    idx_c = rng.choice(len(pts_c), size=N, replace=False)

    gap_n = spectral_gap(pts_n[idx_n])
    gap_c = spectral_gap(pts_c[idx_c])
    ratio = gap_c / gap_n if gap_n > 0 else float("inf")

    print("\n" + "=" * 70)
    print(f"{'Hasta':<10}{'Hucre tipi':<20}{'N':<8}{'normal gap':<14}{'carcinoma gap':<16}{'oran':<8}")
    print("-" * 70)
    print(f"{'Patient3':<10}{CELL_TYPE:<20}{N:<8}{gap_n:<14.5f}{gap_c:<16.5f}{ratio:<8.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
