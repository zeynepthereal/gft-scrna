"""External CRC epithelial dataset: per-donor spectral gap, normal vs colorectal cancer, N-matched."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc

from gft.graph import build_graph
from gft.embedding import get_eigenvalues

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "crc_epithelial.h5ad")
N_COMPS = 50
K_NN = 15
K_MAX = 10
METRIC = "cosine"
SEED = 42
MIN_N = K_NN + 5  # need enough cells for a stable k=15 graph + a handful of eigenvalues


def spectral_gap(pts):
    W = build_graph(pts, k_nn=K_NN, method=METRIC)
    eigs = get_eigenvalues(W, k_max=min(K_MAX, len(pts) - 3))
    return float(eigs[1] - eigs[0])


def main():
    adata = sc.read_h5ad(PATH)
    print(adata)

    print(f"\n1) PCA (n_comps={N_COMPS}) on log-normalized X (all {adata.n_vars} genes, no HVG mask)...")
    sc.pp.pca(adata, n_comps=N_COMPS, mask_var=None)
    pts_all = adata.obsm["X_pca"][:, :N_COMPS]

    donor = adata.obs["donor_id"].values
    disease = adata.obs["disease"].values

    donors = sorted(adata.obs["donor_id"].unique())
    rng = np.random.default_rng(SEED)

    rows = []
    for d in donors:
        mask_n = (donor == d) & (disease == "normal")
        mask_c = (donor == d) & (disease == "colorectal cancer")
        n_normal, n_cancer = int(mask_n.sum()), int(mask_c.sum())

        if n_normal == 0 or n_cancer == 0:
            print(f"SKIP {d}: missing one condition (normal={n_normal}, cancer={n_cancer})")
            continue

        N = min(n_normal, n_cancer)
        if N < MIN_N:
            print(f"SKIP {d}: N={N} too small (normal={n_normal}, cancer={n_cancer})")
            continue

        pts_n_full, pts_c_full = pts_all[mask_n], pts_all[mask_c]
        idx_n = rng.choice(n_normal, size=N, replace=False)
        idx_c = rng.choice(n_cancer, size=N, replace=False)

        gap_n = spectral_gap(pts_n_full[idx_n])
        gap_c = spectral_gap(pts_c_full[idx_c])
        ratio = gap_c / gap_n if gap_n > 0 else float("inf")

        rows.append({"donor": d, "N": N, "n_normal_full": n_normal, "n_cancer_full": n_cancer,
                     "gap_normal": gap_n, "gap_cancer": gap_c, "ratio": ratio})
        print(f"{d}: N={N} (full normal={n_normal}, full cancer={n_cancer})  "
              f"gap_normal={gap_n:.5f}  gap_cancer={gap_c:.5f}  ratio={ratio:.2f}x")

    print("\n" + "=" * 90)
    print(f"{'Donor':<14}{'N':<8}{'normal gap':<14}{'cancer gap':<14}{'oran':<8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['donor']:<14}{r['N']:<8}{r['gap_normal']:<14.5f}{r['gap_cancer']:<14.5f}{r['ratio']:<8.2f}")
    print("=" * 90)

    n_cancer_higher = sum(1 for r in rows if r["ratio"] > 1)
    print(f"\nCancer gap > Normal gap: {n_cancer_higher}/{len(rows)} donor")


if __name__ == "__main__":
    main()
