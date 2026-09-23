"""Malignant-cell-only spectral gap per donor, plus a direct N-matched Primary-vs-Metastasis comparison."""
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
MIN_N = 50


def spectral_gap(pts):
    W = build_graph(pts, k_nn=K_NN, method=METRIC)
    eigs = get_eigenvalues(W, k_max=min(K_MAX, len(pts) - 3))
    return float(eigs[1] - eigs[0])


def tissue_label(n_primary, n_met, n_nontumor, total, pure_thresh=0.9):
    if n_primary / total >= pure_thresh:
        return "Primer"
    if n_met / total >= pure_thresh:
        return "Metastaz"
    return f"Karisik (P={n_primary/total:.0%}, M={n_met/total:.0%}, NT={n_nontumor/total:.0%})"


def main():
    adata = sc.read_h5ad(PATH)
    print(f"1) PCA (n_comps={N_COMPS}) on log-normalized X (no HVG mask)...")
    sc.pp.pca(adata, n_comps=N_COMPS, mask_var=None)
    pts_all = adata.obsm["X_pca"][:, :N_COMPS]

    is_mal = (adata.obs["cell_type"] == "malignant cell").values
    donor = adata.obs["donor_id"].values
    status = adata.obs["Tumor Status"].values

    donors = sorted(np.unique(donor[is_mal]))
    rng = np.random.default_rng(SEED)

    print("\n=== 1) Per-donor: malignant-cell-only spectral gap ===")
    rows = []
    for d in donors:
        mask = is_mal & (donor == d)
        N = int(mask.sum())
        if N < MIN_N:
            print(f"SKIP {d}: N={N} < {MIN_N}")
            continue
        n_primary = int((status[mask] == "Primary Tumor").sum())
        n_met = int((status[mask] == "Metastasis").sum())
        n_nontumor = int((status[mask] == "Non-Tumor").sum())
        tissue = tissue_label(n_primary, n_met, n_nontumor, N)

        gap = spectral_gap(pts_all[mask])
        rows.append({"donor": d, "N": N, "gap": gap, "tissue": tissue})
        print(f"{d}: N={N}  gap={gap:.5f}  tissue={tissue}")

    print("\n" + "=" * 100)
    print(f"{'Donor':<12}{'N malignant':<14}{'spectral gap':<16}{'tissue'}")
    print("-" * 100)
    for r in rows:
        print(f"{r['donor']:<12}{r['N']:<14}{r['gap']:<16.5f}{r['tissue']}")
    print("=" * 100)

    print("\n=== 2) Direct paired comparison: Primary-only vs Metastasis-only malignant cells ===")
    print("    (donors with >=50 malignant cells in BOTH Primary Tumor and Metastasis, N-matched)")
    rows2 = []
    for d in donors:
        mask_p = is_mal & (donor == d) & (status == "Primary Tumor")
        mask_m = is_mal & (donor == d) & (status == "Metastasis")
        n_p, n_m = int(mask_p.sum()), int(mask_m.sum())
        if n_p < MIN_N or n_m < MIN_N:
            continue
        N = min(n_p, n_m)
        idx_p = rng.choice(n_p, size=N, replace=False)
        idx_m = rng.choice(n_m, size=N, replace=False)
        gap_p = spectral_gap(pts_all[mask_p][idx_p])
        gap_m = spectral_gap(pts_all[mask_m][idx_m])
        ratio = gap_m / gap_p if gap_p > 0 else float("inf")
        rows2.append({"donor": d, "N": N, "gap_primary": gap_p, "gap_met": gap_m, "ratio": ratio})
        print(f"{d}: N={N} (full primary={n_p}, full met={n_m})  "
              f"gap_primary={gap_p:.5f}  gap_met={gap_m:.5f}  ratio={ratio:.2f}x")

    print("\n" + "=" * 90)
    print(f"{'Donor':<12}{'N':<8}{'gap_primary':<14}{'gap_metastaz':<14}{'oran':<8}")
    print("-" * 90)
    for r in rows2:
        print(f"{r['donor']:<12}{r['N']:<8}{r['gap_primary']:<14.5f}{r['gap_met']:<14.5f}{r['ratio']:<8.2f}")
    print("=" * 90)

    if rows2:
        higher_met = sum(1 for r in rows2 if r["ratio"] > 1)
        print(f"\nMetastasis gap > Primary gap: {higher_met}/{len(rows2)} donor")


if __name__ == "__main__":
    main()
