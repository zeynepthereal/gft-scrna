"""Per cell-type spectral gap comparison, normal vs carcinoma, N-matched, Patient1 & Patient2."""
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

SAMPLES = {
    ("Patient1", "normal"): "GSM4904237",
    ("Patient1", "carcinoma"): "GSM4904236",
    ("Patient2", "normal"): "GSM4904240",
    ("Patient2", "carcinoma"): "GSM4904239",
}
CELL_TYPES = ["T cell", "B cell", "Macrophage/Monocyte", "Epithelial/Tumor"]


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

    rows = []
    for patient in ["Patient1", "Patient2"]:
        gsm_normal = SAMPLES[(patient, "normal")]
        gsm_carcinoma = SAMPLES[(patient, "carcinoma")]

        for ct in CELL_TYPES:
            mask_n = (gsm == gsm_normal) & (cell_type == ct)
            mask_c = (gsm == gsm_carcinoma) & (cell_type == ct)
            pts_n = pts_all[mask_n]
            pts_c = pts_all[mask_c]

            N = min(len(pts_n), len(pts_c))
            if N < K_NN + 5:
                print(f"SKIP {patient} {ct}: N={N} too small (normal={len(pts_n)}, carcinoma={len(pts_c)})")
                continue

            idx_n = rng.choice(len(pts_n), size=N, replace=False)
            idx_c = rng.choice(len(pts_c), size=N, replace=False)

            gap_n = spectral_gap(pts_n[idx_n])
            gap_c = spectral_gap(pts_c[idx_c])
            ratio = gap_c / gap_n if gap_n > 0 else float("inf")

            rows.append({
                "patient": patient, "cell_type": ct, "N": N,
                "gap_normal": gap_n, "gap_carcinoma": gap_c, "ratio": ratio,
            })
            print(f"{patient:<10}{ct:<22}N={N:<6}gap_normal={gap_n:.5f}  "
                  f"gap_carcinoma={gap_c:.5f}  ratio={ratio:.2f}x")

    print("\n" + "=" * 90)
    print(f"{'Hasta':<10}{'Hucre tipi':<22}{'N':<8}{'normal gap':<14}{'carcinoma gap':<16}{'oran':<8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['patient']:<10}{r['cell_type']:<22}{r['N']:<8}{r['gap_normal']:<14.5f}"
              f"{r['gap_carcinoma']:<16.5f}{r['ratio']:<8.2f}")
    print("=" * 90)

    print("\nOrtalama oran (2 hasta) hucre tipine gore:")
    for ct in CELL_TYPES:
        ratios = [r["ratio"] for r in rows if r["cell_type"] == ct]
        if ratios:
            print(f"  {ct:<22}: {np.mean(ratios):.2f}x  (n_hasta={len(ratios)})")


if __name__ == "__main__":
    main()
