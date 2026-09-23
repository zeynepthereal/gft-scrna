"""CRC epithelial dataset: per-donor PAGA spectral gap, normal vs cancer (N>=100 in both groups)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import scanpy as sc
from scipy.sparse import csr_matrix, save_npz

from gft.embedding import get_eigenvalues

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "crc_epithelial.h5ad")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
N_COMPS = 50
N_NEIGHBORS = 15
K_MAX = 5  # only need lambda2, lambda3 for spectral gap -> smaller cluster-count requirement
MIN_CLUSTERS = K_MAX + 2
MIN_N = 100


def leiden_enough_clusters(adata, min_clusters=MIN_CLUSTERS, start_res=1.0, max_tries=6):
    res = start_res
    for _ in range(max_tries):
        sc.tl.leiden(adata, resolution=res, key_added="leiden",
                     flavor="igraph", n_iterations=2, directed=False)
        n = adata.obs["leiden"].nunique()
        if n >= min_clusters:
            return n, res
        res += 0.5
    raise RuntimeError(f"Could not reach {min_clusters} clusters after {max_tries} tries")


def analyze(adata_full, mask, tag):
    adata = adata_full[mask].copy()
    sc.pp.neighbors(adata, n_pcs=N_COMPS, n_neighbors=N_NEIGHBORS)
    n_clusters, res = leiden_enough_clusters(adata)
    sc.tl.paga(adata, groups="leiden")
    W = csr_matrix(adata.uns["paga"]["connectivities"])

    os.makedirs(OUT_DIR, exist_ok=True)
    save_npz(os.path.join(OUT_DIR, f"paga_connectivity_{tag}.npz"), W)

    eigs = get_eigenvalues(W, k_max=min(K_MAX, W.shape[0] - 3))
    gap = float(eigs[1] - eigs[0])
    print(f"    {tag}: n_cells={adata.n_obs}, n_clusters={n_clusters} (res={res:.1f}), gap={gap:.5f}")
    return gap, n_clusters


def main():
    adata_all = sc.read_h5ad(PATH)
    print(f"1) PCA (n_comps={N_COMPS}) on log-normalized X (no HVG mask)...")
    sc.pp.pca(adata_all, n_comps=N_COMPS, mask_var=None)
    adata_all.raw = None

    donor = adata_all.obs["donor_id"].values
    disease = adata_all.obs["disease"].values
    donors = sorted(adata_all.obs["donor_id"].unique())

    rows = []
    for d in donors:
        mask_n = (donor == d) & (disease == "normal")
        mask_c = (donor == d) & (disease == "colorectal cancer")
        n_normal, n_cancer = int(mask_n.sum()), int(mask_c.sum())

        if n_normal < MIN_N or n_cancer < MIN_N:
            print(f"SKIP {d}: normal={n_normal}, cancer={n_cancer} (need >= {MIN_N} in both)")
            continue

        print(f"\n=== {d} (normal={n_normal}, cancer={n_cancer}) ===")
        gap_n, ncl_n = analyze(adata_all, mask_n, f"{d}_normal")
        gap_c, ncl_c = analyze(adata_all, mask_c, f"{d}_cancer")
        direction = "down" if gap_c < gap_n else "up"
        rows.append({"donor": d, "gap_normal": gap_n, "gap_cancer": gap_c, "direction": direction})

    print("\n" + "=" * 70)
    print(f"{'Donor':<14}{'normal gap':<14}{'cancer gap':<14}{'yon':<8}")
    print("-" * 70)
    for r in rows:
        arrow = "↓ (cancer < normal)" if r["direction"] == "down" else "↑ (cancer > normal)"
        print(f"{r['donor']:<14}{r['gap_normal']:<14.5f}{r['gap_cancer']:<14.5f}{arrow}")
    print("=" * 70)

    n_down = sum(1 for r in rows if r["direction"] == "down")
    print(f"\ncancer gap < normal gap: {n_down}/{len(rows)} donor")


if __name__ == "__main__":
    main()
