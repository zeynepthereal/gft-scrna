"""GFT vs PCA K-Means on CMV cohort and breast TME datasets (for paper Table 1)."""
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

from gft import build_graph, gft_embed
from gft import ari_multi_seed, heterogeneity_score

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
CMV_PATH = os.path.join(PROJECT_DIR, "data", "cmv_cohort.h5ad")
TME_PATH = os.path.join(PROJECT_DIR, "data", "breast_tme.h5ad")
OUT_PATH = os.path.join(PROJECT_DIR, "results", "cmv_tme_gft_vs_pca.txt")

SEED = 42
K_EIG = 5
K_NN = 20
METRIC = "cosine"
N_SEEDS = 10


def load_cmv():
    adata = sc.read_h5ad(CMV_PATH)
    le = LabelEncoder()
    y = le.fit_transform(adata.obs["predicted_AIFI_L2"].astype(str).values)
    pts = adata.obsm["X_pca_harmony"][:, :50]
    return pts, y, le.classes_


def load_tme():
    adata = sc.read_h5ad(TME_PATH)
    counts = adata.obs["cell_type"].value_counts()
    keep = counts[counts >= 10].index
    mask = adata.obs["cell_type"].isin(keep).values
    adata = adata[mask].copy()
    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"].astype(str).values)
    pts = adata.obsm["X_pca"][:, :50]
    return pts, y, le.classes_


def run(name, pts, y, classes, log):
    k = len(classes)
    N = len(pts)
    het = heterogeneity_score(pts, y)
    print(f"\n=== {name} ===", file=log)
    print(f"N={N}  k={k}  heterogeneity_score={het:.2f}", file=log)
    print(f"classes: {list(classes)}", file=log)

    W = build_graph(pts, k_nn=K_NN, method=METRIC)
    emb = gft_embed(W, k_eig=K_EIG)

    pca_mean, pca_std = ari_multi_seed(pts, y, k, n_seeds=N_SEEDS, seed=SEED)
    gft_mean, gft_std = ari_multi_seed(emb, y, k, n_seeds=N_SEEDS, seed=SEED)
    ratio = gft_mean / max(pca_mean, 1e-6)

    print(f"PCA ARI: {pca_mean:.4f} +/- {pca_std:.4f}", file=log)
    print(f"GFT ARI: {gft_mean:.4f} +/- {gft_std:.4f}", file=log)
    print(f"Ratio (GFT/PCA): {ratio:.2f}x", file=log)

    return {"name": name, "N": N, "k": k, "het": het, "pca_mean": pca_mean, "pca_std": pca_std,
            "gft_mean": gft_mean, "gft_std": gft_std, "ratio": ratio}


def main():
    np.random.seed(SEED)
    buf = io.StringIO()

    print("Loading CMV cohort (predicted_AIFI_L2 labels, X_pca_harmony)...", file=buf)
    pts_cmv, y_cmv, cls_cmv = load_cmv()
    r_cmv = run("CMV cohort", pts_cmv, y_cmv, cls_cmv, buf)

    print("\nLoading breast TME (cell_type labels, X_pca, N>=10 per type)...", file=buf)
    pts_tme, y_tme, cls_tme = load_tme()
    r_tme = run("Breast TME", pts_tme, y_tme, cls_tme, buf)

    print("\n\n" + "=" * 100, file=buf)
    print("SUMMARY TABLE", file=buf)
    print("=" * 100, file=buf)
    header = f"{'Dataset':<16}{'N':<10}{'k':<6}{'Het.score':<12}{'PCA ARI':<18}{'GFT ARI':<18}{'Ratio'}"
    print(header, file=buf)
    print("-" * 100, file=buf)
    for r in [r_cmv, r_tme]:
        print(f"{r['name']:<16}{r['N']:<10}{r['k']:<6}{r['het']:<12.2f}"
              f"{r['pca_mean']:.4f}+/-{r['pca_std']:.4f}   {r['gft_mean']:.4f}+/-{r['gft_std']:.4f}   "
              f"{r['ratio']:.2f}x", file=buf)
    print("=" * 100, file=buf)

    text = buf.getvalue()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
