"""Verify GFT K-Means std with n_init=10 across all three GFT datasets (CMV, TME, CRC)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from gft import build_graph, gft_embed

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
SEED = 42
K_EIG = 5
K_NN = 20
N_SEEDS = 10


def ari_ninit10(emb, y, k, n_seeds=N_SEEDS, seed=SEED):
    aris = [adjusted_rand_score(y, KMeans(k, random_state=s, n_init=10).fit_predict(emb))
            for s in range(seed, seed + n_seeds)]
    return float(np.mean(aris)), float(np.std(aris))


def load_cmv():
    adata = sc.read_h5ad(os.path.join(PROJECT_DIR, "data", "cmv_cohort.h5ad"))
    le = LabelEncoder()
    y = le.fit_transform(adata.obs["predicted_AIFI_L2"].astype(str).values)
    return adata.obsm["X_pca_harmony"][:, :50], y, len(le.classes_)


def load_crc():
    adata = sc.read_h5ad(os.path.join(PROJECT_DIR, "data", "GSE161277_all.h5ad"), backed="r")
    y_raw = adata.obs["cell_type"].astype(str).values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    pts = adata.obsm["X_pca"][:, :50]
    return pts, y, len(le.classes_)


def load_tme():
    adata = sc.read_h5ad(os.path.join(PROJECT_DIR, "data", "breast_tme.h5ad"))
    counts = adata.obs["cell_type"].value_counts()
    keep = counts[counts >= 10].index
    mask = adata.obs["cell_type"].isin(keep).values
    adata = adata[mask].copy()
    le = LabelEncoder()
    y = le.fit_transform(adata.obs["cell_type"].astype(str).values)
    return adata.obsm["X_pca"][:, :50], y, len(le.classes_)


def main():
    for name, loader in [("CMV cohort", load_cmv), ("Breast TME", load_tme), ("CRC (GSE161277)", load_crc)]:
        pts, y, k = loader()
        W = build_graph(pts, k_nn=K_NN, method="cosine")
        emb = gft_embed(W, k_eig=K_EIG)
        gft_mean, gft_std = ari_ninit10(emb, y, k)
        pca_mean, pca_std = ari_ninit10(pts, y, k)
        ratio = gft_mean / max(pca_mean, 1e-6)
        print(f"{name}: N={len(pts)} k={k}")
        print(f"  PCA ARI (n_init=10, {N_SEEDS} seeds) = {pca_mean:.4f} +/- {pca_std:.4f}")
        print(f"  GFT ARI (n_init=10, {N_SEEDS} seeds) = {gft_mean:.4f} +/- {gft_std:.4f}")
        print(f"  Ratio = {ratio:.2f}x")


if __name__ == "__main__":
    main()
