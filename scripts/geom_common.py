"""Shared utilities for scripts/geom_ppca_potential.py, geom_principal_angles.py, geom_dnb.py.

Not one of the three deliverable scripts itself -- a small helper module the three import from,
to avoid re-deriving the dual-space (cohort PCA vs normal-only PCA) construction three times.

Robustness check R1 (common to Components 1 and 2): every analysis is repeated in
  (i)  the cohort 50-D PCA (data/GSE161277-style cache: here, computed directly on the full
       crc_epithelial.h5ad log-normalized X, all genes, same as sc.pp.pca(n_comps=50,
       mask_var=None)), and
  (ii) a NORMAL-ONLY space: per donor, top-2000 HVGs selected from that donor's normal cells
       only, 50-D PCA fit on normal cells only, malignant cells projected with the same
       transform (same genes, same mean, same loadings) -- no malignant cell influences gene
       selection or the fitted subspace.
"""
import os

import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5AD = os.path.join(PROJECT_DIR, "data", "crc_epithelial.h5ad")
NPZ = os.path.join(PROJECT_DIR, "_ricci_tmp", "crc_pca50.npz")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

DONORS = ["HTA8_6002", "HTA8_6009", "HTA8_6013", "HTA8_6016", "HTA8_6018", "HTA8_6028"]
N_HVG = 2000
N_PCA = 50
MIN_N = 50


class Buf:
    def __init__(self):
        self.lines = []

    def log(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.lines.append(msg)

    def text(self):
        return "\n".join(self.lines) + "\n"

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text())
        print(f"\nSaved -> {path}")


def svd_subspace(pts, q):
    """Full SVD of centered pts; returns (U: d x q top loadings, mean, eigvals_desc: d,
    the FULL sorted-descending eigenvalue spectrum of the sample covariance, for sigma^2/PPCA)."""
    mean = pts.mean(axis=0)
    Xc = pts - mean
    n = Xc.shape[0]
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / max(n - 1, 1)
    return Vt[:q].T, mean, eigvals


def load_adata():
    return sc.read_h5ad(H5AD)


def load_cohort_pca():
    """Space (i): cohort-wide 50-D PCA, cached (same subspace as sc.pp.pca(n_comps=50,
    mask_var=None) over ALL cells, all genes) -- see scripts/extract_pca50.py."""
    d = np.load(NPZ, allow_pickle=True)
    return d["pcs"].astype(np.float64), d["n_genes"].astype(float), d["donor"], d["disease"], d["cell_type"]


def donor_masks_cohort(donor, disease, ctype, dn):
    dmask = donor == dn
    is_norm = (disease == "normal") & (ctype != "malignant cell")
    is_mal = (disease == "colorectal cancer") & (ctype == "malignant cell")
    return dmask & is_norm, dmask & is_mal


def donor_masks_full(adata, dn):
    obs = adata.obs
    dmask = (obs["donor_id"].astype(str) == dn).values
    is_norm = ((obs["disease"] == "normal") & (obs["cell_type"] != "malignant cell")).values
    is_mal = ((obs["disease"] == "colorectal cancer") & (obs["cell_type"] == "malignant cell")).values
    return dmask & is_norm, dmask & is_mal


def build_normal_only_space(adata, dn, n_hvg=N_HVG, n_pca=N_PCA, seed=0):
    """Space (ii) for one donor: top-n_hvg HVGs from normal cells only (Seurat flavor on
    log-normalized adata.X restricted to normal cells), 50-D PCA fit on normal cells only,
    malignant cells of the same donor projected with the identical genes/mean/loadings.

    Returns dict: pts_normal (n_normal, n_pca), pts_malignant (n_malignant, n_pca),
    hvg_names (gene symbols), loadings (n_hvg, n_pca) -- for mapping PCA directions back to genes,
    hvg_mean (n_hvg,).
    """
    mask_n, mask_m = donor_masks_full(adata, dn)
    ad_n = adata[mask_n].copy()
    sc.pp.highly_variable_genes(ad_n, n_top_genes=n_hvg, flavor="seurat")
    hvg_mask = ad_n.var["highly_variable"].values
    hvg_names = adata.var["feature_name"].astype(str).values[hvg_mask]
    gene_idx = np.where(hvg_mask)[0]

    Xn = np.asarray(adata[mask_n][:, gene_idx].X.todense()) if hasattr(adata.X, "todense") else \
        np.asarray(adata[mask_n][:, gene_idx].X)
    Xm = np.asarray(adata[mask_m][:, gene_idx].X.todense()) if hasattr(adata.X, "todense") else \
        np.asarray(adata[mask_m][:, gene_idx].X)

    mean = Xn.mean(axis=0)
    Xn_c = Xn - mean
    U, S, Vt = np.linalg.svd(Xn_c, full_matrices=False)
    loadings = Vt[:n_pca].T  # (n_hvg, n_pca)
    pts_normal = Xn_c @ loadings
    pts_malignant = (Xm - mean) @ loadings

    return {
        "pts_normal": pts_normal, "pts_malignant": pts_malignant,
        "hvg_names": hvg_names, "loadings": loadings, "hvg_mean": mean,
        "n_normal": mask_n.sum(), "n_malignant": mask_m.sum(),
    }


SPACES = ["cohort_pca", "normal_only"]
