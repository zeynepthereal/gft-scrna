"""Patient2 & Patient3: normal-epithelial PCA subspace, carcinoma projection residual, marker check."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc

K_EIG = 20
MARKER_GENES = ["ASCL2", "SOX4", "MKI67"]

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")

PATIENTS = {
    "Patient2": {"normal": "GSM4904240", "carcinoma": "GSM4904239"},
    "Patient3": {"normal": "GSM4904246", "carcinoma": "GSM4904245"},
}


def u_normal_subspace(pts_normal, k_eig=K_EIG):
    mean = pts_normal.mean(axis=0)
    Xc = pts_normal - mean
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    U = eigvecs[:, order][:, :k_eig]
    explained = eigvals[order][:k_eig].sum() / eigvals.sum()
    return U, mean, explained


def residual(pts, U, mean):
    X = pts - mean
    proj = X @ U @ U.T
    return np.linalg.norm(X - proj, axis=1)


def gene_expr(raw, symbol_to_id, gene, mask):
    if gene not in symbol_to_id:
        return None, None
    gene_id = symbol_to_id[gene]
    expr = raw.X[:, raw.var_names.get_loc(gene_id)]
    expr = np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()
    sub = expr[mask]
    return sub.mean(), (sub > 0).mean() * 100


def analyze(adata, raw, symbol_to_id, patient, normal_gsm, carcinoma_gsm):
    print(f"\n{'='*70}\n{patient}: normal={normal_gsm}, carcinoma={carcinoma_gsm}\n{'='*70}")

    pts_all = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values
    cell_type = adata.obs["cell_type"].values
    leiden = adata.obs["leiden"].values

    mask_normal = (gsm == normal_gsm) & (cell_type == "Epithelial/Tumor")
    mask_carcinoma = (gsm == carcinoma_gsm) & (cell_type == "Epithelial/Tumor")
    pts_normal = pts_all[mask_normal]
    pts_carcinoma = pts_all[mask_carcinoma]
    print(f"Normal epithelial: n={len(pts_normal)}, Carcinoma epithelial: n={len(pts_carcinoma)}")

    print("\n1) U_normal = PCA-of-PCA subspace (top-20 eigenvectors of normal cells' 50D-PCA covariance)...")
    U, mean, explained = u_normal_subspace(pts_normal)
    print(f"   U_normal: {U.shape}, explained variance = {explained:.3f}")

    print("\n2) Projection residuals...")
    res_carcinoma = residual(pts_carcinoma, U, mean)
    res_normal_self = residual(pts_normal, U, mean)
    print(f"   Normal (self): median={np.median(res_normal_self):.4f}  mean={res_normal_self.mean():.4f}  std={res_normal_self.std():.4f}")
    print(f"   Carcinoma     : median={np.median(res_carcinoma):.4f}  mean={res_carcinoma.mean():.4f}  std={res_carcinoma.std():.4f}")
    print(f"   Ratio (carcinoma/normal mean): {res_carcinoma.mean()/res_normal_self.mean():.2f}x")

    print("\n3) Top-20 / Bottom-20 residual carcinoma cells -- Leiden clusters...")
    carc_leiden = leiden[mask_carcinoma]
    order_res = np.argsort(res_carcinoma)
    bottom20_idx = order_res[:20]
    top20_idx = order_res[-20:]

    bottom_counts = pd.Series(carc_leiden[bottom20_idx]).value_counts()
    top_counts = pd.Series(carc_leiden[top20_idx]).value_counts()
    print(f"   Bottom-20 (lowest residual) leiden dist:\n{bottom_counts}")
    print(f"   Top-20 (highest residual) leiden dist:\n{top_counts}")

    dominant_high_cluster = top_counts.idxmax()
    dominant_high_frac = top_counts.max() / 20
    dominant_low_cluster = bottom_counts.idxmax()
    dominant_low_frac = bottom_counts.max() / 20
    print(f"   Dominant HIGH-residual cluster: {dominant_high_cluster} ({dominant_high_frac:.0%} of top-20)")
    print(f"   Dominant LOW-residual cluster:  {dominant_low_cluster} ({dominant_low_frac:.0%} of bottom-20)")

    print(f"\n4) Marker expression in dominant high-residual cluster ({dominant_high_cluster}) "
          f"vs dominant low-residual cluster ({dominant_low_cluster}), within carcinoma epithelial cells...")
    mask_high_cluster = mask_carcinoma & (leiden == dominant_high_cluster)
    mask_low_cluster = mask_carcinoma & (leiden == dominant_low_cluster)
    n_high, n_low = mask_high_cluster.sum(), mask_low_cluster.sum()
    print(f"   n(high-residual cluster {dominant_high_cluster})={n_high}, n(low-residual cluster {dominant_low_cluster})={n_low}")

    rows = []
    for gene in MARKER_GENES:
        mean_high, pct_high = gene_expr(raw, symbol_to_id, gene, mask_high_cluster)
        mean_low, pct_low = gene_expr(raw, symbol_to_id, gene, mask_low_cluster)
        rows.append({"gene": gene, "mean_high_cluster": mean_high, "pct_high_cluster": pct_high,
                     "mean_low_cluster": mean_low, "pct_low_cluster": pct_low})
        print(f"   {gene}: high-cluster mean={mean_high:.3f} (%+={pct_high:.1f}%)  |  "
              f"low-cluster mean={mean_low:.3f} (%+={pct_low:.1f}%)  |  "
              f"{'HIGH > low' if mean_high > mean_low else 'high <= LOW'}")

    return {
        "patient": patient, "res_carcinoma_mean": res_carcinoma.mean(), "res_normal_mean": res_normal_self.mean(),
        "ratio": res_carcinoma.mean() / res_normal_self.mean(),
        "dominant_high_cluster": dominant_high_cluster, "dominant_high_frac": dominant_high_frac,
        "dominant_low_cluster": dominant_low_cluster, "dominant_low_frac": dominant_low_frac,
        "markers": rows,
    }


def main():
    adata = sc.read_h5ad(PATH)
    raw = adata.raw
    symbol_to_id = {}
    for sym, gid in zip(raw.var["gene_symbol"], raw.var_names):
        symbol_to_id[sym] = gid

    results = []
    for patient, gsms in PATIENTS.items():
        r = analyze(adata, raw, symbol_to_id, patient, gsms["normal"], gsms["carcinoma"])
        results.append(r)

    print("\n\n" + "=" * 90)
    print("OZET -- Patient2 & Patient3")
    print("=" * 90)
    for r in results:
        print(f"\n{r['patient']}:")
        print(f"  Residual ratio (carcinoma/normal): {r['ratio']:.2f}x")
        print(f"  High-residual dominant cluster: {r['dominant_high_cluster']} ({r['dominant_high_frac']:.0%} of top-20)")
        print(f"  Low-residual dominant cluster:  {r['dominant_low_cluster']} ({r['dominant_low_frac']:.0%} of bottom-20)")
        for m in r["markers"]:
            direction = "HIGH-cluster up" if m["mean_high_cluster"] > m["mean_low_cluster"] else "low-cluster up/equal"
            print(f"    {m['gene']}: high={m['mean_high_cluster']:.3f} (%+{m['pct_high_cluster']:.1f}) "
                  f"vs low={m['mean_low_cluster']:.3f} (%+{m['pct_low_cluster']:.1f}) -> {direction}")
    print("=" * 90)


if __name__ == "__main__":
    main()
