"""Permutation test for the cancer/normal residual-ratio finding, across every sample analyzed so far.

For each sample: pool the (already-computed, fixed) per-cell residuals from the normal
self-projection group and the cancer projection group. Shuffle the cancer/normal group
labels over this pooled set 1000 times (group sizes held fixed), recomputing
median(shuffled 'cancer' group) / median(shuffled 'normal' group) each time. This tests
whether the true group labels carry more signal than random labels of the same sizes,
holding the underlying residual values (and thus the U_normal subspace used to produce
them) fixed -- the standard, tractable form of this test (re-deriving 1000 SVD subspaces
per sample would be needlessly expensive and is not what the group-label shuffle tests).

Empirical p-value = fraction of permutations with shuffled ratio >= observed ratio.
95% CI = [2.5th, 97.5th] percentile of the null (permutation) distribution of ratios.
"""
import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scanpy as sc

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
GSE_PATH = os.path.join(PROJECT_DIR, "data", "GSE161277_all.h5ad")
CRC_PATH = os.path.join(PROJECT_DIR, "data", "crc_epithelial.h5ad")
OUT_PATH = os.path.join(PROJECT_DIR, "results", "permutation_test.txt")

K_EIG = 20
N_PERM = 1000
SEED = 42

GSE_PATIENTS = {
    "Patient1": {"normal": "GSM4904237", "carcinoma": "GSM4904236"},
    "Patient2": {"normal": "GSM4904240", "carcinoma": "GSM4904239"},
    "Patient3": {"normal": "GSM4904246", "carcinoma": "GSM4904245"},
}
CRC_DONORS = ["HTA8_6002", "HTA8_6009", "HTA8_6013", "HTA8_6016", "HTA8_6018", "HTA8_6028"]


def subspace_svd(pts_normal, k_eig=K_EIG):
    mean = pts_normal.mean(axis=0)
    Xc = pts_normal - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k_eig].T, mean


def residual(pts, U, mean):
    X = pts - mean
    return np.linalg.norm(X - X @ U @ U.T, axis=1)


def permutation_test(res_normal, res_cancer, n_perm, rng):
    observed_ratio = np.median(res_cancer) / np.median(res_normal)
    pooled = np.concatenate([res_normal, res_cancer])
    n_normal, n_cancer = len(res_normal), len(res_cancer)

    null_ratios = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        shuf_normal, shuf_cancer = perm[:n_normal], perm[n_normal:]
        med_n = np.median(shuf_normal)
        null_ratios[i] = np.median(shuf_cancer) / med_n if med_n > 0 else np.inf

    p_value = float(np.mean(null_ratios >= observed_ratio))
    ci_low, ci_high = np.percentile(null_ratios, [2.5, 97.5])
    return observed_ratio, p_value, ci_low, ci_high, null_ratios


def run_sample(name, pts_normal, pts_cancer, rng, log):
    U, mean = subspace_svd(pts_normal)
    res_normal = residual(pts_normal, U, mean)
    res_cancer = residual(pts_cancer, U, mean)
    obs, p, lo, hi, null = permutation_test(res_normal, res_cancer, N_PERM, rng)
    print(f"{name}: n_normal={len(pts_normal)}, n_cancer={len(pts_cancer)}", file=log)
    print(f"  observed ratio = {obs:.3f}x", file=log)
    print(f"  permutation p-value (n_perm={N_PERM}) = {p:.4f}", file=log)
    print(f"  null distribution 95% CI = [{lo:.3f}, {hi:.3f}]", file=log)
    print(f"  null distribution mean={null.mean():.3f}, std={null.std():.3f}", file=log)
    return {"sample": name, "n_normal": len(pts_normal), "n_cancer": len(pts_cancer),
            "observed_ratio": obs, "p_value": p, "ci_low": lo, "ci_high": hi}


def main():
    rng = np.random.default_rng(SEED)
    buf = io.StringIO()
    rows = []

    print("=" * 90, file=buf)
    print("GSE161277 patients", file=buf)
    print("=" * 90, file=buf)
    adata_gse = sc.read_h5ad(GSE_PATH, backed="r")
    pts_all = adata_gse.obsm["X_pca"][:, :50]
    gsm = adata_gse.obs["gsm_id"].values
    cell_type = adata_gse.obs["cell_type"].values
    for patient, gsms in GSE_PATIENTS.items():
        mask_normal = (gsm == gsms["normal"]) & (cell_type == "Epithelial/Tumor")
        mask_carcinoma = (gsm == gsms["carcinoma"]) & (cell_type == "Epithelial/Tumor")
        r = run_sample(f"GSE161277/{patient}", pts_all[mask_normal], pts_all[mask_carcinoma], rng, buf)
        r["source"] = "GSE161277"
        rows.append(r)

    print("\n" + "=" * 90, file=buf)
    print("CRC epithelial cohort donors", file=buf)
    print("=" * 90, file=buf)
    adata_crc = sc.read_h5ad(CRC_PATH)
    sc.pp.pca(adata_crc, n_comps=50, mask_var=None)
    pts_crc = adata_crc.obsm["X_pca"][:, :50]
    donor = adata_crc.obs["donor_id"].values
    disease = adata_crc.obs["disease"].values
    tissue = adata_crc.obs["tissue"].values

    for d in CRC_DONORS:
        mask_normal = (donor == d) & (disease == "normal")
        mask_cancer = (donor == d) & (disease == "colorectal cancer")
        r = run_sample(f"CRC_epith/{d}", pts_crc[mask_normal], pts_crc[mask_cancer], rng, buf)
        r["source"] = "CRC_epith"
        rows.append(r)

    print("\n" + "=" * 90, file=buf)
    print("HTA8_6016 primary/metastasis split (normal subspace = sigmoid colon normal cells)", file=buf)
    print("=" * 90, file=buf)
    mask_normal_6016 = (donor == "HTA8_6016") & (disease == "normal")
    mask_primary = (donor == "HTA8_6016") & (disease == "colorectal cancer") & (tissue == "sigmoid colon")
    mask_met = (donor == "HTA8_6016") & (disease == "colorectal cancer") & (tissue == "liver")
    r = run_sample("HTA8_6016/primary(sigmoid_colon)", pts_crc[mask_normal_6016], pts_crc[mask_primary], rng, buf)
    r["source"] = "CRC_epith_split"
    rows.append(r)
    r = run_sample("HTA8_6016/metastasis(liver)", pts_crc[mask_normal_6016], pts_crc[mask_met], rng, buf)
    r["source"] = "CRC_epith_split"
    rows.append(r)

    print("\n\n" + "=" * 110, file=buf)
    print(f"SUMMARY TABLE ({len(rows)} samples, {N_PERM} permutations each, seed={SEED})", file=buf)
    print("=" * 110, file=buf)
    header = f"{'Source':<18}{'Sample':<32}{'N_normal':<10}{'N_cancer':<10}{'Obs.ratio':<12}{'Perm.p':<10}{'Null 95% CI'}"
    print(header, file=buf)
    print("-" * 110, file=buf)
    for r in rows:
        print(f"{r['source']:<18}{r['sample']:<32}{r['n_normal']:<10}{r['n_cancer']:<10}"
              f"{r['observed_ratio']:<12.3f}{r['p_value']:<10.4f}[{r['ci_low']:.3f}, {r['ci_high']:.3f}]", file=buf)
    print("=" * 110, file=buf)

    n_sig = sum(1 for r in rows if r["p_value"] < 0.05)
    print(f"\nSamples with permutation p < 0.05: {n_sig}/{len(rows)}", file=buf)
    print(f"Max permutation p-value across all samples: {max(r['p_value'] for r in rows):.4f}", file=buf)

    text = buf.getvalue()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
