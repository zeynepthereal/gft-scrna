"""Verify the rebuilt data/GSE161277_all.h5ad against the paper's reported numbers.

Writes results/rebuild_verification_mac.txt. No data is modified.
Residual method is the one in scripts/paper_final_verification.py (the script that produced
the paper numbers): U = top-20 right singular vectors of mean-centered normal epithelial
X_pca[:, :50]; r_i = ||(x_i - mu) - U U^T (x_i - mu)||.
"""
import io
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_gse161277_all import annotate  # same marker scoring as the build

REPO = Path(__file__).resolve().parent.parent
H5AD = REPO / "data" / "GSE161277_all.h5ad"
OUT = REPO / "results" / "rebuild_verification_mac.txt"
K_EIG = 20

EXPECTED_CT = {
    "Epithelial/Tumor": 21969, "T cell": 13176, "B cell": 8377, "NK cell": 3467,
    "Macrophage/Monocyte": 2876, "Fibroblast/CAF": 529, "Endothelial": 124,
}
PATIENTS = {
    "Patient1": {"normal": "GSM4904237", "carcinoma": "GSM4904236",
                 "exp": {"n_normal": 834, "n_cancer": 2972, "med_n": 2.20, "med_c": 8.02, "ratio": 3.64}},
    "Patient2": {"normal": "GSM4904240", "carcinoma": "GSM4904239",
                 "exp": {"n_normal": 3584, "n_cancer": 3304, "med_n": 2.59, "med_c": 3.58, "ratio": 1.38}},
}


def residuals(pts_normal, pts_cancer, center=True):
    mu = pts_normal.mean(axis=0) if center else np.zeros(pts_normal.shape[1])
    _, _, Vt = np.linalg.svd(pts_normal - mu, full_matrices=False)
    U = Vt[:K_EIG].T
    r = lambda P: np.linalg.norm((P - mu) - (P - mu) @ U @ U.T, axis=1)
    return r(pts_normal), r(pts_cancer)


def row(out, name, exp, got, fmt="{:,}"):
    diff = got - exp
    rel = 100 * diff / exp if exp else float("nan")
    print(f"{name:<38}{fmt.format(exp):>12}{fmt.format(got):>12}{('{:+,}' if fmt == '{:,}' else '{:+.2f}').format(diff):>10}{rel:>+9.2f}%", file=out)


def main():
    out = io.StringIO()
    adata = sc.read_h5ad(H5AD)
    seed = adata.uns.get("rebuild_leiden_seed", "?")
    print("GSE161277_all.h5ad rebuild verification (Mac)", file=out)
    print(f"platform={platform.platform()}  python={platform.python_version()}", file=out)
    print("versions: " + ", ".join(f"{p}={version(p)}" for p in
          ["scanpy", "anndata", "numpy", "scipy", "scikit-learn", "leidenalg", "igraph", "pandas"]), file=out)
    print(f"Leiden: resolution=0.5, random_state={seed}, flavor=leidenalg (scanpy default), "
          f"n_clusters={adata.obs['leiden'].nunique()}\n", file=out)

    hdr = f"{'Quantity':<38}{'Expected':>12}{'Obtained':>12}{'Diff':>10}{'Rel':>10}"
    print(hdr, file=out); print("-" * len(hdr), file=out)
    row(out, "Cells after filtering", 50518, adata.n_obs)
    row(out, "Genes after filtering", 24111, adata.raw.n_vars)
    counts = adata.obs["cell_type"].value_counts()
    for ct, e in EXPECTED_CT.items():
        row(out, f"cell_type: {ct}", e, int(counts.get(ct, 0)))

    pts = adata.obsm["X_pca"][:, :50]
    gsm = adata.obs["gsm_id"].values
    epi = (adata.obs["cell_type"] == "Epithelial/Tumor").values
    uncentered = {}
    for p, d in PATIENTS.items():
        pn, pc = pts[(gsm == d["normal"]) & epi], pts[(gsm == d["carcinoma"]) & epi]
        rn, rc = residuals(pn, pc, center=True)
        e = d["exp"]
        row(out, f"{p} normal epithelial", e["n_normal"], len(pn))
        row(out, f"{p} carcinoma epithelial", e["n_cancer"], len(pc))
        row(out, f"{p} residual median normal", e["med_n"], float(np.median(rn)), "{:.2f}")
        row(out, f"{p} residual median cancer", e["med_c"], float(np.median(rc)), "{:.2f}")
        row(out, f"{p} residual ratio (median)", e["ratio"], float(np.median(rc) / np.median(rn)), "{:.2f}")
        un, uc = residuals(pn, pc, center=False)
        uncentered[p] = (np.median(un), np.median(uc))
    print("\n(Patient2 medians 2.59 / 3.58 taken from results/paper_final_verification.txt; "
          "the task spec listed only the 1.38x ratio.)", file=out)

    # Sensitivity: the task spec says Leiden random_state=42, the original script used the default (0).
    # Re-run Leiden on the same stored neighbor graph with seed 42 and re-annotate (not saved).
    sc.tl.leiden(adata, resolution=0.5, random_state=42, key_added="leiden_s42")
    lab42 = adata.obs["leiden_s42"].map(annotate(adata, key="leiden_s42"))
    c42 = lab42.value_counts()
    print(f"\nSensitivity: Leiden random_state=42 on the same graph "
          f"(n_clusters={adata.obs['leiden_s42'].nunique()}; not saved)", file=out)
    print(f"{'cell_type':<24}{'Expected':>10}{'seed=0':>10}{'seed=42':>10}", file=out)
    for ct, e in EXPECTED_CT.items():
        print(f"{ct:<24}{e:>10,}{int(counts.get(ct, 0)):>10,}{int(c42.get(ct, 0)):>10,}", file=out)
    agree = (lab42.astype(str).values == adata.obs["cell_type"].astype(str).values).mean()
    print(f"cell_type agreement seed 0 vs 42: {100 * agree:.2f}% of cells", file=out)

    print("\nResidual without mean-centering (formula exactly as written in the task, "
          "r=||x-UU^T x||; paper script centers):", file=out)
    for p, (mn, mc) in uncentered.items():
        print(f"  {p}: median normal={mn:.2f} cancer={mc:.2f} ratio={mc / mn:.2f}x", file=out)

    text = out.getvalue()
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
