"""Reviewer checks on the GFT clustering analysis (three critiques).

Uses gft.embedding.gft_embed (imported, not copied -- fixed v0 seeded by random_state=42,
default already matches) and gft.graph.build_graph (cosine kNN, k=20) throughout. K-Means:
10 seeds (42..51), n_init=10 (the module's own gft.metrics.ari_multi_seed hardcodes n_init=5,
so a small local helper is used instead to match this task's explicit n_init=10 requirement --
not a reimplementation of anything else). heterogeneity_score is imported unmodified from
gft.metrics.

Datasets:
  - data/GSE161277_all.h5ad (Mac rebuild): labels=cell_type, X_pca[:, :50] (already computed,
    reused as-is).
  - PBMC3k negative control: scanpy.datasets.pbmc3k_processed(), labels=obs['louvain'], its
    stored X_pca (50 dims already present; would be recomputed to 50 PCs if fewer were stored).

TEST A -- label-free heterogeneity score
  Critique: heterogeneity_score needs true labels, so it can't guide method choice a priori.
  For both datasets: heterogeneity_score computed with (1) true labels, (2) unsupervised Leiden
  labels (PCA neighbor graph, n_neighbors=15, resolution=0.5, random_state=42). For PBMC3k only,
  also recompute GFT-vs-PCA K-Means ARI with both label sets (k = number of types in each).
  DECISION, stated before the numbers: the label-free score is usable if it stays on the SAME
  side of the threshold 7 as the label-based score, for BOTH datasets (CRC above 7, PBMC below 7).

TEST B -- Leiden resolution sweep (GSE161277)
  Critique: k_eig was varied but not Leiden resolution. Representations: PCA(50), GFT(k_eig=5),
  GFT(k_eig=20) -- all three built from ONE cosine-20nn graph on the same 50-D PCA points (GFT5
  and GFT20 reuse that graph; PCA representation is graphed separately via its own neighbor
  graph for Leiden, as is standard). For each: sc.pp.neighbors(use_rep=..., n_neighbors=15),
  Leiden(random_state=42) at resolutions 0.02/0.05/0.1/0.2/0.3/0.5/1.0. Reports cluster count and
  ARI per resolution, best ARI per representation.
  DECISION, stated before the numbers: the "structural incompatibility" claim survives only if
  the best GFT Leiden ARI (across k_eig=5 and k_eig=20, across all resolutions) stays CLEARLY
  below PCA's best Leiden ARI (operationalized as PCA_best - GFT_best >= 0.02).

TEST C -- Harmony baseline (GSE161277)
  Critique: PCA is not the modern standard; compare against a batch-corrected embedding.
  Harmony (harmonypy, batch=sample) on the same 50-D PCA. Reports K-Means ARI (mean+-std, 10
  seeds, n_init=10) for PCA, Harmony, GFT-on-PCA-graph, GFT-on-Harmony-graph; and best Leiden ARI
  (same 7 resolutions) for PCA and Harmony.

Output: results/gft_reviewer_checks.txt
"""
import os
import sys
import time
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gft import build_graph, gft_embed  # noqa: E402
from gft.metrics import heterogeneity_score  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
GSE_H5AD = os.path.join(REPO, "data", "GSE161277_all.h5ad")

SEED = 42
KMEANS_SEEDS = list(range(42, 52))  # 10 seeds
KMEANS_N_INIT = 10
K_NN = 20
K_EIG_DEFAULT = 5
RESOLUTIONS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


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
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text())
        print(f"\nSaved -> {path}")


def kmeans_ari(emb, y, k, buf=None, tag=""):
    aris = [adjusted_rand_score(y, KMeans(k, random_state=s, n_init=KMEANS_N_INIT).fit_predict(emb))
            for s in KMEANS_SEEDS]
    mean, std = float(np.mean(aris)), float(np.std(aris))
    if buf is not None:
        buf.log(f"  {tag} K-Means ARI (10 seeds, n_init=10): {mean:.4f} +/- {std:.4f}")
    return mean, std


def leiden_labels(pts, n_neighbors=15, resolution=0.5, seed=SEED):
    ad = sc.AnnData(np.zeros((len(pts), 1), dtype=np.float32))
    ad.obsm["rep"] = pts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.neighbors(ad, use_rep="rep", n_neighbors=n_neighbors, random_state=seed)
        sc.tl.leiden(ad, resolution=resolution, random_state=seed, flavor="igraph",
                     n_iterations=2, directed=False)
    return ad.obs["leiden"].values


def leiden_sweep(pts, y_true, resolutions, rep_name, buf, n_neighbors=15, seed=SEED):
    ad = sc.AnnData(np.zeros((len(pts), 1), dtype=np.float32))
    ad.obsm["rep"] = pts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.neighbors(ad, use_rep="rep", n_neighbors=n_neighbors, random_state=seed)
    buf.log(f"\n  [{rep_name}] Leiden resolution sweep:")
    buf.log(f"  {'resolution':<12}{'n_clusters':<12}{'ARI'}")
    rows = []
    for res in resolutions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.tl.leiden(ad, resolution=res, random_state=seed, key_added=f"leiden_{res}",
                         flavor="igraph", n_iterations=2, directed=False)
        labels = ad.obs[f"leiden_{res}"].values
        ari = adjusted_rand_score(y_true, labels)
        n_cl = len(pd.unique(labels))
        buf.log(f"  {res:<12}{n_cl:<12}{ari:.4f}")
        rows.append({"resolution": res, "n_clusters": n_cl, "ari": ari})
    best = max(rows, key=lambda r: r["ari"])
    buf.log(f"  [{rep_name}] BEST: resolution={best['resolution']}  n_clusters={best['n_clusters']}  "
            f"ARI={best['ari']:.4f}")
    return rows, best


def test_a(buf):
    buf.log("\n" + "=" * 96)
    buf.log("TEST A -- label-free heterogeneity score")
    buf.log("=" * 96)
    buf.log("DECISION (stated before the numbers): label-free score is usable if it stays on the")
    buf.log("SAME side of threshold 7 as the label-based score, for BOTH datasets (CRC above 7, "
            "PBMC below 7).\n")

    results = {}

    buf.log("--- GSE161277 (Mac rebuild) ---")
    t0 = time.time()
    adata = sc.read_h5ad(GSE_H5AD)
    pts_pca = adata.obsm["X_pca"][:, :50]
    y_true = adata.obs["cell_type"].astype(str).values
    het_true = heterogeneity_score(pts_pca, y_true)
    y_leiden = leiden_labels(pts_pca, n_neighbors=15, resolution=0.5, seed=SEED)
    het_leiden = heterogeneity_score(pts_pca, y_leiden)
    buf.log(f"  n_cells={len(pts_pca)}  n_true_types={len(np.unique(y_true))}  "
            f"n_leiden_clusters={len(np.unique(y_leiden))}")
    buf.log(f"  heterogeneity_score: true_labels={het_true:.3f}  leiden_labels={het_leiden:.3f}")
    same_side = (het_true > 7) == (het_leiden > 7)
    buf.log(f"  Same side of threshold 7? {'YES' if same_side else 'NO'} "
            f"(true {'above' if het_true>7 else 'below'} / leiden "
            f"{'above' if het_leiden>7 else 'below'})  [{time.time()-t0:.0f}s]")
    results["GSE161277"] = {"het_true": het_true, "het_leiden": het_leiden, "same_side": same_side}

    buf.log("\n--- PBMC3k (negative control) ---")
    t0 = time.time()
    pbmc = sc.datasets.pbmc3k_processed()
    pts_pbmc = pbmc.obsm["X_pca"]
    if pts_pbmc.shape[1] < 50:
        buf.log(f"  Stored X_pca has only {pts_pbmc.shape[1]} dims -- recomputing 50 PCs.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.pca(pbmc, n_comps=50)
        pts_pbmc = pbmc.obsm["X_pca"]
    else:
        pts_pbmc = pts_pbmc[:, :50]
    y_true_pbmc = pbmc.obs["louvain"].astype(str).values
    het_true_pbmc = heterogeneity_score(pts_pbmc, y_true_pbmc)
    y_leiden_pbmc = leiden_labels(pts_pbmc, n_neighbors=15, resolution=0.5, seed=SEED)
    het_leiden_pbmc = heterogeneity_score(pts_pbmc, y_leiden_pbmc)
    buf.log(f"  n_cells={len(pts_pbmc)}  n_true_types={len(np.unique(y_true_pbmc))}  "
            f"n_leiden_clusters={len(np.unique(y_leiden_pbmc))}")
    buf.log(f"  heterogeneity_score: true_labels={het_true_pbmc:.3f}  leiden_labels={het_leiden_pbmc:.3f}")
    same_side_pbmc = (het_true_pbmc > 7) == (het_leiden_pbmc > 7)
    buf.log(f"  Same side of threshold 7? {'YES' if same_side_pbmc else 'NO'} "
            f"(true {'above' if het_true_pbmc>7 else 'below'} / leiden "
            f"{'above' if het_leiden_pbmc>7 else 'below'})  [{time.time()-t0:.0f}s]")
    results["PBMC3k"] = {"het_true": het_true_pbmc, "het_leiden": het_leiden_pbmc,
                          "same_side": same_side_pbmc}

    buf.log("\n  Recomputing GFT vs PCA K-Means ARI on PBMC3k with both label sets:")
    W_pbmc = build_graph(pts_pbmc, k_nn=K_NN, method="cosine")
    gft_pbmc = gft_embed(W_pbmc, k_eig=K_EIG_DEFAULT, random_state=SEED)
    for label_name, y in [("true_labels", y_true_pbmc), ("leiden_labels", y_leiden_pbmc)]:
        k = len(np.unique(y))
        buf.log(f"  [{label_name}, k={k}]")
        pca_mean, pca_std = kmeans_ari(pts_pbmc, y, k, buf, "    PCA")
        gft_mean, gft_std = kmeans_ari(gft_pbmc, y, k, buf, "    GFT")
        buf.log(f"    ratio GFT/PCA = {gft_mean/max(pca_mean,1e-9):.2f}x")

    buf.log(f"\nTEST A VERDICT: label-free score usable = "
            f"{'YES' if results['GSE161277']['same_side'] and results['PBMC3k']['same_side'] else 'NO'} "
            f"(both datasets must stay on the same side)")
    return results


def test_b(buf, pts_pca, y_true):
    buf.log("\n" + "=" * 96)
    buf.log("TEST B -- Leiden resolution sweep (GSE161277)")
    buf.log("=" * 96)
    buf.log("DECISION (stated before the numbers): 'structural incompatibility' survives only if")
    buf.log("GFT's best Leiden ARI (either k_eig, any resolution) stays CLEARLY below PCA's best")
    buf.log("(operationalized as PCA_best - GFT_best >= 0.02).\n")

    t0 = time.time()
    W = build_graph(pts_pca, k_nn=K_NN, method="cosine")
    buf.log(f"Cosine k={K_NN} graph built on PCA(50) points [{time.time()-t0:.0f}s], reused for "
            f"both GFT(k_eig=5) and GFT(k_eig=20).")
    gft5 = gft_embed(W, k_eig=5, random_state=SEED)
    gft20 = gft_embed(W, k_eig=20, random_state=SEED)

    reps = {"PCA(50)": pts_pca, "GFT(k_eig=5)": gft5, "GFT(k_eig=20)": gft20}
    bests = {}
    all_rows = {}
    for name, pts in reps.items():
        t0 = time.time()
        rows, best = leiden_sweep(pts, y_true, RESOLUTIONS, name, buf)
        buf.log(f"  [{name}] sweep time: {time.time()-t0:.0f}s")
        bests[name] = best
        all_rows[name] = rows

    pca_best = bests["PCA(50)"]["ari"]
    gft_best = max(bests["GFT(k_eig=5)"]["ari"], bests["GFT(k_eig=20)"]["ari"])
    gap = pca_best - gft_best
    survives = gap >= 0.02
    buf.log(f"\nBest ARI: PCA={pca_best:.4f}  GFT(best of k=5,20)={gft_best:.4f}  gap={gap:+.4f}")
    buf.log(f"TEST B VERDICT: structural incompatibility claim "
            f"{'SURVIVES' if survives else 'DOES NOT SURVIVE'} (gap >= 0.02 required)")
    return {"bests": bests, "rows": all_rows, "gft5": gft5, "gft20": gft20, "W_pca": W}


def test_c(buf, adata, pts_pca, y_true, test_b_results):
    buf.log("\n" + "=" * 96)
    buf.log("TEST C -- Harmony baseline (GSE161277)")
    buf.log("=" * 96)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ho = harmonypy.run_harmony(pts_pca, adata.obs, ["sample"], random_state=SEED)
    Z = np.asarray(ho.Z_corr)
    pts_harmony = Z.T if Z.shape[0] == pts_pca.shape[1] else Z
    buf.log(f"Harmony (batch=sample, {adata.obs['sample'].nunique()} batches) on PCA(50) "
            f"[{time.time()-t0:.0f}s]. Embedding shape: {pts_harmony.shape}")

    t0 = time.time()
    W_harmony = build_graph(pts_harmony, k_nn=K_NN, method="cosine")
    gft_harmony = gft_embed(W_harmony, k_eig=K_EIG_DEFAULT, random_state=SEED)
    buf.log(f"GFT(k_eig={K_EIG_DEFAULT}) on Harmony graph built [{time.time()-t0:.0f}s]")

    k = len(np.unique(y_true))
    buf.log(f"\nK-Means ARI (k={k}, 10 seeds, n_init=10):")
    reps = {
        "PCA": pts_pca,
        "Harmony": pts_harmony,
        "GFT-on-PCA-graph (k_eig=5)": test_b_results["gft5"],
        "GFT-on-Harmony-graph (k_eig=5)": gft_harmony,
    }
    kmeans_results = {}
    for name, pts in reps.items():
        mean, std = kmeans_ari(pts, y_true, k, buf, name)
        kmeans_results[name] = (mean, std)

    buf.log("\nBest Leiden ARI, same 7 resolutions:")
    buf.log(f"  PCA best (from Test B): resolution={test_b_results['bests']['PCA(50)']['resolution']}  "
            f"ARI={test_b_results['bests']['PCA(50)']['ari']:.4f}")
    _, harmony_best = leiden_sweep(pts_harmony, y_true, RESOLUTIONS, "Harmony", buf)

    buf.log(f"\nSummary: PCA best Leiden ARI={test_b_results['bests']['PCA(50)']['ari']:.4f}  "
            f"Harmony best Leiden ARI={harmony_best['ari']:.4f}")
    buf.log(f"K-Means: PCA={kmeans_results['PCA'][0]:.4f}  Harmony={kmeans_results['Harmony'][0]:.4f}  "
            f"GFT-on-PCA={kmeans_results['GFT-on-PCA-graph (k_eig=5)'][0]:.4f}  "
            f"GFT-on-Harmony={kmeans_results['GFT-on-Harmony-graph (k_eig=5)'][0]:.4f}")
    best_km = max(kmeans_results.items(), key=lambda kv: kv[1][0])
    buf.log(f"Best K-Means ARI overall: {best_km[0]} ({best_km[1][0]:.4f})")

    return {"kmeans": kmeans_results, "harmony_leiden_best": harmony_best, "pts_harmony": pts_harmony}


def main():
    buf = Buf()
    buf.log("GFT reviewer checks (label-free heterogeneity, Leiden resolution sweep, Harmony baseline)")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["numpy", "scipy", "scikit-learn", "scanpy", "harmonypy", "leidenalg"]))
    buf.log(f"seed={SEED}  k_nn={K_NN}  kmeans_seeds={KMEANS_SEEDS}  kmeans_n_init={KMEANS_N_INIT}  "
            f"resolutions={RESOLUTIONS}\n")

    t_start = time.time()
    results_a = test_a(buf)

    buf.log("\n\nLoading GSE161277 for Tests B and C...")
    adata = sc.read_h5ad(GSE_H5AD)
    pts_pca = adata.obsm["X_pca"][:, :50]
    y_true = adata.obs["cell_type"].astype(str).values

    results_b = test_b(buf, pts_pca, y_true)
    results_c = test_c(buf, adata, pts_pca, y_true, results_b)

    buf.log("\n" + "=" * 96)
    buf.log(f"TOTAL RUNTIME: {time.time()-t_start:.0f}s")
    buf.log("=" * 96)

    buf.log("\n" + "=" * 96)
    buf.log("OVERALL SUMMARY")
    buf.log("=" * 96)
    buf.log(f"Test A: label-free heterogeneity score usable = "
            f"{'YES' if results_a['GSE161277']['same_side'] and results_a['PBMC3k']['same_side'] else 'NO'}")
    pca_best_b = results_b["bests"]["PCA(50)"]["ari"]
    gft_best_b = max(results_b["bests"]["GFT(k_eig=5)"]["ari"], results_b["bests"]["GFT(k_eig=20)"]["ari"])
    buf.log(f"Test B: structural incompatibility claim "
            f"{'SURVIVES' if pca_best_b - gft_best_b >= 0.02 else 'DOES NOT SURVIVE'} "
            f"(PCA best={pca_best_b:.4f}, GFT best={gft_best_b:.4f})")
    harmony_km = results_c["kmeans"]["Harmony"][0]
    pca_km = results_c["kmeans"]["PCA"][0]
    buf.log(f"Test C: Harmony K-Means ARI ({harmony_km:.4f}) vs PCA K-Means ARI ({pca_km:.4f}) -- "
            f"Harmony {'improves on' if harmony_km > pca_km else 'does not improve on'} PCA as a baseline")

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- All ARI values are relative to a single label set per dataset (cell_type for")
    buf.log("  GSE161277, louvain for PBMC3k); those labels are themselves the product of an")
    buf.log("  earlier clustering/annotation pipeline, not ground truth in an absolute sense.")
    buf.log("- The threshold of 7 for heterogeneity_score, and the >=0.02 margin used to call a Test")
    buf.log("  B gap 'clear', are both operationalizations chosen for this report, not externally")
    buf.log("  validated cutoffs.")
    buf.log("- Harmony is run once (random_state=42, single run); no seed-repeat variance is")
    buf.log("  reported for the Harmony correction step itself, only for the downstream K-Means.")
    buf.log("- GFT, PCA and Harmony embeddings are all derived from the same expression matrix and")
    buf.log("  the same original 50-D PCA; comparisons are internal to this pipeline.")

    buf.save(os.path.join(RESULTS, "gft_reviewer_checks.txt"))


if __name__ == "__main__":
    main()
