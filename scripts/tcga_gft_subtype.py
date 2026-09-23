"""TCGA GFT embedding vs molecular SUBTYPE: ARI, cluster separation, log-rank survival test."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from lifelines.statistics import multivariate_logrank_test

from gft import build_graph, gft_embed

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "coadread_tcga_pan_can_atlas_2018")
N_HVG = 2000
K_NN = 20
K_EIG = 20
K_CLUSTERS = 4
SEED = 42


def main():
    print("1) Loading + preprocessing mRNA-seq matrix (same pipeline as before)...")
    mrna = pd.read_csv(os.path.join(DATA_DIR, "data_mrna_seq_v2_rsem.txt"), sep="\t")
    mrna = mrna.dropna(subset=["Hugo_Symbol"]).drop_duplicates(subset=["Hugo_Symbol"])
    sample_cols = [c for c in mrna.columns if c not in ("Hugo_Symbol", "Entrez_Gene_Id")]
    expr = mrna.set_index("Hugo_Symbol")[sample_cols]
    expr = expr.dropna(axis=0, how="any")
    expr_log = np.log2(expr.astype(float) + 1)

    variances = expr_log.var(axis=1)
    hvg = variances.sort_values(ascending=False).index[:N_HVG]
    expr_hvg = expr_log.loc[hvg]
    pts = expr_hvg.T.values
    patient_ids = [c[:12] for c in expr_hvg.columns]
    print(f"   {expr_hvg.shape[0]} genes x {expr_hvg.shape[1]} patients")

    print(f"\n2) Cosine kNN graph (k={K_NN}) + GFT embedding (k_eig={K_EIG})...")
    W = build_graph(pts, k_nn=K_NN, method="cosine")
    emb = gft_embed(W, k_eig=K_EIG)

    print("\n3) Merging clinical data...")
    clinical = pd.read_csv(os.path.join(DATA_DIR, "data_clinical_patient.txt"), sep="\t", comment="#")
    clinical = clinical.set_index("PATIENT_ID")

    df = pd.DataFrame({"patient_id": patient_ids})
    df["subtype_raw"] = df["patient_id"].map(clinical["SUBTYPE"])
    df["os_months"] = df["patient_id"].map(clinical["OS_MONTHS"])
    df["os_status_raw"] = df["patient_id"].map(clinical["OS_STATUS"])
    df["event"] = df["os_status_raw"].map(lambda x: 1 if isinstance(x, str) and x.startswith("1") else
                                           (0 if isinstance(x, str) and x.startswith("0") else np.nan))
    for i in range(emb.shape[1]):
        df[f"gft_{i}"] = emb[:, i]

    print("\n   Raw SUBTYPE distribution:")
    print(df["subtype_raw"].value_counts(dropna=False))

    # dataset does NOT actually contain "COAD_INDETERMINATE" as assumed in the request;
    # real categories are COAD_/READ_ x {CIN, MSI, GS, POLE}. Collapse to the 4 molecular
    # subtypes (ignore COAD/READ primary-site prefix) to get a genuine k=4 comparison.
    df["subtype_molecular"] = df["subtype_raw"].str.replace(r"^(COAD|READ)_", "", regex=True)
    print("\n   Collapsed molecular subtype (COAD/READ prefix removed) distribution:")
    print(df["subtype_molecular"].value_counts(dropna=False))

    print(f"\n4) K-Means (k={K_CLUSTERS}) on GFT embedding...")
    km = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10)
    df["gft_cluster"] = km.fit_predict(emb)

    print("\n=== ARI: GFT K-Means cluster vs molecular SUBTYPE (CIN/MSI/GS/POLE) ===")
    mask = df["subtype_molecular"].notna()
    ari = adjusted_rand_score(df.loc[mask, "subtype_molecular"], df.loc[mask, "gft_cluster"])
    print(f"N={mask.sum()}  ARI={ari:.4f}")

    print("\n=== Confusion: GFT cluster x molecular SUBTYPE ===")
    conf = pd.crosstab(df.loc[mask, "gft_cluster"], df.loc[mask, "subtype_molecular"])
    print(conf)

    print("\n=== Separation check: silhouette score of SUBTYPE labels on GFT embedding ===")
    if mask.sum() > 0:
        sil = silhouette_score(emb[mask.values], df.loc[mask, "subtype_molecular"])
        print(f"Silhouette score (SUBTYPE as labels, on GFT embedding): {sil:.4f}  "
              f"(near 0 or negative = no separation, close to 1 = well separated)")

    print("\n=== GFT embedding centroids per SUBTYPE (first 3 dims) ===")
    for st in sorted(df.loc[mask, "subtype_molecular"].unique()):
        sub = df.loc[mask & (df["subtype_molecular"] == st)]
        centroid = sub[[f"gft_{i}" for i in range(3)]].mean().values
        print(f"  {st:<8} n={len(sub):<4} centroid(dim0-2)={np.round(centroid, 4)}")

    print("\n=== Pairwise centroid distances between SUBTYPE groups (full 20D embedding) ===")
    subtypes = sorted(df.loc[mask, "subtype_molecular"].unique())
    centroids = {st: df.loc[mask & (df["subtype_molecular"] == st), [f"gft_{i}" for i in range(K_EIG)]].mean().values
                 for st in subtypes}
    for i, a in enumerate(subtypes):
        for b in subtypes[i + 1:]:
            d = np.linalg.norm(centroids[a] - centroids[b])
            print(f"  {a} <-> {b}: {d:.4f}")
    within_std = {st: df.loc[mask & (df["subtype_molecular"] == st), [f"gft_{i}" for i in range(K_EIG)]].values.std()
                  for st in subtypes}
    print("Within-group std (spread) per subtype:", {k: round(v, 4) for k, v in within_std.items()})

    print("\n=== Log-rank test: survival difference across molecular SUBTYPE groups ===")
    mask_surv = mask & df["os_months"].notna() & df["event"].notna()
    lr = multivariate_logrank_test(
        df.loc[mask_surv, "os_months"], df.loc[mask_surv, "subtype_molecular"], df.loc[mask_surv, "event"]
    )
    print(f"N={mask_surv.sum()}  chi2={lr.test_statistic:.4f}  p-value={lr.p_value:.4f}")
    print("\nMedian OS_MONTHS per SUBTYPE:")
    print(df.loc[mask_surv].groupby("subtype_molecular")["os_months"].median())

    print("\n" + "=" * 70)
    print("OZET")
    print("=" * 70)
    print(f"ARI (GFT k=4 cluster vs molecular SUBTYPE): {ari:.4f}")
    print(f"Silhouette (SUBTYPE labels on GFT embedding): {sil:.4f}")
    print(f"Log-rank (SUBTYPE, survival): chi2={lr.test_statistic:.4f}, p={lr.p_value:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
