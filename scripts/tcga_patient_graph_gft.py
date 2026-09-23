"""TCGA COADREAD: patient-patient cosine kNN graph + GFT embedding, K-Means vs stage, survival by GFT cluster."""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from lifelines.statistics import multivariate_logrank_test

from gft import build_graph, gft_embed

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "coadread_tcga_pan_can_atlas_2018")
N_HVG = 2000
K_NN = 20
K_EIG = 20
K_CLUSTERS = 4
SEED = 42

STAGE_MAIN = {
    "STAGE I": "I", "STAGE IA": "I",
    "STAGE II": "II", "STAGE IIA": "II", "STAGE IIB": "II", "STAGE IIC": "II",
    "STAGE III": "III", "STAGE IIIA": "III", "STAGE IIIB": "III", "STAGE IIIC": "III",
    "STAGE IV": "IV", "STAGE IVA": "IV", "STAGE IVB": "IV",
}


def main():
    print("1) Loading mRNA-seq matrix, dropping Hugo_Symbol NaN rows, log2(x+1)...")
    mrna = pd.read_csv(os.path.join(DATA_DIR, "data_mrna_seq_v2_rsem.txt"), sep="\t")
    mrna = mrna.dropna(subset=["Hugo_Symbol"]).drop_duplicates(subset=["Hugo_Symbol"])
    sample_cols = [c for c in mrna.columns if c not in ("Hugo_Symbol", "Entrez_Gene_Id")]
    expr = mrna.set_index("Hugo_Symbol")[sample_cols]
    n_before = expr.shape[0]
    expr = expr.dropna(axis=0, how="any")  # ~3015 genes are NaN across a batch of 227 samples (platform artifact)
    print(f"   genes after Hugo_Symbol dropna: {n_before}, further dropped {n_before - expr.shape[0]} "
          f"genes with any-sample NaN (batch artifact) -> {expr.shape[0]} genes, samples: {expr.shape[1]}")
    expr_log = np.log2(expr.astype(float) + 1)

    print(f"\n2) Selecting top {N_HVG} highest-variance genes...")
    variances = expr_log.var(axis=1)
    hvg = variances.sort_values(ascending=False).index[:N_HVG]
    expr_hvg = expr_log.loc[hvg]
    print(f"   HVG matrix: {expr_hvg.shape}")

    pts = expr_hvg.T.values  # patients x genes
    patient_ids = [c[:12] for c in expr_hvg.columns]

    print(f"\n3-4) Cosine kNN graph (k={K_NN}) on {len(pts)} patients...")
    W = build_graph(pts, k_nn=K_NN, method="cosine")
    print(f"   W: {W.shape}, nnz={W.nnz}")

    print(f"\n5) Normalized Laplacian eigenvectors (k_eig={K_EIG})...")
    emb = gft_embed(W, k_eig=K_EIG)
    print(f"   embedding shape: {emb.shape}")

    print("\n6) Merging with clinical data on PATIENT_ID...")
    clinical = pd.read_csv(os.path.join(DATA_DIR, "data_clinical_patient.txt"), sep="\t", comment="#")
    clinical = clinical.set_index("PATIENT_ID")

    df = pd.DataFrame({"patient_id": patient_ids})
    df["stage_raw"] = df["patient_id"].map(clinical["AJCC_PATHOLOGIC_TUMOR_STAGE"])
    df["stage_main"] = df["stage_raw"].map(STAGE_MAIN)
    df["os_months"] = df["patient_id"].map(clinical["OS_MONTHS"])
    df["os_status_raw"] = df["patient_id"].map(clinical["OS_STATUS"])
    df["event"] = df["os_status_raw"].map(lambda x: 1 if isinstance(x, str) and x.startswith("1") else
                                           (0 if isinstance(x, str) and x.startswith("0") else np.nan))
    print(f"   matched clinical rows: {df['stage_raw'].notna().sum()}/{len(df)}")
    print(f"   stage_main distribution:\n{df['stage_main'].value_counts(dropna=False)}")

    print(f"\n7) K-Means (k={K_CLUSTERS}) on GFT embedding...")
    km = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10)
    df["gft_cluster"] = km.fit_predict(emb)

    print("\n8a) ARI vs AJCC_PATHOLOGIC_TUMOR_STAGE (main stage I-IV, complete cases only)...")
    mask_stage = df["stage_main"].notna()
    ari = adjusted_rand_score(df.loc[mask_stage, "stage_main"], df.loc[mask_stage, "gft_cluster"])
    print(f"    N={mask_stage.sum()}  ARI={ari:.4f}")

    print("\n8b) Median OS_MONTHS per GFT cluster...")
    med_os = df.groupby("gft_cluster")["os_months"].median()
    counts = df["gft_cluster"].value_counts().sort_index()
    for c in sorted(df["gft_cluster"].unique()):
        print(f"    cluster {c}: n={counts[c]:<5} median OS_MONTHS={med_os[c]:.2f}")

    print("\n8c) Log-rank test across GFT clusters (OS_MONTHS, event=OS_STATUS)...")
    mask_surv = df["os_months"].notna() & df["event"].notna()
    lr = multivariate_logrank_test(
        df.loc[mask_surv, "os_months"], df.loc[mask_surv, "gft_cluster"], df.loc[mask_surv, "event"]
    )
    print(f"    N={mask_surv.sum()}  chi2={lr.test_statistic:.4f}  p-value={lr.p_value:.4f}")

    print("\n" + "=" * 70)
    print("OZET")
    print("=" * 70)
    print(f"ARI (GFT cluster vs AJCC stage I-IV): {ari:.4f}")
    print("Median OS_MONTHS per GFT cluster:")
    for c in sorted(df["gft_cluster"].unique()):
        print(f"  cluster {c}: n={counts[c]:<5} median OS={med_os[c]:.2f} ay")
    print(f"Log-rank test: chi2={lr.test_statistic:.4f}, p-value={lr.p_value:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
