"""Patient1 carcinoma epithelial cells: Leiden cluster 14 vs 15 differential expression + marker panel."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "GSE161277_all.h5ad")
CARCINOMA_GSM = "GSM4904236"

MARKER_PANEL = {
    "Proliferasyon": ["MKI67", "TOP2A", "PCNA"],
    "Stemness": ["LGR5", "ASCL2", "SOX9"],
    "EMT (malignite)": ["VIM", "CDH2", "SNAI1", "ZEB1"],
    "Diferansiasyon": ["KRT20", "MUC2", "FABP1"],
}


def main():
    adata = sc.read_h5ad(PATH)
    raw = adata.raw  # keep a reference; avoid ever calling AnnData.copy() with .raw attached (OOM on this machine)

    mask = ((adata.obs["gsm_id"] == CARCINOMA_GSM) & (adata.obs["cell_type"] == "Epithelial/Tumor")
            & adata.obs["leiden"].isin(["14", "15"])).values
    n14 = ((adata.obs["gsm_id"].values == CARCINOMA_GSM) & (adata.obs["cell_type"].values == "Epithelial/Tumor")
           & (adata.obs["leiden"].values == "14")).sum()
    n15 = ((adata.obs["gsm_id"].values == CARCINOMA_GSM) & (adata.obs["cell_type"].values == "Epithelial/Tumor")
           & (adata.obs["leiden"].values == "15")).sum()
    print(f"Patient1 carcinoma epithelial cells -- Cluster 14: n={n14}, Cluster 15: n={n15}")

    # build a fresh small AnnData directly from the raw (full gene, log-normalized) matrix,
    # bypassing AnnData.copy()'s internal raw-copy path entirely
    sub_X = raw.X[mask]
    sub_obs = adata.obs.loc[mask, ["leiden"]].copy()
    sub_obs["leiden"] = sub_obs["leiden"].cat.remove_unused_categories()
    adata_sub = ad.AnnData(X=sub_X, obs=sub_obs, var=raw.var.copy())
    adata_sub.var_names = raw.var_names

    print("\n2) rank_genes_groups (Wilcoxon, cluster 14 vs 15, on raw log-normalized data)...")
    sc.tl.rank_genes_groups(adata_sub, groupby="leiden", groups=["14", "15"], reference="rest",
                             method="wilcoxon", use_raw=False)

    for group in ["14", "15"]:
        df = sc.get.rank_genes_groups_df(adata_sub, group=group).head(20)
        print(f"\n=== Top 20 marker genes for cluster {group} (vs the other cluster) ===")
        print(df[["names", "logfoldchanges", "pvals_adj", "scores"]].to_string(index=False))

    print("\n3) Marker panel expression (mean log-normalized) per cluster...")
    symbol_to_id = dict(zip(adata_sub.var["gene_symbol"], adata_sub.var_names))
    leiden_labels = adata_sub.obs["leiden"].values

    rows = []
    for category, genes in MARKER_PANEL.items():
        for gene in genes:
            if gene not in symbol_to_id:
                print(f"   WARNING: {gene} not found in gene set")
                continue
            gene_id = symbol_to_id[gene]
            expr = adata_sub[:, gene_id].X
            expr = np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()
            mean14 = expr[leiden_labels == "14"].mean()
            mean15 = expr[leiden_labels == "15"].mean()
            pct14 = (expr[leiden_labels == "14"] > 0).mean() * 100
            pct15 = (expr[leiden_labels == "15"] > 0).mean() * 100
            higher_in_15 = mean15 > mean14
            rows.append({"category": category, "gene": gene, "mean_c14": mean14, "mean_c15": mean15,
                         "pct_c14": pct14, "pct_c15": pct15, "higher_in_15": higher_in_15})

    df_markers = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(f"{'Kategori':<20}{'Gen':<10}{'mean_c14':<12}{'mean_c15':<12}{'%+_c14':<10}{'%+_c15':<10}{'c15>c14'}")
    print("-" * 100)
    for _, r in df_markers.iterrows():
        print(f"{r['category']:<20}{r['gene']:<10}{r['mean_c14']:<12.3f}{r['mean_c15']:<12.3f}"
              f"{r['pct_c14']:<10.1f}{r['pct_c15']:<10.1f}{'YES' if r['higher_in_15'] else 'no'}")
    print("=" * 100)

    print("\n4) Kategori bazinda ozet: kac gen kume 15'te yuksek?")
    for category in MARKER_PANEL:
        sub = df_markers[df_markers["category"] == category]
        n_higher_15 = sub["higher_in_15"].sum()
        print(f"   {category}: {n_higher_15}/{len(sub)} gen kume 15'te daha yuksek")


if __name__ == "__main__":
    main()
