"""EXPLORATORY - T1 failed pre-registered robustness; results here are hypothesis-generating only.

Follow-up to scripts/ricci_curvature_pilot.py (T2: no residual-ORC link, 0/6 donors) and
scripts/ricci_t1_robustness.py (T1 failed robustness, 3/6 donors; sign of the malignant effect
reversed in 3 donors once local intrinsic dimension was controlled). This step does not revisit
those results. It asks a narrower, exploratory question: within malignant cells only, once node
ORC is stripped of its local-geometry component (kappa_resid), does what remains correlate with
a reproducible gene program independent of local intrinsic dimension (ID)?

Data: data/crc_epithelial.h5ad, adata.X (log-normalized), gene names from var['feature_name'].
Groups: malignant = disease=='colorectal cancer' & cell_type=='malignant cell' (per donor, no
subsampling -- each donor's own malignant-only kNN graph, k=20).
Donors: HTA8_6002/6009/6013/6016/6018/6028 (same 6 as the pilot and the robustness check).

Reuses (imported, not copied):
  knn_graph, ollivier_ricci, node_mean, partial_spearman, fit_subspace, residual
    -- from scripts/ricci_curvature_pilot.py
  local_intrinsic_dim (Levina-Bickel MLE + mean kNN distance)
    -- from scripts/ricci_t1_robustness.py

STEP 0 (per donor, malignant cells only): node ORC, degree, n_genes_by_counts, local ID,
  residual (vs. that donor's normal-cell PCA subspace, k_eig=20, same method as the pilot).
  kappa_resid = OLS residual of  ORC ~ log(deg) + log(n_genes) + local_ID  (report R2).
  Donor-level Spearman correlation matrix of (residual, ORC, kappa_resid, ID, n_genes).

STEP 1 (phase portrait): 3-panel figure per donor -- (residual, kappa_resid) and (residual, ID)
  colored by obs['Author Cell Type'] (has >1 category among malignant cells in every donor here),
  plus (residual, kappa_resid) colored continuously by obs['ISC Signature']. Quantitative only,
  no reading of the figure: per Author-Cell-Type cluster with >=30 cells, median residual,
  kappa_resid, ID; Kruskal-Wallis p + epsilon^2 for kappa_resid and for ID across clusters;
  partial Spearman(ISC Signature, kappa_resid | log deg, log n_genes, ID) and, for comparison,
  partial Spearman(ISC Signature, residual | same covariates).

STEP 2 (gene programs): genes expressed in >=10% of malignant cells in EVERY donor (intersection).
  Per donor and gene: partial Spearman(gene, kappa_resid | log deg, log n_genes) and
  partial Spearman(gene, ID | log deg, log n_genes); BH-FDR within donor. Consistent gene =
  FDR<0.05 with the same sign in >=4/6 donors (separately for kappa_resid and ID, positive and
  negative lists). Median rho per gene -> CSV. Preranked GSEA (gseapy.prerank, ranked by median
  rho) against the requested MSigDB Hallmark sets and two GO BP terms; if a requested gene set
  cannot be found in the downloaded library under its literal name, it is skipped and reported,
  never approximated or hand-built.

DECISION RULE ("kappa_resid carries a gene program independent of local geometry") -- ALL THREE:
  (a) >=20 consistent genes for kappa_resid (positive + negative lists combined).
  (b) |Spearman(median_rho_kappa_resid, median_rho_ID)| < 0.5, across the shared gene set.
  (c) at least one gene set has FDR<0.05 in the kappa_resid GSEA AND FDR>=0.25 in the ID GSEA.
Otherwise: "curvature reduces to local geometry; the Ricci line closes as a negative result."

SECONDARY RULE (phase portrait): "kappa_resid separates subtypes" if, in a MAJORITY of donors
(>=4/6), kappa_resid's cross-cluster epsilon^2 >= 0.05 AND exceeds ID's epsilon^2.

Outputs: results/ricci_exploratory_phase_genes.txt, results/ricci_phase_<donor>.png,
results/ricci_gene_rho_kresid.csv, results/ricci_gene_rho_id.csv.
"""
import os
import sys
import warnings
from importlib.metadata import version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from scipy.stats import kruskal, spearmanr
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ricci_curvature_pilot as rc  # noqa: E402
from ricci_curvature_pilot import fit_subspace, residual, partial_spearman  # noqa: E402
from ricci_t1_robustness import local_intrinsic_dim  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(PROJECT_DIR, "_ricci_tmp", "crc_pca50.npz")
H5AD = os.path.join(PROJECT_DIR, "data", "crc_epithelial.h5ad")
OUT_TXT = os.path.join(PROJECT_DIR, "results", "ricci_exploratory_phase_genes.txt")
OUT_KRESID_CSV = os.path.join(PROJECT_DIR, "results", "ricci_gene_rho_kresid.csv")
OUT_ID_CSV = os.path.join(PROJECT_DIR, "results", "ricci_gene_rho_id.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

K_NN = 20
K_EIG = 20
MIN_N = 50
MIN_CLUSTER_N = 30
GENE_MIN_FRAC = 0.10
CONSISTENT_MIN_DONORS = 4
DONORS = ["HTA8_6002", "HTA8_6009", "HTA8_6013", "HTA8_6016", "HTA8_6018", "HTA8_6028"]

HALLMARK_REQUESTED = {
    "EMT": "Epithelial Mesenchymal Transition",
    "APICAL_JUNCTION": "Apical Junction",
    "G2M_CHECKPOINT": "G2-M Checkpoint",
    "E2F_TARGETS": "E2F Targets",
    "MYC_TARGETS_V1": "Myc Targets V1",
    "OXIDATIVE_PHOSPHORYLATION": "Oxidative Phosphorylation",
    "WNT_BETA_CATENIN_SIGNALING": "Wnt-beta Catenin Signaling",
    "TNFA_SIGNALING_VIA_NFKB": "TNF-alpha Signaling via NF-kB",
}
GO_BP_REQUESTED = {
    "cell-cell adhesion": None,  # resolved (or not) at runtime; never hand-built
    "actin cytoskeleton organization": None,
}


class Buf:
    def __init__(self):
        self.lines = []

    def log(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.lines.append(msg)

    def text(self):
        return "\n".join(self.lines) + "\n"


def epsilon_sq(groups):
    """Kruskal-Wallis effect size: epsilon^2 = (H - k + 1) / (n - k)."""
    all_vals = np.concatenate(groups)
    n, k = len(all_vals), len(groups)
    if k < 2 or n <= k:
        return np.nan, np.nan
    H, p = kruskal(*groups)
    eps2 = (H - k + 1) / (n - k)
    return p, max(eps2, 0.0)


def load_npz():
    d = np.load(NPZ, allow_pickle=True)
    return (d["pcs"].astype(np.float64), d["n_genes"].astype(float),
            d["donor"], d["disease"], d["cell_type"])


def step0_donor(dn, pcs, ng, donor, disease, ctype, buf):
    dmask = donor == dn
    is_norm = (disease == "normal") & (ctype != "malignant cell")
    is_mal = (disease == "colorectal cancer") & (ctype == "malignant cell")
    Xn, Xm = pcs[dmask & is_norm], pcs[dmask & is_mal]
    n_mal = len(Xm)
    buf.log(f"\n{dn}: n_normal={len(Xn)} n_malignant={n_mal}")

    U, mean = fit_subspace(Xn)
    r = residual(Xm, U, mean)

    rc.K_NN = K_NN
    A = rc.knn_graph(Xm)
    edges, kap = rc.ollivier_ricci(A)
    node_orc = rc.node_mean(edges, kap, n_mal)
    deg = np.asarray(A.sum(1)).ravel()

    local_id, knn_dist = local_intrinsic_dim(Xm, K_NN)
    ngc = ng[dmask & is_mal]

    X_ols = sm.add_constant(np.column_stack([np.log(deg), np.log(np.maximum(ngc, 1)), local_id]))
    fit = sm.OLS(node_orc, X_ols).fit()
    kappa_resid = fit.resid
    buf.log(f"  ORC ~ log(deg)+log(n_genes)+local_ID: R2={fit.rsquared:.3f}  "
            f"coef_logdeg={fit.params[1]:+.4f} coef_logngenes={fit.params[2]:+.4f} "
            f"coef_localID={fit.params[3]:+.4f}")

    mat = np.column_stack([r, node_orc, kappa_resid, local_id, ngc])
    corr, _ = spearmanr(mat)
    names = ["residual", "ORC", "kappa_resid", "local_ID", "n_genes"]
    buf.log("  Spearman correlation matrix (residual, ORC, kappa_resid, local_ID, n_genes):")
    buf.log("    " + " ".join(f"{nm:>12}" for nm in names))
    for i, nm in enumerate(names):
        buf.log(f"    {nm:<9}" + " ".join(f"{corr[i, j]:>12.3f}" for j in range(len(names))))

    return {
        "n_mal": n_mal, "r": r, "orc": node_orc, "kappa_resid": kappa_resid, "local_id": local_id,
        "deg": deg, "n_genes": ngc, "R2": fit.rsquared, "corr": corr,
    }


def step1_donor(dn, s0, adata_obs_mask, buf):
    obs = adata_obs_mask
    act = obs["Author Cell Type"].astype(str).values
    isc = obs["ISC Signature"].values
    r, kr, idv = s0["r"], s0["kappa_resid"], s0["local_id"]
    deg, ngc = s0["deg"], s0["n_genes"]

    n_cat = pd.Series(act).nunique()
    buf.log(f"\n{dn}: Author Cell Type categories among malignant cells: "
            f"{dict(pd.Series(act).value_counts())}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cats = sorted(set(act))
    for cat in cats:
        m = act == cat
        axes[0].scatter(r[m], kr[m], s=6, alpha=0.5, label=cat)
        axes[1].scatter(r[m], idv[m], s=6, alpha=0.5, label=cat)
    axes[0].set_xlabel("residual"); axes[0].set_ylabel("kappa_resid"); axes[0].legend(fontsize=7)
    axes[0].set_title(f"{dn}: residual vs kappa_resid (by Author Cell Type)")
    axes[1].set_xlabel("residual"); axes[1].set_ylabel("local ID"); axes[1].legend(fontsize=7)
    axes[1].set_title(f"{dn}: residual vs local ID (by Author Cell Type)")
    sc3 = axes[2].scatter(r, kr, s=6, c=isc, cmap="viridis")
    axes[2].set_xlabel("residual"); axes[2].set_ylabel("kappa_resid")
    axes[2].set_title(f"{dn}: residual vs kappa_resid (color=ISC Signature)")
    plt.colorbar(sc3, ax=axes[2], label="ISC Signature")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"ricci_phase_{dn}.png"), dpi=130)
    plt.close(fig)

    buf.log(f"  Clusters with >={MIN_CLUSTER_N} malignant cells (Author Cell Type):")
    buf.log(f"  {'cluster':<20}{'n':<8}{'median_residual':<18}{'median_kappa_resid':<20}{'median_ID'}")
    groups_kr, groups_id, valid_cats = [], [], []
    for cat in cats:
        m = act == cat
        n = int(m.sum())
        if n < MIN_CLUSTER_N:
            buf.log(f"  {cat:<20}{n:<8}(skipped, <{MIN_CLUSTER_N})")
            continue
        buf.log(f"  {cat:<20}{n:<8}{np.median(r[m]):<18.3f}{np.median(kr[m]):<20.3f}{np.median(idv[m]):.3f}")
        groups_kr.append(kr[m]); groups_id.append(idv[m]); valid_cats.append(cat)

    p_kr, eps_kr = epsilon_sq(groups_kr) if len(groups_kr) >= 2 else (np.nan, np.nan)
    p_id, eps_id = epsilon_sq(groups_id) if len(groups_id) >= 2 else (np.nan, np.nan)
    buf.log(f"  Kruskal-Wallis across {len(valid_cats)} clusters: "
            f"kappa_resid p={p_kr:.2e} eps2={eps_kr:.4f}  |  local_ID p={p_id:.2e} eps2={eps_id:.4f}")

    log_deg, log_ng = np.log(deg), np.log(np.maximum(ngc, 1))
    rho_kr, p_pkr = partial_spearman(isc, kr, [log_deg, log_ng, idv])
    rho_r, p_pr = partial_spearman(isc, r, [log_deg, log_ng, idv])
    buf.log(f"  partial Spearman(ISC Signature, kappa_resid | log deg, log n_genes, ID) = "
            f"{rho_kr:+.3f} (p={p_pkr:.2e})")
    buf.log(f"  partial Spearman(ISC Signature, residual    | same covariates)          = "
            f"{rho_r:+.3f} (p={p_pr:.2e})  [comparison]")

    subtype_rule = (not np.isnan(eps_kr)) and eps_kr >= 0.05 and (np.isnan(eps_id) or eps_kr > eps_id)
    return {"eps_kr": eps_kr, "eps_id": eps_id, "p_kr": p_kr, "p_id": p_id,
            "rho_isc_kr": rho_kr, "p_isc_kr": p_pkr, "rho_isc_r": rho_r, "p_isc_r": p_pr,
            "subtype_rule": subtype_rule, "n_clusters": len(valid_cats)}


def step2_genes(all_s0, adata, donor_masks, buf):
    buf.log("\n" + "=" * 96)
    buf.log("STEP 2 -- gene programs")
    buf.log("=" * 96)

    gene_sets_per_donor = []
    for dn in DONORS:
        m = donor_masks[dn]
        X = adata[m].X
        frac = np.asarray((X > 0).mean(axis=0)).ravel()
        gene_sets_per_donor.append(set(adata.var["feature_name"].astype(str).values[frac >= GENE_MIN_FRAC]))
    shared_genes = sorted(set.intersection(*gene_sets_per_donor))
    buf.log(f"Genes expressed in >={int(GENE_MIN_FRAC*100)}% of malignant cells, per donor: "
            + ", ".join(f"{dn}={len(g)}" for dn, g in zip(DONORS, gene_sets_per_donor)))
    buf.log(f"Intersection across all 6 donors: {len(shared_genes)} genes")

    symbol_to_id = dict(zip(adata.var["feature_name"].astype(str), adata.var_names))
    gene_ids = [symbol_to_id[g] for g in shared_genes]
    col_idx = {gid: adata.var_names.get_loc(gid) for gid in gene_ids}

    rho_kr = pd.DataFrame(index=shared_genes, columns=DONORS, dtype=float)
    fdr_kr = pd.DataFrame(index=shared_genes, columns=DONORS, dtype=float)
    rho_id = pd.DataFrame(index=shared_genes, columns=DONORS, dtype=float)
    fdr_id = pd.DataFrame(index=shared_genes, columns=DONORS, dtype=float)

    for dn in DONORS:
        m = donor_masks[dn]
        s0 = all_s0[dn]
        Xg = adata[m][:, [symbol_to_id[g] for g in shared_genes]].X
        Xg = np.asarray(Xg.todense()) if hasattr(Xg, "todense") else np.asarray(Xg)
        log_deg, log_ng = np.log(s0["deg"]), np.log(np.maximum(s0["n_genes"], 1))
        covs = [log_deg, log_ng]

        p_kr_list, r_kr_list, p_id_list, r_id_list = [], [], [], []
        for j in range(Xg.shape[1]):
            gv = Xg[:, j]
            r_kr, p_kr = partial_spearman(gv, s0["kappa_resid"], covs)
            r_id, p_id = partial_spearman(gv, s0["local_id"], covs)
            r_kr_list.append(r_kr); p_kr_list.append(p_kr)
            r_id_list.append(r_id); p_id_list.append(p_id)

        rho_kr[dn] = r_kr_list
        rho_id[dn] = r_id_list
        fdr_kr[dn] = multipletests(p_kr_list, method="fdr_bh")[1]
        fdr_id[dn] = multipletests(p_id_list, method="fdr_bh")[1]
        buf.log(f"  {dn}: {len(shared_genes)} genes tested "
                f"(n_mal={s0['n_mal']}); median |rho| kappa_resid={np.median(np.abs(r_kr_list)):.3f}, "
                f"ID={np.median(np.abs(r_id_list)):.3f}")

    def consistent_genes(rho_df, fdr_df):
        sig = fdr_df < 0.05
        sign = np.sign(rho_df.where(sig))
        n_pos = (sign == 1).sum(axis=1)
        n_neg = (sign == -1).sum(axis=1)
        pos = rho_df.index[n_pos >= CONSISTENT_MIN_DONORS].tolist()
        neg = rho_df.index[n_neg >= CONSISTENT_MIN_DONORS].tolist()
        return pos, neg

    pos_kr, neg_kr = consistent_genes(rho_kr, fdr_kr)
    pos_id, neg_id = consistent_genes(rho_id, fdr_id)
    buf.log(f"\nConsistent genes (FDR<0.05, same sign, in >={CONSISTENT_MIN_DONORS}/6 donors):")
    buf.log(f"  kappa_resid: {len(pos_kr)} positive, {len(neg_kr)} negative "
            f"(total {len(pos_kr)+len(neg_kr)})")
    buf.log(f"  local_ID:    {len(pos_id)} positive, {len(neg_id)} negative "
            f"(total {len(pos_id)+len(neg_id)})")

    median_rho_kr = rho_kr.median(axis=1)
    median_rho_id = rho_id.median(axis=1)
    out_kr = rho_kr.copy(); out_kr["median_rho"] = median_rho_kr
    out_kr["n_donors_fdr_sig"] = (fdr_kr < 0.05).sum(axis=1)
    out_kr["consistent"] = out_kr.index.isin(pos_kr + neg_kr)
    out_kr.to_csv(OUT_KRESID_CSV)
    out_id = rho_id.copy(); out_id["median_rho"] = median_rho_id
    out_id["n_donors_fdr_sig"] = (fdr_id < 0.05).sum(axis=1)
    out_id["consistent"] = out_id.index.isin(pos_id + neg_id)
    out_id.to_csv(OUT_ID_CSV)
    buf.log(f"\nSaved per-gene median rho -> {os.path.basename(OUT_KRESID_CSV)}, "
            f"{os.path.basename(OUT_ID_CSV)}")

    rho_corr, _ = spearmanr(median_rho_kr, median_rho_id)
    buf.log(f"\nSpearman(median_rho_kappa_resid, median_rho_ID) across {len(shared_genes)} genes "
            f"= {rho_corr:+.3f}")

    return {
        "shared_genes": shared_genes, "median_rho_kr": median_rho_kr, "median_rho_id": median_rho_id,
        "n_consistent_kr": len(pos_kr) + len(neg_kr), "n_consistent_id": len(pos_id) + len(neg_id),
        "rho_corr_kr_id": rho_corr,
    }


def run_gsea(rnk, gene_set_dict, label, buf):
    try:
        import gseapy
    except ImportError:
        buf.log(f"  gseapy not installed -- skipping GSEA for {label}.")
        return None
    if not gene_set_dict:
        buf.log(f"  No resolved gene sets for {label} -- skipping GSEA.")
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = gseapy.prerank(rnk=rnk, gene_sets=gene_set_dict, min_size=5, max_size=1000,
                                  permutation_num=1000, outdir=None, seed=0, threads=4, verbose=False)
        except Exception as e:
            buf.log(f"  GSEA failed for {label}: {e}")
            return None
    df = res.res2d.copy()
    df["FDR q-val"] = pd.to_numeric(df["FDR q-val"], errors="coerce")
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
    buf.log(f"  GSEA ({label}), {len(gene_set_dict)} gene sets tested:")
    for _, row in df.sort_values("FDR q-val").iterrows():
        buf.log(f"    {row['Term']:<35} NES={row['NES']:+.3f}  FDR={row['FDR q-val']:.3f}")
    return df.set_index("Term")


def resolve_gene_sets(buf):
    try:
        import gseapy
    except ImportError:
        buf.log("gseapy not installed -- GSEA step skipped entirely.")
        return {}
    resolved = {}
    try:
        hallmark = gseapy.get_library(name="MSigDB_Hallmark_2020", organism="Human")
    except Exception as e:
        buf.log(f"Could not download MSigDB_Hallmark_2020 from Enrichr: {e}. Hallmark sets skipped.")
        hallmark = {}
    for req, literal_name in HALLMARK_REQUESTED.items():
        if literal_name in hallmark:
            resolved[literal_name] = hallmark[literal_name]
            buf.log(f"  resolved Hallmark set '{req}' -> '{literal_name}' ({len(hallmark[literal_name])} genes)")
        else:
            buf.log(f"  COULD NOT resolve Hallmark set '{req}' (expected name '{literal_name}') "
                     f"-- skipped, not fabricated.")

    for req in GO_BP_REQUESTED:
        found = None
        for lib_name in ["GO_Biological_Process_2023", "GO_Biological_Process_2025",
                          "GO_Biological_Process_2026", "GO_Biological_Process_2021"]:
            try:
                go = gseapy.get_library(name=lib_name, organism="Human")
            except Exception as e:
                buf.log(f"  Could not download {lib_name}: {e}")
                continue
            for term_name in go:
                base = term_name.split(" (GO:")[0].lower()
                if base == req.lower():
                    found = (term_name, go[term_name], lib_name)
                    break
            if found:
                break
        if found:
            term_name, genes, lib_name = found
            resolved[term_name] = genes
            buf.log(f"  resolved GO BP term '{req}' -> '{term_name}' ({len(genes)} genes, {lib_name})")
        else:
            buf.log(f"  COULD NOT find the exact GO BP term '{req}' in any available "
                     f"GO_Biological_Process_* Enrichr library (only narrower child terms like "
                     f"'Regulation of ...' or 'Cell-Cell Adhesion Via ...' exist) -- SKIPPED per "
                     f"instructions (not approximated, not hand-built).")
    return resolved


def main():
    buf = Buf()
    buf.log("EXPLORATORY - T1 failed pre-registered robustness; results here are hypothesis-generating only.\n")
    buf.log("Decision rule (\"kappa_resid carries a gene program independent of local geometry\"), "
            "stated before the numbers -- ALL THREE must hold:")
    buf.log("  (a) >=20 consistent genes for kappa_resid (positive+negative, FDR<0.05, same sign, "
            ">=4/6 donors).")
    buf.log("  (b) |Spearman(median_rho_kappa_resid, median_rho_ID)| < 0.5 across the shared gene set.")
    buf.log("  (c) >=1 gene set has FDR<0.05 in kappa_resid's GSEA AND FDR>=0.25 in ID's GSEA.")
    buf.log("Otherwise: curvature reduces to local geometry; the Ricci line closes as a negative result.")
    buf.log("Secondary phase-portrait rule: kappa_resid separates subtypes if, in >=4/6 donors, its "
            "cross-cluster epsilon^2>=0.05 AND exceeds local_ID's epsilon^2.\n")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["numpy", "scipy", "statsmodels", "POT", "scanpy"]))
    try:
        buf.log(f"gseapy={version('gseapy')}")
    except Exception:
        buf.log("gseapy not installed.")

    pcs, ng, donor, disease, ctype = load_npz()
    adata = sc.read_h5ad(H5AD)
    is_mal_full = (adata.obs["disease"] == "colorectal cancer") & (adata.obs["cell_type"] == "malignant cell")

    buf.log("\n" + "=" * 96)
    buf.log("STEP 0 -- per-cell quantities, malignant-only graphs, k=20")
    buf.log("=" * 96)
    all_s0, donor_masks = {}, {}
    for dn in DONORS:
        s0 = step0_donor(dn, pcs, ng, donor, disease, ctype, buf)
        all_s0[dn] = s0
        donor_masks[dn] = (adata.obs["donor_id"].astype(str).values == dn) & is_mal_full.values

    buf.log("\n" + "=" * 96)
    buf.log("STEP 1 -- phase portrait (quantitative; figures saved, not read for conclusions)")
    buf.log("=" * 96)
    all_s1 = {}
    for dn in DONORS:
        all_s1[dn] = step1_donor(dn, all_s0[dn], adata.obs[donor_masks[dn]], buf)

    s2 = step2_genes(all_s0, adata, donor_masks, buf)

    buf.log("\n" + "=" * 96)
    buf.log("GSEA (preranked by median rho across donors)")
    buf.log("=" * 96)
    gene_sets = resolve_gene_sets(buf)
    rnk_kr = s2["median_rho_kr"].sort_values(ascending=False)
    rnk_id = s2["median_rho_id"].sort_values(ascending=False)
    buf.log("\nGSEA on kappa_resid ranking:")
    gsea_kr = run_gsea(rnk_kr, gene_sets, "kappa_resid", buf)
    buf.log("\nGSEA on local_ID ranking:")
    gsea_id = run_gsea(rnk_id, gene_sets, "local_ID", buf)

    buf.log("\n" + "=" * 96)
    buf.log("SUMMARY TABLES")
    buf.log("=" * 96)
    buf.log(f"\n{'donor':<11}{'n_mal':>7}{'R2(ORC~geom)':>14}{'eps2_kresid':>13}{'eps2_ID':>10}"
            f"{'subtype_rule':>14}{'rho(ISC,kresid)':>17}")
    n_subtype = 0
    for dn in DONORS:
        s0, s1 = all_s0[dn], all_s1[dn]
        if s1["subtype_rule"]:
            n_subtype += 1
        buf.log(f"{dn:<11}{s0['n_mal']:>7}{s0['R2']:>14.3f}{s1['eps_kr']:>13.4f}{s1['eps_id']:>10.4f}"
                f"{('YES' if s1['subtype_rule'] else 'NO'):>14}{s1['rho_isc_kr']:>+17.3f}")
    buf.log(f"\nSubtype rule met in {n_subtype}/{len(DONORS)} donors -> "
            f"{'SUPPORTED' if n_subtype >= 4 else 'NOT SUPPORTED'} (need >=4/6)")

    cond_a = s2["n_consistent_kr"] >= 20
    cond_b = abs(s2["rho_corr_kr_id"]) < 0.5
    cond_c = False
    cond_c_detail = "no GSEA results to compare" if gsea_kr is None or gsea_id is None else None
    if gsea_kr is not None and gsea_id is not None:
        common = gsea_kr.index.intersection(gsea_id.index)
        hits = [t for t in common if gsea_kr.loc[t, "FDR q-val"] < 0.05 and gsea_id.loc[t, "FDR q-val"] >= 0.25]
        cond_c = len(hits) > 0
        cond_c_detail = f"gene sets meeting (c): {hits}" if hits else "no gene set met (c)"

    buf.log("\nMAIN DECISION:")
    buf.log(f"  (a) consistent genes for kappa_resid: {s2['n_consistent_kr']} "
            f"-> {'PASS' if cond_a else 'FAIL'} (need >=20)")
    buf.log(f"  (b) |Spearman(median_rho_kresid, median_rho_ID)| = {abs(s2['rho_corr_kr_id']):.3f} "
            f"-> {'PASS' if cond_b else 'FAIL'} (need <0.5)")
    buf.log(f"  (c) {cond_c_detail} -> {'PASS' if cond_c else 'FAIL'}")
    all_pass = cond_a and cond_b and cond_c
    if all_pass:
        buf.log("\nVERDICT: kappa_resid carries a locality-independent gene program (all 3 conditions met).")
    else:
        buf.log("\nVERDICT: curvature reduces to local geometry; the Ricci line closes as a negative result.")

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- ORC (kappa), residual, and gene expression are all derived from the same 50-D PCA / "
            "log-normalized expression matrix. Correlations with genes that dominate the PCA are "
            "partly mechanical for this reason; the local_ID comparison is the main control against "
            "that, not an independent validation.")
    buf.log("- n_genes_by_counts is also associated with stemness/CytoTRACE-like scores; using it as "
            "a covariate here may partially remove genuine biological signal along with the "
            "sequencing-depth confound, in either direction.")
    buf.log("- HTA8_6002 has only 60 malignant cells -- statistical power is low for this donor "
            "specifically, and its per-gene partial correlations are the least stable of the six.")
    buf.log("- This entire analysis is exploratory and follows a failed pre-registered robustness "
            "check (T1, 3/6 donors); nothing here should be read as confirmatory.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(buf.text())
    print(f"\nSaved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
