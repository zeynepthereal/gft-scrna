"""CNV-clone vs projection-residual test (clone hypothesis vs state hypothesis).

Per patient (Patient1, Patient2 -- Patient3 skipped, known myeloid contamination):
  0. Genomic positions from a GENCODE GTF onto adata.raw's Ensembl gene ids.
  1. infercnvpy CNV inference: carcinoma epithelial (observation) vs normal epithelial +
     immune cells (reference), window_size=250.
  2. Sanity checks: CNV score carcinoma vs normal; known CRC arm-level gains/losses.
  3. CNV clones: Leiden on PCA of X_cnv (carcinoma cells only), lowest resolution in
     {0.3, 0.5, 0.8} giving >=2 clusters with distinct arm profiles.
  4. Projection residual: EXACT function from scripts/paper_final_verification.py.
  5. Clone-vs-state tests: Spearman(residual, cnv_score); crosstab (residual group x clone)
     chi2 + adjusted MI; eta-squared (one-way ANOVA) of residual by clone; per-clone
     ASCL2+/SOX4+ rates; observed vs expected share of top-decile cells per clone.

Outputs: results/cnv_clone_vs_residual_patient{1,2}.txt and figures in results/.
"""
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import infercnvpy as cnv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr, f_oneway, chi2_contingency
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_final_verification import subspace_svd, residual as paper_residual  # noqa: E402 - exact reuse

REPO = Path(__file__).resolve().parent.parent
H5AD = REPO / "data" / "GSE161277_all.h5ad"
GTF = REPO / "data" / "reference" / "gencode.v44.basic.annotation.gtf.gz"
RESULTS = REPO / "results"

IMMUNE_TYPES = ["T cell", "B cell", "NK cell", "Macrophage/Monocyte"]
K_EIG = 20  # matches paper_final_verification.py
RESOLUTIONS = [0.3, 0.5, 0.8]
RANDOM_STATE = 42
ARM_DISTINCT_THRESHOLD = 0.01  # mean |arm-vector diff| to call two clusters' CNV profiles "distinct"
                                # (calibrated to the ~0.005-0.03 scale of infercnvpy's noise-
                                # thresholded, cell-median-centered smoothed values seen here)

# Approximate GRCh38 centromere midpoints (bp) -- only for the arms this task checks.
CENTROMERE_MID = {
    "7": 59_500_000, "8": 45_000_000, "13": 17_000_000, "17": 24_850_000,
    "18": 18_160_000, "20": 27_600_000,
}
ARM_EVENTS = {  # (chromosome, arm or None for whole chr) -> expected direction
    ("7", None): "gain", ("8", "q"): "gain", ("13", None): "gain", ("20", "q"): "gain",
    ("8", "p"): "loss", ("17", "p"): "loss", ("18", None): "loss",
}

PATIENTS = {
    "Patient1": {"normal_sample": "Patient1_normal", "carcinoma_sample": "Patient1_carcinoma"},
    "Patient2": {"normal_sample": "Patient2_normal", "carcinoma_sample": "Patient2_carcinoma"},
}


class Buf:
    """Collects lines for both stdout and the per-patient report file."""
    def __init__(self):
        self.lines = []

    def log(self, *args):
        s = " ".join(str(a) for a in args)
        print(s)
        self.lines.append(s)

    def text(self):
        return "\n".join(self.lines) + "\n"


def setup_positions(raw_adata, buf):
    """Attach chromosome/start/end from the GTF to raw_adata.var; report coverage. Returns a
    new AnnData restricted to genes that got a position."""
    ex = raw_adata.var_names[0]
    is_ensembl = str(ex).startswith("ENSG")
    buf.log(f"var_names format: {'Ensembl gene id' if is_ensembl else 'gene symbol'} (e.g. {ex})")

    if not GTF.exists():
        raise FileNotFoundError(f"GTF not found at {GTF}; download a GENCODE GTF first.")

    cnv.io.genomic_position_from_gtf(
        str(GTF), adata=raw_adata,
        gtf_gene_id="gene_id" if is_ensembl else "gene_name",
        adata_gene_id=None, inplace=True,
    )
    n_total = raw_adata.n_vars
    n_positioned = int(raw_adata.var["chromosome"].notna().sum())
    buf.log(f"Genes with genomic position: {n_positioned}/{n_total} "
            f"({100 * n_positioned / n_total:.1f}%); dropping the rest.")
    return raw_adata[:, raw_adata.var["chromosome"].notna()].copy()


def arm_of(chrom, start):
    c = str(chrom).replace("chr", "")
    if c not in CENTROMERE_MID:
        return None
    return "p" if start < CENTROMERE_MID[c] else "q"


_ARM_CACHE = {}


def _arm_series(var):
    key = id(var)
    if key not in _ARM_CACHE:
        _ARM_CACHE.clear()  # only ever need one patient's var at a time
        chrom_clean = var["chromosome"].astype(str).str.replace("chr", "", regex=False)
        arm = [arm_of(c, s) for c, s in zip(var["chromosome"], var["start"])]
        _ARM_CACHE[key] = (chrom_clean.values, np.array(arm, dtype=object))
    return _ARM_CACHE[key]


def arm_level_means(gene_values, var, cell_mask):
    """Mean smoothed per-gene CNV value per (chrom, arm) event of interest, over cell_mask."""
    chrom_clean, arm = _arm_series(var)
    out = {}
    for (chrom, a) in ARM_EVENTS:
        gmask = (chrom_clean == chrom) if a is None else ((chrom_clean == chrom) & (arm == a))
        gidx = np.where(gmask)[0]
        if len(gidx) == 0:
            out[(chrom, a)] = np.nan
            continue
        vals = gene_values[np.ix_(cell_mask, gidx)]
        out[(chrom, a)] = float(np.nanmean(vals))
    return out


def arm_level_percell(gene_values, var, cell_mask):
    """Per-cell mean smoothed CNV value per (chrom, arm) event -- one array of length
    cell_mask.sum() per event, for statistical testing (Mann-Whitney) between groups."""
    chrom_clean, arm = _arm_series(var)
    out = {}
    for (chrom, a) in ARM_EVENTS:
        gmask = (chrom_clean == chrom) if a is None else ((chrom_clean == chrom) & (arm == a))
        gidx = np.where(gmask)[0]
        if len(gidx) == 0:
            out[(chrom, a)] = np.full(int(np.sum(cell_mask)), np.nan)
            continue
        out[(chrom, a)] = np.nanmean(gene_values[np.ix_(cell_mask, gidx)], axis=1)
    return out


def build_patient_adata(raw_pos, obs_all, patient, samples):
    """Cells: this patient's normal+carcinoma epithelial + this patient's immune cells (any sample)."""
    prefix = patient + "_"
    is_patient = obs_all["sample"].str.startswith(prefix)
    normal_epi = (obs_all["sample"] == samples["normal_sample"]) & (obs_all["cell_type"] == "Epithelial/Tumor")
    carc_epi = (obs_all["sample"] == samples["carcinoma_sample"]) & (obs_all["cell_type"] == "Epithelial/Tumor")
    immune = is_patient & obs_all["cell_type"].isin(IMMUNE_TYPES)

    keep = (normal_epi | carc_epi | immune).values
    sub = raw_pos[keep].copy()
    grp = np.select(
        [normal_epi.values[keep], carc_epi.values[keep], immune.values[keep]],
        ["normal_epithelial", "carcinoma_epithelial", "immune"],
        default="",
    )
    sub.obs["cnv_group"] = grp
    sub.obs["cnv_ref"] = np.where(np.isin(grp, ["normal_epithelial", "immune"]), "reference", "observation")
    return sub


def get_marker_expr(raw_ref, obs_names, symbol):
    """Log-normalized expression of `symbol` for the given cells, from the full raw matrix."""
    symbol_to_id = raw_ref["symbol_to_id"]
    if symbol not in symbol_to_id:
        return np.full(len(obs_names), np.nan)
    gid = symbol_to_id[symbol]
    col = raw_ref["var_names"].get_loc(gid)
    rows = raw_ref["row_idx"].loc[obs_names].values
    X = raw_ref["X"][rows, col]
    return np.asarray(X.todense()).flatten() if hasattr(X, "todense") else np.asarray(X).flatten()


def make_figures(patient, adata, carc):
    try:
        cnv.pl.chromosome_heatmap(adata, groupby="cnv_group", show=False)
        plt.savefig(RESULTS / f"cnv_chromosome_heatmap_{patient.lower()}.png", dpi=150, bbox_inches="tight")
    except Exception as e:
        print(f"  [warn] chromosome_heatmap failed for {patient}: {e}")
    plt.close("all")

    fig, ax = plt.subplots(figsize=(6, 4))
    clones = list(carc.obs["cnv_clone"].cat.categories)
    data = [carc.obs.loc[carc.obs.cnv_clone == cl, "residual"].values for cl in clones]
    ax.boxplot(data, tick_labels=clones)
    ax.set_xlabel("CNV clone")
    ax.set_ylabel("Projection residual")
    ax.set_title(f"{patient}: residual by CNV clone")
    fig.tight_layout()
    fig.savefig(RESULTS / f"residual_boxplot_{patient.lower()}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    pct = carc.obs.groupby("cnv_clone", observed=True)["ASCL2_expr"].apply(lambda x: 100 * (x > 0).mean())
    ax.bar(pct.index.astype(str), pct.values)
    ax.set_xlabel("CNV clone")
    ax.set_ylabel("% ASCL2+ cells")
    ax.set_title(f"{patient}: ASCL2+ fraction by CNV clone")
    fig.tight_layout()
    fig.savefig(RESULTS / f"ascl2_pct_by_clone_{patient.lower()}.png", dpi=150)
    plt.close(fig)


def run_patient(patient, adata, raw_ref, buf):
    m_norm = (adata.obs.cnv_group == "normal_epithelial").values
    m_carc = (adata.obs.cnv_group == "carcinoma_epithelial").values
    buf.log(f"Cells: normal_epithelial={m_norm.sum()}  carcinoma_epithelial={m_carc.sum()}  "
            f"immune={(adata.obs.cnv_group == 'immune').sum()}  "
            f"(reference={(adata.obs.cnv_ref == 'reference').sum()}, "
            f"observation={(adata.obs.cnv_ref == 'observation').sum()})")

    # --- STEP 1: infer CNV ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cnv.tl.infercnv(adata, reference_key="cnv_ref", reference_cat=["reference"],
                         window_size=250, calculate_gene_values=True, chunksize=100)
    cell_key = "_cnv_score_cell_tmp"
    adata.obs[cell_key] = adata.obs_names.astype(str)  # per-cell "cluster" -> true per-cell score
    cnv.tl.cnv_score(adata, groupby=cell_key, use_rep="cnv", key_added="cnv_score")
    del adata.obs[cell_key]

    gv = adata.layers["gene_values_cnv"]
    var = adata.var

    buf.log("\n--- Sanity check (a): CNV score, carcinoma vs normal epithelial ---")
    s_norm, s_carc = adata.obs.cnv_score.values[m_norm], adata.obs.cnv_score.values[m_carc]
    u, p = mannwhitneyu(s_carc, s_norm, alternative="two-sided")
    buf.log(f"  normal median={np.median(s_norm):.4f}  carcinoma median={np.median(s_carc):.4f}  "
            f"ratio={np.median(s_carc) / max(np.median(s_norm), 1e-9):.2f}x  Mann-Whitney p={p:.3e}")

    buf.log("\n--- Sanity check (b): known CRC chromosome-arm events ---")
    buf.log("(per-cell arm-mean smoothed CNV, carcinoma vs normal epithelial; Mann-Whitney U test;")
    buf.log(" 'present' = p<0.01 AND direction matches expected event)")
    percell_carc = arm_level_percell(gv, var, m_carc)
    percell_norm = arm_level_percell(gv, var, m_norm)
    detected = []
    buf.log(f"{'event':<14}{'expected':<10}{'carc_mean':<12}{'norm_mean':<12}{'p-value':<12}{'present?'}")
    for (chrom, arm), expected in ARM_EVENTS.items():
        label = f"{chrom}{arm or ''}"
        vc_arr, vn_arr = percell_carc[(chrom, arm)], percell_norm[(chrom, arm)]
        vc, vn = np.nanmean(vc_arr), np.nanmean(vn_arr)
        if np.all(np.isnan(vc_arr)) or np.all(np.isnan(vn_arr)):
            buf.log(f"{label:<14}{expected:<10}{'nan':<12}{'nan':<12}{'--':<12}no (no genes in arm)")
            continue
        _, p_arm = mannwhitneyu(vc_arr, vn_arr, alternative="two-sided", nan_policy="omit")
        direction_ok = (expected == "gain" and vc > vn) or (expected == "loss" and vc < vn)
        present = direction_ok and p_arm < 0.01
        if present:
            detected.append(label)
        buf.log(f"{label:<14}{expected:<10}{vc:<12.4f}{vn:<12.4f}{p_arm:<12.3e}{'YES' if present else 'no'}")
    buf.log(f"Arm events detected in carcinoma: {len(detected)}/{len(ARM_EVENTS)} ({', '.join(detected) or 'none'})")

    global_cnv_up = np.median(s_carc) > np.median(s_norm) and p < 0.01
    no_signal = (not global_cnv_up) and len(detected) == 0
    if no_signal:
        buf.log("\n*** No clear CNV structure detected (score similar to normal, no arm-level events). ***")
        buf.log("*** STOPPING for this patient: possibly MSI; clone test not applicable. ***")
        return None

    # --- STEP 2: CNV clones (carcinoma epithelial cells only) ---
    buf.log("\n--- STEP 2: CNV clones (carcinoma epithelial cells, PCA(X_cnv) -> neighbors -> Leiden) ---")
    carc = adata[m_carc].copy()
    X_cnv_carc = carc.obsm["X_cnv"]
    X_cnv_carc = np.asarray(X_cnv_carc.todense()) if hasattr(X_cnv_carc, "todense") else np.asarray(X_cnv_carc)
    gv_carc = gv[m_carc]

    n_pcs = max(2, min(30, X_cnv_carc.shape[1] - 1, X_cnv_carc.shape[0] - 1))
    pca_cnv = PCA(n_components=n_pcs, random_state=RANDOM_STATE).fit_transform(X_cnv_carc)
    carc.obsm["X_pca_cnv"] = pca_cnv
    sc.pp.neighbors(carc, use_rep="X_pca_cnv", random_state=RANDOM_STATE)

    chosen_res, chosen_labels, chosen_n = None, None, None
    for res in RESOLUTIONS:
        key = f"cnv_leiden_{res}"
        sc.tl.leiden(carc, resolution=res, random_state=RANDOM_STATE, key_added=key,
                     flavor="igraph", n_iterations=2, directed=False)
        labels = carc.obs[key]
        n_cl = labels.nunique()
        buf.log(f"  resolution={res}: n_clusters={n_cl}  sizes={dict(labels.value_counts())}")
        if n_cl >= 2 and chosen_res is None:
            arm_vecs = {}
            for cl in labels.cat.categories:
                mask = (labels == cl).values
                arm_vecs[cl] = np.array([arm_level_means(gv_carc, var, mask)[k] for k in ARM_EVENTS])
            cls = list(arm_vecs.keys())
            diffs = [np.abs(arm_vecs[a] - arm_vecs[b]).mean() for i, a in enumerate(cls) for b in cls[i + 1:]]
            if max(diffs, default=0) > ARM_DISTINCT_THRESHOLD:
                chosen_res, chosen_labels, chosen_n = res, labels.astype(str).values, n_cl

    if chosen_res is None:
        chosen_res = RESOLUTIONS[0]
        chosen_labels = carc.obs[f"cnv_leiden_{chosen_res}"].astype(str).values
        chosen_n = carc.obs[f"cnv_leiden_{chosen_res}"].nunique()
        buf.log(f"  No resolution gave clusters with clearly distinct (>{ARM_DISTINCT_THRESHOLD}) arm profiles; "
                f"using lowest resolution {chosen_res} ({chosen_n} clusters) as-is.")

    carc.obs["cnv_clone"] = pd.Categorical(chosen_labels)
    buf.log(f"\nChosen resolution={chosen_res}: {chosen_n} CNV clones, sizes:")
    for cl, n in carc.obs["cnv_clone"].value_counts().items():
        buf.log(f"  clone {cl}: n={n}")
        mask = (carc.obs["cnv_clone"] == cl).values
        arm_vals = arm_level_means(gv_carc, var, mask)
        events = [f"{c}{a or ''}:{v:+.3f}" for (c, a), v in arm_vals.items() if abs(v) > ARM_DISTINCT_THRESHOLD]
        buf.log(f"    arm-level (|mean|>0.05): {', '.join(events) if events else 'none notable'}")

    # --- STEP 3: projection residual (exact reuse) ---
    buf.log("\n--- STEP 3: projection residual (exact function from paper_final_verification.py) ---")
    pts_all = adata.obsm["X_pca_full"][:, :50]
    pts_norm, pts_carc = pts_all[m_norm], pts_all[m_carc]
    U, mean = subspace_svd(pts_norm, k_eig=K_EIG)
    res_carc = paper_residual(pts_carc, U, mean)
    res_norm = paper_residual(pts_norm, U, mean)
    ratio = np.median(res_carc) / np.median(res_norm)
    ref_ratio = "~3.67x" if "Patient1" in patient or patient == "Patient1" else "~1.39x"
    buf.log(f"  normal median={np.median(res_norm):.4f}  carcinoma median={np.median(res_carc):.4f}  "
            f"ratio={ratio:.2f}x  (Mac rebuild verification reference: {ref_ratio})")

    carc.obs["residual"] = res_carc
    cnv_score_carc = adata.obs.cnv_score.values[m_carc]

    # --- STEP 4: clone vs state tests ---
    buf.log("\n--- STEP 4: clone vs state tests ---")
    buf.log("Decision rules (stated before results):")
    buf.log("  Clone hypothesis: top-decile residual cells concentrated in ONE CNV clone (>70%) "
            "AND CNV clone explains most residual variance.")
    buf.log("  State hypothesis: top-decile residual and ASCL2+ cells spread across clones "
            "~proportionally to clone size, AND within-clone residual variance is large.")

    rho, p_rho = spearmanr(res_carc, cnv_score_carc)
    buf.log(f"\n(a) Spearman(residual, cnv_score) = {rho:.3f}  p={p_rho:.3e}")

    q10, q90 = np.quantile(res_carc, [0.10, 0.90])
    res_group = np.where(res_carc >= q90, "top10", np.where(res_carc <= q10, "bottom10", "middle80"))
    carc.obs["residual_group"] = res_group
    ct = pd.crosstab(carc.obs["residual_group"], carc.obs["cnv_clone"])
    chi2, p_chi2, _, _ = chi2_contingency(ct)
    ami = adjusted_mutual_info_score(carc.obs["residual_group"], carc.obs["cnv_clone"])
    buf.log("\n(b) Crosstab residual_group x cnv_clone:")
    buf.log(ct.to_string())
    buf.log(f"    chi2 p={p_chi2:.3e}  adjusted_mutual_info={ami:.4f}")

    groups = [res_carc[carc.obs["cnv_clone"].values == cl] for cl in carc.obs["cnv_clone"].cat.categories]
    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
        f_stat, p_anova = f_oneway(*groups)
        grand_mean = res_carc.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((res_carc - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total
    else:
        f_stat, p_anova, eta_sq = np.nan, np.nan, np.nan
    buf.log(f"\n(c) One-way ANOVA residual ~ cnv_clone: F={f_stat:.3f}  p={p_anova:.3e}  eta^2={eta_sq:.4f}")

    ascl2_expr = get_marker_expr(raw_ref, carc.obs_names, "ASCL2")
    sox4_expr = get_marker_expr(raw_ref, carc.obs_names, "SOX4")
    carc.obs["ASCL2_expr"] = ascl2_expr
    carc.obs["SOX4_expr"] = sox4_expr

    buf.log("\n(d) Per-clone summary:")
    buf.log(f"{'clone':<8}{'n':<8}{'med_resid':<12}{'%ASCL2+':<10}{'mean_ASCL2':<12}{'%SOX4+'}")
    for cl in carc.obs["cnv_clone"].cat.categories:
        m = (carc.obs["cnv_clone"] == cl).values
        n = int(m.sum())
        buf.log(f"{cl:<8}{n:<8}{np.median(res_carc[m]):<12.3f}"
                f"{100 * (ascl2_expr[m] > 0).mean():<10.1f}{np.nanmean(ascl2_expr[m]):<12.3f}"
                f"{100 * (sox4_expr[m] > 0).mean():.1f}")

    buf.log("\n(e) Observed vs expected share of top-decile residual cells, per clone:")
    n_total = len(carc)
    n_top = int((carc.obs["residual_group"] == "top10").sum())
    buf.log(f"{'clone':<8}{'clone_size':<12}{'expected_top10':<16}{'observed_top10':<16}{'obs/exp'}")
    for cl in carc.obs["cnv_clone"].cat.categories:
        m = (carc.obs["cnv_clone"] == cl).values
        clone_size = int(m.sum())
        expected = n_top * clone_size / n_total
        observed = int(((carc.obs["cnv_clone"] == cl) & (carc.obs["residual_group"] == "top10")).sum())
        ratio_oe = observed / expected if expected > 0 else np.nan
        buf.log(f"{cl:<8}{clone_size:<12}{expected:<16.2f}{observed:<16}{ratio_oe:.2f}")

    top10_by_clone = carc.obs.loc[carc.obs.residual_group == "top10", "cnv_clone"].value_counts(normalize=True)
    max_concentration = float(top10_by_clone.max()) if len(top10_by_clone) else float("nan")
    buf.log(f"\nMax concentration of top-decile cells in a single clone: {100 * max_concentration:.1f}%")
    clone_supported = max_concentration > 0.70 and eta_sq > 0.5
    state_supported = max_concentration <= 0.70 and eta_sq < 0.3
    if clone_supported:
        verdict = "CLONE hypothesis better supported by the stated rules."
    elif state_supported:
        verdict = "STATE hypothesis better supported by the stated rules."
    else:
        verdict = "Neither rule set is cleanly satisfied -- mixed/ambiguous result."
    buf.log(f"Verdict per stated decision rules: {verdict}")

    make_figures(patient, adata, carc)
    return {"adata": adata, "carc": carc, "m_norm": m_norm, "m_carc": m_carc, "gv": gv, "var": var}


def main():
    RESULTS.mkdir(exist_ok=True)
    full = sc.read_h5ad(H5AD)
    obs_all = full.obs

    raw_all = full.raw.to_adata()
    raw_all.obs = full.obs.copy()

    setup_buf = Buf()
    setup_buf.log("STEP 0 -- Setup")
    versions_line = "versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["scanpy", "infercnvpy", "leidenalg", "numpy", "anndata", "scipy"])
    setup_buf.log(versions_line)
    raw_pos = setup_positions(raw_all, setup_buf)

    symbol_to_id = dict(zip(raw_all.var["gene_symbol"], raw_all.var_names))
    raw_ref = {
        "symbol_to_id": symbol_to_id,
        "var_names": raw_all.var_names,
        "X": raw_all.X,
        "row_idx": pd.Series(np.arange(raw_all.n_obs), index=raw_all.obs_names),
    }

    pca_full = pd.DataFrame(full.obsm["X_pca"], index=full.obs_names)

    for patient, samples in PATIENTS.items():
        buf = Buf()
        buf.log(setup_buf.text().rstrip())
        buf.log("=" * 90)
        buf.log(patient)
        buf.log("=" * 90)

        adata_p = build_patient_adata(raw_pos, obs_all, patient, samples)
        adata_p.obsm["X_pca_full"] = pca_full.loc[adata_p.obs_names].values

        run_patient(patient, adata_p, raw_ref, buf)

        out_path = RESULTS / f"cnv_clone_vs_residual_{patient.lower()}.txt"
        out_path.write_text(buf.text(), encoding="utf-8")
        print(f"\nSaved -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
