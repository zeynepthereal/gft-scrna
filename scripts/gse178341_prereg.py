"""Pre-registered analysis of GSE178341 (original pre-registration + Amendments 1 and 2,
docs/prereg_gse178341.md). Runs H1, the H2 gate, H2, and H3 exactly as specified there.

Frozen pipeline primitives are IMPORTED, not reimplemented:
  thin_to_depth          <- scripts/geom_dnb.py
  svd_subspace            <- scripts/geom_common.py
  fit_subspace, residual  <- scripts/ricci_curvature_pilot.py
  test_a_sample            <- scripts/confound_and_detector.py
  setup_positions          <- scripts/cnv_clone_vs_residual.py (GTF gene-position merge)
infercnvpy itself provides the CNV score (external library, not project code to reuse/copy).

Memory: the 1.2 GB h5 count matrix is read ONE PATIENT AT A TIME. All 62 patients occupy a
single CONTIGUOUS column range each in the stored (gene x cell, CSC) matrix (verified before
writing this script); each patient's raw counts are loaded via one contiguous HDF5 slice over
indptr/data/indices, never the full matrix.

See docs/prereg_gse178341.md for the exact decision rules (quoted verbatim at the top of the
output file) and both amendments (channel/chemistry eligibility, CNV reference definition).

Output: results/gse178341_prereg_results.txt
"""
import os
import sys
import time
import warnings
from importlib.metadata import version

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import infercnvpy as cnv
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402
from geom_dnb import thin_to_depth  # noqa: E402
from ricci_curvature_pilot import fit_subspace, residual  # noqa: E402
from confound_and_detector import test_a_sample  # noqa: E402
from cnv_clone_vs_residual import setup_positions  # noqa: E402

warnings.filterwarnings("ignore", message=".*slogdet.*")

REPO = gc.PROJECT_DIR
DATA_DIR = os.path.join(REPO, "data", "GSE178341")
H5 = os.path.join(DATA_DIR, "GSE178341_crc10x_full_c295v4_submit.h5")
META_CSV = os.path.join(DATA_DIR, "GSE178341_crc10x_full_c295v4_submit_metatables.csv")
CLUSTER_CSV = os.path.join(DATA_DIR, "GSE178341_crc10x_full_c295v4_submit_cluster.csv")

SEED = 0
N_HVG = 3000
Q_PCA = 50
Q_SUB = 20
THIN_PCT = 10
MIN_GROUP_N = 50          # eligibility: >=50 normal and >=50 tumor epithelial cells
MIN_CHANNEL_N = 50        # C-ctrl: channel must have >=50 epithelial cells to qualify
MIN_CHANNELS_FOR_CCTRL = 2
CNV_WINDOW = 250

DECISION_RULES_VERBATIM = """\
DECISION RULES, quoted verbatim from docs/prereg_gse178341.md
================================================================================================

[Original] H1 (blind validation). Malignant-proxy cells have higher held-out-normal-referenced
residual than held-out normal cells, across GSE178341 as a whole (both MMR strata pooled).
Supported if Test A's "survives" rule passes in >=70% of the 35 eligible patients.

[Original] H2 (MMRd). Among MMRd patients, residual separates authors'-proxy-malignant from
normal epithelial cells better than an infercnvpy CNV score does (AUROC comparison). Supported
if residual AUROC > CNV-score AUROC in >=70% of eligible MMRd patients (of the 20 MMRd patients
in the eligible pool) AND the median AUROC difference (residual - CNV) across those patients is
> 0.05.

[Original] H3 (control). Among MMRp patients, CNV-score AUROC is expected to be high. Residual
vs CNV-score AUROC will be reported for comparison, with no pass/fail threshold.

[Amendment 1] Eligibility (supersedes the original "35/62"): patients are eligible for H1/H2/H3
only if their tumor and normal epithelium come from the same 10x chemistry, on top of the
original >=50-normal AND >=50-tumor-epithelial(-proxy) requirement. 34/62 patients remain
eligible (20 MMRd, 14 MMRp).

[Amendment 1] Channel-effect control (C-ctrl), for the 15 patients (8 MMRd, 7 MMRp) with >=2
normal channels each with >=50 epithelial cells: fit the normal subspace on normal channel A,
compute held-out residuals for channel A and residuals for normal channel B (every ordered
pair). Channel ratio = median(B) / median(A held-out). Also compute residual AUROC for
separating channel B from channel A.

[Amendment 1] H1 (revised). Supported only if, in >=70% of the 34 eligible patients: (a) Test A
passes, AND (b) the tumor/normal residual ratio exceeds the 95th percentile of the pooled
C-ctrl channel-ratio distribution.

[Amendment 1] H2 (revised). Interpretable only if the median C-ctrl residual AUROC (pooled
across the 15-patient distribution) is below 0.6. If >=0.6, H2 is declared untestable in this
dataset, regardless of the AUROC results. If the gate clears, the original H2 thresholds apply.

[Amendment 2] CNV reference definition: infercnvpy run per patient with ONLY non-epithelial
cells (T/NK, myeloid, stromal; pooled from both tumor and normal channels) as the reference.
Normal epithelial cells are NOT in the reference. window_size=250, genomic positions from the
GENCODE GTF at data/reference/.
================================================================================================
"""


class Buf(gc.Buf):
    pass


# ---------------------------------------------------------------------- metadata / eligibility
def load_metadata():
    meta = pd.read_csv(META_CSV, usecols=["cellID", "SPECIMEN_TYPE", "SINGLECELL_TYPE", "PID", "MMRStatus"])
    cl = pd.read_csv(CLUSTER_CSV, usecols=["sampleID", "batchID", "clTopLevel"])
    assert (meta["cellID"].values == cl["sampleID"].values).all()
    df = pd.concat([meta, cl[["batchID", "clTopLevel"]]], axis=1)
    df["pos"] = np.arange(len(df))
    df["is_epi"] = df["clTopLevel"].values == "Epi"
    return df


def patient_column_ranges(df):
    g = df.groupby("PID")["pos"].agg(["min", "max", "count"])
    assert ((g["max"] - g["min"] + 1) == g["count"]).all(), "patients are not contiguous blocks"
    return {pid: (int(r["min"]), int(r["max"]) + 1) for pid, r in g.iterrows()}  # [start, end)


def compute_eligibility(df):
    pid_mmr = (df[df["SPECIMEN_TYPE"] == "T"].groupby("PID")["MMRStatus"]
               .agg(lambda x: x.dropna().mode().iat[0] if len(x.dropna()) else "NA"))
    chan = df.groupby(["PID", "batchID", "SPECIMEN_TYPE", "SINGLECELL_TYPE"])["is_epi"].sum().reset_index(name="n_epi")

    rows = []
    for pid, g in chan.groupby("PID"):
        t_chem = set(g.loc[g["SPECIMEN_TYPE"] == "T", "SINGLECELL_TYPE"])
        n_chem = set(g.loc[g["SPECIMEN_TYPE"] == "N", "SINGLECELL_TYPE"])
        same_chem = len(n_chem) > 0 and len(t_chem) > 0 and t_chem == n_chem and len(t_chem) == 1
        n_epi_total = int(g.loc[g["SPECIMEN_TYPE"] == "N", "n_epi"].sum())
        t_epi_total = int(g.loc[g["SPECIMEN_TYPE"] == "T", "n_epi"].sum())
        n_chan_ge50 = int((g[(g["SPECIMEN_TYPE"] == "N") & (g["n_epi"] >= MIN_CHANNEL_N)]).shape[0])
        rows.append(dict(PID=pid, MMRStatus=pid_mmr.get(pid, "NA"), same_chem=same_chem,
                          n_epi_total=n_epi_total, t_epi_total=t_epi_total,
                          n_normal_channels_ge50=n_chan_ge50))
    tab = pd.DataFrame(rows)
    tab["eligible34"] = tab["same_chem"] & (tab["n_epi_total"] >= MIN_GROUP_N) & (tab["t_epi_total"] >= MIN_GROUP_N)
    tab["cctrl_eligible"] = tab["n_normal_channels_ge50"] >= MIN_CHANNELS_FOR_CCTRL
    return tab.set_index("PID")


# ---------------------------------------------------------------------- h5 per-patient loader
def load_patient_raw(col_start, col_end):
    """Raw counts (cells x genes, CSR), genuinely memory-safe: reads only this patient's
    contiguous column range from the on-disk CSC matrix."""
    with h5py.File(H5, "r") as f:
        n_genes, n_cells_total = f["matrix/shape"][:]
        indptr = f["matrix/indptr"][col_start:col_end + 1]
        d0, d1 = int(indptr[0]), int(indptr[-1])
        data = f["matrix/data"][d0:d1]
        indices = f["matrix/indices"][d0:d1]
        local_indptr = indptr - d0
        gene_id = np.array([x.decode() for x in f["matrix/features/id"][:]])
        gene_name = np.array([x.decode() for x in f["matrix/features/name"][:]])
    X = sp.csc_matrix((data, indices, local_indptr), shape=(n_genes, col_end - col_start))
    return X.T.tocsr(), gene_id, gene_name


def clean_gene_ids(gene_id_raw):
    return np.array([g.split(".")[0] for g in gene_id_raw])


# ---------------------------------------------------------------------- global gene positions
def build_gene_position_table(buf):
    with h5py.File(H5, "r") as f:
        gene_id_raw = np.array([x.decode() for x in f["matrix/features/id"][:]])
        gene_name = np.array([x.decode() for x in f["matrix/features/name"][:]])
    ensg = clean_gene_ids(gene_id_raw)
    dummy = ad.AnnData(X=np.zeros((1, len(ensg)), dtype="float32"))
    dummy.var_names = ensg
    dummy.var["gene_symbol"] = gene_name
    dummy.var_names_make_unique()
    pos_adata = setup_positions(dummy, buf)
    buf.log(f"Gene positions resolved for {pos_adata.n_vars}/{len(ensg)} genes (shared across "
            f"all patients, computed once).")
    return pos_adata.var_names.values, pos_adata.var[["chromosome", "start", "end", "gene_symbol"]], ensg


# ---------------------------------------------------------------------- residual pipeline (patient-level)
def residual_pipeline(raw_n, raw_m, channel_n, buf, tag):
    """Thinning -> normalize+log1p -> label-blind 3000-HVG -> 50D PCA (all included cells) ->
    q=20 normal subspace -> 5-fold held-out normal residual, full-fit malignant residual.
    Returns per-cell arrays for normal (incl. channel id) and malignant groups."""
    rng = np.random.default_rng(SEED)
    combined = sp.vstack([raw_n, raw_m]).tocsr()
    total = np.asarray(combined.sum(axis=1)).ravel()
    target = np.percentile(total, THIN_PCT)
    keep_idx = np.where(total >= target)[0]
    thinned = thin_to_depth(combined, keep_idx, target, rng)
    is_n = np.r_[np.ones(raw_n.shape[0], bool), np.zeros(raw_m.shape[0], bool)][keep_idx]
    is_m = ~is_n
    channel_kept = np.r_[channel_n, np.array([None] * raw_m.shape[0])][keep_idx]
    ng = np.asarray((thinned > 0).sum(axis=1)).ravel()

    var = pd.DataFrame(index=[f"g{i}" for i in range(thinned.shape[1])])
    adata = sc.AnnData(X=thinned, var=var)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat")
    X_all = np.asarray(adata[:, adata.var["highly_variable"].values].X.todense())

    V, mean, _ = gc.svd_subspace(X_all, Q_PCA)
    pts_all = (X_all - mean) @ V
    pts_n, pts_m = pts_all[is_n], pts_all[is_m]
    ng_n, ng_m = ng[is_n], ng[is_m]
    ch_n = channel_kept[is_n]

    n = len(pts_n)
    folds = rng.permutation(n) % 5
    r_n = np.empty(n)
    for f in range(5):
        Uf, meanf = fit_subspace(pts_n[folds != f])
        r_n[folds == f] = residual(pts_n[folds == f], Uf, meanf)
    U_all, mean_all = fit_subspace(pts_n)
    r_m = residual(pts_m, U_all, mean_all)

    buf.log(f"    [{tag}] thinning target(pct={THIN_PCT})={target:.0f}; kept {len(keep_idx)}/"
            f"{combined.shape[0]} (n_normal={is_n.sum()}, n_tumor={is_m.sum()})")
    return {"r_n": r_n, "r_m": r_m, "ng_n": ng_n, "ng_m": ng_m, "pts_n": pts_n, "pts_m": pts_m,
            "ch_n": ch_n, "n_before": (raw_n.shape[0], raw_m.shape[0]),
            "n_after": (int(is_n.sum()), int(is_m.sum()))}


def cctrl_for_patient(res, buf, patient):
    """Ordered normal-channel pairs within the SAME 50D embedding already fit for this
    patient's main residual pipeline (pts_n, ch_n)."""
    channels, counts = np.unique(res["ch_n"][res["ch_n"] != None], return_counts=True)  # noqa: E711
    qualifying = [c for c, n in zip(channels, counts) if n >= MIN_CHANNEL_N]
    if len(qualifying) < 2:
        buf.log(f"  C-ctrl: {patient} has <2 qualifying (>= {MIN_CHANNEL_N} cells) normal "
                f"channels after thinning -- skipped.")
        return []
    rows = []
    rng = np.random.default_rng(SEED)
    for a in qualifying:
        for b in qualifying:
            if a == b:
                continue
            pts_a = res["pts_n"][res["ch_n"] == a]
            pts_b = res["pts_n"][res["ch_n"] == b]
            n = len(pts_a)
            folds = rng.permutation(n) % 5
            r_a_heldout = np.empty(n)
            for f in range(5):
                Uf, meanf = fit_subspace(pts_a[folds != f])
                r_a_heldout[folds == f] = residual(pts_a[folds == f], Uf, meanf)
            U_all, mean_all = fit_subspace(pts_a)
            r_b = residual(pts_b, U_all, mean_all)
            ratio = np.median(r_b) / np.median(r_a_heldout)
            y = np.r_[np.zeros(len(r_a_heldout)), np.ones(len(r_b))]
            auc = roc_auc_score(y, np.r_[r_a_heldout, r_b])
            rows.append({"patient": patient, "A": a, "B": b, "n_A": len(pts_a), "n_B": len(pts_b),
                         "ratio": ratio, "auroc": auc})
            buf.log(f"  C-ctrl [{patient}] A={a} (n={len(pts_a)}) B={b} (n={len(pts_b)}): "
                    f"ratio={ratio:.3f}x  AUROC={auc:.3f}")
    return rows


# ---------------------------------------------------------------------- CNV pipeline (Amendment 2)
def cnv_pipeline(raw_full, gene_id_raw, gene_pos_names, gene_pos_var, mask_nonepi, mask_norm_epi,
                  mask_tum_epi, buf, tag):
    ensg = clean_gene_ids(gene_id_raw)
    ad_full = sc.AnnData(X=raw_full, var=pd.DataFrame(index=ensg))
    ad_full.var_names_make_unique()
    keep = ad_full.var_names.isin(gene_pos_names)
    ad_sub = ad_full[:, keep].copy()
    ad_sub.var = ad_sub.var.join(gene_pos_var, how="left")

    obs_mask = mask_nonepi | mask_norm_epi | mask_tum_epi
    ad_cnv = ad_sub[obs_mask].copy()
    grp = np.select([mask_nonepi[obs_mask], mask_norm_epi[obs_mask], mask_tum_epi[obs_mask]],
                     ["nonepi_ref", "normal_epi", "tumor_epi"], default="")
    ad_cnv.obs["cnv_group"] = grp
    ad_cnv.obs["cnv_ref"] = np.where(ad_cnv.obs["cnv_group"] == "nonepi_ref", "reference", "observation")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.normalize_total(ad_cnv, target_sum=1e4)
        sc.pp.log1p(ad_cnv)
        cnv.tl.infercnv(ad_cnv, reference_key="cnv_ref", reference_cat=["reference"],
                         window_size=CNV_WINDOW)
    cell_key = "_cnv_cell_tmp"
    ad_cnv.obs[cell_key] = ad_cnv.obs_names.astype(str)
    cnv.tl.cnv_score(ad_cnv, groupby=cell_key, use_rep="cnv", key_added="cnv_score")

    buf.log(f"    [{tag}] CNV: reference(non-epi)={int((ad_cnv.obs.cnv_group=='nonepi_ref').sum())}  "
            f"normal_epi={int((ad_cnv.obs.cnv_group=='normal_epi').sum())}  "
            f"tumor_epi={int((ad_cnv.obs.cnv_group=='tumor_epi').sum())}  "
            f"genes_used={ad_cnv.n_vars}")

    cnv_norm = ad_cnv.obs.loc[ad_cnv.obs.cnv_group == "normal_epi", "cnv_score"]
    cnv_tum = ad_cnv.obs.loc[ad_cnv.obs.cnv_group == "tumor_epi", "cnv_score"]
    return cnv_norm, cnv_tum  # pandas Series, indexed by obs_names (original cell order within group)


# ---------------------------------------------------------------------- main per-patient driver
def process_patient(pid, col_range, df_pid, gene_pos_names, gene_pos_var, buf):
    t0 = time.time()
    start, end = col_range
    raw_full, gene_id_raw, gene_name = load_patient_raw(start, end)
    d = df_pid.reset_index(drop=True)
    assert len(d) == raw_full.shape[0]

    mask_norm_epi = ((d["SPECIMEN_TYPE"] == "N") & d["is_epi"]).values
    mask_tum_epi = ((d["SPECIMEN_TYPE"] == "T") & d["is_epi"]).values
    mask_nonepi = (~d["is_epi"]).values
    channel_n = d.loc[mask_norm_epi, "batchID"].values

    buf.log(f"  n_cells={raw_full.shape[0]}  normal_epi={mask_norm_epi.sum()}  "
            f"tumor_epi={mask_tum_epi.sum()}  non_epi={mask_nonepi.sum()}")

    res = residual_pipeline(raw_full[mask_norm_epi], raw_full[mask_tum_epi], channel_n, buf, pid)

    cnv_norm, cnv_tum = cnv_pipeline(raw_full, gene_id_raw, gene_pos_names, gene_pos_var,
                                      mask_nonepi, mask_norm_epi, mask_tum_epi, buf, pid)
    # align CNV scores to the SAME (thinned, surviving) cells used for the residual, for a fair
    # AUROC comparison -- restrict to the keep-order via positional alignment.
    # cnv_norm/cnv_tum are indexed by original per-group positional order (0..n_group-1 as
    # obs_names were cell positions before thinning); need the same keep mask used in
    # residual_pipeline. Recompute it identically (same seed, same inputs) for alignment.
    combined = sp.vstack([raw_full[mask_norm_epi], raw_full[mask_tum_epi]]).tocsr()
    total = np.asarray(combined.sum(axis=1)).ravel()
    target = np.percentile(total, THIN_PCT)
    keep_idx = np.where(total >= target)[0]
    n_norm_total = int(mask_norm_epi.sum())
    is_n_keep = keep_idx < n_norm_total
    norm_keep_pos = keep_idx[is_n_keep]
    tum_keep_pos = keep_idx[~is_n_keep] - n_norm_total

    cnv_norm_arr = cnv_norm.values[norm_keep_pos]
    cnv_tum_arr = cnv_tum.values[tum_keep_pos]

    y = np.r_[np.zeros(len(res["r_n"])), np.ones(len(res["r_m"]))]
    aurocs = {
        "residual": roc_auc_score(y, np.r_[res["r_n"], res["r_m"]]),
        "cnv_score": roc_auc_score(y, np.r_[cnv_norm_arr, cnv_tum_arr]),
        "n_genes": roc_auc_score(y, np.r_[res["ng_n"], res["ng_m"]]),
        "euclid_centroid": roc_auc_score(y, np.r_[
            np.linalg.norm(res["pts_n"] - res["pts_n"].mean(0), axis=1),
            np.linalg.norm(res["pts_m"] - res["pts_n"].mean(0), axis=1)]),
    }
    buf.log(f"    [{pid}] AUROC: residual={aurocs['residual']:.3f}  cnv={aurocs['cnv_score']:.3f}  "
            f"n_genes={aurocs['n_genes']:.3f}  euclid={aurocs['euclid_centroid']:.3f}  "
            f"[{time.time()-t0:.0f}s]")

    testA = test_a_sample(pid, res["r_n"], res["ng_n"], res["r_m"], res["ng_m"], buf)
    cctrl_rows = cctrl_for_patient(res, buf, pid)

    return {"res": res, "aurocs": aurocs, "testA": testA, "cctrl_rows": cctrl_rows,
            "n_before": res["n_before"], "n_after": res["n_after"]}


def main():
    buf = Buf()
    buf.log(DECISION_RULES_VERBATIM)
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["numpy", "scipy", "scikit-learn", "scanpy", "infercnvpy", "anndata"]))
    buf.log(f"seed={SEED}  thin_pct={THIN_PCT}  N_HVG={N_HVG}  Q_PCA={Q_PCA}  Q_SUB={Q_SUB}  "
            f"cnv_window={CNV_WINDOW}\n")

    df = load_metadata()
    ranges = patient_column_ranges(df)
    elig = compute_eligibility(df)
    eligible34 = elig[elig["eligible34"]].index.tolist()
    cctrl15 = elig[elig["cctrl_eligible"]].index.tolist()
    buf.log(f"Eligible (H1/H2/H3), recomputed from source files: {len(eligible34)} "
            f"({sum(elig.loc[eligible34,'MMRStatus']=='MMRd')} MMRd, "
            f"{sum(elig.loc[eligible34,'MMRStatus']=='MMRp')} MMRp)")
    buf.log(f"C-ctrl eligible: {len(cctrl15)} "
            f"({sum(elig.loc[cctrl15,'MMRStatus']=='MMRd')} MMRd, "
            f"{sum(elig.loc[cctrl15,'MMRStatus']=='MMRp')} MMRp)")
    to_process = sorted(set(eligible34) | set(cctrl15))
    buf.log(f"Patients to process (union of both sets): {len(to_process)}\n")

    buf.log("Building global gene-position table (once)...")
    gene_pos_names, gene_pos_var, _ = build_gene_position_table(buf)
    buf.log("")

    patient_results = {}
    all_cctrl_rows = []
    for pid in to_process:
        buf.log("=" * 96)
        buf.log(f"{pid}  (MMR={elig.loc[pid,'MMRStatus']}  eligible34={pid in eligible34}  "
                f"cctrl={pid in cctrl15})")
        buf.log("=" * 96)
        df_pid = df[df["PID"] == pid]
        r = process_patient(pid, ranges[pid], df_pid, gene_pos_names, gene_pos_var, buf)
        patient_results[pid] = r
        all_cctrl_rows.extend(r["cctrl_rows"])

    # ---------------------------------------------------------------- C-ctrl pooled distributions
    buf.log("\n" + "=" * 96)
    buf.log("C-CTRL POOLED DISTRIBUTION (all ordered normal-channel pairs, all qualifying patients)")
    buf.log("=" * 96)
    cctrl_df = pd.DataFrame(all_cctrl_rows)
    ratios = cctrl_df["ratio"].values
    aurocs_cctrl = cctrl_df["auroc"].values
    buf.log(f"n_pairs={len(cctrl_df)}  from {cctrl_df['patient'].nunique()} patients")
    buf.log(f"Channel ratio: min={ratios.min():.3f}  median={np.median(ratios):.3f}  "
            f"p95={np.percentile(ratios,95):.3f}  max={ratios.max():.3f}")
    buf.log(f"Channel AUROC: min={aurocs_cctrl.min():.3f}  median={np.median(aurocs_cctrl):.3f}  "
            f"p95={np.percentile(aurocs_cctrl,95):.3f}  max={aurocs_cctrl.max():.3f}")
    ratio_p95 = np.percentile(ratios, 95)
    median_cctrl_auroc = np.median(aurocs_cctrl)

    # ---------------------------------------------------------------- per-patient summary table (eligible34)
    buf.log("\n" + "=" * 96)
    buf.log("PER-PATIENT SUMMARY (34 eligible patients)")
    buf.log("=" * 96)
    hdr = (f"{'PID':<8}{'MMR':<6}{'n_norm_bef':<11}{'n_tum_bef':<10}{'n_norm_aft':<11}{'n_tum_aft':<10}"
           f"{'med_resid_n':<13}{'med_resid_t':<13}{'ratio':<8}{'quint_ratio':<12}{'OLS_coef':<10}"
           f"{'OLS_p':<11}{'res_AUROC':<10}{'cnv_AUROC':<10}{'ng_AUROC':<9}{'euc_AUROC':<10}{'TestA':<6}")
    buf.log(hdr)
    rows_summary = []
    for pid in eligible34:
        r = patient_results[pid]
        res, t, a = r["res"], r["testA"], r["aurocs"]
        med_n, med_t = np.median(res["r_n"]), np.median(res["r_m"])
        ratio = med_t / med_n
        h1b = ratio > ratio_p95
        row = dict(PID=pid, MMR=elig.loc[pid, "MMRStatus"], n_norm_bef=r["n_before"][0],
                   n_tum_bef=r["n_before"][1], n_norm_aft=r["n_after"][0], n_tum_aft=r["n_after"][1],
                   med_resid_n=med_n, med_resid_t=med_t, ratio=ratio,
                   quint_ratio=t["median_quintile_ratio"], ols_coef=t["ols_coef"], ols_p=t["ols_p"],
                   res_auroc=a["residual"], cnv_auroc=a["cnv_score"], ng_auroc=a["n_genes"],
                   euc_auroc=a["euclid_centroid"], testA_survives=t["survives"], h1b_survives=h1b)
        rows_summary.append(row)
        qr = row["quint_ratio"] if row["quint_ratio"] is not None else float("nan")
        buf.log(f"{pid:<8}{row['MMR']:<6}{row['n_norm_bef']:<11}{row['n_tum_bef']:<10}"
                f"{row['n_norm_aft']:<11}{row['n_tum_aft']:<10}{med_n:<13.3f}{med_t:<13.3f}{ratio:<8.2f}"
                f"{qr:<12.2f}{row['ols_coef']:<10.4f}{row['ols_p']:<11.2e}{a['residual']:<10.3f}"
                f"{a['cnv_score']:<10.3f}{a['n_genes']:<9.3f}{a['euclid_centroid']:<10.3f}"
                f"{('Y' if t['survives'] else 'N'):<6}")
    summary = pd.DataFrame(rows_summary)

    # ---------------------------------------------------------------- H1
    buf.log("\n" + "=" * 96)
    buf.log("H1 VERDICT")
    buf.log("=" * 96)
    n_h1_full = int((summary["testA_survives"] & summary["h1b_survives"]).sum())
    n_total = len(summary)
    buf.log(f"Test A passes: {int(summary['testA_survives'].sum())}/{n_total}")
    buf.log(f"Tumor/normal ratio > C-ctrl 95th pct ({ratio_p95:.3f}x): "
            f"{int(summary['h1b_survives'].sum())}/{n_total}")
    buf.log(f"BOTH conditions: {n_h1_full}/{n_total} ({100*n_h1_full/n_total:.1f}%)")
    h1_verdict = "SUPPORTED" if n_h1_full / n_total >= 0.70 else "NOT SUPPORTED"
    buf.log(f"H1 VERDICT: {h1_verdict} (need >=70%)")

    # ---------------------------------------------------------------- H2 gate + H2
    buf.log("\n" + "=" * 96)
    buf.log("H2 GATE and H2 VERDICT")
    buf.log("=" * 96)
    gate_pass = median_cctrl_auroc < 0.6
    buf.log(f"Median C-ctrl residual AUROC (normal channel B vs A) = {median_cctrl_auroc:.3f} "
            f"-> gate {'CLEARS' if gate_pass else 'FAILS'} (<0.6 required)")
    if not gate_pass:
        buf.log("H2 VERDICT: UNTESTABLE IN THIS DATASET (gate failed; AUROC results below are "
                "reported for completeness only and do NOT constitute an H2 test).")
    mmrd = summary[summary["MMR"] == "MMRd"]
    n_mmrd = len(mmrd)
    n_res_gt_cnv = int((mmrd["res_auroc"] > mmrd["cnv_auroc"]).sum())
    diffs = (mmrd["res_auroc"] - mmrd["cnv_auroc"]).values
    buf.log(f"\nMMRd patients (n={n_mmrd}): residual AUROC > CNV AUROC in {n_res_gt_cnv}/{n_mmrd} "
            f"({100*n_res_gt_cnv/max(n_mmrd,1):.1f}%)")
    buf.log(f"Median AUROC difference (residual - CNV) = {np.median(diffs):+.3f}")
    h2_criteria_met = (n_res_gt_cnv / max(n_mmrd, 1) >= 0.70) and (np.median(diffs) > 0.05)
    if gate_pass:
        buf.log(f"H2 VERDICT: {'SUPPORTED' if h2_criteria_met else 'NOT SUPPORTED'} "
                f"(need >=70% AND median diff>0.05)")
    else:
        buf.log(f"[reference only, gate failed] criteria as computed: "
                f"{'would have been SUPPORTED' if h2_criteria_met else 'would NOT have been SUPPORTED'}")

    # ---------------------------------------------------------------- H3
    buf.log("\n" + "=" * 96)
    buf.log("H3 (MMRp, control, no pass/fail threshold)")
    buf.log("=" * 96)
    mmrp = summary[summary["MMR"] == "MMRp"]
    buf.log(f"{'PID':<8}{'res_AUROC':<11}{'cnv_AUROC':<11}{'diff':<8}{'ratio_vs_cctrl_p95'}")
    for _, row in mmrp.iterrows():
        buf.log(f"{row['PID']:<8}{row['res_auroc']:<11.3f}{row['cnv_auroc']:<11.3f}"
                f"{row['res_auroc']-row['cnv_auroc']:<8.3f}"
                f"{'above' if row['h1b_survives'] else 'below'} p95 ({ratio_p95:.2f}x)")
    buf.log(f"\nMMRp (n={len(mmrp)}): median residual AUROC={mmrp['res_auroc'].median():.3f}  "
            f"median CNV AUROC={mmrp['cnv_auroc'].median():.3f}  "
            f"residual>CNV in {int((mmrp['res_auroc']>mmrp['cnv_auroc']).sum())}/{len(mmrp)}")

    # ---------------------------------------------------------------- Deviations
    buf.log("\n" + "=" * 96)
    buf.log("DEVIATIONS (implementation choices not covered by the pre-registration; no threshold changed)")
    buf.log("=" * 96)
    buf.log("- CNV score is computed on standard normalize_total(1e4)+log1p expression from the")
    buf.log("  patient's UNTHINNED raw counts (not binomial-thinned). The pre-registration and its")
    buf.log("  amendments specify thinning for the residual pipeline only; CNV scoring elsewhere in")
    buf.log("  this project (scripts/cnv_clone_vs_residual.py) was likewise never thinned.")
    buf.log("- AUROC comparisons (residual vs CNV vs n_genes vs Euclidean) are all restricted to the")
    buf.log("  SAME cell set: the cells that survived the residual pipeline's binomial thinning.")
    buf.log("  CNV scores for the dropped (below-depth-target) cells are computed but not used in")
    buf.log("  the AUROC, so all four scores are compared on identical cells per patient.")
    buf.log("- Gene IDs in the h5 (e.g. 'ENSG00000243485.5_4') carry both an Ensembl version and an")
    buf.log("  extra disambiguation suffix; both are stripped to the bare ENSG ID before GTF")
    buf.log("  matching, then var_names_make_unique() is applied (same convention as")
    buf.log("  scripts/rebuild_gse161277_all.py's load_sample for GSE161277).")
    buf.log("- Gene genomic positions (GTF merge) are computed ONCE globally (not per patient) and")
    buf.log("  reused for all 35 patients processed, since gene coordinates do not depend on the")
    buf.log("  patient; this is a performance optimization with no effect on any result.")
    buf.log("- C-ctrl for C162 (channel-eligible but excluded from the 34 by Amendment 1's chemistry")
    buf.log("  rule) is still computed, since the pre-registration specifies C-ctrl for \"the 15")
    buf.log("  patients with >=2 normal channels\" without re-stating the chemistry-eligibility")
    buf.log("  filter for that specific list; C162's C-ctrl rows are included in the pooled")
    buf.log("  distribution but it does not appear in the per-patient H1/H2/H3 summary table.")
    buf.log("- The H1(a)+H1(b) \"both conditions\" verdict is computed per patient as stated; the")
    buf.log("  pre-registration does not specify whether (a) and (b) must hold jointly per patient")
    buf.log("  or be satisfied as two separate >=70% rates. This script requires them jointly per")
    buf.log("  patient (the stricter, more literal reading of \"Test A passes AND ... ratio exceeds")
    buf.log("  ... in >=70% of eligible patients\") and also reports each condition's own rate above")
    buf.log("  so the reader can apply the looser reading if preferred.")

    out_path = os.path.join(gc.RESULTS_DIR, "gse178341_prereg_results.txt")
    buf.save(out_path)
    summary.to_csv(os.path.join(gc.RESULTS_DIR, "gse178341_prereg_summary.csv"), index=False)
    cctrl_df.to_csv(os.path.join(gc.RESULTS_DIR, "gse178341_cctrl_pairs.csv"), index=False)


if __name__ == "__main__":
    main()
