"""State-matched control analysis, pre-registered in docs/prereg_gse178341_statematched.md.
Runs H-SM1 and H-SM2 exactly as pre-registered. Does not touch
results/gse178341_prereg_results.txt, results/gse178341_posthoc.txt, or either
pre-registration document.

Per-cell residual and cross-fitted CNV values are NOT stored on disk by
scripts/gse178341_prereg.py or scripts/gse178341_posthoc.py (only per-patient AUROC
summaries were saved); they are recomputed here with IDENTICAL parameters, by importing
the same primitives those two scripts import (not reimplementing them):
  thin_to_depth           <- scripts/geom_dnb.py
  svd_subspace             <- scripts/geom_common.py
  fit_subspace, residual   <- scripts/ricci_curvature_pilot.py
  crossfit_cnv_patient     <- scripts/gse178341_posthoc.py (imported unmodified, called as-is)
  metadata/eligibility/h5-loading/gene-position helpers <- scripts/gse178341_prereg.py

scripts/gse178341_prereg.py's own `residual_pipeline` only tracks channel-of-origin for
surviving normal cells (needed for its C-ctrl test), not differentiation state for both
groups. Since state labels are needed for BOTH normal and tumor cells here, and the
pre-registration forbids changing the frozen script, a new orchestration function
(`residual_pipeline_with_state` below) mirrors `residual_pipeline`'s exact steps (same
thinning target definition, same seed, same HVG/PCA/subspace/fold logic) while also
threading state labels through -- it calls the same imported primitives, not a
reimplementation of the pipeline's logic.

Output: results/gse178341_statematched.txt
"""
import os
import sys
import time
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402
from geom_dnb import thin_to_depth  # noqa: E402
from ricci_curvature_pilot import fit_subspace, residual  # noqa: E402
from gse178341_prereg import (  # noqa: E402
    DATA_DIR, load_metadata, patient_column_ranges, compute_eligibility, load_patient_raw,
    build_gene_position_table, THIN_PCT, N_HVG, Q_PCA, SEED, Buf,
)
from gse178341_posthoc import crossfit_cnv_patient  # noqa: E402

warnings.filterwarnings("ignore", message=".*slogdet.*")

RESULTS = gc.RESULTS_DIR
MIN_STATE_N = 30
MIN_PAIRS_POWERED = 10
STEMTA_STATES = {"cE01", "cE02", "cE03"}
STATE_DESC = {
    "cE01": "Stem/TA-like", "cE02": "Stem/TA-like/Immature Goblet", "cE03": "Stem/TA-like prolif",
    "cE04": "Enterocyte 1", "cE05": "Enterocyte 2", "cE06": "Immature Goblet",
    "cE07": "Goblet/Enterocyte", "cE08": "Goblet", "cE09": "Best4", "cE10": "Tuft",
    "cE11": "Enteroendocrine",
}

DECISION_RULES_VERBATIM = """\
DECISION RULES, quoted verbatim from docs/prereg_gse178341_statematched.md
================================================================================================

Analysis unit: a (patient, state) pair is eligible if that patient has >=30 tumor-channel AND
>=30 normal-channel epithelial cells assigned to that state (cE01-cE11).

H-SM1 (residual detects malignancy beyond differentiation state). Supported if, across the
eligible (patient, state) pairs: the median within-state residual AUROC is > 0.75 AND
within-state residual AUROC is > 0.6 in >= 70% of eligible pairs.

H-SM2 (state-matched residual vs CNV, descriptive, no threshold). Report median within-state
(residual AUROC - cross-fitted CNV AUROC), overall and split by MMR status.

Underpowered rule: if fewer than 10 eligible pairs existed, the test would be declared
underpowered and reported descriptively only. (136 pairs are eligible; this does not apply.)

"What is NOT pre-registered": no fallback/secondary threshold if H-SM1's two conditions
disagree beyond reporting both numbers and declaring H-SM1 "not supported" whenever either
condition fails.

[Descriptive only, not part of any decision, pre-specified separately in this run's task]:
median within-state residual AUROC and CNV AUROC per state and per MMR status, and the same
for the three stem/TA-like states (cE01, cE02, cE03) pooled versus all other states pooled.
================================================================================================
"""


def load_metadata_with_state():
    meta = pd.read_csv(os.path.join(DATA_DIR, "GSE178341_crc10x_full_c295v4_submit_metatables.csv"),
                        usecols=["cellID", "SPECIMEN_TYPE", "SINGLECELL_TYPE", "PID", "MMRStatus"])
    cl = pd.read_csv(os.path.join(DATA_DIR, "GSE178341_crc10x_full_c295v4_submit_cluster.csv"),
                      usecols=["sampleID", "batchID", "clTopLevel", "cl295v11SubShort"])
    assert (meta["cellID"].values == cl["sampleID"].values).all()
    df = pd.concat([meta, cl[["batchID", "clTopLevel", "cl295v11SubShort"]]], axis=1)
    df["pos"] = np.arange(len(df))
    df["is_epi"] = df["clTopLevel"].values == "Epi"
    return df


def compute_eligible_pairs(df, eligible34):
    """Reproduces the exact 136-pair count from docs/prereg_gse178341_statematched.md
    (pre-thinning cell counts, as pre-registered -- eligibility is NOT recomputed on
    post-thinning survivor counts)."""
    epi = df[df["is_epi"] & df["PID"].isin(eligible34)]
    ct = epi.groupby(["PID", "cl295v11SubShort", "SPECIMEN_TYPE"]).size().unstack(fill_value=0)
    ct = ct.reindex(columns=["T", "N"], fill_value=0).reset_index()
    ct["eligible_pair"] = (ct["T"] >= MIN_STATE_N) & (ct["N"] >= MIN_STATE_N)
    return ct[ct["eligible_pair"]][["PID", "cl295v11SubShort", "T", "N"]].reset_index(drop=True)


def residual_pipeline_with_state(raw_n, raw_m, state_n, state_m, buf, tag):
    """Mirrors scripts/gse178341_prereg.py's residual_pipeline exactly (same thinning target,
    seed, HVG/PCA/subspace/fold steps -- same imported primitives), additionally tracking
    differentiation state for BOTH surviving normal and tumor cells."""
    rng = np.random.default_rng(SEED)
    combined = sp.vstack([raw_n, raw_m]).tocsr()
    total = np.asarray(combined.sum(axis=1)).ravel()
    target = np.percentile(total, THIN_PCT)
    keep_idx = np.where(total >= target)[0]
    thinned = thin_to_depth(combined, keep_idx, target, rng)
    is_n = np.r_[np.ones(raw_n.shape[0], bool), np.zeros(raw_m.shape[0], bool)][keep_idx]
    is_m = ~is_n
    state_kept = np.r_[state_n, state_m][keep_idx]
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
    state_n_kept, state_m_kept = state_kept[is_n], state_kept[is_m]

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
    return {"r_n": r_n, "r_m": r_m, "state_n": state_n_kept, "state_m": state_m_kept,
            "keep_idx": keep_idx, "n_normal_total": raw_n.shape[0]}


def process_patient(pid, col_range, df_pid, gene_pos_names, gene_pos_var, buf):
    t0 = time.time()
    start, end = col_range
    raw_full, gene_id_raw, _ = load_patient_raw(start, end)
    d = df_pid.reset_index(drop=True)
    assert len(d) == raw_full.shape[0]

    mask_norm_epi = ((d["SPECIMEN_TYPE"] == "N") & d["is_epi"]).values
    mask_tum_epi = ((d["SPECIMEN_TYPE"] == "T") & d["is_epi"]).values
    mask_nonepi = (~d["is_epi"]).values
    state_norm = d.loc[mask_norm_epi, "cl295v11SubShort"].values
    state_tum = d.loc[mask_tum_epi, "cl295v11SubShort"].values

    res = residual_pipeline_with_state(raw_full[mask_norm_epi], raw_full[mask_tum_epi],
                                        state_norm, state_tum, buf, pid)

    cnv_norm_ho, cnv_tum_avg = crossfit_cnv_patient(
        raw_full, gene_id_raw, gene_pos_names, gene_pos_var,
        mask_nonepi, mask_norm_epi, mask_tum_epi, buf, pid)

    # restrict CNV scores to the same post-thinning surviving cells, same convention as
    # scripts/gse178341_prereg.py's AUROC alignment (see its own Deviations section).
    keep_idx, n_norm_total = res["keep_idx"], res["n_normal_total"]
    is_n_keep = keep_idx < n_norm_total
    norm_keep_pos = keep_idx[is_n_keep]
    tum_keep_pos = keep_idx[~is_n_keep] - n_norm_total
    cnv_n_kept = cnv_norm_ho[norm_keep_pos]
    cnv_m_kept = cnv_tum_avg[tum_keep_pos]

    buf.log(f"    [{pid}] done [{time.time()-t0:.0f}s]")
    return {"r_n": res["r_n"], "r_m": res["r_m"], "state_n": res["state_n"], "state_m": res["state_m"],
            "cnv_n": cnv_n_kept, "cnv_m": cnv_m_kept}


def main():
    buf = Buf()
    buf.log(DECISION_RULES_VERBATIM)
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["numpy", "scipy", "scikit-learn", "scanpy", "infercnvpy"]))
    buf.log(f"seed={SEED}  thin_pct={THIN_PCT}  N_HVG={N_HVG}  Q_PCA={Q_PCA}  "
            f"min_state_n={MIN_STATE_N}\n")

    df_base = load_metadata()
    elig = compute_eligibility(df_base)
    eligible34 = elig[elig["eligible34"]].index.tolist()
    ranges = patient_column_ranges(df_base)

    df = load_metadata_with_state()
    pairs = compute_eligible_pairs(df, eligible34)
    buf.log(f"Eligible (patient, state) pairs (pre-thinning counts, as pre-registered): "
            f"{len(pairs)}")
    assert len(pairs) == 136, f"pair count changed: {len(pairs)} != 136 pre-registered"
    if len(pairs) < MIN_PAIRS_POWERED:
        buf.log("UNDERPOWERED (<10 pairs) -- would be reported descriptively only. Not the case here.\n")
    else:
        buf.log(f">= {MIN_PAIRS_POWERED} -- not underpowered; proceeding with H-SM1/H-SM2.\n")

    buf.log("Building global gene-position table (once)...")
    gene_pos_names, gene_pos_var, _ = build_gene_position_table(buf)
    buf.log("")

    patients_needed = sorted(pairs["PID"].unique())
    patient_data = {}
    for pid in patients_needed:
        buf.log("=" * 96)
        buf.log(f"{pid}  (MMR={elig.loc[pid,'MMRStatus']})")
        buf.log("=" * 96)
        df_pid = df[df["PID"] == pid]
        patient_data[pid] = process_patient(pid, ranges[pid], df_pid, gene_pos_names, gene_pos_var, buf)

    # ---------------------------------------------------------------- per-pair AUROC
    buf.log("\n" + "=" * 96)
    buf.log("PER-PAIR WITHIN-STATE AUROC")
    buf.log("=" * 96)
    hdr = (f"{'PID':<8}{'state':<7}{'MMR':<6}{'n_norm_prereg':<14}{'n_tum_prereg':<13}"
           f"{'n_norm_post_thin':<17}{'n_tum_post_thin':<16}{'res_AUROC':<11}{'cnv_AUROC':<10}")
    buf.log(hdr)
    rows = []
    for _, prow in pairs.iterrows():
        pid, state = prow["PID"], prow["cl295v11SubShort"]
        pd_ = patient_data[pid]
        mn = pd_["state_n"] == state
        mt = pd_["state_m"] == state
        n_n, n_t = int(mn.sum()), int(mt.sum())
        mmr = elig.loc[pid, "MMRStatus"]
        if n_n == 0 or n_t == 0:
            buf.log(f"{pid:<8}{state:<7}{mmr:<6}{prow['N']:<14}{prow['T']:<13}{n_n:<17}{n_t:<16}"
                    f"{'n/a (0 post-thin)':<11}{'n/a':<10}")
            rows.append({"PID": pid, "state": state, "MMR": mmr, "n_norm_prereg": prow["N"],
                         "n_tum_prereg": prow["T"], "n_norm_post_thin": n_n, "n_tum_post_thin": n_t,
                         "res_auroc": np.nan, "cnv_auroc": np.nan})
            continue
        y = np.r_[np.zeros(n_n), np.ones(n_t)]
        res_auc = roc_auc_score(y, np.r_[pd_["r_n"][mn], pd_["r_m"][mt]])
        cnv_auc = roc_auc_score(y, np.r_[pd_["cnv_n"][mn], pd_["cnv_m"][mt]])
        buf.log(f"{pid:<8}{state:<7}{mmr:<6}{prow['N']:<14}{prow['T']:<13}{n_n:<17}{n_t:<16}"
                f"{res_auc:<11.3f}{cnv_auc:<10.3f}")
        rows.append({"PID": pid, "state": state, "MMR": mmr, "n_norm_prereg": prow["N"],
                     "n_tum_prereg": prow["T"], "n_norm_post_thin": n_n, "n_tum_post_thin": n_t,
                     "res_auroc": res_auc, "cnv_auroc": cnv_auc})
    pair_df = pd.DataFrame(rows)
    n_nan = int(pair_df["res_auroc"].isna().sum())
    if n_nan:
        buf.log(f"\nNOTE: {n_nan}/{len(pair_df)} pre-registered pairs had 0 cells in one group AFTER")
        buf.log("binomial thinning (thinning drops cells below the per-patient depth target; the")
        buf.log("pre-registered eligibility count used PRE-thinning cell numbers, as specified in")
        buf.log("docs/prereg_gse178341_statematched.md). These pairs are excluded from H-SM1/H-SM2")
        buf.log("(AUROC undefined with one empty group) and reported here for transparency.")
    valid = pair_df.dropna(subset=["res_auroc"])

    # ---------------------------------------------------------------- H-SM1
    buf.log("\n" + "=" * 96)
    buf.log("H-SM1 VERDICT")
    buf.log("=" * 96)
    med_res = valid["res_auroc"].median()
    frac_gt06 = (valid["res_auroc"] > 0.6).mean()
    buf.log(f"Median within-state residual AUROC across {len(valid)} valid pairs: {med_res:.3f} "
            f"(need > 0.75)")
    buf.log(f"Fraction of pairs with residual AUROC > 0.6: {100*frac_gt06:.1f}% "
            f"({int((valid['res_auroc']>0.6).sum())}/{len(valid)}) (need >= 70%)")
    cond1 = med_res > 0.75
    cond2 = frac_gt06 >= 0.70
    verdict_sm1 = "SUPPORTED" if (cond1 and cond2) else "NOT SUPPORTED"
    buf.log(f"Condition 1 (median>0.75): {'met' if cond1 else 'NOT met'}")
    buf.log(f"Condition 2 (>=70% pairs >0.6): {'met' if cond2 else 'NOT met'}")
    buf.log(f"H-SM1 VERDICT: {verdict_sm1}")

    # ---------------------------------------------------------------- H-SM2
    buf.log("\n" + "=" * 96)
    buf.log("H-SM2 (descriptive, no threshold)")
    buf.log("=" * 96)
    gap = valid["res_auroc"] - valid["cnv_auroc"]
    buf.log(f"Overall median (residual AUROC - CNV AUROC) = {gap.median():+.3f}  (n={len(valid)})")
    for mmr, g in valid.groupby("MMR"):
        gap_g = g["res_auroc"] - g["cnv_auroc"]
        buf.log(f"  {mmr} (n={len(g)}): median gap = {gap_g.median():+.3f}  "
                f"(median residual={g['res_auroc'].median():.3f}, median CNV={g['cnv_auroc'].median():.3f})")

    # ---------------------------------------------------------------- Descriptive: per state / MMR
    buf.log("\n" + "=" * 96)
    buf.log("DESCRIPTIVE ONLY -- median within-state AUROC per state and per MMR status "
            "(NOT part of any decision)")
    buf.log("=" * 96)
    buf.log(f"{'state':<7}{'desc':<30}{'n_pairs':<9}{'median_res_AUROC':<18}{'median_cnv_AUROC'}")
    for state, g in valid.groupby("state"):
        buf.log(f"{state:<7}{STATE_DESC.get(state,''):<30}{len(g):<9}"
                f"{g['res_auroc'].median():<18.3f}{g['cnv_auroc'].median():.3f}")

    buf.log(f"\n{'state':<7}{'MMR':<6}{'n_pairs':<9}{'median_res_AUROC':<18}{'median_cnv_AUROC'}")
    for (state, mmr), g in valid.groupby(["state", "MMR"]):
        buf.log(f"{state:<7}{mmr:<6}{len(g):<9}{g['res_auroc'].median():<18.3f}{g['cnv_auroc'].median():.3f}")

    # ---------------------------------------------------------------- Descriptive: stem/TA vs other
    buf.log("\n" + "=" * 96)
    buf.log("DESCRIPTIVE ONLY -- stem/TA-like (cE01,cE02,cE03) pooled vs all other states pooled "
            "(NOT part of any decision)")
    buf.log("=" * 96)
    is_stemta = valid["state"].isin(STEMTA_STATES)
    stemta, other = valid[is_stemta], valid[~is_stemta]
    buf.log(f"Stem/TA-like (n={len(stemta)} pairs): median residual AUROC={stemta['res_auroc'].median():.3f}  "
            f"median CNV AUROC={stemta['cnv_auroc'].median():.3f}")
    buf.log(f"Other states  (n={len(other)} pairs): median residual AUROC={other['res_auroc'].median():.3f}  "
            f"median CNV AUROC={other['cnv_auroc'].median():.3f}")
    buf.log(f"Difference in median residual AUROC (stem/TA - other): "
            f"{stemta['res_auroc'].median()-other['res_auroc'].median():+.3f}")

    # ---------------------------------------------------------------- Deviations
    buf.log("\n" + "=" * 96)
    buf.log("DEVIATIONS (implementation choices not covered by the pre-registration; no threshold changed)")
    buf.log("=" * 96)
    buf.log("- Per-cell residual and cross-fitted CNV values were not stored by the earlier scripts")
    buf.log("  (only per-patient AUROC summaries were saved), so both were recomputed here using the")
    buf.log("  same imported primitives and identical parameters (seed=0, thin_pct=10, N_HVG=3000,")
    buf.log("  Q_PCA=50, Q_SUB=20, 5-fold CV, CNV window_size=250) as scripts/gse178341_prereg.py and")
    buf.log("  scripts/gse178341_posthoc.py. The residual pipeline is non-deterministic only in the")
    buf.log("  sense that it depends solely on the seeded RNG and the input data, both unchanged, so")
    buf.log("  recomputed values should match the original per-patient AUROCs; this was not")
    buf.log("  separately verified cell-for-cell against the stored summary CSVs.")
    buf.log("- Pair eligibility (>=30/>=30) was computed on PRE-thinning cell counts, exactly as")
    buf.log("  pre-registered. Binomial thinning (applied only to the residual pipeline, not to raw")
    buf.log("  cell counts) can leave FEWER than 30 cells per group surviving in a given state for a")
    buf.log("  given pair; such pairs are reported with their post-thinning n alongside the")
    buf.log("  pre-registered n, and pairs where thinning leaves ZERO cells in either group (AUROC")
    buf.log("  undefined) are excluded from H-SM1/H-SM2 and flagged by count above.")
    buf.log("- CNV scores are restricted to the same post-thinning surviving cells as the residual,")
    buf.log("  for a fair within-pair AUROC comparison -- the same convention")
    buf.log("  scripts/gse178341_prereg.py used for its own main AUROC comparisons.")
    buf.log("- Where the pre-registration's AND-combined H-SM1 conditions could disagree, this script")
    buf.log("  declares H-SM1 'not supported' if either condition fails, as the pre-registration")
    buf.log("  states under \"What is NOT pre-registered.\"")

    out_path = os.path.join(RESULTS, "gse178341_statematched.txt")
    buf.save(out_path)
    pair_df.to_csv(os.path.join(RESULTS, "gse178341_statematched_pairs.csv"), index=False)


if __name__ == "__main__":
    main()
