"""POST HOC analyses on GSE178341 (NOT pre-registered). Motivated by the pre-registered run's
H3 anomaly: CNV AUROC was weak even in MMRp (median 0.72), where CNV-based detection is
expected to work well. Does not modify results/gse178341_prereg_results.txt or
docs/prereg_gse178341.md.

PART A -- baselines, from the stored pre-registered run (results/gse178341_prereg_summary.csv),
  no recomputation.
PART B -- cross-fitted CNV: per eligible patient, 5-fold split of normal epithelial cells
  (seed 0); each fold's infercnvpy reference = non-epithelial cells + the OTHER 4 training
  folds of normal epithelium (never the held-out fold); scores the held-out normal fold and all
  tumor epithelium (tumor scores averaged over the 5 fold-runs, same convention used for
  malignant cells' residual elsewhere in this project). Fixes the Amendment-2 reference's
  structural advantage for normal cells (they were previously scored in-sample).
PART C -- mechanism test (Mann-Whitney, cross-fitted CNV AUROC and the residual-CNV gap, MMRd
  vs MMRp), with the decision rule stated before the numbers.
PART D -- per C-ctrl-patient channel-AUROC vs tumor-AUROC check, from the stored
  results/gse178341_cctrl_pairs.csv, no recomputation.

Reuses (imported, not copied) scripts/gse178341_prereg.py's metadata/eligibility/h5-loading/
gene-position functions, and the same frozen-pipeline primitives it itself imports.

Output: results/gse178341_posthoc.txt
"""
import os
import sys
import time
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import scanpy as sc
import infercnvpy as cnv
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402
from gse178341_prereg import (  # noqa: E402
    load_metadata, patient_column_ranges, compute_eligibility, load_patient_raw,
    clean_gene_ids, build_gene_position_table, THIN_PCT, CNV_WINDOW, SEED, Buf,
)

warnings.filterwarnings("ignore", message=".*slogdet.*")

RESULTS = gc.RESULTS_DIR
N_FOLDS = 5
SIG_ALPHA = 0.05


def crossfit_cnv_patient(raw_full, gene_id_raw, gene_pos_names, gene_pos_var,
                          mask_nonepi, mask_norm_epi, mask_tum_epi, buf, tag):
    """5-fold cross-fitted CNV score: reference = non-epi + 4/5 training folds of normal
    epithelium (never the held-out fold); observation = held-out normal fold + ALL tumor
    epithelium. Tumor scores are averaged over the 5 fold-runs (each fold uses a slightly
    different reference composition, since none ever includes malignant-proxy cells)."""
    ensg = clean_gene_ids(gene_id_raw)
    ad_full = sc.AnnData(X=raw_full, var=pd.DataFrame(index=ensg))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad_full.var_names_make_unique()
    keep = ad_full.var_names.isin(gene_pos_names)
    ad_sub = ad_full[:, keep].copy()
    ad_sub.var = ad_sub.var.join(gene_pos_var, how="left")

    n_norm = int(mask_norm_epi.sum())
    rng = np.random.default_rng(SEED)
    folds = rng.permutation(n_norm) % N_FOLDS
    norm_idx_global = np.where(mask_norm_epi)[0]
    tum_idx_global = np.where(mask_tum_epi)[0]
    nonepi_idx_global = np.where(mask_nonepi)[0]

    cnv_norm_heldout = np.full(n_norm, np.nan)
    cnv_tum_per_fold = np.full((N_FOLDS, len(tum_idx_global)), np.nan)

    for f in range(N_FOLDS):
        train_mask = folds != f
        held_mask = folds == f
        train_idx = norm_idx_global[train_mask]
        held_idx = norm_idx_global[held_mask]

        obs_idx = np.concatenate([nonepi_idx_global, train_idx, held_idx, tum_idx_global])
        ad_f = ad_sub[obs_idx].copy()
        n_nonepi, n_train, n_held, n_tum = (len(nonepi_idx_global), len(train_idx),
                                             len(held_idx), len(tum_idx_global))
        grp = np.array((["nonepi_ref"] * n_nonepi) + (["normal_train_ref"] * n_train)
                       + (["normal_heldout"] * n_held) + (["tumor"] * n_tum))
        ad_f.obs["cnv_group"] = grp
        ad_f.obs["cnv_ref"] = np.where(np.isin(grp, ["nonepi_ref", "normal_train_ref"]),
                                        "reference", "observation")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.normalize_total(ad_f, target_sum=1e4)
            sc.pp.log1p(ad_f)
            cnv.tl.infercnv(ad_f, reference_key="cnv_ref", reference_cat=["reference"],
                             window_size=CNV_WINDOW)
        cell_key = "_cnv_cell_tmp"
        ad_f.obs[cell_key] = ad_f.obs_names.astype(str)
        cnv.tl.cnv_score(ad_f, groupby=cell_key, use_rep="cnv", key_added="cnv_score")

        held_scores = ad_f.obs.loc[ad_f.obs.cnv_group == "normal_heldout", "cnv_score"].values
        tum_scores = ad_f.obs.loc[ad_f.obs.cnv_group == "tumor", "cnv_score"].values
        cnv_norm_heldout[held_mask] = held_scores
        cnv_tum_per_fold[f] = tum_scores
        buf.log(f"    [{tag}] fold {f}: ref(nonepi+train_normal)={n_nonepi+n_train}  "
                f"held_normal={n_held}  tumor={n_tum}")

    cnv_tum_avg = np.nanmean(cnv_tum_per_fold, axis=0)
    assert not np.any(np.isnan(cnv_norm_heldout))
    return cnv_norm_heldout, cnv_tum_avg


def part_a(summary, buf):
    buf.log("\n" + "=" * 96)
    buf.log("PART A [POST HOC] -- baselines from the stored pre-registered run (no recomputation)")
    buf.log("=" * 96)
    hdr = f"{'PID':<8}{'MMR':<6}{'res_AUROC':<11}{'cnv_AUROC':<11}{'euc_AUROC':<11}{'ng_AUROC':<10}"
    buf.log(hdr)
    for _, r in summary.iterrows():
        buf.log(f"{r['PID']:<8}{r['MMR']:<6}{r['res_auroc']:<11.3f}{r['cnv_auroc']:<11.3f}"
                f"{r['euc_auroc']:<11.3f}{r['ng_auroc']:<10.3f}")

    buf.log("\nMedians by MMR status:")
    for mmr, g in summary.groupby("MMR"):
        buf.log(f"  {mmr} (n={len(g)}): residual={g['res_auroc'].median():.3f}  "
                f"CNV={g['cnv_auroc'].median():.3f}  euclid={g['euc_auroc'].median():.3f}  "
                f"n_genes={g['ng_auroc'].median():.3f}")

    diff = summary["res_auroc"] - summary["euc_auroc"]
    n_gt = int((diff > 0.02).sum())
    buf.log(f"\nPatients where residual AUROC exceeds Euclidean-distance AUROC by >0.02: "
            f"{n_gt}/{len(summary)}")
    buf.log(f"  (median residual-euclid gap = {diff.median():+.3f})")


def part_b(df, ranges, eligible34, gene_pos_names, gene_pos_var, summary, buf):
    buf.log("\n" + "=" * 96)
    buf.log("PART B [POST HOC] -- cross-fitted CNV (fair, held-out comparison)")
    buf.log("=" * 96)
    rows = []
    for pid in eligible34:
        t0 = time.time()
        buf.log(f"\n  {pid}:")
        start, end = ranges[pid]
        raw_full, gene_id_raw, _ = load_patient_raw(start, end)
        d = df[df["PID"] == pid].reset_index(drop=True)
        mask_norm_epi = ((d["SPECIMEN_TYPE"] == "N") & d["is_epi"]).values
        mask_tum_epi = ((d["SPECIMEN_TYPE"] == "T") & d["is_epi"]).values
        mask_nonepi = (~d["is_epi"]).values

        cnv_norm_ho, cnv_tum_avg = crossfit_cnv_patient(
            raw_full, gene_id_raw, gene_pos_names, gene_pos_var,
            mask_nonepi, mask_norm_epi, mask_tum_epi, buf, pid)

        y = np.r_[np.zeros(len(cnv_norm_ho)), np.ones(len(cnv_tum_avg))]
        auc_xfit = roc_auc_score(y, np.r_[cnv_norm_ho, cnv_tum_avg])
        res_auroc = summary.loc[summary.PID == pid, "res_auroc"].iat[0]
        amend2_auroc = summary.loc[summary.PID == pid, "cnv_auroc"].iat[0]
        mmr = summary.loc[summary.PID == pid, "MMR"].iat[0]
        buf.log(f"  [{pid}] cross-fitted CNV AUROC={auc_xfit:.3f}  (Amendment-2 CNV AUROC was "
                f"{amend2_auroc:.3f})  residual AUROC={res_auroc:.3f}  [{time.time()-t0:.0f}s]")
        rows.append({"PID": pid, "MMR": mmr, "res_auroc": res_auroc,
                     "cnv_auroc_crossfit": auc_xfit, "cnv_auroc_amend2": amend2_auroc})

    xfit_df = pd.DataFrame(rows)
    buf.log("\n" + "-" * 96)
    buf.log("PART B SUMMARY TABLE")
    buf.log("-" * 96)
    hdr = f"{'PID':<8}{'MMR':<6}{'res_AUROC':<11}{'CNV_crossfit':<13}{'CNV_amend2':<11}{'xfit-amend2'}"
    buf.log(hdr)
    for _, r in xfit_df.iterrows():
        buf.log(f"{r['PID']:<8}{r['MMR']:<6}{r['res_auroc']:<11.3f}{r['cnv_auroc_crossfit']:<13.3f}"
                f"{r['cnv_auroc_amend2']:<11.3f}{r['cnv_auroc_crossfit']-r['cnv_auroc_amend2']:+.3f}")
    buf.log("\nMedians by MMR status:")
    for mmr, g in xfit_df.groupby("MMR"):
        buf.log(f"  {mmr} (n={len(g)}): residual={g['res_auroc'].median():.3f}  "
                f"CNV_crossfit={g['cnv_auroc_crossfit'].median():.3f}  "
                f"CNV_amend2={g['cnv_auroc_amend2'].median():.3f}")
    return xfit_df


def part_c(xfit_df, buf):
    buf.log("\n" + "=" * 96)
    buf.log("PART C [POST HOC] -- mechanism test")
    buf.log("=" * 96)
    buf.log("DECISION RULE, stated before the numbers: the \"residual helps where CNV is blind\"")
    buf.log("claim is supported only if cross-fitted CNV AUROC is SIGNIFICANTLY LOWER in MMRd AND")
    buf.log(f"the residual-CNV gap is SIGNIFICANTLY LARGER in MMRd (both at alpha={SIG_ALPHA}, two-sided")
    buf.log("Mann-Whitney).\n")

    mmrd = xfit_df[xfit_df["MMR"] == "MMRd"]
    mmrp = xfit_df[xfit_df["MMR"] == "MMRp"]
    gap = xfit_df["res_auroc"] - xfit_df["cnv_auroc_crossfit"]
    xfit_df = xfit_df.assign(gap=gap)
    gap_d = xfit_df.loc[xfit_df.MMR == "MMRd", "gap"]
    gap_p = xfit_df.loc[xfit_df.MMR == "MMRp", "gap"]

    u1, p1 = mannwhitneyu(mmrd["cnv_auroc_crossfit"], mmrp["cnv_auroc_crossfit"], alternative="two-sided")
    buf.log(f"Cross-fitted CNV AUROC: MMRd median={mmrd['cnv_auroc_crossfit'].median():.3f} (n={len(mmrd)})  "
            f"MMRp median={mmrp['cnv_auroc_crossfit'].median():.3f} (n={len(mmrp)})")
    buf.log(f"  Mann-Whitney U={u1:.1f}  p={p1:.3e}  "
            f"{'MMRd lower (as biology predicts)' if mmrd['cnv_auroc_crossfit'].median() < mmrp['cnv_auroc_crossfit'].median() else 'MMRd NOT lower'}")
    sig1 = p1 < SIG_ALPHA and mmrd["cnv_auroc_crossfit"].median() < mmrp["cnv_auroc_crossfit"].median()

    u2, p2 = mannwhitneyu(gap_d, gap_p, alternative="two-sided")
    buf.log(f"\nResidual-CNV gap (res_auroc - cnv_auroc_crossfit): MMRd median={gap_d.median():+.3f} "
            f"(n={len(gap_d)})  MMRp median={gap_p.median():+.3f} (n={len(gap_p)})")
    buf.log(f"  Mann-Whitney U={u2:.1f}  p={p2:.3e}  "
            f"{'MMRd larger gap (as hypothesis predicts)' if gap_d.median() > gap_p.median() else 'MMRd NOT larger'}")
    sig2 = p2 < SIG_ALPHA and gap_d.median() > gap_p.median()

    buf.log(f"\nVERDICT: \"residual helps where CNV is blind\" claim is "
            f"{'SUPPORTED' if sig1 and sig2 else 'NOT SUPPORTED'} "
            f"(condition 1 {'met' if sig1 else 'NOT met'}, condition 2 {'met' if sig2 else 'NOT met'})")
    return xfit_df


def part_d(cctrl_df, summary, buf):
    buf.log("\n" + "=" * 96)
    buf.log("PART D [POST HOC] -- per-patient channel check (from stored C-ctrl pairs, no recomputation)")
    buf.log("=" * 96)
    per_patient_chan_auc = cctrl_df.groupby("patient")["auroc"].median().rename("channel_auroc_median")
    per_patient_chan_max = cctrl_df.groupby("patient")["auroc"].max().rename("channel_auroc_max")
    merged = per_patient_chan_auc.to_frame().join(per_patient_chan_max)
    merged = merged.join(summary.set_index("PID")[["MMR", "res_auroc"]], how="left")
    buf.log(f"{'patient':<10}{'MMR':<6}{'tumor_vs_normal_res_AUROC':<28}{'channel_AUROC_median':<22}"
            f"{'channel_AUROC_max':<19}{'flag (>0.75)'}")
    flagged = []
    for pid, r in merged.sort_values("channel_auroc_max", ascending=False).iterrows():
        flag = r["channel_auroc_max"] > 0.75
        if flag:
            flagged.append(pid)
        res_str = f"{r['res_auroc']:.3f}" if pd.notna(r["res_auroc"]) else "n/a (not in eligible-34)"
        buf.log(f"{pid:<10}{str(r['MMR']):<6}{res_str:<28}{r['channel_auroc_median']:<22.3f}"
                f"{r['channel_auroc_max']:<19.3f}{'YES' if flag else 'no'}")
    buf.log(f"\nPatients with max channel AUROC > 0.75: {len(flagged)}/{len(merged)} -> {flagged}")


def main():
    buf = Buf()
    buf.log("POST HOC analyses on GSE178341 -- NOT pre-registered.")
    buf.log("Motivated by the pre-registered run's H3 anomaly: CNV AUROC was weak even in MMRp")
    buf.log("(median 0.72, see results/gse178341_prereg_results.txt), where CNV-based detection is")
    buf.log("expected to work well. Does not modify results/gse178341_prereg_results.txt or")
    buf.log("docs/prereg_gse178341.md. Every section below is explicitly labeled [POST HOC].\n")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["numpy", "scipy", "scikit-learn", "scanpy", "infercnvpy"]))
    buf.log(f"seed={SEED}  n_folds={N_FOLDS}  thin_pct={THIN_PCT}  cnv_window={CNV_WINDOW}\n")

    summary = pd.read_csv(os.path.join(RESULTS, "gse178341_prereg_summary.csv"))
    cctrl_df = pd.read_csv(os.path.join(RESULTS, "gse178341_cctrl_pairs.csv"))

    part_a(summary, buf)

    df = load_metadata()
    ranges = patient_column_ranges(df)
    elig = compute_eligibility(df)
    eligible34 = elig[elig["eligible34"]].index.tolist()
    assert sorted(eligible34) == sorted(summary["PID"].tolist()), "eligibility mismatch vs stored run"

    buf.log("\nBuilding global gene-position table (once)...")
    gene_pos_names, gene_pos_var, _ = build_gene_position_table(buf)

    xfit_df = part_b(df, ranges, eligible34, gene_pos_names, gene_pos_var, summary, buf)
    xfit_df = part_c(xfit_df, buf)
    part_d(cctrl_df, summary, buf)

    buf.log("\n" + "=" * 96)
    buf.log("[POST HOC] CAVEATS")
    buf.log("=" * 96)
    buf.log("- All analyses in this file are POST HOC: they were designed after seeing the")
    buf.log("  pre-registered run's results (specifically the H3 anomaly) and are exploratory,")
    buf.log("  not confirmatory. They do not alter the pre-registered H1/H2/H3 verdicts.")
    buf.log("- Cross-fitted CNV (Part B) still shares genomic positions, window size, and the")
    buf.log("  non-epithelial reference cells with the Amendment-2 CNV score; only the normal-")
    buf.log("  epithelial portion of the reference is now held out per fold.")
    buf.log("- Tumor epithelial CNV scores in Part B are averaged over 5 independent fold-runs")
    buf.log("  (each with a very slightly different reference, varying only in which 80% of normal")
    buf.log("  cells are included) -- this is a minor smoothing effect, not a held-out score in the")
    buf.log("  same sense as the normal fold scores.")
    buf.log("- Part C's Mann-Whitney tests use n=20 (MMRd) and n=14 (MMRp) eligible patients --")
    buf.log("  modest sample sizes for a two-group nonparametric test.")

    out_path = os.path.join(RESULTS, "gse178341_posthoc.txt")
    buf.save(out_path)
    xfit_df.to_csv(os.path.join(RESULTS, "gse178341_posthoc_crossfit_cnv.csv"), index=False)


if __name__ == "__main__":
    main()
