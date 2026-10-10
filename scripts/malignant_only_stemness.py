"""Test whether the residual-stemness (ASCL2) association survives when restricted to
CNV-confirmed malignant cells only.

STEP 1 (Patient1, Patient2 -- GSE161277): reuse the CNV pipeline (scripts/cnv_clone_vs_
  residual.py's run_patient) and the malignant-like/normal-like clone labeling rule from
  scripts/cnv_followup_checks.py's Check A. Restrict to malignant-like cells and report
  residual ratio, Spearman(residual, ASCL2), Spearman(residual, SOX4), ASCL2+ % top/bottom
  decile.
STEP 2 (CRC epithelial cohort, data/crc_epithelial.h5ad): per donor with >=50 normal and
  >=50 malignant cells, same residual + stemness tests. Skipped if the file is absent.
STEP 3: summary table across all samples.
STEP 4: quality confound -- Spearman(residual, pct_counts_mt) and
  Spearman(residual, n_genes_by_counts) within malignant cells.

Output: results/malignant_only_stemness.txt
"""
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnv_clone_vs_residual import (  # noqa: E402 - reuse exact pipeline
    REPO, RESULTS, H5AD, PATIENTS, ARM_EVENTS, Buf,
    setup_positions, build_patient_adata, run_patient,
    arm_level_percell,
)
from paper_final_verification import subspace_svd, residual as paper_residual  # noqa: E402

DATA_DIR = REPO / "data" / "GSE161277"
CRC_PATH = REPO / "data" / "crc_epithelial.h5ad"
CRC_MIN_N = 50
MARKERS = ["ASCL2", "SOX4"]


def label_malignant_clones(carc, adata, gv, var, m_norm, m_carc, buf):
    """Same rule as cnv_followup_checks.py Check A: malignant-like = CNV score > 1.1x normal
    median (p<0.01) AND >=4/7 CRC arm events significant vs normal."""
    cnv_score_norm = adata.obs.cnv_score.values[m_norm]
    percell_norm = arm_level_percell(gv, var, m_norm)
    carc_idx_in_adata = np.where(m_carc)[0]

    clones = list(carc.obs["cnv_clone"].cat.categories)
    labels = {}
    buf.log(f"\n{'clone':<8}{'n':<8}{'cnv_score_med':<16}{'vs_normal_p':<14}{'n_arm_events':<14}{'label'}")
    for cl in clones:
        cl_mask_in_carc = (carc.obs["cnv_clone"] == cl).values
        cl_idx_in_adata = carc_idx_in_adata[cl_mask_in_carc]
        n = len(cl_idx_in_adata)

        score_cl = adata.obs.cnv_score.values[cl_idx_in_adata]
        _, p_score = mannwhitneyu(score_cl, cnv_score_norm, alternative="two-sided")
        score_up = np.median(score_cl) > np.median(cnv_score_norm) * 1.1 and p_score < 0.01

        gv_cl = gv[cl_idx_in_adata]
        percell_cl = arm_level_percell(gv_cl, var, np.ones(n, dtype=bool))
        n_events = 0
        for (chrom, arm), expected in ARM_EVENTS.items():
            vc_arr, vn_arr = percell_cl[(chrom, arm)], percell_norm[(chrom, arm)]
            if np.all(np.isnan(vc_arr)) or np.all(np.isnan(vn_arr)):
                continue
            _, p_arm = mannwhitneyu(vc_arr, vn_arr, alternative="two-sided", nan_policy="omit")
            direction_ok = (expected == "gain" and np.nanmean(vc_arr) > np.nanmean(vn_arr)) or \
                           (expected == "loss" and np.nanmean(vc_arr) < np.nanmean(vn_arr))
            if direction_ok and p_arm < 0.01:
                n_events += 1

        label = "malignant-like" if (score_up and n_events >= 4) else "normal-like"
        labels[cl] = label
        buf.log(f"{cl:<8}{n:<8}{np.median(score_cl):<16.4f}{p_score:<14.3e}{n_events:<14}{label}")
    return labels


def stemness_stats(res, ascl2, sox4):
    rho_a, p_a = spearmanr(res, ascl2)
    rho_s, p_s = spearmanr(res, sox4)
    q10, q90 = np.quantile(res, [0.10, 0.90])
    top_pct = 100 * (ascl2[res >= q90] > 0).mean()
    bot_pct = 100 * (ascl2[res <= q10] > 0).mean()
    return {"rho_ascl2": rho_a, "p_ascl2": p_a, "rho_sox4": rho_s, "p_sox4": p_s,
            "ascl2_top_pct": top_pct, "ascl2_bot_pct": bot_pct}


def load_raw_mt_pct(gsm_id, label):
    """Raw-count %MT for one sample, indexed by the same obs_names used in GSE161277_all.h5ad."""
    prefix = f"{gsm_id}_{label}"
    matrix = scipy.io.mmread(DATA_DIR / f"{prefix}_matrix.mtx.gz").T.tocsr()
    barcodes = pd.read_csv(DATA_DIR / f"{prefix}_barcodes.tsv.gz", header=None, sep="\t")[0].values
    features = pd.read_csv(DATA_DIR / f"{prefix}_features.tsv.gz", header=None, sep="\t")
    symbols = features[1].values
    mt_mask = pd.Series(symbols).astype(str).str.upper().str.startswith("MT-").values

    total = np.asarray(matrix.sum(axis=1)).flatten()
    mt_sum = np.asarray(matrix[:, mt_mask].sum(axis=1)).flatten()
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_mt = np.where(total > 0, 100 * mt_sum / total, np.nan)
    obs_names = [f"{gsm_id}_{bc}" for bc in barcodes]
    return pd.Series(pct_mt, index=obs_names), int(mt_mask.sum())


def step1_patient(patient, samples, raw_pos, obs_all, raw_ref, pca_full, full_obs, buf):
    buf.log("\n" + "=" * 90)
    buf.log(f"STEP 1 -- {patient}")
    buf.log("=" * 90)

    adata_p = build_patient_adata(raw_pos, obs_all, patient, samples)
    adata_p.obsm["X_pca_full"] = pca_full.loc[adata_p.obs_names].values
    pipe_buf = Buf()
    print(f"\n=== Re-running CNV pipeline for {patient} ===")
    pipeline = run_patient(patient, adata_p, raw_ref, pipe_buf)
    if pipeline is None:
        buf.log(f"{patient}: no CNV structure detected in the original pipeline -- cannot label clones.")
        return None

    adata, carc = pipeline["adata"], pipeline["carc"]
    gv, var = pipeline["gv"], pipeline["var"]
    m_norm, m_carc = pipeline["m_norm"], pipeline["m_carc"]

    buf.log("\nCNV clone labeling (malignant-like vs normal-like, same rule as Check A):")
    labels = label_malignant_clones(carc, adata, gv, var, m_norm, m_carc, buf)
    malignant_clones = [cl for cl, lab in labels.items() if lab == "malignant-like"]
    mal_mask = carc.obs["cnv_clone"].isin(malignant_clones).values
    frac_normal_like = 1 - mal_mask.sum() / len(carc)
    buf.log(f"\nMalignant-like clone(s): {malignant_clones}  "
            f"n_malignant={int(mal_mask.sum())}/{len(carc)}  "
            f"fraction normal-like = {100 * frac_normal_like:.1f}%")

    if not malignant_clones:
        buf.log(f"{patient}: no malignant-like clone -- cannot compute malignant-only residual stats.")
        return None

    pts_all = adata.obsm["X_pca_full"][:, :50]
    pts_norm = pts_all[m_norm]
    U, mean = subspace_svd(pts_norm, k_eig=20)
    res_norm = paper_residual(pts_norm, U, mean)

    # CHECK 1 -- formula sanity check: carc.obs["residual"] was set inside run_patient() using
    # subspace_svd/residual imported directly from paper_final_verification.py (the centered
    # formula: r_i = ||(x_i-mu) - UU^T(x_i-mu)||). Confirm that computing the ratio on ALL
    # carcinoma epithelial cells (no malignant restriction), via this exact same code path,
    # reproduces the Mac rebuild verification value (~3.67x Patient1, ~1.39x Patient2).
    res_all_carc = carc.obs["residual"].values
    ratio_all = np.median(res_all_carc) / np.median(res_norm)
    ref_ratio = 3.67 if patient == "Patient1" else 1.39
    formula_ok = abs(ratio_all - ref_ratio) < 0.05
    buf.log(f"\nCHECK 1 -- formula sanity check: residual function used is "
            f"paper_final_verification.residual() (centered: subtracts the normal-cell mean "
            f"before projecting), imported directly, not reimplemented.")
    buf.log(f"  Ratio on ALL carcinoma epithelial cells (same code path, no malignant "
            f"restriction) = {ratio_all:.2f}x  (Mac rebuild reference: ~{ref_ratio}x)  -> "
            f"{'MATCH: centered formula confirmed, no fix needed.' if formula_ok else 'MISMATCH: investigate formula.'}")
    if not formula_ok:
        buf.log("  WARNING: computed ratio does not match the Mac rebuild reference within 0.05. "
                 "This script already imports residual()/subspace_svd() directly from "
                 "paper_final_verification.py (no local reimplementation exists to fix), so a "
                 "mismatch here would point to a difference in the *input* PCA points, cell "
                 "selection, or k_eig -- not a wrong-formula bug in this script.")

    res_mal = carc.obs["residual"].values[mal_mask]
    ratio = np.median(res_mal) / np.median(res_norm)

    ascl2_mal = carc.obs["ASCL2_expr"].values[mal_mask]
    sox4_mal = carc.obs["SOX4_expr"].values[mal_mask]
    stats = stemness_stats(res_mal, ascl2_mal, sox4_mal)

    buf.log(f"\nWithin malignant-like cells only (n={int(mal_mask.sum())}):")
    buf.log(f"  residual median = {np.median(res_mal):.4f}  (normal median = {np.median(res_norm):.4f}, "
            f"ratio = {ratio:.2f}x)")
    buf.log(f"  Spearman(residual, ASCL2) = {stats['rho_ascl2']:.3f}  p={stats['p_ascl2']:.3e}")
    buf.log(f"  Spearman(residual, SOX4)  = {stats['rho_sox4']:.3f}  p={stats['p_sox4']:.3e}")
    buf.log(f"  %ASCL2+ top decile = {stats['ascl2_top_pct']:.1f}%   bottom decile = {stats['ascl2_bot_pct']:.1f}%")

    # STEP 4 inputs: n_genes_by_counts (from the original filter_cells column) and pct_counts_mt
    # (recomputed from the raw 10x matrix, since only log-normalized data is stored in the h5ad).
    mal_names = carc.obs_names[mal_mask]
    n_genes = full_obs.loc[mal_names, "n_genes"].values
    pct_mt, n_mt_genes = load_raw_mt_pct(samples["carcinoma_gsm"], samples["carcinoma_label"])
    pct_mt_mal = pct_mt.reindex(mal_names).values
    n_missing_mt = int(np.isnan(pct_mt_mal).sum())

    rho_mt, p_mt = spearmanr(res_mal, pct_mt_mal, nan_policy="omit")
    rho_ng, p_ng = spearmanr(res_mal, n_genes)
    buf.log(f"\nQuality confound check (malignant-like cells only, {n_mt_genes} MT genes found; "
            f"{n_missing_mt} cells missing raw-count match):")
    buf.log(f"  Spearman(residual, pct_counts_mt)     = {rho_mt:.3f}  p={p_mt:.3e}")
    buf.log(f"  Spearman(residual, n_genes_by_counts) = {rho_ng:.3f}  p={p_ng:.3e}")

    return {
        "sample": patient, "n_malignant": int(mal_mask.sum()), "ratio": ratio,
        "rho_ascl2": stats["rho_ascl2"], "p_ascl2": stats["p_ascl2"],
        "rho_sox4": stats["rho_sox4"], "p_sox4": stats["p_sox4"],
        "ascl2_top_pct": stats["ascl2_top_pct"], "ascl2_bot_pct": stats["ascl2_bot_pct"],
        "rho_mt": rho_mt, "p_mt": p_mt, "rho_ngenes": rho_ng, "p_ngenes": p_ng,
        "frac_normal_like": frac_normal_like,
    }


PRIMARY_TISSUES = ["sigmoid colon", "hepatic flexure of colon", "caecum"]
MET_TISSUES = ["liver"]


def step2_crc_cohort(buf):
    buf.log("\n" + "=" * 90)
    buf.log("STEP 2 -- CRC epithelial cohort (data/crc_epithelial.h5ad)")
    buf.log("=" * 90)
    if not CRC_PATH.exists():
        buf.log(f"\ndata/crc_epithelial.h5ad not found at {CRC_PATH} -- skipping Step 2.")
        return []

    adata = sc.read_h5ad(CRC_PATH)
    buf.log(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes (Moorman et al. 2025, Nature; "
            f"HTAN CRC epithelial)")
    if "donor_id" not in adata.obs.columns or "cell_type" not in adata.obs.columns:
        buf.log("Missing 'donor_id' or 'cell_type' in obs -- cannot run per-donor analysis. Skipping Step 2.")
        return []

    # This dataset's cell_type has no "normal" category (it has epithelial subtypes like
    # colonocyte, secretory cell, etc.); the normal/tumor label instead lives in `disease`.
    if "disease" not in adata.obs.columns:
        buf.log("Missing 'disease' column (expected 'normal' vs 'colorectal cancer') -- skipping Step 2.")
        return []
    buf.log("Note: normal cells = obs['disease']=='normal' (this dataset's cell_type has no "
            "'normal' category, only epithelial subtypes); malignant = cell_type=='malignant cell'.")

    symbol_to_id = None
    for col in ["feature_name", "gene_symbol", "gene"]:
        if col in adata.var.columns:
            cand = dict(zip(adata.var[col].astype(str), adata.var_names))
            if all(g in cand for g in MARKERS):
                symbol_to_id = cand
                break
    if symbol_to_id is None:
        buf.log("Could not resolve ASCL2/SOX4 gene symbols in adata.var -- skipping Step 2.")
        return []

    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=50, mask_var=None)
    pts_all = adata.obsm["X_pca"][:, :50]
    donor = adata.obs["donor_id"].values
    ctype = adata.obs["cell_type"].values
    disease = adata.obs["disease"].values
    tissue = adata.obs["tissue"].values if "tissue" in adata.obs.columns else None

    # QC metrics (pct_counts_mt, n_genes_by_counts) aren't precomputed in obs, but adata.raw
    # holds raw counts -- compute them from there so the quality-confound check can run here too.
    has_qc = False
    if adata.raw is not None:
        try:
            raw_counts = adata.raw.to_adata()
            raw_counts.var["mt"] = raw_counts.var["feature_name"].astype(str).str.upper().str.startswith("MT-")
            sc.pp.calculate_qc_metrics(raw_counts, qc_vars=["mt"], inplace=True,
                                        percent_top=None, log1p=False)
            adata.obs["pct_counts_mt"] = raw_counts.obs["pct_counts_mt"].values
            adata.obs["n_genes_by_counts"] = raw_counts.obs["n_genes_by_counts"].values
            has_qc = True
            buf.log(f"QC metrics computed from adata.raw counts ({int(raw_counts.var['mt'].sum())} MT genes found).")
        except Exception as e:
            buf.log(f"Could not compute QC metrics from adata.raw: {e}")
    if not has_qc:
        buf.log("Note: pct_counts_mt / n_genes_by_counts unavailable -- quality check skipped for this cohort.")

    def expr_of(gene, mask):
        gid = symbol_to_id[gene]
        col_idx = adata.var_names.get_loc(gid)
        e = adata[mask, :].X[:, col_idx]
        return np.asarray(e.todense()).flatten() if hasattr(e, "todense") else np.asarray(e).flatten()

    rows = []
    tissue_rows = []
    donors = sorted(adata.obs["donor_id"].unique())
    for d in donors:
        mask_n = (donor == d) & (disease == "normal")
        mask_m = (donor == d) & (ctype == "malignant cell")
        n_normal, n_mal = int(mask_n.sum()), int(mask_m.sum())
        if n_normal < CRC_MIN_N or n_mal < CRC_MIN_N:
            continue

        pts_norm, pts_mal = pts_all[mask_n], pts_all[mask_m]
        U, mean = subspace_svd(pts_norm, k_eig=20)
        res_norm = paper_residual(pts_norm, U, mean)
        res_mal = paper_residual(pts_mal, U, mean)
        ratio = np.median(res_mal) / np.median(res_norm)

        ascl2_mal, sox4_mal = expr_of("ASCL2", mask_m), expr_of("SOX4", mask_m)
        stats = stemness_stats(res_mal, ascl2_mal, sox4_mal)

        buf.log(f"\n{d}: n_normal={n_normal}  n_malignant={n_mal}  residual ratio={ratio:.2f}x")
        buf.log(f"  Spearman(residual, ASCL2)={stats['rho_ascl2']:.3f} p={stats['p_ascl2']:.3e}  "
                f"Spearman(residual, SOX4)={stats['rho_sox4']:.3f} p={stats['p_sox4']:.3e}")
        buf.log(f"  %ASCL2+ top decile={stats['ascl2_top_pct']:.1f}%  bottom decile={stats['ascl2_bot_pct']:.1f}%")

        row = {"sample": d, "n_malignant": n_mal, "ratio": ratio, **stats}
        if has_qc:
            pct_mt_mal = adata.obs["pct_counts_mt"].values[mask_m]
            ngenes_mal = adata.obs["n_genes_by_counts"].values[mask_m]
            rho_mt, p_mt = spearmanr(res_mal, pct_mt_mal)
            rho_ng, p_ng = spearmanr(res_mal, ngenes_mal)
            buf.log(f"  Spearman(residual, pct_counts_mt)={rho_mt:.3f} p={p_mt:.3e}  "
                    f"Spearman(residual, n_genes_by_counts)={rho_ng:.3f} p={p_ng:.3e}")
            row.update({"rho_mt": rho_mt, "p_mt": p_mt, "rho_ngenes": rho_ng, "p_ngenes": p_ng})
        rows.append(row)

        # tissue split: primary colon vs liver metastasis, each with >=50 malignant cells,
        # using this donor's normal-subspace residual (already computed above), just re-grouped.
        if tissue is not None:
            tissue_mal = tissue[mask_m]
            res_mal_by_idx = res_mal  # aligned with mask_m order
            is_primary = np.isin(tissue_mal, PRIMARY_TISSUES)
            is_met = np.isin(tissue_mal, MET_TISSUES)
            n_primary, n_met = int(is_primary.sum()), int(is_met.sum())
            if n_primary >= CRC_MIN_N and n_met >= CRC_MIN_N:
                buf.log(f"  Tissue split (n_primary={n_primary}, n_met={n_met}):")
                for tlabel, tmask in [("primary_colon", is_primary), ("liver_met", is_met)]:
                    res_t = res_mal_by_idx[tmask]
                    ascl2_t, sox4_t = ascl2_mal[tmask], sox4_mal[tmask]
                    st = stemness_stats(res_t, ascl2_t, sox4_t)
                    buf.log(f"    {tlabel}: n={int(tmask.sum())}  median_residual={np.median(res_t):.3f}  "
                            f"rho_ASCL2={st['rho_ascl2']:.3f} p={st['p_ascl2']:.3e}  "
                            f"rho_SOX4={st['rho_sox4']:.3f} p={st['p_sox4']:.3e}  "
                            f"ASCL2%_top={st['ascl2_top_pct']:.1f}  ASCL2%_bot={st['ascl2_bot_pct']:.1f}")
                    tissue_rows.append({"sample": f"{d}_{tlabel}", "n_malignant": int(tmask.sum()),
                                         "ratio": np.median(res_t) / np.median(res_norm),
                                         "median_residual": float(np.median(res_t)), **st})
            else:
                buf.log(f"  Tissue split not possible for {d}: n_primary={n_primary}, n_met={n_met} "
                        f"(need >={CRC_MIN_N} each).")

    if not rows:
        buf.log(f"\nNo donor had >={CRC_MIN_N} normal AND >={CRC_MIN_N} malignant cells.")
    buf.log(f"\nDonors passing >={CRC_MIN_N}/{CRC_MIN_N} normal/malignant filter: {len(rows)}/{len(donors)} "
            f"({', '.join(r['sample'] for r in rows)})")
    if tissue_rows:
        buf.log(f"Donors with a valid primary-vs-metastasis tissue split (>={CRC_MIN_N} each): "
                f"{len(tissue_rows)//2}")
    return rows, tissue_rows


def main():
    RESULTS.mkdir(exist_ok=True)
    buf = Buf()
    buf.log("Malignant-only residual-stemness test")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["scanpy", "infercnvpy", "leidenalg", "numpy", "anndata", "scipy"]))

    full = sc.read_h5ad(H5AD)
    obs_all = full.obs
    raw_all = full.raw.to_adata()
    raw_all.obs = full.obs.copy()

    setup_buf = Buf()
    raw_pos = setup_positions(raw_all, setup_buf)
    buf.log(setup_buf.text().rstrip())

    symbol_to_id = dict(zip(raw_all.var["gene_symbol"], raw_all.var_names))
    raw_ref = {
        "symbol_to_id": symbol_to_id, "var_names": raw_all.var_names, "X": raw_all.X,
        "row_idx": pd.Series(np.arange(raw_all.n_obs), index=raw_all.obs_names),
    }
    pca_full = pd.DataFrame(full.obsm["X_pca"], index=full.obs_names)

    patients_meta = {
        "Patient1": {**PATIENTS["Patient1"], "carcinoma_gsm": "GSM4904236", "carcinoma_label": "Patient1_carcinoma"},
        "Patient2": {**PATIENTS["Patient2"], "carcinoma_gsm": "GSM4904239", "carcinoma_label": "Patient2_carcinoma"},
    }

    summary_rows = []
    for patient, samples in patients_meta.items():
        row = step1_patient(patient, samples, raw_pos, obs_all, raw_ref, pca_full, obs_all, buf)
        if row is not None:
            summary_rows.append(row)

    crc_rows, crc_tissue_rows = step2_crc_cohort(buf)
    summary_rows.extend(crc_rows)

    buf.log("\n" + "=" * 90)
    buf.log("STEP 3 -- Summary table across all samples")
    buf.log("=" * 90)
    buf.log("\nDecision rule (stated before the numbers): the stemness claim -- that high-residual")
    buf.log("cells are enriched for ASCL2 (stemness marker) -- survives restriction to confirmed")
    buf.log("malignant cells only if Spearman(residual, ASCL2) is POSITIVE and significant (p<0.05)")
    buf.log("in MOST samples below.")

    hdr = f"{'sample':<14}{'N_malignant':<13}{'resid_ratio':<13}{'rho_ASCL2':<11}{'p':<11}" \
          f"{'rho_SOX4':<11}{'p':<11}{'ASCL2%_top':<12}{'ASCL2%_bot'}"
    buf.log("\n" + hdr)
    buf.log("-" * len(hdr))
    n_pos_sig_ascl2 = 0
    for r in summary_rows:
        buf.log(f"{r['sample']:<14}{r['n_malignant']:<13}{r['ratio']:<13.2f}{r['rho_ascl2']:<11.3f}"
                f"{r['p_ascl2']:<11.3e}{r['rho_sox4']:<11.3f}{r['p_sox4']:<11.3e}"
                f"{r['ascl2_top_pct']:<12.1f}{r['ascl2_bot_pct']:.1f}")
        if r["rho_ascl2"] > 0 and r["p_ascl2"] < 0.05:
            n_pos_sig_ascl2 += 1
    n_total = len(summary_rows)
    buf.log(f"\n{n_pos_sig_ascl2}/{n_total} samples show positive AND significant "
            f"Spearman(residual, ASCL2) within malignant cells only.")
    verdict = "SURVIVES" if n_total > 0 and n_pos_sig_ascl2 > n_total / 2 else "DOES NOT SURVIVE"
    buf.log(f"By the stated decision rule: the stemness claim {verdict} restriction to malignant-only cells.")

    if crc_tissue_rows:
        buf.log("\n--- Tissue-split summary: primary colon vs liver metastasis (Moorman et al. predict")
        buf.log("    the ASCL2/stem-like association should differ: primary tumors LGR5+ stem-like,")
        buf.log("    metastases lose intestinal identity) ---")
        hdr_t = f"{'sample':<24}{'N':<8}{'resid_med':<11}{'rho_ASCL2':<11}{'p':<11}{'rho_SOX4':<11}{'p':<11}" \
                f"{'ASCL2%_top':<12}{'ASCL2%_bot'}"
        buf.log(hdr_t)
        buf.log("-" * len(hdr_t))
        for r in crc_tissue_rows:
            buf.log(f"{r['sample']:<24}{r['n_malignant']:<8}{r.get('median_residual', float('nan')):<11.3f}"
                    f"{r['rho_ascl2']:<11.3f}{r['p_ascl2']:<11.3e}{r['rho_sox4']:<11.3f}{r['p_sox4']:<11.3e}"
                    f"{r['ascl2_top_pct']:<12.1f}{r['ascl2_bot_pct']:.1f}")

    buf.log("\n" + "=" * 90)
    buf.log("STEP 4 -- Quality confound summary (already reported per-sample above; consolidated here)")
    buf.log("=" * 90)
    hdr2 = f"{'sample':<14}{'rho_pct_mt':<13}{'p':<11}{'rho_n_genes':<13}{'p'}"
    buf.log(hdr2)
    for r in summary_rows:
        if "rho_mt" in r:
            buf.log(f"{r['sample']:<14}{r['rho_mt']:<13.3f}{r['p_mt']:<11.3e}"
                    f"{r['rho_ngenes']:<13.3f}{r['p_ngenes']:.3e}")
        else:
            buf.log(f"{r['sample']:<14}{'n/a':<13}{'':<11}{'n/a':<13}")

    out_path = RESULTS / "malignant_only_stemness.txt"
    out_path.write_text(buf.text(), encoding="utf-8")
    print(f"\nSaved -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
