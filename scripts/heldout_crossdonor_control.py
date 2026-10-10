"""REVIEWER CRITIQUE: normal-cell residuals were computed IN-SAMPLE (the 20-D subspace was fit on
the same normal cells being scored), while malignant cells are always out-of-sample. Higher
malignant residuals may be partly -- or entirely -- expected by construction rather than by
disease. This script tests that directly with two controls:

TEST 1 -- held-out normal control (per sample): 5-fold CV on normal cells (fixed seed=42). Per
  fold, fit the 20-D subspace on the training normal cells only; score the held-out normal fold
  AND all malignant cells with that fit. Each normal cell gets exactly one held-out residual (from
  the fold where it was excluded); each malignant cell gets 5 residuals (one per fold's fit,
  since malignant cells are never in the training set to begin with) -- these 5 are averaged per
  malignant cell to give one number per cell, on the same out-of-sample footing as the normal
  cells. Reports in-sample vs held-out normal medians (the gap IS the in-sample advantage), the
  malignant/held-out-normal ratio, and reruns the n_genes quintile stratification test
  (scripts/confound_and_detector.py's test_a_sample, imported unmodified) using held-out normal
  residuals.

TEST 2 -- cross-donor normal control (within each dataset): for every ordered donor pair (A, B)
  in the same dataset, fit the subspace on donor A's normal cells and score donor B's (a
  different, healthy person's) normal cells with it. This isolates how much residual is just
  "being a different individual" vs "being malignant." Ratio = median(B-scored-on-A) /
  median(A's own held-out normal, from TEST 1) -- the natural person-to-person baseline the
  malignant/held-out-normal ratio should be compared against.

Residual function: EXACT centered formula from scripts/paper_final_verification.py (subspace_svd,
residual), imported, not reimplemented.

Malignant cells:
  Patient1, Patient2 (GSE161277): CNV malignant-like cells, same pipeline as
    scripts/cnv_followup_checks.py's Check A (run_patient from scripts/cnv_clone_vs_residual.py,
    label_malignant_clones from scripts/malignant_only_stemness.py -- the reusable function
    implementing that exact rule).
  CRC cohort (data/crc_epithelial.h5ad): cell_type=='malignant cell', normal=disease=='normal'.
Only samples with >=100 normal cells are included (stated requirement; no separate malignant-count
threshold was specified, so all qualifying samples' actual malignant N is reported as-is).

DECISION RULE, stated before the numbers: the disease-deviation claim survives only if, in MOST
samples, malignant/held-out-normal ratios are CLEARLY above 1 (>1.2x) AND clearly exceed the
typical (median) cross-donor normal ratio for that sample's dataset. If a malignant cell deviates
no more than a healthy cell from a different person, the claim fails for that sample.

Output: results/heldout_crossdonor_control.txt
"""
import os
import sys
import warnings
from importlib.metadata import version

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402 -- for Buf, RESULTS_DIR only (no space-duplication logic used)
from paper_final_verification import subspace_svd, residual as paper_residual  # noqa: E402
from confound_and_detector import test_a_sample  # noqa: E402
from cnv_clone_vs_residual import (  # noqa: E402
    REPO, PATIENTS as GSE_PATIENTS, setup_positions, build_patient_adata, run_patient,
)
from malignant_only_stemness import label_malignant_clones  # noqa: E402

warnings.filterwarnings("ignore", message=".*slogdet.*")

SEED = 42  # fixed, as specified
K_EIG = 20
N_FOLDS = 5
MIN_NORMAL_N = 100
CRC_PATH = REPO / "data" / "crc_epithelial.h5ad"
GSE_H5AD = REPO / "data" / "GSE161277_all.h5ad"


def held_out_normal_and_malignant(pts_n, pts_m, seed=SEED, n_folds=N_FOLDS, k_eig=K_EIG):
    """5-fold CV: each normal cell scored once (held-out); each malignant cell scored n_folds
    times (once per fold's fit) and averaged per cell. Also returns the plain in-sample normal
    residual (subspace fit on ALL normal cells, scored on the same cells) for comparison."""
    rng = np.random.default_rng(seed)
    n = len(pts_n)
    folds = rng.permutation(n) % n_folds

    r_n_heldout = np.empty(n)
    r_m_per_fold = np.empty((n_folds, len(pts_m)))
    for f in range(n_folds):
        U, mean = subspace_svd(pts_n[folds != f], k_eig=k_eig)
        r_n_heldout[folds == f] = paper_residual(pts_n[folds == f], U, mean)
        r_m_per_fold[f] = paper_residual(pts_m, U, mean)
    r_m_avg = r_m_per_fold.mean(axis=0)

    U_all, mean_all = subspace_svd(pts_n, k_eig=k_eig)
    r_n_insample = paper_residual(pts_n, U_all, mean_all)

    return {"r_n_insample": r_n_insample, "r_n_heldout": r_n_heldout, "r_m": r_m_avg,
            "U_all": U_all, "mean_all": mean_all}


def run_sample(sample_name, pts_n, pts_m, ng_n, ng_m, buf):
    buf.log("\n" + "=" * 96)
    buf.log(sample_name)
    buf.log("=" * 96)
    buf.log(f"n_normal={len(pts_n)}  n_malignant={len(pts_m)}")

    res = held_out_normal_and_malignant(pts_n, pts_m)
    med_insample = np.median(res["r_n_insample"])
    med_heldout = np.median(res["r_n_heldout"])
    med_mal = np.median(res["r_m"])
    ratio_mal_heldout = med_mal / med_heldout
    insample_gap = med_heldout / med_insample

    buf.log(f"  median normal residual: in-sample={med_insample:.4f}  held-out={med_heldout:.4f}  "
            f"(held-out/in-sample gap = {insample_gap:.3f}x -- the in-sample advantage itself)")
    buf.log(f"  median malignant residual (avg over {N_FOLDS} fold-fits per cell) = {med_mal:.4f}")
    buf.log(f"  RATIO malignant / held-out-normal = {ratio_mal_heldout:.3f}x")
    buf.log(f"  [for reference] RATIO malignant / in-sample-normal (the original, biased "
            f"comparison) = {med_mal/med_insample:.3f}x")

    buf.log(f"\n  Test A (n_genes quintile stratification + OLS) using HELD-OUT normal residuals:")
    testA = test_a_sample(f"{sample_name}_heldout", res["r_n_heldout"], ng_n, res["r_m"], ng_m, buf)

    return {"res": res, "med_insample": med_insample, "med_heldout": med_heldout, "med_mal": med_mal,
            "ratio_mal_heldout": ratio_mal_heldout, "insample_gap": insample_gap, "testA": testA,
            "pts_n": pts_n}


def cross_donor_test(dataset_name, sample_results, buf):
    buf.log("\n" + "-" * 96)
    buf.log(f"TEST 2 -- cross-donor normal control, within {dataset_name}")
    buf.log("-" * 96)
    names = list(sample_results.keys())
    if len(names) < 2:
        buf.log("  Only one sample in this dataset -- cross-donor test not possible.")
        return {}

    ratios = {}
    buf.log(f"  {'A (train)':<14}{'B (test)':<14}{'median(B on A)':<16}{'A held-out-norm':<18}{'ratio'}")
    all_ratios = []
    for a in names:
        for b in names:
            if a == b:
                continue
            U, mean = subspace_svd(sample_results[a]["pts_n"], k_eig=K_EIG)
            r_b_on_a = paper_residual(sample_results[b]["pts_n"], U, mean)
            med_b_on_a = np.median(r_b_on_a)
            ratio = med_b_on_a / sample_results[a]["med_heldout"]
            ratios[(a, b)] = ratio
            all_ratios.append(ratio)
            buf.log(f"  {a:<14}{b:<14}{med_b_on_a:<16.4f}{sample_results[a]['med_heldout']:<18.4f}"
                    f"{ratio:.3f}x")

    all_ratios = np.array(all_ratios)
    buf.log(f"\n  Cross-donor normal ratio distribution ({dataset_name}, n={len(all_ratios)} "
            f"ordered pairs): min={all_ratios.min():.3f}  median={np.median(all_ratios):.3f}  "
            f"max={all_ratios.max():.3f}")
    if len(names) == 2:
        buf.log(f"  NOTE: only {len(names)} donors in this dataset -> only 2 ordered pairs; this")
        buf.log(f"  distribution is not statistically meaningful, reported for completeness only.")

    buf.log(f"\n  Comparison: malignant/held-out-normal ratio vs typical (median) cross-donor ratio:")
    for s in names:
        mal_ratio = sample_results[s]["ratio_mal_heldout"]
        exceeds_median = mal_ratio > np.median(all_ratios)
        exceeds_max = mal_ratio > all_ratios.max()
        buf.log(f"    {s}: malignant/held-out-normal={mal_ratio:.3f}x  vs  cross-donor median="
                f"{np.median(all_ratios):.3f}x (exceeds? {'YES' if exceeds_median else 'NO'})  "
                f"vs cross-donor max={all_ratios.max():.3f}x (exceeds? {'YES' if exceeds_max else 'NO'})")

    return {"ratios": ratios, "all_ratios": all_ratios}


def gse161277_samples(buf):
    """Patient1, Patient2 malignant-like cells via the CNV pipeline (same as
    scripts/cnv_followup_checks.py's Check A)."""
    full = sc.read_h5ad(GSE_H5AD)
    obs_all = full.obs
    raw_all = full.raw.to_adata()
    raw_all.obs = full.obs.copy()
    setup_buf = gc.Buf()
    raw_pos = setup_positions(raw_all, setup_buf)
    buf.log(setup_buf.text().rstrip())

    symbol_to_id = dict(zip(raw_all.var["gene_symbol"], raw_all.var_names))
    row_idx = pd.Series(np.arange(raw_all.n_obs), index=raw_all.obs_names)
    pca_full = pd.DataFrame(full.obsm["X_pca"], index=full.obs_names)

    samples = {}
    for patient, samp in GSE_PATIENTS.items():
        adata_p = build_patient_adata(raw_pos, obs_all, patient, samp)
        adata_p.obsm["X_pca_full"] = pca_full.loc[adata_p.obs_names].values
        pipe_buf = gc.Buf()
        print(f"\n=== Running CNV pipeline for {patient} (GSE161277) ===")
        pipeline = run_patient(patient, adata_p, {"symbol_to_id": symbol_to_id,
                                                   "var_names": raw_all.var_names, "X": raw_all.X,
                                                   "row_idx": row_idx}, pipe_buf)
        if pipeline is None:
            buf.log(f"{patient}: no CNV structure detected -- skipped.")
            continue
        carc, adata_full = pipeline["carc"], pipeline["adata"]
        m_norm, m_carc = pipeline["m_norm"], pipeline["m_carc"]
        labels = label_malignant_clones(carc, adata_full, pipeline["gv"], pipeline["var"],
                                         m_norm, m_carc, gc.Buf())
        malignant_clones = [cl for cl, lab in labels.items() if lab == "malignant-like"]
        mal_mask = carc.obs["cnv_clone"].isin(malignant_clones).values

        pts_all = adata_full.obsm["X_pca_full"][:, :50]
        pts_n = pts_all[m_norm]
        pts_m = pts_all[m_carc][mal_mask]
        if pts_n.shape[0] < MIN_NORMAL_N:
            buf.log(f"{patient}: n_normal={pts_n.shape[0]} < {MIN_NORMAL_N} -- excluded.")
            continue

        ng_full = adata_full.obs["n_genes"].values if "n_genes" in adata_full.obs.columns else None
        if ng_full is None:
            raw_counts = adata_full.raw.to_adata() if adata_full.raw is not None else None
            ng_full = np.asarray((adata_full.X > 0).sum(axis=1)).ravel()
        ng_n = ng_full[m_norm]
        ng_m = ng_full[m_carc][mal_mask]

        samples[patient] = (pts_n, pts_m, ng_n, ng_m)
    return samples


def crc_samples(buf):
    if not CRC_PATH.exists():
        buf.log(f"data/crc_epithelial.h5ad not found at {CRC_PATH} -- skipping CRC cohort.")
        return {}
    adata = sc.read_h5ad(CRC_PATH)
    if "X_pca" not in adata.obsm:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.pca(adata, n_comps=50, mask_var=None)

    if adata.raw is not None:
        raw_counts = adata.raw.to_adata()
        raw_counts.var["mt"] = raw_counts.var["feature_name"].astype(str).str.upper().str.startswith("MT-")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.calculate_qc_metrics(raw_counts, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
        adata.obs["n_genes_by_counts"] = raw_counts.obs["n_genes_by_counts"].values
    else:
        adata.obs["n_genes_by_counts"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

    pts_all = adata.obsm["X_pca"][:, :50]
    donor = adata.obs["donor_id"].values
    disease = adata.obs["disease"].values
    ctype = adata.obs["cell_type"].values
    ng_all = adata.obs["n_genes_by_counts"].values

    samples = {}
    for dn in sorted(pd.unique(donor)):
        mask_n = (donor == dn) & (disease == "normal")
        mask_m = (donor == dn) & (ctype == "malignant cell")
        n_normal = int(mask_n.sum())
        if n_normal < MIN_NORMAL_N:
            continue
        samples[dn] = (pts_all[mask_n], pts_all[mask_m], ng_all[mask_n], ng_all[mask_m])
    buf.log(f"CRC cohort donors with >={MIN_NORMAL_N} normal cells: {list(samples.keys())}")
    return samples


def main():
    buf = gc.Buf()
    buf.log("Held-out and cross-donor normal controls for the malignant-vs-normal residual claim")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["numpy", "scipy", "scikit-learn", "scanpy", "infercnvpy"]))
    buf.log(f"seed={SEED}  n_folds={N_FOLDS}  k_eig={K_EIG}  min_normal_n={MIN_NORMAL_N}\n")
    buf.log("DECISION RULE, stated before the numbers: the disease-deviation claim survives only if,")
    buf.log("in MOST samples, malignant/held-out-normal ratios are CLEARLY above 1 (>1.2x) AND")
    buf.log("clearly exceed the typical (median) cross-donor normal ratio for that dataset. If a")
    buf.log("malignant cell deviates no more than a healthy cell from a different person, the claim")
    buf.log("fails for that sample.\n")

    buf.log("=" * 96)
    buf.log("TEST 1 -- held-out normal control, per sample")
    buf.log("=" * 96)

    all_sample_results = {}
    dataset_membership = {}

    buf.log("\n--- GSE161277 (Patient1, Patient2) ---")
    gse_samples = gse161277_samples(buf)
    for name, (pts_n, pts_m, ng_n, ng_m) in gse_samples.items():
        all_sample_results[name] = run_sample(name, pts_n, pts_m, ng_n, ng_m, buf)
        dataset_membership[name] = "GSE161277"

    buf.log("\n--- CRC epithelial cohort ---")
    crc = crc_samples(buf)
    for name, (pts_n, pts_m, ng_n, ng_m) in crc.items():
        if len(pts_m) == 0:
            buf.log(f"\n{name}: 0 malignant cells -- skipped.")
            continue
        all_sample_results[name] = run_sample(name, pts_n, pts_m, ng_n, ng_m, buf)
        dataset_membership[name] = "CRC_epithelial"

    buf.log("\n" + "=" * 96)
    buf.log("TEST 2 -- cross-donor normal control")
    buf.log("=" * 96)
    cross_donor_results = {}
    for dataset in sorted(set(dataset_membership.values())):
        members = {n: all_sample_results[n] for n, d in dataset_membership.items() if d == dataset}
        cross_donor_results[dataset] = cross_donor_test(dataset, members, buf)

    buf.log("\n" + "=" * 96)
    buf.log("SUMMARY")
    buf.log("=" * 96)
    hdr = (f"{'sample':<12}{'dataset':<16}{'med_insample':<13}{'med_heldout':<13}{'med_mal':<10}"
           f"{'mal/heldout':<13}{'insample_gap':<13}{'testA_survives':<15}")
    buf.log(hdr)
    n_clear_ratio = 0
    n_exceeds_crossdonor = 0
    n_total = len(all_sample_results)
    for name, r in all_sample_results.items():
        dataset = dataset_membership[name]
        cdr = cross_donor_results.get(dataset, {}).get("all_ratios")
        clear_ratio = r["ratio_mal_heldout"] > 1.2
        exceeds_cd = (cdr is not None and len(cdr) > 0 and r["ratio_mal_heldout"] > np.median(cdr))
        if clear_ratio:
            n_clear_ratio += 1
        if exceeds_cd:
            n_exceeds_crossdonor += 1
        buf.log(f"{name:<12}{dataset:<16}{r['med_insample']:<13.4f}{r['med_heldout']:<13.4f}"
                f"{r['med_mal']:<10.4f}{r['ratio_mal_heldout']:<13.3f}{r['insample_gap']:<13.3f}"
                f"{('YES' if r['testA']['survives'] else 'NO'):<15}")

    buf.log(f"\n{n_clear_ratio}/{n_total} samples: malignant/held-out-normal ratio > 1.2x")
    buf.log(f"{n_exceeds_crossdonor}/{n_total} samples: ratio also exceeds the typical (median) "
            f"cross-donor normal ratio for that dataset")

    both = 0
    for name, r in all_sample_results.items():
        cdr = cross_donor_results.get(dataset_membership[name], {}).get("all_ratios")
        clears_crossdonor = cdr is not None and len(cdr) > 0 and r["ratio_mal_heldout"] > np.median(cdr)
        if r["ratio_mal_heldout"] > 1.2 and clears_crossdonor:
            both += 1
    survives = both > n_total / 2
    buf.log(f"\n{both}/{n_total} samples satisfy BOTH conditions.")
    buf.log(f"VERDICT: the disease-deviation claim {'SURVIVES' if survives else 'DOES NOT SURVIVE'} "
            f"this control (need a majority of samples satisfying both conditions).")

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- GSE161277 has only 2 donors -- its cross-donor distribution (2 ordered pairs) is not")
    buf.log("  statistically meaningful and is reported for completeness only, not as a robust")
    buf.log("  baseline the way the 7-donor CRC cohort distribution is.")
    buf.log("- Malignant cells' held-out residual is an AVERAGE over 5 different fold-fits per cell")
    buf.log("  (since malignant cells are never in the training set regardless of fold); normal")
    buf.log("  cells get exactly one held-out value each, from the single fold where they were")
    buf.log("  excluded. This asymmetry is a reasonable way to put both groups on equal out-of-")
    buf.log("  sample footing, but is not identical machinery for both groups.")
    buf.log("- CNV clone labels (GSE161277 malignant-like cells) and the residual are derived from")
    buf.log("  the same expression matrix -- internal consistency, not independent ground truth.")
    buf.log("- Cross-donor comparisons mix different individuals, sequencing runs and possibly batch")
    buf.log("  effects; a large cross-donor ratio does not distinguish biological inter-individual")
    buf.log("  variation from technical batch variation.")

    buf.save(os.path.join(gc.RESULTS_DIR, "heldout_crossdonor_control.txt"))


if __name__ == "__main__":
    main()
