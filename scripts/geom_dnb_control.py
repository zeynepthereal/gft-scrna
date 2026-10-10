"""DNB monotonic-increase HETEROGENEITY CONTROL.

Prior finding (results/geom_dnb.txt): module coordination score I = SD_in*PCC_in/PCC_out rises
monotonically with residual quantile in 5/6 donors, most consistently for Glycolysis, Hypoxia,
TNFA_SIGNALING_VIA_NFKB, MTORC1_SIGNALING and UV_RESPONSE_UP.
Suspicion: higher-residual bins may simply have a WIDER RANGE of module activity (more
heterogeneous cells), which mechanically inflates correlation estimates (PCC_in) even with no
change in true intra-module coordination -- a latent-factor-variance artefact, not a genuine
coordination increase.

Method: for each of the 5 pre-registered CONFIRMATORY modules (the other 45 Hallmark modules are
EXPLORATORY, reported separately, not part of the decision rule), match each bin's per-cell
module-score distribution to bin Q1's distribution (10 deciles of Q1's own score, equal cell
counts drawn from each decile range in every bin, 20 resampling reps) BEFORE recomputing PCC_in.
If PCC_in still rises with bin order after this matching, the coordination increase is not fully
explained by activity-range heterogeneity between bins.

Reuses (imported, not copied): prep_donor, assign_bins, module_score, resolve_module_gene_idx
from scripts/geom_dnb.py (thinning, gene filter, bins, and the I-score formula are identical to
the original DNB run); fit_subspace, residual from scripts/ricci_curvature_pilot.py (residual);
scripts/geom_common.py for cohort PCA loading.

DECISION RULE (B3'), stated before the numbers: in a confirmatory module, "coordination increase
survives the heterogeneity control" if, AFTER score-distribution matching, Spearman(PCC_in, bin
order) > 0.8 (NOT I -- matching already equalizes SD, so the residual test is on PCC_in alone),
in >=4/6 donors. No Bonferroni across the 5 confirmatory modules (pre-registered, not multiple
comparisons in the exploratory sense), but the number of modules passing is reported plainly.
If no module passes: "the monotonic increase is explained by heterogeneity in module activity,
not a genuine coordination increase." HTA8_6002 uses 3 bins (terciles), as in the original run.

Output: results/geom_dnb_control.txt
"""
import os
import sys
import warnings

import numpy as np
import scanpy as sc
from scipy.stats import spearmanr, zscore

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402
from geom_dnb import prep_donor, assign_bins, module_score, resolve_module_gene_idx  # noqa: E402
from ricci_curvature_pilot import fit_subspace, residual  # noqa: E402

SEED = 0
N_SUBSAMPLE_REPS = 20
N_DECILES = 10
N_OUT_GENES = 500
MIN_MODULE_GENES = 15

CONFIRMATORY = ["Glycolysis", "Hypoxia", "TNF-alpha Signaling via NF-kB",
                "mTORC1 Signaling", "UV Response Up"]


def module_score_detailed(expr, gene_idx, out_idx, cell_idx):
    sub = expr[np.ix_(cell_idx, gene_idx)]
    sd_in = sub.std(axis=0, ddof=1).mean()
    C = np.corrcoef(sub, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    pcc_in = np.nanmean(np.abs(C[iu]))
    out = expr[np.ix_(cell_idx, out_idx)]
    cross = np.corrcoef(np.hstack([sub, out]), rowvar=False)[: sub.shape[1], sub.shape[1]:]
    pcc_out = max(np.nanmean(np.abs(cross)), 1e-6)
    I = sd_in * pcc_in / pcc_out
    return I, sd_in, pcc_in, pcc_out


def per_cell_module_zscore(expr, gene_idx):
    """Mean per-gene z-score across the module's genes, per cell -- the 'module score' used for
    distribution matching (z-scored so genes with different scales contribute comparably)."""
    sub = expr[:, gene_idx]
    z = zscore(sub, axis=0, ddof=1, nan_policy="omit")
    z = np.nan_to_num(z, nan=0.0)
    return z.mean(axis=1)


def match_and_recompute(expr, gene_idx, out_idx, module_score_all, group_idx_list, rng,
                         n_reps=N_SUBSAMPLE_REPS, n_deciles=N_DECILES):
    """Match every bin's module-score distribution to bin index 1's (Q1's) deciles; recompute
    PCC_in (and I) on the matched sample, n_reps times; return per-bin median PCC_in, I."""
    ref_scores = module_score_all[group_idx_list[1]]  # Q1 (index 0 is 'normal')
    edges = np.quantile(ref_scores, np.linspace(0, 1, n_deciles + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_decile = np.digitize(ref_scores, edges[1:-1], right=True)
    ref_counts = np.array([np.sum(ref_decile == k) for k in range(n_deciles)])

    pcc_in_by_bin = np.empty((len(group_idx_list), n_reps))
    I_by_bin = np.empty((len(group_idx_list), n_reps))
    shortfall_report = []
    for gi, pool in enumerate(group_idx_list):
        pool_scores = module_score_all[pool]
        pool_decile = np.digitize(pool_scores, edges[1:-1], right=True)
        avail_by_decile = [pool[pool_decile == k] for k in range(n_deciles)]
        short = sum(max(0, ref_counts[k] - len(avail_by_decile[k])) for k in range(n_deciles))
        if short > 0:
            shortfall_report.append((gi, int(short)))
        for rep in range(n_reps):
            matched = []
            for k in range(n_deciles):
                avail = avail_by_decile[k]
                want = ref_counts[k]
                if want == 0:
                    continue
                replace = len(avail) < want
                matched.append(rng.choice(avail, size=want, replace=replace) if len(avail) > 0
                               else rng.choice(pool, size=want, replace=True))
            matched = np.concatenate(matched)
            I, sd, pcc_in, pcc_out = module_score_detailed(expr, gene_idx, out_idx, matched)
            pcc_in_by_bin[gi, rep] = pcc_in
            I_by_bin[gi, rep] = I
    return np.median(pcc_in_by_bin, axis=1), np.median(I_by_bin, axis=1), shortfall_report


def run_donor(dn, adata, hallmark_raw, buf):
    rng = np.random.default_rng(SEED)
    buf.log("\n" + "=" * 96)
    buf.log(dn)
    buf.log("=" * 96)

    prep = prep_donor(adata, dn, rng)
    expr, genes = prep["expr"], prep["genes"]
    is_norm, is_mal = prep["is_norm"], prep["is_mal"]
    name_to_idx = {g: i for i, g in enumerate(genes)}

    pcs, ng, donor_arr, disease_arr, ctype_arr = gc.load_cohort_pca()
    mask_n_c, mask_m_c = gc.donor_masks_cohort(donor_arr, disease_arr, ctype_arr, dn)
    U, mean = fit_subspace(pcs[mask_n_c])
    r_all = residual(pcs[mask_m_c], U, mean)
    import pandas as pd
    r_by_obsname = pd.Series(r_all, index=adata.obs_names[mask_m_c])
    mal_names_kept = prep["obs_names_kept"][is_mal]
    r_mal = r_by_obsname.reindex(mal_names_kept).values
    valid = ~np.isnan(r_mal)

    n_bins = 3 if dn == "HTA8_6002" else 5
    bins = assign_bins(r_mal[valid], n_bins)
    mal_idx_all = np.where(is_mal)[0][valid]
    bin_cell_idx = [mal_idx_all[bins == b] for b in range(n_bins)]
    norm_cell_idx = np.where(is_norm)[0]
    all_group_idx = [norm_cell_idx] + bin_cell_idx
    group_names = ["normal"] + [f"Q{i+1}" for i in range(n_bins)]
    buf.log(f"bin sizes: {dict(zip(group_names, [len(g) for g in all_group_idx]))}"
            + (" (3 bins/terciles, per instructions for this small donor)" if dn == "HTA8_6002" else ""))

    all_gene_idx = set(range(len(genes)))
    module_results = {}
    for name in CONFIRMATORY + [k for k in hallmark_raw if k not in CONFIRMATORY]:
        gset = hallmark_raw.get(name)
        if gset is None:
            continue
        gidx = resolve_module_gene_idx(gset, genes, name_to_idx)
        if len(gidx) < MIN_MODULE_GENES:
            continue
        others = np.array(list(all_gene_idx - set(gidx.tolist())))
        out_idx = rng.choice(others, size=min(N_OUT_GENES, len(others)), replace=False)

        # (i) raw, unmatched
        raw_I, raw_sd, raw_pcc_in, raw_pcc_out, raw_mod_mean = [], [], [], [], []
        mscore_all = per_cell_module_zscore(expr, gidx)
        for pool in all_group_idx:
            I, sd, pin, pout = module_score_detailed(expr, gidx, out_idx, pool)
            raw_I.append(I); raw_sd.append(sd); raw_pcc_in.append(pin); raw_pcc_out.append(pout)
            raw_mod_mean.append(mscore_all[pool].mean())

        # (ii)+(iii) matched
        pcc_in_matched, I_matched, shortfall = match_and_recompute(
            expr, gidx, out_idx, mscore_all, all_group_idx, np.random.default_rng(SEED))

        order = np.arange(len(all_group_idx))
        rho_raw_I, _ = spearmanr(raw_I, order)
        rho_matched_pcc, p_matched = spearmanr(pcc_in_matched, order)
        rho_matched_I, _ = spearmanr(I_matched, order)

        is_confirmatory = name in CONFIRMATORY
        module_results[name] = {
            "raw_I": raw_I, "raw_sd": raw_sd, "raw_pcc_in": raw_pcc_in, "raw_pcc_out": raw_pcc_out,
            "raw_mod_mean": raw_mod_mean, "pcc_in_matched": pcc_in_matched, "I_matched": I_matched,
            "rho_raw_I": rho_raw_I, "rho_matched_pcc": rho_matched_pcc, "rho_matched_I": rho_matched_I,
            "survives": rho_matched_pcc > 0.8, "confirmatory": is_confirmatory, "shortfall": shortfall,
        }

        if is_confirmatory:
            buf.log(f"\n  [CONFIRMATORY] {name}")
            buf.log(f"    raw I({','.join(group_names)})       = " + ",".join(f"{v:.3f}" for v in raw_I))
            buf.log(f"    raw SD_in                            = " + ",".join(f"{v:.3f}" for v in raw_sd))
            buf.log(f"    raw PCC_in                            = " + ",".join(f"{v:.3f}" for v in raw_pcc_in))
            buf.log(f"    raw PCC_out                           = " + ",".join(f"{v:.3f}" for v in raw_pcc_out))
            buf.log(f"    raw module mean z-score               = " + ",".join(f"{v:.3f}" for v in raw_mod_mean))
            buf.log(f"    MATCHED PCC_in (median of 20 reps)    = " + ",".join(f"{v:.3f}" for v in pcc_in_matched))
            buf.log(f"    MATCHED I (median of 20 reps)         = " + ",".join(f"{v:.3f}" for v in I_matched))
            buf.log(f"    Spearman(raw I, bin order)            = {rho_raw_I:+.3f}")
            buf.log(f"    Spearman(MATCHED PCC_in, bin order)   = {rho_matched_pcc:+.3f}  p={p_matched:.3e}  "
                    f"-> {'SURVIVES (>0.8)' if rho_matched_pcc > 0.8 else 'does not survive'}")
            buf.log(f"    Spearman(MATCHED I, bin order)        = {rho_matched_I:+.3f}  [reference only]")
            if shortfall:
                buf.log(f"    NOTE: decile-count shortfall (sampled with replacement) in bins {shortfall}")

    return module_results, group_names


def main():
    buf = gc.Buf()
    buf.log("DNB monotonic-increase HETEROGENEITY CONTROL (score-distribution matching)")
    buf.log(f"n_subsample_reps={N_SUBSAMPLE_REPS}; n_deciles={N_DECILES}; seed={SEED}\n")
    buf.log("DECISION RULE (B3'), stated before the numbers: a confirmatory module 'survives the")
    buf.log("heterogeneity control' if, AFTER matching every bin's module-score distribution to Q1's")
    buf.log("(10 deciles, equal per-decile counts, 20 resampling reps), Spearman(PCC_in, bin order)")
    buf.log("> 0.8 (PCC_in, not I -- matching already equalizes SD_in) in >=4/6 donors. No Bonferroni")
    buf.log("across the 5 pre-registered confirmatory modules; the pass count is reported plainly.")
    buf.log("The other 45 Hallmark modules are EXPLORATORY and reported separately, not part of this")
    buf.log("decision rule. If no confirmatory module passes: 'the monotonic increase is explained by")
    buf.log("heterogeneity in module activity, not a genuine coordination increase.'")
    buf.log("HTA8_6002 uses 3 bins (terciles), as in the original DNB run.\n")

    try:
        import gseapy
        hallmark_raw = gseapy.get_library(name="MSigDB_Hallmark_2020", organism="Human")
        buf.log(f"Loaded {len(hallmark_raw)} MSigDB Hallmark gene sets from Enrichr.\n")
    except Exception as e:
        buf.log(f"Could not download MSigDB Hallmark gene sets: {e}. Cannot proceed.")
        buf.save(os.path.join(gc.RESULTS_DIR, "geom_dnb_control.txt"))
        return

    adata = gc.load_adata()
    all_results = {}
    for dn in gc.DONORS:
        module_results, group_names = run_donor(dn, adata, hallmark_raw, buf)
        all_results[dn] = module_results

    buf.log("\n" + "=" * 96)
    buf.log("CROSS-DONOR SUMMARY -- CONFIRMATORY MODULES")
    buf.log("=" * 96)
    hdr = f"{'module':<32}{'n_donors_survive':>17}  donors"
    buf.log(hdr)
    survive_counts = {}
    for name in CONFIRMATORY:
        donors_pass = [dn for dn in gc.DONORS if name in all_results[dn] and all_results[dn][name]["survives"]]
        survive_counts[name] = donors_pass
        buf.log(f"{name:<32}{len(donors_pass):>17}  {donors_pass}")

    n_modules_passing = sum(1 for v in survive_counts.values() if len(v) >= 4)
    buf.log(f"\n{n_modules_passing}/5 confirmatory modules pass B3' (>=4/6 donors with "
            f"Spearman(matched PCC_in, bin order)>0.8).")

    buf.log("\n" + "=" * 96)
    buf.log("PER-DONOR DETAIL -- raw vs matched Spearman, confirmatory modules")
    buf.log("=" * 96)
    hdr2 = f"{'donor':<11}{'module':<32}{'rho_raw_I':>11}{'rho_matched_PCC':>17}{'rho_matched_I':>15}{'survives?'}"
    buf.log(hdr2)
    for dn in gc.DONORS:
        for name in CONFIRMATORY:
            if name not in all_results[dn]:
                continue
            r = all_results[dn][name]
            buf.log(f"{dn:<11}{name:<32}{r['rho_raw_I']:>+11.3f}{r['rho_matched_pcc']:>+17.3f}"
                    f"{r['rho_matched_I']:>+15.3f}{('YES' if r['survives'] else 'no'):>10}")

    buf.log("\n" + "=" * 96)
    buf.log("EXPLORATORY -- remaining Hallmark modules (not part of the decision rule)")
    buf.log("=" * 96)
    all_names = sorted(set().union(*[set(r.keys()) for r in all_results.values()]) - set(CONFIRMATORY))
    for name in all_names:
        donors_pass = [dn for dn in gc.DONORS if name in all_results[dn] and all_results[dn][name]["survives"]]
        if donors_pass:
            buf.log(f"{name:<35} survives (matched PCC_in rho>0.8) in {len(donors_pass)}/6: {donors_pass}")

    buf.log("\nVERDICT (B3'):")
    if n_modules_passing > 0:
        passing_names = [k for k, v in survive_counts.items() if len(v) >= 4]
        buf.log(f"  Coordination increase SURVIVES the heterogeneity control for: {passing_names}")
    else:
        buf.log("  No confirmatory module survives in >=4/6 donors after matching.")
        buf.log("  The monotonic increase is explained by heterogeneity in module activity between")
        buf.log("  bins, not a genuine increase in intra-module coordination.")

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- Distribution matching equalizes SD_in (and the module-score range) between bins by")
    buf.log("  construction, so it is expected to shrink both I and PCC_in toward the reference (Q1)")
    buf.log("  bin's values; a genuine reduction in the raw-to-matched gap does not by itself prove")
    buf.log("  the raw finding was purely an artefact, only that heterogeneity explains part of it.")
    buf.log("- When a bin lacks enough cells in one of Q1's score deciles, cells are resampled WITH")
    buf.log("  replacement to keep counts equal across reps; donors/modules where this occurred are")
    buf.log("  flagged inline.")
    buf.log("- Residual, module scores and gene expression are derived from the same expression")
    buf.log("  matrix; this is an internal-consistency test, not independent validation.")
    buf.log("- Residual quantiles in this snapshot dataset are not a time axis.")
    buf.log("- HTA8_6002 has only 60 malignant cells and uses 3 bins; it has by far the lowest power.")

    buf.save(os.path.join(gc.RESULTS_DIR, "geom_dnb_control.txt"))


if __name__ == "__main__":
    main()
