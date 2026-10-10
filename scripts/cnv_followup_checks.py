"""Follow-up checks on scripts/cnv_clone_vs_residual.py's clone-vs-state analysis.

CHECK A (Patient2): are the non-CNV-typical clones just normal epithelial cells inside the
  carcinoma sample? Label each clone malignant-like / normal-like vs the normal epithelial
  reference, report normal-colon-marker expression per clone, and re-run the clone-vs-state
  tests restricted to the malignant-like clone(s).
CHECK B (Patient1): are the 7 clones real or over-split? Hierarchically cluster clones on
  their arm-level profiles (correlation distance, average linkage), merge at correlation>0.9,
  and re-run the tests on the merged clones.
CHECK C (both): cell-cycle confound -- score S/G2M, correlate with residual, check whether
  top-decile residual cells are cycling-phase enriched.

Reuses the exact pipeline (CNV inference, clustering, residual) from cnv_clone_vs_residual.py
by calling its run_patient() so clone assignments are identical to the original report.

Output: results/cnv_followup_checks.txt
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu, spearmanr, f_oneway, chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnv_clone_vs_residual import (  # noqa: E402 - reuse exact pipeline
    REPO, RESULTS, H5AD, PATIENTS, ARM_EVENTS, RANDOM_STATE,
    Buf, setup_positions, build_patient_adata, run_patient,
    arm_level_percell, arm_level_means, get_marker_expr,
)
from paper_final_verification import residual as paper_residual  # noqa: E402
from importlib.metadata import version

NORMAL_COLON_MARKERS = ["CA1", "CA2", "SLC26A3", "AQP8", "GUCA2A"]

# Standard Tirosh/Regev cell-cycle gene lists (scanpy tutorial set).
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
           "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP",
           "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2",
           "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2",
           "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M_GENES = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2",
             "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2",
             "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1",
             "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
             "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1",
             "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]


def check_a_patient2(patient, pipeline, raw_ref, buf):
    """Are Patient2's non-CNV-typical clones actually normal epithelial cells?"""
    buf.log("\n" + "#" * 90)
    buf.log("CHECK A (Patient2): malignant-like vs normal-like CNV clones")
    buf.log("#" * 90)
    buf.log("\nWhat would change the earlier verdict: the earlier Patient2 verdict was 'neither")
    buf.log("rule cleanly satisfied' (max clone concentration 73.1%, eta^2=0.252 -- just under the")
    buf.log("clone rule's eta^2>0.5 bar). If some of the 4 clones are actually normal epithelial")
    buf.log("cells misassigned into the carcinoma sample (e.g. adjacent normal mucosa captured in")
    buf.log("the 10x carcinoma sample), diluting them out should raise eta^2 and clone concentration")
    buf.log("among the remaining, genuinely malignant clone(s) -- that would push the verdict toward")
    buf.log("CLONE. If instead a malignant-like clone still shows large internal residual spread and")
    buf.log("ASCL2/residual correlation *within* it, that continues to support a within-clone STATE")
    buf.log("component that clone identity alone doesn't explain.")

    adata, carc, gv, var = pipeline["adata"], pipeline["carc"], pipeline["gv"], pipeline["var"]
    m_norm = pipeline["m_norm"]
    gv_norm = gv[m_norm]
    cnv_score_norm = adata.obs.cnv_score.values[m_norm]
    percell_norm = arm_level_percell(gv, var, m_norm)

    clones = list(carc.obs["cnv_clone"].cat.categories)
    labels = {}
    buf.log(f"\n{'clone':<8}{'n':<8}{'cnv_score_med':<16}{'vs_normal_p':<14}{'n_arm_events':<14}{'label'}")
    # gv rows correspond to `adata` order, not `carc` order -- map clone cells back to adata indices.
    carc_idx_in_adata = np.where(pipeline["m_carc"])[0]
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
            vc_arr = percell_cl[(chrom, arm)]
            vn_arr = percell_norm[(chrom, arm)]
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

    buf.log(f"\nLabel rule: malignant-like = CNV score > 1.1x normal median (p<0.01) AND "
            f">=4/{len(ARM_EVENTS)} CRC arm events significant vs normal (same test as main script).")

    buf.log("\nNormal colon epithelial marker means per clone (log-normalized, adata.raw):")
    buf.log(f"{'clone':<8}{'label':<16}" + "".join(f"{m:<10}" for m in NORMAL_COLON_MARKERS))
    for cl in clones:
        cl_mask_in_carc = (carc.obs["cnv_clone"] == cl).values
        names = carc.obs_names[cl_mask_in_carc]
        vals = [np.nanmean(get_marker_expr(raw_ref, names, m)) for m in NORMAL_COLON_MARKERS]
        buf.log(f"{cl:<8}{labels[cl]:<16}" + "".join(f"{v:<10.3f}" for v in vals))
    # normal epithelial reference row for comparison
    normal_names = adata.obs_names[m_norm]
    vals_norm = [np.nanmean(get_marker_expr(raw_ref, normal_names, m)) for m in NORMAL_COLON_MARKERS]
    buf.log(f"{'(normal ref)':<8}{'--':<16}" + "".join(f"{v:<10.3f}" for v in vals_norm))

    malignant_clones = [cl for cl, lab in labels.items() if lab == "malignant-like"]
    buf.log(f"\nMalignant-like clone(s): {malignant_clones} "
            f"({sum((carc.obs.cnv_clone.isin(malignant_clones)))}/{len(carc)} cells, "
            f"{100 * sum(carc.obs.cnv_clone.isin(malignant_clones)) / len(carc):.1f}%)")

    if len(malignant_clones) == 0:
        buf.log("\nNo malignant-like clone found -- cannot re-run clone-vs-state tests.")
        return
    mal_mask = carc.obs["cnv_clone"].isin(malignant_clones).values
    res_mal = carc.obs["residual"].values[mal_mask]
    ascl2_mal = carc.obs["ASCL2_expr"].values[mal_mask]

    if len(malignant_clones) == 1:
        buf.log(f"\nOnly one malignant-like clone ({malignant_clones[0]}, n={mal_mask.sum()}) -- "
                f"reporting WITHIN-clone residual distribution and ASCL2-vs-residual correlation")
        buf.log(f"  instead of a between-clone eta^2/crosstab (which needs >=2 groups).")
        buf.log(f"  residual: median={np.median(res_mal):.3f}  IQR=[{np.percentile(res_mal,25):.3f}, "
                f"{np.percentile(res_mal,75):.3f}]  min={res_mal.min():.3f}  max={res_mal.max():.3f}  "
                f"std={res_mal.std():.3f}")
        rho, p_rho = spearmanr(res_mal, ascl2_mal)
        buf.log(f"  Spearman(residual, ASCL2 expression) within malignant clone: rho={rho:.3f}  p={p_rho:.3e}")
        q10, q90 = np.quantile(res_mal, [0.10, 0.90])
        top_ascl2 = 100 * (ascl2_mal[res_mal >= q90] > 0).mean()
        bot_ascl2 = 100 * (ascl2_mal[res_mal <= q10] > 0).mean()
        buf.log(f"  %ASCL2+ in within-clone top-decile residual cells: {top_ascl2:.1f}%  "
                f"vs bottom-decile: {bot_ascl2:.1f}%")
        buf.log(f"\n  Interpretation (facts only): a large within-clone residual spread "
                f"(std={res_mal.std():.3f}, range {res_mal.max()-res_mal.min():.3f}) and "
                f"{'a positive' if rho > 0.1 and p_rho < 0.05 else 'no clear'} residual-ASCL2 "
                f"association within this single genetic clone means residual/ASCL2 status is "
                f"{'not fully explained by' if rho < 0.5 else 'substantially explained by'} "
                f"clone identity alone.")
    else:
        buf.log(f"\n{len(malignant_clones)} malignant-like clones -- re-running eta^2, "
                f"top-decile distribution, ASCL2% restricted to these clones.")
        groups = [res_mal[carc.obs["cnv_clone"].values[mal_mask] == cl] for cl in malignant_clones]
        if all(len(g) > 1 for g in groups):
            f_stat, p_anova = f_oneway(*groups)
            grand_mean = res_mal.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = ((res_mal - grand_mean) ** 2).sum()
            eta_sq = ss_between / ss_total
            buf.log(f"  One-way ANOVA (malignant clones only): F={f_stat:.3f}  p={p_anova:.3e}  eta^2={eta_sq:.4f}")
        q10, q90 = np.quantile(res_mal, [0.10, 0.90])
        rg = np.where(res_mal >= q90, "top10", np.where(res_mal <= q10, "bottom10", "middle80"))
        sub_clone = carc.obs["cnv_clone"].values[mal_mask]
        ct = pd.crosstab(rg, sub_clone)
        buf.log("  Crosstab residual_group x malignant clone:\n" + ct.to_string())
        n_top = (rg == "top10").sum()
        buf.log(f"  {'clone':<8}{'size':<10}{'expected_top10':<16}{'observed_top10':<16}{'obs/exp'}")
        for cl in malignant_clones:
            m = sub_clone == cl
            size = m.sum()
            expected = n_top * size / mal_mask.sum()
            observed = int(((sub_clone == cl) & (rg == "top10")).sum())
            buf.log(f"  {cl:<8}{size:<10}{expected:<16.2f}{observed:<16}{(observed/expected if expected>0 else float('nan')):.2f}")
        buf.log(f"\n  Per-clone ASCL2%:")
        for cl in malignant_clones:
            m = sub_clone == cl
            buf.log(f"    clone {cl}: n={m.sum()}  %ASCL2+={100*(ascl2_mal[m]>0).mean():.1f}  "
                    f"med_residual={np.median(res_mal[m]):.3f}")


def check_b_patient1(patient, pipeline, buf):
    """Are Patient1's 7 clones real or over-split?"""
    buf.log("\n" + "#" * 90)
    buf.log("CHECK B (Patient1): merging over-split CNV clones (correlation distance, avg linkage)")
    buf.log("#" * 90)
    buf.log("\nWhat would change the earlier verdict: the earlier Patient1 verdict was STATE")
    buf.log("(max clone concentration 44.6%, eta^2=0.071, 7 clones). If merging near-identical")
    buf.log("arm profiles collapses this into e.g. 2-3 clones with eta^2 rising substantially")
    buf.log("(>0.3-0.5) and one merged clone concentrating >70% of top-decile cells, that would")
    buf.log("shift the verdict toward CLONE -- i.e. the original 'state' result would have been an")
    buf.log("over-clustering artifact. If eta^2 and concentration stay low after merging, that")
    buf.log("confirms the earlier STATE verdict is not a clustering artifact.")

    carc, gv, var = pipeline["carc"], pipeline["gv"], pipeline["var"]
    m_carc = pipeline["m_carc"]
    gv_carc = gv[m_carc]
    clones = list(carc.obs["cnv_clone"].cat.categories)
    event_keys = list(ARM_EVENTS.keys())

    arm_matrix = np.array([[arm_level_means(gv_carc, var, (carc.obs["cnv_clone"] == cl).values)[k]
                             for k in event_keys] for cl in clones])
    buf.log(f"\nArm-level profile matrix ({len(clones)} clones x {len(event_keys)} events):")
    header = f"{'clone':<8}" + "".join(f"{c}{a or '':<9}" for c, a in event_keys)
    buf.log(header)
    for cl, row in zip(clones, arm_matrix):
        buf.log(f"{cl:<8}" + "".join(f"{v:<10.3f}" for v in row))

    # correlation distance, average linkage
    corr = np.corrcoef(arm_matrix)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    dist_condensed = squareform((dist + dist.T) / 2, checks=False)
    Z = linkage(dist_condensed, method="average")
    merged = fcluster(Z, t=0.10, criterion="distance")  # distance<0.10 <=> correlation>0.9

    merge_map = {clones[i]: f"merged{merged[i]}" for i in range(len(clones))}
    buf.log(f"\nMerge threshold: correlation > 0.9 (distance < 0.10), average linkage.")
    buf.log(f"Original clone -> merged clone: {merge_map}")
    n_merged = len(set(merged))
    buf.log(f"Merged clone count: {n_merged} (from {len(clones)} original clones)")

    carc.obs["cnv_clone_merged"] = carc.obs["cnv_clone"].astype(str).map(merge_map)
    for mc in sorted(set(merge_map.values())):
        orig = [k for k, v in merge_map.items() if v == mc]
        mask = carc.obs["cnv_clone_merged"] == mc
        n = int(mask.sum())
        arm_vals = arm_level_means(gv_carc, var, mask.values)
        events = [f"{c}{a or ''}:{v:+.3f}" for (c, a), v in arm_vals.items() if abs(v) > 0.01]
        buf.log(f"  {mc} (from clones {orig}): n={n}  arm events: {', '.join(events) or 'none notable'}")

    # re-run tests on merged clones
    res = carc.obs["residual"].values
    ascl2 = carc.obs["ASCL2_expr"].values
    groups = [res[carc.obs["cnv_clone_merged"].values == mc] for mc in sorted(set(merge_map.values()))]
    if n_merged >= 2 and all(len(g) > 1 for g in groups):
        f_stat, p_anova = f_oneway(*groups)
        grand_mean = res.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((res - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total
        buf.log(f"\nOne-way ANOVA residual ~ merged_clone: F={f_stat:.3f}  p={p_anova:.3e}  eta^2={eta_sq:.4f}")
    else:
        eta_sq = np.nan
        buf.log(f"\nMerging collapsed to {n_merged} clone(s) -- eta^2 undefined (need >=2 groups).")

    q10, q90 = np.quantile(res, [0.10, 0.90])
    rg = np.where(res >= q90, "top10", np.where(res <= q10, "bottom10", "middle80"))
    ct = pd.crosstab(rg, carc.obs["cnv_clone_merged"])
    buf.log("\nCrosstab residual_group x merged_clone:\n" + ct.to_string())
    n_top = (rg == "top10").sum()
    n_total = len(carc)
    buf.log(f"\n{'merged_clone':<14}{'size':<10}{'expected_top10':<16}{'observed_top10':<16}{'%ASCL2+'}")
    top_share = {}
    for mc in sorted(set(merge_map.values())):
        m = (carc.obs["cnv_clone_merged"] == mc).values
        size = int(m.sum())
        expected = n_top * size / n_total
        observed = int(((carc.obs["cnv_clone_merged"] == mc) & (rg == "top10")).sum())
        top_share[mc] = observed
        buf.log(f"{mc:<14}{size:<10}{expected:<16.2f}{observed:<16}{100*(ascl2[m]>0).mean():.1f}")
    max_conc = max(top_share.values()) / max(sum(top_share.values()), 1)
    buf.log(f"\nMax concentration of top-decile cells in a single merged clone: {100*max_conc:.1f}%")

    clone_supported = max_conc > 0.70 and (not np.isnan(eta_sq)) and eta_sq > 0.5
    state_supported = max_conc <= 0.70 and (np.isnan(eta_sq) or eta_sq < 0.3)
    if clone_supported:
        verdict = "CLONE hypothesis better supported after merging."
    elif state_supported:
        verdict = "STATE hypothesis better supported after merging (confirms original verdict)."
    else:
        verdict = "Neither rule set cleanly satisfied after merging (same as before)."
    buf.log(f"Verdict after merging: {verdict}")


def check_c_cell_cycle(patient, pipeline, raw_ref, buf):
    buf.log("\n" + "#" * 90)
    buf.log(f"CHECK C ({patient}): cell-cycle confound")
    buf.log("#" * 90)
    buf.log("\nWhat would change the earlier verdict: if residual correlates strongly with S/G2M")
    buf.log("scores (|rho| notably >0.3-0.4) and top-decile residual cells are strongly enriched for")
    buf.log("cycling phases, a meaningful part of the 'high residual = distinct state' signal could")
    buf.log("instead reflect proliferation rather than a CNV-independent malignant identity -- this")
    buf.log("would qualify (not overturn) the STATE reading: residual heterogeneity would still argue")
    buf.log("against a single clone explaining it, but part of what the residual captures would be")
    buf.log("cell cycle rather than a stable phenotype. If correlation is weak, cell cycle is ruled")
    buf.log("out as a confound for the earlier findings.")

    adata, carc, m_carc = pipeline["adata"], pipeline["carc"], pipeline["m_carc"]
    symbol_to_id = raw_ref["symbol_to_id"]
    s_ids = [symbol_to_id[g] for g in S_GENES if g in symbol_to_id]
    g2m_ids = [symbol_to_id[g] for g in G2M_GENES if g in symbol_to_id]
    s_ids = [g for g in s_ids if g in adata.var_names]
    g2m_ids = [g for g in g2m_ids if g in adata.var_names]
    buf.log(f"\nS genes used: {len(s_ids)}/{len(S_GENES)}  G2M genes used: {len(g2m_ids)}/{len(G2M_GENES)}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_ids, g2m_genes=g2m_ids, random_state=RANDOM_STATE)

    s_score = adata.obs["S_score"].values[m_carc]
    g2m_score = adata.obs["G2M_score"].values[m_carc]
    phase = adata.obs["phase"].values[m_carc]
    res = carc.obs["residual"].values

    rho_s, p_s = spearmanr(res, s_score)
    rho_g2m, p_g2m = spearmanr(res, g2m_score)
    buf.log(f"\nSpearman(residual, S_score)   = {rho_s:.3f}  p={p_s:.3e}")
    buf.log(f"Spearman(residual, G2M_score) = {rho_g2m:.3f}  p={p_g2m:.3e}")

    buf.log(f"\nPhase distribution among carcinoma epithelial cells: "
            + ", ".join(f"{ph}={int((phase==ph).sum())} ({100*(phase==ph).mean():.1f}%)"
                        for ph in sorted(set(phase))))

    q10, q90 = np.quantile(res, [0.10, 0.90])
    rg = np.where(res >= q90, "top10", np.where(res <= q10, "bottom10", "middle80"))
    ct = pd.crosstab(rg, phase)
    chi2, p_chi2, _, _ = chi2_contingency(ct)
    buf.log("\nCrosstab residual_group x cell-cycle phase:\n" + ct.to_string())
    buf.log(f"chi2 p={p_chi2:.3e}")

    cycling = np.isin(phase, ["S", "G2M"])
    pct_cycling_top = 100 * cycling[rg == "top10"].mean()
    pct_cycling_rest = 100 * cycling[rg != "top10"].mean()
    buf.log(f"\n% cycling (S+G2M) in top-decile residual cells: {pct_cycling_top:.1f}%  "
            f"vs rest: {pct_cycling_rest:.1f}%")


def main():
    RESULTS.mkdir(exist_ok=True)
    full = sc.read_h5ad(H5AD)
    obs_all = full.obs
    raw_all = full.raw.to_adata()
    raw_all.obs = full.obs.copy()

    buf = Buf()
    buf.log("Follow-up checks on scripts/cnv_clone_vs_residual.py")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["scanpy", "infercnvpy", "leidenalg", "numpy", "anndata", "scipy"]))

    setup_buf = Buf()
    raw_pos = setup_positions(raw_all, setup_buf)
    buf.log(setup_buf.text().rstrip())

    symbol_to_id = dict(zip(raw_all.var["gene_symbol"], raw_all.var_names))
    raw_ref = {
        "symbol_to_id": symbol_to_id,
        "var_names": raw_all.var_names,
        "X": raw_all.X,
        "row_idx": pd.Series(np.arange(raw_all.n_obs), index=raw_all.obs_names),
    }
    pca_full = pd.DataFrame(full.obsm["X_pca"], index=full.obs_names)

    pipelines = {}
    for patient, samples in PATIENTS.items():
        pipe_buf = Buf()
        adata_p = build_patient_adata(raw_pos, obs_all, patient, samples)
        adata_p.obsm["X_pca_full"] = pca_full.loc[adata_p.obs_names].values
        print(f"\n=== Re-running pipeline (same as original report) for {patient} ===")
        result = run_patient(patient, adata_p, raw_ref, pipe_buf)
        pipelines[patient] = result

    buf.log("\n" + "=" * 90)
    buf.log("Reused pipeline results are identical in method to results/cnv_clone_vs_residual_"
            "patient{1,2}.txt (same code path, same random_state); only the checks below are new.")
    buf.log("=" * 90)

    check_a_patient2("Patient2", pipelines["Patient2"], raw_ref, buf)
    check_b_patient1("Patient1", pipelines["Patient1"], buf)
    check_c_cell_cycle("Patient1", pipelines["Patient1"], raw_ref, buf)
    check_c_cell_cycle("Patient2", pipelines["Patient2"], raw_ref, buf)

    out_path = RESULTS / "cnv_followup_checks.txt"
    out_path.write_text(buf.text(), encoding="utf-8")
    print(f"\nSaved -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
