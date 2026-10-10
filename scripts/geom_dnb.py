"""COMPONENT 3 -- Dynamic Network Biomarker (DNB) signature vs. residual (Chen et al. 2012).

Hypothesis: near a critical transition, a gene module shows a within-module SD increase,
within-module |PCC| increase, and between-module |PCC| decrease. Composite score:
  I = SD_in * PCC_in / PCC_out
computed per module, per residual-ordered bin (normal reference, then malignant residual
quintiles Q1..Q5 -- terciles Q1..Q3 for HTA8_6002, which has only 60 malignant cells).

Depth control (mandatory): adata.raw counts are binomially thinned to a common target depth
(10th percentile of total counts among the donor's included cells); cells below target are
dropped; normalize_total + log1p applied after thinning. Without this, n_genes_by_counts rises
with residual (established in results/confound_and_detector.txt) and inflates correlation
estimates for high-residual bins purely through depth.

Residual: the standard cohort-PCA-based projection residual (fit_subspace/residual imported from
scripts/ricci_curvature_pilot.py, q=20) -- this component is NOT part of the R1 dual-space check
(that applies only to Components 1 and 2, per the task).

Modules:
  PRIMARY: all 50 MSigDB Hallmark sets, filtered per donor to genes passing the donor's >=10%-
    expressed filter; kept if >=15 genes remain.
  SECONDARY (EXPLORATORY): hierarchical-clustering modules from normal cells' gene-gene
    correlation, restricted to the top 2000 most variable donor-expressed genes (a computational
    cap on top of the >=10%-expressed filter, noted explicitly as a simplification) -- kept if
    >=15 genes per cluster.

Per module/bin: median I over 20 subsamples (all bins sampled down to the smallest bin's size).
"Transition peak": max I in Q2-Q4 exceeds both Q1's and Q5's I (Q1/Q3 for HTA8_6002).
"Monotonic increase": Spearman(I, bin order) > 0.8.
Permutation (200 reps, 1 subsample/rep): shuffle malignant residual-bin labels (bin sizes fixed,
normal reference untouched); per-permutation statistic = max over the 50 modules of
[max(I_Q2..Q4) - max(I_Q1,I_Q5)]; family-wise p-value per module from this shared null.
Also reports the normal-reference-vs-Q5(last bin) I comparison directly, per module.

DECISION RULE (B3), stated before the numbers:
  "DNB transition signature exists": the SAME Hallmark module(s) show a transition peak with
  family-wise permutation p<0.05 in >=4/6 donors.
  "Monotonic increase" (reported separately, NOT the same claim -- DNB theory predicts a peak,
  not monotonicity): the same module(s) show Spearman(I, bin order)>0.8 in >=4/6 donors.
  If neither holds for any module, this is reported as-is.

Outputs: results/geom_dnb.txt
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_common as gc  # noqa: E402
from ricci_curvature_pilot import fit_subspace, residual  # noqa: E402

SEED = 0
N_SUBSAMPLE_REPS = 20
N_PERM = 200
MIN_MODULE_GENES = 15
N_TOP_VAR_GENES_SECONDARY = 2000
N_SECONDARY_CLUSTERS = 20
GENE_MIN_FRAC = 0.10
N_OUT_GENES = 500
MIN_N = 50


def thin_to_depth(raw_X, keep_mask, target, rng):
    """Binomially thin raw_X rows (restricted to keep_mask, all >= target) down to `target`
    total counts each. Returns a dense/csr matrix of thinned counts, same row order as keep_mask."""
    X = raw_X[keep_mask].tocsr().copy()
    total = np.asarray(X.sum(axis=1)).ravel()
    p = target / total
    row_idx = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    p_per_entry = p[row_idx]
    data_int = X.data.astype(np.int64)
    thinned = rng.binomial(data_int, np.clip(p_per_entry, 0, 1))
    X.data = thinned.astype(np.float32)
    X.eliminate_zeros()
    return X


def prep_donor(adata, dn, rng):
    """Depth-thin + normalize + log1p this donor's normal+malignant cells; compute residual for
    malignant cells; return dict with dense expr matrix (cells x genes-passing-filter), gene
    names, group labels, residual."""
    mask_n, mask_m = gc.donor_masks_full(adata, dn)
    mask_incl = mask_n | mask_m
    raw_X = adata.raw.X.tocsr()
    total_incl = np.asarray(raw_X[mask_incl].sum(axis=1)).ravel()
    target = np.percentile(total_incl, 10)

    keep_local = total_incl >= target
    incl_idx = np.where(mask_incl)[0]
    keep_idx = incl_idx[keep_local]
    n_dropped = int((~keep_local).sum())

    thinned = thin_to_depth(raw_X, keep_idx, target, rng)
    ad_thin = sc.AnnData(X=thinned, var=adata.var.copy())
    ad_thin.obs_names = adata.obs_names[keep_idx]
    sc.pp.normalize_total(ad_thin, target_sum=1e4)
    sc.pp.log1p(ad_thin)

    frac = np.asarray((ad_thin.X > 0).mean(axis=0)).ravel()
    gene_keep = frac >= GENE_MIN_FRAC
    expr = np.asarray(ad_thin.X[:, gene_keep].todense())
    gene_names = adata.var["feature_name"].astype(str).values[gene_keep]
    gene_names, uniq_idx = np.unique(gene_names, return_index=True)
    expr = expr[:, uniq_idx]

    is_norm_kept = mask_n[keep_idx]
    is_mal_kept = mask_m[keep_idx]

    return {
        "expr": expr, "genes": gene_names, "is_norm": is_norm_kept, "is_mal": is_mal_kept,
        "target_depth": target, "n_dropped": n_dropped, "n_kept": len(keep_idx),
        "obs_names_kept": adata.obs_names[keep_idx],
    }


def assign_bins(residual_mal, n_bins):
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(residual_mal, q)
    bins = np.digitize(residual_mal, edges[1:-1], right=True)
    return np.clip(bins, 0, n_bins - 1)


def module_score(expr, gene_idx, out_idx, cell_idx):
    """I = SD_in * PCC_in / PCC_out for one module, one set of cells."""
    sub = expr[np.ix_(cell_idx, gene_idx)]
    sd_in = sub.std(axis=0, ddof=1).mean()
    if sub.shape[1] < 2:
        pcc_in = 0.0
    else:
        C = np.corrcoef(sub, rowvar=False)
        iu = np.triu_indices_from(C, k=1)
        pcc_in = np.nanmean(np.abs(C[iu]))
    out = expr[np.ix_(cell_idx, out_idx)]
    cross = np.corrcoef(np.hstack([sub, out]), rowvar=False)[: sub.shape[1], sub.shape[1]:]
    pcc_out = np.nanmean(np.abs(cross))
    pcc_out = max(pcc_out, 1e-6)
    return sd_in * pcc_in / pcc_out


def resolve_module_gene_idx(gene_set_symbols, gene_names_arr, name_to_idx):
    idx = [name_to_idx[g] for g in gene_set_symbols if g in name_to_idx]
    return np.array(idx, dtype=int)


def secondary_modules(expr, genes, is_norm, rng):
    """Hierarchical clustering modules from normal cells' gene-gene correlation, restricted to
    the top N_TOP_VAR_GENES_SECONDARY most variable donor-expressed genes (computational cap)."""
    Xn = expr[is_norm]
    var = Xn.var(axis=0)
    top = np.argsort(-var)[:N_TOP_VAR_GENES_SECONDARY]
    Xn_top = Xn[:, top]
    C = np.corrcoef(Xn_top, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    dist = 1 - C
    np.fill_diagonal(dist, 0)
    condensed = squareform((dist + dist.T) / 2, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=N_SECONDARY_CLUSTERS, criterion="maxclust")
    modules = {}
    for lab in np.unique(labels):
        gidx = top[labels == lab]
        if len(gidx) >= MIN_MODULE_GENES:
            modules[f"cluster_{lab}"] = gidx
    return modules


def run_donor(dn, adata, hallmark_raw, buf):
    rng = np.random.default_rng(SEED)
    buf.log("\n" + "=" * 96)
    buf.log(dn)
    buf.log("=" * 96)

    prep = prep_donor(adata, dn, rng)
    expr, genes = prep["expr"], prep["genes"]
    is_norm, is_mal = prep["is_norm"], prep["is_mal"]
    name_to_idx = {g: i for i, g in enumerate(genes)}
    buf.log(f"Depth target (10th pct of included cells' total counts) = {prep['target_depth']:.0f}; "
            f"dropped {prep['n_dropped']} cells below target; kept {prep['n_kept']}; "
            f"{len(genes)} genes pass >={int(GENE_MIN_FRAC*100)}% expressed filter")

    pcs, ng, donor_arr, disease_arr, ctype_arr = gc.load_cohort_pca()
    mask_n_c, mask_m_c = gc.donor_masks_cohort(donor_arr, disease_arr, ctype_arr, dn)
    U, mean = fit_subspace(pcs[mask_n_c])
    r_all = residual(pcs[mask_m_c], U, mean)
    r_by_obsname = pd.Series(r_all, index=adata.obs_names[mask_m_c])
    mal_names_kept = prep["obs_names_kept"][is_mal]
    r_mal = r_by_obsname.reindex(mal_names_kept).values
    valid = ~np.isnan(r_mal)
    if (~valid).any():
        buf.log(f"  WARNING: {(~valid).sum()} malignant cells kept after thinning have no residual "
                f"match (unexpected) -- dropped from binning.")

    n_bins = 3 if dn == "HTA8_6002" else 5
    if dn == "HTA8_6002":
        buf.log("  HTA8_6002: only 60 malignant cells -- using 3 bins (terciles) instead of 5, per instructions.")
    bins = assign_bins(r_mal[valid], n_bins)
    mal_idx_all = np.where(is_mal)[0][valid]
    bin_cell_idx = [mal_idx_all[bins == b] for b in range(n_bins)]
    norm_cell_idx = np.where(is_norm)[0]
    all_group_idx = [norm_cell_idx] + bin_cell_idx
    group_names = ["normal"] + [f"Q{i+1}" for i in range(n_bins)]
    sizes = [len(g) for g in all_group_idx]
    min_size = min(sizes)
    buf.log(f"  bin sizes: {dict(zip(group_names, sizes))}; subsampling all bins to n={min_size}, "
            f"{N_SUBSAMPLE_REPS} reps")

    frac = (expr > 0).mean(axis=0)
    modules_primary = {}
    for name, gset in hallmark_raw.items():
        gidx = resolve_module_gene_idx(gset, genes, name_to_idx)
        if len(gidx) >= MIN_MODULE_GENES:
            modules_primary[name] = gidx
    buf.log(f"  Hallmark modules with >={MIN_MODULE_GENES} genes after donor gene filter: "
            f"{len(modules_primary)}/{len(hallmark_raw)}")

    modules_secondary = secondary_modules(expr, genes, is_norm, rng)
    buf.log(f"  Secondary (exploratory) hierarchical-clustering modules with >={MIN_MODULE_GENES} genes: "
            f"{len(modules_secondary)} (from top {N_TOP_VAR_GENES_SECONDARY} variable genes, "
            f"{N_SECONDARY_CLUSTERS}-way cut)")

    all_gene_idx = set(range(len(genes)))

    def compute_I_matrix(group_idx_list, n_reps, rng_local, modules):
        """Returns dict module_name -> array (n_groups,) of median-over-reps I scores."""
        out = {}
        for name, gidx in modules.items():
            others = np.array(list(all_gene_idx - set(gidx.tolist())))
            scores = np.empty((len(group_idx_list), n_reps))
            for gi, cell_pool in enumerate(group_idx_list):
                for rep in range(n_reps):
                    cell_idx = rng_local.choice(cell_pool, size=min_size, replace=False)
                    out_idx = rng_local.choice(others, size=min(N_OUT_GENES, len(others)), replace=False)
                    scores[gi, rep] = module_score(expr, gidx, out_idx, cell_idx)
            out[name] = np.median(scores, axis=1)
        return out

    buf.log("\n  Computing I scores (primary Hallmark modules)...")
    I_primary = compute_I_matrix(all_group_idx, N_SUBSAMPLE_REPS, rng, modules_primary)
    buf.log("  Computing I scores (secondary exploratory modules)...")
    I_secondary = compute_I_matrix(all_group_idx, N_SUBSAMPLE_REPS, rng, modules_secondary)

    def peak_stat(I_vec):
        mid = I_vec[1:-1]  # Q2..Q4 (or the single middle bin for n_bins=3)
        return (mid.max() if len(mid) else -np.inf) - max(I_vec[0], I_vec[-1])

    def is_peak(I_vec):
        mid = I_vec[1:-1]
        if len(mid) == 0:
            return False
        return mid.max() > I_vec[0] and mid.max() > I_vec[-1]

    def is_monotone(I_vec):
        rho, _ = spearmanr(I_vec[1:], np.arange(len(I_vec) - 1))
        return rho, rho > 0.8

    buf.log("\n  Permutation null (200 reps, 1 subsample/rep, malignant bin-label shuffles; "
            "family-wise max over the 50 Hallmark modules)...")
    perm_max = np.empty(N_PERM)
    rng_perm = np.random.default_rng(SEED + 1)
    mal_idx_flat = np.concatenate(bin_cell_idx)
    bin_sizes = [len(b) for b in bin_cell_idx]
    for p in range(N_PERM):
        shuffled = rng_perm.permutation(mal_idx_flat)
        cuts = np.cumsum(bin_sizes)[:-1]
        perm_bins = np.split(shuffled, cuts)
        perm_groups = [norm_cell_idx] + perm_bins
        I_perm = compute_I_matrix(perm_groups, 1, rng_perm, modules_primary)
        stats = [peak_stat(v) for v in I_perm.values()]
        perm_max[p] = max(stats) if stats else -np.inf

    buf.log(f"\n  {'module':<35}{'I(' + ','.join(group_names) + ')':<55}{'peak?':>7}{'peak_stat':>11}"
            f"{'fw_p':>9}{'rho(mono)':>11}{'mono?':>7}")
    module_results = {}
    for name, I_vec in I_primary.items():
        pk = is_peak(I_vec)
        pstat = peak_stat(I_vec)
        fw_p = (1 + np.sum(perm_max >= pstat)) / (N_PERM + 1)
        rho, mono = is_monotone(I_vec)
        vec_str = ",".join(f"{v:.3f}" for v in I_vec)
        buf.log(f"  {name:<35}{vec_str:<55}{('YES' if pk else 'no'):>7}{pstat:>11.4f}{fw_p:>9.3f}"
                f"{rho:>+11.3f}{('YES' if mono else 'no'):>7}")
        module_results[name] = {"I": I_vec, "peak": pk, "peak_stat": pstat, "fw_p": fw_p,
                                 "rho": rho, "mono": mono}

    buf.log(f"\n  Secondary (EXPLORATORY) modules -- not part of the pre-registered B3 decision rule:")
    for name, I_vec in I_secondary.items():
        pk = is_peak(I_vec)
        rho, mono = is_monotone(I_vec)
        vec_str = ",".join(f"{v:.3f}" for v in I_vec)
        buf.log(f"    {name:<20}{vec_str:<55}{('peak' if pk else '')}{' mono' if mono else ''}")

    buf.log(f"\n  Normal-reference vs Q{n_bins} (last bin) I comparison, primary modules:")
    for name, I_vec in I_primary.items():
        buf.log(f"    {name:<35} I_normal={I_vec[0]:.4f}  I_Q{n_bins}={I_vec[-1]:.4f}  "
                f"ratio={I_vec[-1]/max(I_vec[0],1e-9):.2f}x")

    return module_results, group_names


def main():
    buf = gc.Buf()
    buf.log("COMPONENT 3 -- Dynamic Network Biomarker (DNB) signature vs. residual (Chen et al. 2012)")
    buf.log(f"n_subsample_reps={N_SUBSAMPLE_REPS}; n_perm={N_PERM}; seed={SEED}; "
            f"min_module_genes={MIN_MODULE_GENES}\n")
    buf.log("DECISION RULE (B3), stated before the numbers:")
    buf.log("  'DNB transition signature exists': the SAME Hallmark module(s) show a transition peak")
    buf.log("  (max I in the middle bin(s) > both edge bins) with family-wise permutation p<0.05,")
    buf.log("  in >=4/6 donors.")
    buf.log("  'Monotonic increase' (reported separately -- NOT the same claim; DNB theory predicts a")
    buf.log("  peak, not monotonicity): the same module(s) show Spearman(I, bin order)>0.8 in >=4/6")
    buf.log("  donors.")
    buf.log("  If neither holds for any module, reported as-is (no signature).\n")

    try:
        import gseapy
        hallmark_raw = gseapy.get_library(name="MSigDB_Hallmark_2020", organism="Human")
        buf.log(f"Loaded {len(hallmark_raw)} MSigDB Hallmark gene sets from Enrichr.\n")
    except Exception as e:
        buf.log(f"Could not download MSigDB Hallmark gene sets: {e}. PRIMARY module analysis "
                f"cannot proceed -- skipping Component 3 entirely.")
        buf.save(os.path.join(gc.RESULTS_DIR, "geom_dnb.txt"))
        return

    adata = gc.load_adata()
    all_results = {}
    for dn in gc.DONORS:
        module_results, group_names = run_donor(dn, adata, hallmark_raw, buf)
        all_results[dn] = module_results

    buf.log("\n" + "=" * 96)
    buf.log("CROSS-DONOR SUMMARY (primary Hallmark modules)")
    buf.log("=" * 96)
    all_module_names = sorted(set().union(*[set(r.keys()) for r in all_results.values()]))
    buf.log(f"{'module':<35}{'n_donors_peak_sig':>18}{'donors':<45}{'n_donors_mono':>15}{'donors'}")
    peak_hits, mono_hits = {}, {}
    for name in all_module_names:
        peak_donors = [dn for dn in gc.DONORS if name in all_results[dn] and
                       all_results[dn][name]["peak"] and all_results[dn][name]["fw_p"] < 0.05]
        mono_donors = [dn for dn in gc.DONORS if name in all_results[dn] and all_results[dn][name]["mono"]]
        peak_hits[name], mono_hits[name] = peak_donors, mono_donors
        if len(peak_donors) > 0 or len(mono_donors) > 0:
            buf.log(f"{name:<35}{len(peak_donors):>18}  {str(peak_donors):<43}{len(mono_donors):>15}  "
                    f"{mono_donors}")

    best_peak_module = max(peak_hits, key=lambda k: len(peak_hits[k]))
    best_mono_module = max(mono_hits, key=lambda k: len(mono_hits[k]))
    n_peak_best = len(peak_hits[best_peak_module])
    n_mono_best = len(mono_hits[best_mono_module])

    buf.log(f"\nBest module for 'transition peak': '{best_peak_module}' in {n_peak_best}/6 donors "
            f"{peak_hits[best_peak_module]} -> {'SUPPORTED' if n_peak_best >= 4 else 'NOT SUPPORTED'}")
    buf.log(f"Best module for 'monotonic increase': '{best_mono_module}' in {n_mono_best}/6 donors "
            f"{mono_hits[best_mono_module]} -> {'SUPPORTED' if n_mono_best >= 4 else 'NOT SUPPORTED'}")

    buf.log("\nVERDICT (B3):")
    if n_peak_best >= 4:
        buf.log(f"  DNB transition signature EXISTS for module '{best_peak_module}'.")
    else:
        buf.log("  No Hallmark module shows a transition peak with family-wise p<0.05 in >=4/6 donors.")
        buf.log("  DNB transition signature: NOT SUPPORTED.")
    if n_mono_best >= 4:
        buf.log(f"  Monotonic increase found separately for module '{best_mono_module}' "
                f"(NOT the same claim as a transition peak).")
    else:
        buf.log("  Monotonic increase: NOT SUPPORTED in >=4/6 donors for any module.")

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- Residual, module scores, and gene expression are all derived from the same expression")
    buf.log("  matrix. This is an internal-consistency test, not independent validation.")
    buf.log("- scRNA dropout depresses correlation estimates. Binomial thinning to a common depth")
    buf.log("  equalizes this effect across bins, but does not remove the dropout floor itself.")
    buf.log("- Residual quantiles in this snapshot dataset are not a time axis. A Q1->Q5 ordering can")
    buf.log("  only be read as a hypothesized trajectory, not an observed one.")
    buf.log("- HTA8_6002 has only 60 malignant cells: 3 bins (terciles) were used instead of 5, and")
    buf.log("  its per-bin subsample size (after matching to the smallest group) is small -- this")
    buf.log("  donor has by far the lowest power in Component 3.")
    buf.log("- The secondary (exploratory) hierarchical-clustering modules are capped to the top")
    buf.log("  2000 most variable donor-expressed genes for computational tractability; this is a")
    buf.log("  narrower gene universe than the full >=10%-expressed filter used for Hallmark modules,")
    buf.log("  and is not part of the pre-registered B3 decision rule.")

    buf.save(os.path.join(gc.RESULTS_DIR, "geom_dnb.txt"))


if __name__ == "__main__":
    main()
