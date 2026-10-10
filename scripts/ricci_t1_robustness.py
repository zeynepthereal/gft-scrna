"""T1 robustness check: is "malignant node ORC < normal node ORC" (results/ricci_curvature_pilot.txt,
T1) a density / spread / intrinsic-dimension artefact of the combined kNN graph, or does it hold in
SEPARATE, equal-size, single-population graphs and survive local-geometry covariates?

Reuses knn_graph, ollivier_ricci, node_mean, partial_spearman from scripts/ricci_curvature_pilot.py
unmodified (imported, not copied) -- same alpha=0.5, exact W1 (POT), hop-distance ground cost.
Only the outer loop over (donor, k, seed, group) is parallelized across processes; the curvature
algorithm itself is untouched.

Groups (per donor, same as the pilot):
  normal            = disease=='normal'            & cell_type!='malignant cell'
  malignant         = disease=='colorectal cancer'  & cell_type=='malignant cell'
  tumor_nonmalignant= disease=='colorectal cancer'  & cell_type!='malignant cell'
Donor filter: >=50 normal AND >=50 malignant cells (same 6 donors as the pilot).

1a. Separate graphs, equal N: sample n=min(N_normal, N_malignant) cells from normal and malignant
    (same rng draw reused for the tumor-nonmalignant group, capped at min(n, N_tumor_nonmalignant),
    when >=50 tumor-nonmalignant cells exist). EACH group gets its OWN kNN graph (no bridging edges
    possible by construction). Compare node-ORC medians. 20 seeds. Report median difference,
    fraction of seeds with malignant median < normal median, median Mann-Whitney p.
1b. Local geometry covariates on the FULL combined (normal+malignant) graph, same construction as
    the pilot's T1 (no subsampling): per cell, mean cosine distance to its k nearest neighbours,
    local intrinsic dimension (Levina-Bickel MLE, same k), and the group's global spread (trace of
    the 50-D covariance, reported once per group). OLS: ORC ~ is_mal + log deg + log n_genes +
    log knn_dist + local_ID.
1c. Repeat 1a and 1b for k in {10, 20, 30}. The k=20 results are the ones used in the decision rule.
1d. Batch/sample control: are tumor-derived non-malignant epithelial cells closer to normal or to
    malignant cells in node ORC? Uses the same per-seed separate graphs computed in 1a (k=20),
    averaged over seeds.

DECISION RULE (T1-robust), stated before the numbers:
T1 is robust if, in a MAJORITY of donors (>=4/6), ALL THREE hold:
  (a) in 1a (k=20): malignant median ORC < normal median ORC in >=80% of the 20 seeds;
  (b) in 1b (k=20): the is_malignant OLS coefficient is <0 with p<0.05;
  (c) in 1d: malignant median ORC < tumor-nonmalignant median ORC (only evaluated for donors that
      have a tumor-nonmalignant group; if a donor has none, (c) is treated as not-applicable and
      does not count against that donor).
Otherwise T1 is treated as an artefact. In that case Step 2 (scripts/ricci_cnv_ascl2.py) is NOT run.

Output: results/ricci_t1_robustness.txt
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version

import numpy as np
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, t as t_dist
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ricci_curvature_pilot as rc  # noqa: E402 -- reuse knn_graph/ollivier_ricci/node_mean/partial_spearman

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(PROJECT_DIR, "_ricci_tmp", "crc_pca50.npz")
OUT = os.path.join(PROJECT_DIR, "results", "ricci_t1_robustness.txt")

K_LIST = [10, 20, 30]
K_PRIMARY = 20
N_SEEDS = 20
MIN_N = 50
N_WORKERS = min(12, os.cpu_count() or 4)


class Buf:
    def __init__(self):
        self.lines = []

    def log(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.lines.append(msg)

    def text(self):
        return "\n".join(self.lines) + "\n"


# ---------------------------------------------------------------- worker (runs in a subprocess)
def _group_node_orc(X, k):
    """Same method as the pilot's T1: knn_graph + ollivier_ricci + node_mean, imported unmodified."""
    rc.K_NN = k
    A = rc.knn_graph(X)
    edges, kap = rc.ollivier_ricci(A)
    deg = np.asarray(A.sum(1)).ravel()
    return rc.node_mean(edges, kap, len(X)), deg


def local_intrinsic_dim(X, k):
    """Levina-Bickel MLE local intrinsic dimension, per point, using the k nearest neighbours
    (cosine distance, same metric as the pilot's graph)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    dist, _ = nn.kneighbors(X)
    dist = dist[:, 1:]  # drop self (distance 0)
    dist = np.maximum(dist, 1e-12)  # avoid log(0) for exact duplicates
    r_k = dist[:, -1:]
    logs = np.log(r_k / dist[:, :-1])
    mean_log = logs.mean(axis=1)
    mean_log = np.maximum(mean_log, 1e-12)
    m_hat = 1.0 / mean_log
    return m_hat, dist.mean(axis=1)  # (local ID, mean knn distance)


def spread_trace(X):
    return float(np.trace(np.cov(X, rowvar=False)))


def load_data():
    d = np.load(NPZ, allow_pickle=True)
    pcs = d["pcs"].astype(np.float64)
    ng = d["n_genes"].astype(float)
    donor, disease, ctype = d["donor"], d["disease"], d["cell_type"]
    is_norm = (disease == "normal") & (ctype != "malignant cell")
    is_mal = (disease == "colorectal cancer") & (ctype == "malignant cell")
    is_tnm = (disease == "colorectal cancer") & (ctype != "malignant cell")
    return pcs, ng, donor, is_norm, is_mal, is_tnm


def run_1a_1d(pool, pcs, ng, donor, is_norm, is_mal, is_tnm, donors, buf):
    """Returns {donor: {k: {...}}} and prints 1a per-donor-per-k tables + 1d table."""
    results = {}
    for dn in donors:
        dmask = donor == dn
        Xn_full, Xm_full = pcs[dmask & is_norm], pcs[dmask & is_mal]
        Xt_full = pcs[dmask & is_tnm]
        n = min(len(Xn_full), len(Xm_full))
        has_third = len(Xt_full) >= MIN_N
        buf.log(f"\n{dn}: N_normal={len(Xn_full)} N_malignant={len(Xm_full)} "
                f"N_tumor_nonmal={len(Xt_full)} (n_matched={n}, third_group={'yes' if has_third else 'no'})")

        results[dn] = {}
        for k in K_LIST:
            t0 = time.time()
            futures = {}
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(seed)
                idx_n = rng.choice(len(Xn_full), size=n, replace=False)
                idx_m = rng.choice(len(Xm_full), size=n, replace=False)
                futures[pool.submit(_group_node_orc, Xn_full[idx_n], k)] = (seed, "normal")
                futures[pool.submit(_group_node_orc, Xm_full[idx_m], k)] = (seed, "malignant")
                if has_third:
                    n3 = min(n, len(Xt_full))
                    idx_t = rng.choice(len(Xt_full), size=n3, replace=False)
                    futures[pool.submit(_group_node_orc, Xt_full[idx_t], k)] = (seed, "tumor_nonmal")

            per_seed = {s: {} for s in range(N_SEEDS)}
            for fut in as_completed(futures):
                seed, grp = futures[fut]
                node_orc, _ = fut.result()
                per_seed[seed][grp] = node_orc

            diffs, mwu_ps, less_flags = [], [], []
            third_meds, mal_meds, norm_meds = [], [], []
            for seed in range(N_SEEDS):
                orc_n, orc_m = per_seed[seed]["normal"], per_seed[seed]["malignant"]
                med_n, med_m = np.median(orc_n), np.median(orc_m)
                diffs.append(med_m - med_n)
                less_flags.append(med_m < med_n)
                mwu_ps.append(mannwhitneyu(orc_m, orc_n).pvalue)
                norm_meds.append(med_n)
                mal_meds.append(med_m)
                if has_third:
                    third_meds.append(np.median(per_seed[seed]["tumor_nonmal"]))

            frac_less = np.mean(less_flags)
            buf.log(f"  k={k}: median diff (mal-norm)={np.median(diffs):+.4f}  "
                    f"frac_seeds_mal<norm={frac_less:.2f}  median MWU p={np.median(mwu_ps):.2e}  "
                    f"({time.time()-t0:.0f}s, {len(futures)} graphs)")
            results[dn][k] = {
                "frac_less": frac_less, "median_diff": np.median(diffs), "median_mwu_p": np.median(mwu_ps),
                "mean_norm_med": np.mean(norm_meds), "mean_mal_med": np.mean(mal_meds),
                "mean_third_med": np.mean(third_meds) if has_third else None,
                "has_third": has_third,
            }
    return results


def run_1b(pool, pcs, ng, donor, is_norm, is_mal, donors, buf):
    buf.log("\n" + "=" * 96)
    buf.log("1b -- local geometry covariates on the FULL combined (normal+malignant) graph")
    buf.log("=" * 96)
    results = {}
    for dn in donors:
        dmask = donor == dn
        mn, mm = dmask & is_norm, dmask & is_mal
        Xn, Xm = pcs[mn], pcs[mm]
        X = np.vstack([Xn, Xm])
        lab = np.r_[np.zeros(len(Xn), int), np.ones(len(Xm), int)]
        ngc = np.r_[ng[mn], ng[mm]]
        buf.log(f"\n{dn}: spread (trace cov, 50D PCA) normal={spread_trace(Xn):.2f} "
                f"malignant={spread_trace(Xm):.2f}")

        results[dn] = {}
        futures = {pool.submit(_group_node_orc, X, k): k for k in K_LIST}
        node_orc_by_k, deg_by_k = {}, {}
        for fut in as_completed(futures):
            k = futures[fut]
            node_orc_by_k[k], deg_by_k[k] = fut.result()

        for k in K_LIST:
            node_orc, deg = node_orc_by_k[k], deg_by_k[k]
            local_id, knn_dist = local_intrinsic_dim(X, k)
            Xo = sm.add_constant(np.column_stack(
                [lab, np.log(deg), np.log(ngc), np.log(knn_dist), local_id]))
            fit = sm.OLS(node_orc, Xo).fit()
            ci = fit.conf_int()[1]
            buf.log(f"  k={k}: mean local_ID={local_id.mean():.2f}  mean knn_dist={knn_dist.mean():.4f}")
            buf.log(f"    OLS ORC ~ is_mal + log_deg + log_ngenes + log_knndist + local_ID: "
                    f"coef_mal={fit.params[1]:+.4f}  CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  p={fit.pvalues[1]:.2e}  "
                    f"coef_localID={fit.params[5]:+.4f} p={fit.pvalues[5]:.2e}  R2={fit.rsquared:.3f}")
            results[dn][k] = {"coef_mal": fit.params[1], "p_mal": fit.pvalues[1],
                               "ci_lo": ci[0], "ci_hi": ci[1]}
    return results


def main():
    buf = Buf()
    buf.log("T1 robustness check (density / spread / intrinsic-dimension artefact test)")
    buf.log("versions: " + ", ".join(
        f"{p}={version(p)}" for p in ["numpy", "scipy", "statsmodels", "scikit-learn", "POT"]))
    buf.log(f"k values: {K_LIST}  (primary/decision k={K_PRIMARY})  n_seeds={N_SEEDS}  "
            f"n_workers={N_WORKERS}\n")
    buf.log("DECISION RULE (stated before the numbers): T1 is robust if, in a MAJORITY of donors")
    buf.log("(>=4/6), ALL THREE hold: (a) 1a k=20: malignant median ORC < normal median ORC in")
    buf.log(">=80% of 20 seeds; (b) 1b k=20: is_malignant OLS coefficient <0, p<0.05; (c) 1d:")
    buf.log("malignant median ORC < tumor-nonmalignant median ORC (only for donors that have a")
    buf.log("tumor-nonmalignant group -- otherwise (c) is not-applicable for that donor).")
    buf.log("Otherwise T1 is treated as an artefact and Step 2 is not run.\n")

    pcs, ng, donor, is_norm, is_mal, is_tnm = load_data()
    donors = sorted(dn for dn in set(donor)
                     if ((donor == dn) & is_norm).sum() >= MIN_N and ((donor == dn) & is_mal).sum() >= MIN_N)
    buf.log(f"Donors passing >={MIN_N}/{MIN_N} normal/malignant filter: {donors}\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        buf.log("=" * 96)
        buf.log("1a -- separate graphs, equal N (removes bridging/boundary effects)")
        buf.log("=" * 96)
        res_1a = run_1a_1d(pool, pcs, ng, donor, is_norm, is_mal, is_tnm, donors, buf)
        res_1b = run_1b(pool, pcs, ng, donor, is_norm, is_mal, donors, buf)

    buf.log("\n" + "=" * 96)
    buf.log("1c -- k-robustness summary (1a and 1b across k=10,20,30)")
    buf.log("=" * 96)
    hdr = f"{'donor':<11}{'k':>4}{'1a frac<':>10}{'1a diff':>10}{'1a MWUp':>10}{'1b coef':>10}{'1b p':>10}"
    buf.log(hdr)
    for dn in donors:
        for k in K_LIST:
            a, b = res_1a[dn][k], res_1b[dn][k]
            buf.log(f"{dn:<11}{k:>4}{a['frac_less']:>10.2f}{a['median_diff']:>10.4f}"
                    f"{a['median_mwu_p']:>10.1e}{b['coef_mal']:>10.4f}{b['p_mal']:>10.1e}")

    buf.log("\n" + "=" * 96)
    buf.log("1d -- batch/sample control: normal vs malignant vs tumor-non-malignant (k=20, mean over 20 seeds)")
    buf.log("=" * 96)
    hdr2 = f"{'donor':<11}{'mean_ORC_normal':>17}{'mean_ORC_malignant':>20}{'mean_ORC_tumor_nonmal':>23}"
    buf.log(hdr2)
    for dn in donors:
        a = res_1a[dn][K_PRIMARY]
        tnm = f"{a['mean_third_med']:.4f}" if a["has_third"] else "n/a (<50 cells)"
        buf.log(f"{dn:<11}{a['mean_norm_med']:>17.4f}{a['mean_mal_med']:>20.4f}{tnm:>23}")

    buf.log("\n" + "=" * 96)
    buf.log("DECISION TABLE (k=20)")
    buf.log("=" * 96)
    hdr3 = f"{'donor':<11}{'(a) frac<0.8?':>15}{'(b) coef<0,p<.05?':>19}{'(c) mal<tnm?':>14}{'all 3?':>9}"
    buf.log(hdr3)
    n_pass, n_applicable_donors = 0, 0
    for dn in donors:
        a = res_1a[dn][K_PRIMARY]
        b = res_1b[dn][K_PRIMARY]
        cond_a = a["frac_less"] >= 0.80
        cond_b = b["coef_mal"] < 0 and b["p_mal"] < 0.05
        if a["has_third"]:
            cond_c = a["mean_mal_med"] < a["mean_third_med"]
            cond_c_str = "YES" if cond_c else "NO"
        else:
            cond_c = True  # not-applicable -> does not count against the donor
            cond_c_str = "N/A"
        all3 = cond_a and cond_b and cond_c
        if all3:
            n_pass += 1
        buf.log(f"{dn:<11}{('YES' if cond_a else 'NO'):>15}{('YES' if cond_b else 'NO'):>19}"
                f"{cond_c_str:>14}{('YES' if all3 else 'NO'):>9}")

    buf.log(f"\n{n_pass}/{len(donors)} donors satisfy all three T1-robust conditions.")
    t1_robust = n_pass >= 4
    buf.log(f"VERDICT: T1 is {'ROBUST' if t1_robust else 'TREATED AS AN ARTEFACT'} "
            f"({'>=4/6' if t1_robust else '<4/6'} donors pass).")
    if not t1_robust:
        buf.log("Step 2 (scripts/ricci_cnv_ascl2.py) will NOT be run, per the decision rule.")

    buf.log("\n" + "=" * 96)
    buf.log("CAVEATS")
    buf.log("=" * 96)
    buf.log("- HTA8_6002 has only 60 malignant cells (smallest donor) and no tumor-non-malignant")
    buf.log("  group (0 cells) -- criterion (c) is not-applicable there, and all its per-seed graphs")
    buf.log("  are built on only 60 cells per group, so its per-donor estimates are noisier than the")
    buf.log("  other 5 donors.")
    buf.log("- The tumor-non-malignant group's sample size is capped at min(n, N_tumor_nonmal); for")
    buf.log("  HTA8_6018 (n=54 available) and HTA8_6009/6013/6028 this is well below the matched")
    buf.log("  normal/malignant n, so its per-seed graphs are smaller and its ORC estimates in 1d")
    buf.log("  are correspondingly less stable than the normal/malignant comparison.")
    buf.log("- 1b's OLS is fit on the same combined graph the pilot used for its own T1 (no")
    buf.log("  subsampling) -- it is not independent of the pilot's original T1 result, only an")
    buf.log("  extension of the same regression with two more covariates.")
    buf.log("- local_ID (Levina-Bickel) and knn_dist are themselves computed from the same 50-D PCA")
    buf.log("  coordinates that the kNN graph and curvature are built from -- they are covariates of")
    buf.log("  the same geometry, not an independent measurement.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(buf.text())
    print(f"\nSaved -> {OUT}")
    return t1_robust


if __name__ == "__main__":
    main()
