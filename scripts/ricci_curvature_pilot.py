"""PILOT -- does projection residual have a local-geometric (graph curvature) counterpart?

Question: in each donor's cell-cell kNN graph, do high-residual malignant cells sit in
negatively curved ("bridge"/bottleneck) regions? If yes, residual (global: distance from the
normal PCA subspace) and Ricci curvature (local: how neighbourhoods overlap) are two views of
the same geometry, and curvature can become a thesis axis. If not, we report it as negative.

Data: CRC epithelial cohort (data/crc_epithelial.h5ad). Because the full h5ad is large, the
50-D PCA (implicitly centred sparse SVD on the log-normalised X, all genes -- same subspace as
scanpy sc.pp.pca(n_comps=50, mask_var=None)), n_genes_by_counts (raw nnz>0 per cell) and obs
labels were pre-extracted to _ricci_tmp/crc_pca50.npz by extract_pca50.py.

Groups (per donor):
  normal    = disease=='normal' AND cell_type != 'malignant cell'
  malignant = disease=='colorectal cancer' AND cell_type == 'malignant cell'
  (non-malignant epithelial cells from tumour samples are excluded from the graph here.)
Donors need >= 50 normal and >= 50 malignant cells.

Method (identical for every donor):
  1. residual: U_normal (50x20) via SVD of centred normal cells; residual = ||x - UU^T x||.
     Normal cells' residual is computed 5-fold HELD-OUT (subspace fit on the other 4/5), which
     removes the in-sample advantage normal cells had in all_samples_residual.py.
  2. graph: cosine kNN (k=20) on the 50-D PCA of normal+malignant cells, symmetrised (union),
     unweighted.
  3. Ollivier-Ricci curvature per edge, alpha=0.5 (lazy random walk), exact W1 via POT emd,
     ground cost = exact shortest-path hop distance between the two 1-hop supports
     (0 / 1 / 2 / 3 -- exact for unweighted graphs since d(a,b) <= d(a,x)+d(x,y)+d(y,b) = 3).
     Node curvature = mean over incident edges.
     Augmented Forman-Ricci per edge: 4 - deg(x) - deg(y) + 3*triangles(x,y) (reported only).
  4. Confounds: node degree (kNN hubness / local density) and n_genes_by_counts.

Tests and decision rule (STATED BEFORE THE NUMBERS):
  T1  malignant vs normal node ORC: OLS  ORC ~ is_malignant + log(deg) + log(n_genes).
  T2  within MALIGNANT cells: Spearman(residual, node ORC), raw and partial (ranks of log deg,
      log n_genes regressed out). Control: same correlation within NORMAL cells.
  T3  bridge test: mean ORC of normal-malignant (cross) edges vs within-group edges;
      label-permutation p-value (edges fixed, labels shuffled, 1000 perms).

  "Residual has a curvature counterpart" is SUPPORTED if, in a MAJORITY of donors,
  T2 partial rho <= -0.10 with p < 0.05 among malignant cells, AND that rho is more negative
  than the corresponding rho among normal cells (otherwise the relation is generic
  graph-periphery geometry, not malignancy-specific).
  T3 "bridge" hypothesis SUPPORTED if cross-edge ORC < within-edge ORC with perm p < 0.05 in a
  majority of donors.
"""
import io
import os
import sys
import time

import numpy as np
import ot
import scipy.sparse as sp
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, spearmanr, rankdata, t as t_dist
from sklearn.neighbors import NearestNeighbors

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.environ.get("RICCI_NPZ", os.path.join(PROJECT_DIR, "_ricci_tmp", "crc_pca50.npz"))
OUT = os.environ.get("RICCI_OUT", os.path.join(PROJECT_DIR, "results", "ricci_curvature_pilot.txt"))

K_EIG, K_NN, ALPHA, MIN_N, N_PERM, SEED = 20, 20, 0.5, 50, 1000, 0


class Buf:
    def __init__(self):
        self.s = io.StringIO()

    def log(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        print(msg, file=self.s)


# ---------------------------------------------------------------- residual
def fit_subspace(pts):
    mean = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - mean, full_matrices=False)
    return Vt[:K_EIG].T, mean


def residual(pts, U, mean):
    X = pts - mean
    return np.linalg.norm(X - X @ U @ U.T, axis=1)


def residuals_heldout(pts_n, pts_m, rng):
    U, mu = fit_subspace(pts_n)
    res_m = residual(pts_m, U, mu)
    res_n = np.empty(len(pts_n))
    folds = rng.permutation(len(pts_n)) % 5
    for f in range(5):
        U_f, mu_f = fit_subspace(pts_n[folds != f])
        res_n[folds == f] = residual(pts_n[folds == f], U_f, mu_f)
    return res_n, res_m


# ---------------------------------------------------------------- graph + curvature
def knn_graph(X):
    nn = NearestNeighbors(n_neighbors=K_NN + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    n = len(X)
    rows = np.repeat(np.arange(n), K_NN)
    cols = idx[:, 1:].ravel()
    A = sp.csr_matrix((np.ones_like(rows, dtype=np.float32), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def ollivier_ricci(A):
    n = A.shape[0]
    nbrs = [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(n)]
    Ab = A.astype(bool).tocsr()
    A2 = (Ab @ Ab).astype(bool).tocsr()  # reachable in exactly 2 steps (incl. self)
    Ad, A2d = Ab, A2
    coo = sp.triu(A, k=1).tocoo()
    edges = np.vstack([coo.row, coo.col]).T
    kappa = np.empty(len(edges))
    for e, (x, y) in enumerate(edges):
        Sx = np.concatenate([[x], nbrs[x]])
        Sy = np.concatenate([[y], nbrs[y]])
        mx = np.full(len(Sx), (1 - ALPHA) / (len(Sx) - 1)); mx[0] = ALPHA
        my = np.full(len(Sy), (1 - ALPHA) / (len(Sy) - 1)); my[0] = ALPHA
        D = np.full((len(Sx), len(Sy)), 3.0)
        D[A2d[Sx][:, Sy].toarray()] = 2.0
        D[Ad[Sx][:, Sy].toarray()] = 1.0
        D[Sx[:, None] == Sy[None, :]] = 0.0
        kappa[e] = 1.0 - ot.emd2(mx, my, D)  # d(x,y)=1
    return edges, kappa


def forman_aug(A, edges):
    deg = np.asarray(A.sum(1)).ravel()
    tri = np.asarray((A @ A)[edges[:, 0], edges[:, 1]]).ravel()
    return 4 - deg[edges[:, 0]] - deg[edges[:, 1]] + 3 * tri


def node_mean(edges, vals, n):
    s = np.bincount(edges[:, 0], vals, n) + np.bincount(edges[:, 1], vals, n)
    c = np.bincount(edges[:, 0], None, n) + np.bincount(edges[:, 1], None, n)
    return s / np.maximum(c, 1)


# ---------------------------------------------------------------- stats
def partial_spearman(x, y, covs):
    rx, ry = rankdata(x), rankdata(y)
    C = sm.add_constant(np.column_stack([rankdata(c) for c in covs]))
    ex = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ey = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    df = len(x) - 2 - len(covs)
    tstat = r * np.sqrt(df / max(1e-12, 1 - r * r))
    return r, 2 * t_dist.sf(abs(tstat), df)


def bridge_test(edges, kappa, lab, rng):
    def gap(l):
        cross = l[edges[:, 0]] != l[edges[:, 1]]
        if cross.sum() == 0:
            return np.nan, 0
        return kappa[cross].mean() - kappa[~cross].mean(), int(cross.sum())
    obs, ncross = gap(lab)
    null = np.array([gap(rng.permutation(lab))[0] for _ in range(N_PERM)])
    p = (1 + np.sum(null <= obs)) / (N_PERM + 1)
    return obs, ncross, p, null.mean()


# ---------------------------------------------------------------- main
def main():
    buf = Buf()
    rng = np.random.default_rng(SEED)
    d = np.load(NPZ, allow_pickle=True)
    pcs, ng = d["pcs"].astype(np.float64), d["n_genes"].astype(float)
    donor, disease, ctype = d["donor"], d["disease"], d["cell_type"]
    is_norm = (disease == "normal") & (ctype != "malignant cell")
    is_mal = (disease == "colorectal cancer") & (ctype == "malignant cell")

    buf.log(__doc__.split("Tests and decision rule")[1].join(["Tests and decision rule", ""]))
    buf.log(f"versions: numpy={np.__version__}, POT={ot.__version__}; k_nn={K_NN}, k_eig={K_EIG}, alpha={ALPHA}\n")

    # sanity: reproduce the published in-sample ratio (disease-based grouping, as in all_samples_residual.py)
    buf.log("Sanity check vs results/all_samples_residual.txt (in-sample normal, disease-based groups):")
    for dn in sorted(set(donor)):
        mn = (donor == dn) & (disease == "normal"); mc = (donor == dn) & (disease == "colorectal cancer")
        if mn.sum() >= MIN_N and mc.sum() >= MIN_N:
            U, mu = fit_subspace(pcs[mn])
            r = np.median(residual(pcs[mc], U, mu)) / np.median(residual(pcs[mn], U, mu))
            buf.log(f"  {dn}: ratio={r:.2f}x")
    buf.log("")

    rows = []
    for dn in sorted(set(donor)):
        mn, mm = (donor == dn) & is_norm, (donor == dn) & is_mal
        if mn.sum() < MIN_N or mm.sum() < MIN_N:
            buf.log(f"SKIP {dn}: n_normal={mn.sum()}, n_malignant={mm.sum()}")
            continue
        t0 = time.time()
        Xn, Xm = pcs[mn], pcs[mm]
        res_n, res_m = residuals_heldout(Xn, Xm, rng)
        X = np.vstack([Xn, Xm]); lab = np.r_[np.zeros(len(Xn), int), np.ones(len(Xm), int)]
        res = np.r_[res_n, res_m]; ngc = np.r_[ng[mn], ng[mm]]
        A = knn_graph(X)
        deg = np.asarray(A.sum(1)).ravel()
        edges, kap = ollivier_ricci(A)
        forman = forman_aug(A, edges)
        node_orc = node_mean(edges, kap, len(X))
        node_for = node_mean(edges, forman, len(X))

        buf.log("=" * 96)
        buf.log(f"{dn}: n_normal={len(Xn)} n_malignant={len(Xm)} edges={len(edges)} ({time.time()-t0:.0f}s)")
        buf.log(f"  held-out residual median: normal={np.median(res_n):.3f} malignant={np.median(res_m):.3f} "
                f"ratio={np.median(res_m)/np.median(res_n):.2f}x")
        buf.log(f"  edge ORC: mean={kap.mean():.3f} min={kap.min():.3f} max={kap.max():.3f}; "
                f"frac negative={np.mean(kap<0):.3f}")

        # T1
        Xo = sm.add_constant(np.column_stack([lab, np.log(deg), np.log(ngc)]))
        fit = sm.OLS(node_orc, Xo).fit()
        _, p_mw = mannwhitneyu(node_orc[lab == 1], node_orc[lab == 0])
        buf.log(f"  T1 node ORC median: normal={np.median(node_orc[lab==0]):.3f} "
                f"malignant={np.median(node_orc[lab==1]):.3f} (MWU p={p_mw:.2e})")
        buf.log(f"     OLS ORC ~ is_mal + log deg + log n_genes: coef_mal={fit.params[1]:.4f} "
                f"p={fit.pvalues[1]:.2e}; coef_logdeg={fit.params[2]:.4f}")

        # T2
        out = {}
        for name, g in [("malignant", 1), ("normal", 0)]:
            s = lab == g
            r_raw, p_raw = spearmanr(res[s], node_orc[s])
            r_p, p_p = partial_spearman(res[s], node_orc[s], [np.log(deg[s]), np.log(ngc[s])])
            r_f, _ = spearmanr(res[s], node_for[s])
            r_deg, _ = spearmanr(res[s], deg[s])
            out[name] = (r_p, p_p)
            buf.log(f"  T2 [{name:<9}] rho(res,ORC) raw={r_raw:+.3f} (p={p_raw:.1e})  "
                    f"partial={r_p:+.3f} (p={p_p:.1e})  | rho(res,Forman)={r_f:+.3f}  rho(res,deg)={r_deg:+.3f}")
        # quartiles within malignant
        s = lab == 1
        q = np.quantile(res[s], [.25, .5, .75])
        bins = np.digitize(res[s], q)
        buf.log("     malignant residual quartile -> median node ORC: " +
                "  ".join(f"Q{b+1}={np.median(node_orc[s][bins==b]):+.3f}" for b in range(4)))

        # T3
        gap, ncross, p_perm, null_mean = bridge_test(edges, kap, lab, rng)
        buf.log(f"  T3 cross edges={ncross}; mean ORC cross - within = {gap:+.4f} "
                f"(perm null mean {null_mean:+.4f}, one-sided p={p_perm:.4f})")

        rows.append(dict(donor=dn, n_n=len(Xn), n_m=len(Xm), rho_m=out["malignant"][0], p_m=out["malignant"][1],
                         rho_n=out["normal"][0], coef=fit.params[1], p_coef=fit.pvalues[1],
                         ncross=ncross, gap=gap, p_perm=p_perm))

    buf.log("\n" + "=" * 110)
    buf.log("SUMMARY")
    buf.log(f"{'donor':<11}{'N_n':>6}{'N_m':>6}{'T1 coef_mal':>13}{'p':>10}{'T2 rho_mal':>12}{'p':>10}"
            f"{'rho_norm':>10}{'T3 gap':>10}{'n_cross':>9}{'perm p':>9}")
    for r in rows:
        buf.log(f"{r['donor']:<11}{r['n_n']:>6}{r['n_m']:>6}{r['coef']:>13.4f}{r['p_coef']:>10.1e}"
                f"{r['rho_m']:>+12.3f}{r['p_m']:>10.1e}{r['rho_n']:>+10.3f}{r['gap']:>+10.4f}{r['ncross']:>9}{r['p_perm']:>9.4f}")
    n = len(rows)
    t2 = sum(r["rho_m"] <= -0.10 and r["p_m"] < 0.05 and r["rho_m"] < r["rho_n"] for r in rows)
    t3 = sum(r["gap"] < 0 and r["p_perm"] < 0.05 for r in rows)
    buf.log(f"\nT2 criterion met in {t2}/{n} donors -> {'SUPPORTED' if t2 > n / 2 else 'NOT SUPPORTED'}")
    buf.log(f"T3 criterion met in {t3}/{n} donors -> {'SUPPORTED' if t3 > n / 2 else 'NOT SUPPORTED'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(buf.s.getvalue())
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
