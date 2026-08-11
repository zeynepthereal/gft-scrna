"""
Deney 02 — N Eşiği: GFT ne zaman kazanmaya başlar?

Bulgu: N >= ~4000'de GFT tutarlı olarak PCA'yı geçer.
N < 1000'de GFT çöküş gösterir (seyrek graf).

Çalıştır:
    python experiments/02_n_sweep.py --cmv data/cmv.h5ad
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import anndata as ad
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

from gft import build_graph, gft_embed
from gft import ari_multi_seed, heterogeneity_score

SEED = 42
K_EIG = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmv', required=True)
    parser.add_argument('--out', default='results/02_n_sweep.json')
    args = parser.parse_args()

    np.random.seed(SEED)

    adata = ad.read_h5ad(args.cmv)
    cnt   = adata.obs['predicted_AIFI_L2'].value_counts()
    adata = adata[adata.obs['predicted_AIFI_L2'].isin(cnt[cnt >= 10].index)].copy()
    le    = LabelEncoder()
    y     = le.fit_transform(adata.obs['predicted_AIFI_L2'].values)
    pts   = adata.obsm['X_pca_harmony'][:, :50]
    k     = len(le.classes_)

    print(f"N Sweep | Dataset: CMV Kohort | k={k}")
    print(f"{'N':>6} | {'GFT':>8} | {'PCA':>8} | {'Ratio':>7} | {'λ₂':>8} | {'Het':>6}")
    print("-" * 55)

    results = []
    for N in [300, 500, 1000, 2000, 4000, len(pts)]:
        idx  = np.random.choice(len(pts), min(N, len(pts)), replace=False)
        p, yy = pts[idx], y[idx]
        k_s  = len(np.unique(yy))
        k_nn = min(20, N // 10)

        W   = build_graph(p, k_nn=k_nn, method='cosine')
        emb = gft_embed(W, k_eig=K_EIG)

        a_g, _ = ari_multi_seed(emb, yy, k_s, n_seeds=5, seed=SEED)
        a_p, _ = ari_multi_seed(p,   yy, k_s, n_seeds=5, seed=SEED)
        het     = heterogeneity_score(p, yy)

        # Algebraic connectivity
        from gft.embedding import algebraic_connectivity
        l2 = algebraic_connectivity(W)

        ratio = a_g / max(a_p, 1e-6)
        winner = "GFT" if ratio > 1 else "PCA"
        print(f"{N:>6} | {a_g:>8.4f} | {a_p:>8.4f} | {ratio:>7.2f}x | "
              f"{l2:>8.4f} | {het:>6.1f}  ← {winner}")

        results.append({'N': N, 'ari_gft': a_g, 'ari_pca': a_p,
                        'ratio': ratio, 'lambda2': l2, 'het': het})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSonuçlar: {args.out}")


if __name__ == '__main__':
    main()
