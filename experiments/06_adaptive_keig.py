"""
Deney 06 — Adaptive k_eig Seçimi

Bulgu: Eigengap ve energy-based yöntemler k=5'i geçemedi.
Biyolojik sinyal spektrumun ilk 5 bileşeninde yoğunlaşıyor.
k=5 heuristic olarak önerilir.

Çalıştır:
    python experiments/06_adaptive_keig.py --cmv data/cmv.h5ad --tme data/tme.h5ad
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import anndata as ad
import h5py
from sklearn.preprocessing import LabelEncoder

from gft import build_graph, gft_embed, ari_multi_seed
from gft.embedding import adaptive_k_eig, get_eigenvalues

SEED = 42
K_NN = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmv', required=True)
    parser.add_argument('--tme', required=True)
    parser.add_argument('--out', default='results/06_adaptive_keig.json')
    args = parser.parse_args()

    np.random.seed(SEED)

    # CMV
    adata = ad.read_h5ad(args.cmv)
    cnt   = adata.obs['predicted_AIFI_L2'].value_counts()
    adata = adata[adata.obs['predicted_AIFI_L2'].isin(cnt[cnt >= 10].index)].copy()
    le    = LabelEncoder()
    y_cmv = le.fit_transform(adata.obs['predicted_AIFI_L2'].values)
    pts_cmv = adata.obsm['X_pca_harmony'][:, :50]
    k_cmv   = len(le.classes_)

    # TME (subset)
    f     = h5py.File(args.tme, 'r')
    pts_t = f['obsm']['X_pca'][:][:, :50]
    cats  = [c.decode() if isinstance(c, bytes) else str(c)
             for c in f['obs']['cell_type']['categories']]
    codes = f['obs']['cell_type']['codes'][:]
    f.close()
    labels = np.array([cats[c] for c in codes if c >= 0])
    pts_t  = pts_t[codes >= 0]
    uniq, cnts = np.unique(labels, return_counts=True)
    keep = set(uniq[cnts >= 10])
    mask = np.array([l in keep for l in labels])
    labels, pts_t = labels[mask], pts_t[mask]
    le2   = LabelEncoder()
    y_tme = le2.fit_transform(labels)
    k_tme = len(le2.classes_)
    idx   = np.random.choice(len(y_tme), 3000, replace=False)
    pts_tme, y_tme_s = pts_t[idx], y_tme[idx]

    results = {}
    for ds_name, pts, y, k in [
        ('CMV', pts_cmv[:3000], y_cmv[:3000], k_cmv),
        ('TME', pts_tme, y_tme_s, k_tme),
    ]:
        W    = build_graph(pts, k_nn=K_NN, method='cosine')
        eigs = get_eigenvalues(W, k_max=30)

        k_gap = adaptive_k_eig(W, method='eigengap')
        k_e80 = adaptive_k_eig(W, method='energy80')

        sweep = []
        print(f"\n{ds_name} — k_eig sweep:")
        print(f"  {'k_eig':>6} | {'ARI':>8} | Not")
        for ke in [2, 3, 5, 7, 10, 15, 20]:
            emb = gft_embed(W, k_eig=ke)
            ari, _ = ari_multi_seed(emb, y, k, n_seeds=5, seed=SEED)
            note = " ← eigengap" if ke == k_gap else \
                   " ← energy80" if ke == k_e80 else \
                   " ← önerilen" if ke == 5 else ""
            print(f"  {ke:>6} | {ari:>8.4f} |{note}")
            sweep.append({'k_eig': ke, 'ari': ari})

        results[ds_name] = {
            'k_eigengap': k_gap, 'k_energy80': k_e80,
            'eigenvalues': eigs[:15].tolist(), 'sweep': sweep
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSonuçlar: {args.out}")


if __name__ == '__main__':
    main()
