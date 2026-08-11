"""
Deney 03 — Graf Topolojisi Robustness

Farklı graf inşa yöntemleri ve k_nn değerleri test edilir.

Bulgular:
  - Cosine kNN en kararlı (k=10-30 aralığında)
  - k=5 çöküş noktası (seyrek graf)
  - Mutual kNN ve Epsilon-ball yakınsama sorunu

Çalıştır:
    python experiments/03_robustness.py --tme data/tme.h5ad
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder

from gft import build_graph, gft_embed, ari_multi_seed

SEED  = 42
K_EIG = 5


def load_tme(path):
    f     = h5py.File(path, 'r')
    pts   = f['obsm']['X_pca'][:][:, :50]
    cats  = [c.decode() if isinstance(c, bytes) else str(c)
             for c in f['obs']['cell_type']['categories']]
    codes = f['obs']['cell_type']['codes'][:]
    f.close()
    labels = np.array([cats[c] for c in codes if c >= 0])
    pts    = pts[codes >= 0]
    uniq, cnts = np.unique(labels, return_counts=True)
    keep = set(uniq[cnts >= 10])
    mask = np.array([l in keep for l in labels])
    labels, pts = labels[mask], pts[mask]
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    return pts, y, len(le.classes_)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tme', required=True)
    parser.add_argument('--n_sub', type=int, default=3000,
                        help='Alt örnekleme boyutu (hız için)')
    parser.add_argument('--out', default='results/03_robustness.json')
    args = parser.parse_args()

    np.random.seed(SEED)
    pts_full, y_full, k = load_tme(args.tme)

    idx = np.random.choice(len(pts_full), min(args.n_sub, len(pts_full)),
                            replace=False)
    pts, y = pts_full[idx], y_full[idx]
    k_s = len(np.unique(y))

    results = {'k_nn_sweep': [], 'method_comparison': []}

    # k_nn sweep — cosine
    print("k_nn Sweep (Cosine):")
    print(f"  {'k_nn':>6} | {'ARI':>8}")
    for kn in [5, 10, 15, 20, 30, 50]:
        W   = build_graph(pts, k_nn=kn, method='cosine')
        emb = gft_embed(W, k_eig=K_EIG)
        ari, std = ari_multi_seed(emb, y, k_s, n_seeds=5, seed=SEED)
        print(f"  {kn:>6} | {ari:>8.4f}")
        results['k_nn_sweep'].append({'k_nn': kn, 'ari': ari, 'std': std})

    # Yöntem karşılaştırması
    print("\nYöntem Karşılaştırması (k_nn=20):")
    print(f"  {'Yöntem':>20} | {'ARI':>8}")
    for method in ['cosine', 'gaussian', 'adaptive']:
        try:
            W   = build_graph(pts, k_nn=20, method=method)
            emb = gft_embed(W, k_eig=K_EIG)
            ari, std = ari_multi_seed(emb, y, k_s, n_seeds=5, seed=SEED)
            print(f"  {method:>20} | {ari:>8.4f}")
            results['method_comparison'].append(
                {'method': method, 'ari': ari, 'std': std})
        except Exception as e:
            print(f"  {method:>20} | HATA: {str(e)[:40]}")
            results['method_comparison'].append(
                {'method': method, 'ari': None, 'error': str(e)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSonuçlar: {args.out}")


if __name__ == '__main__':
    main()
