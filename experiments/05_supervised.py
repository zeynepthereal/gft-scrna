"""
Deney 05 — Supervised Sınıflandırma (BAC ve Macro-F1)

Bulgu: Supervised görevde PCA daha iyi (Macro-F1: 0.973 vs 0.953).
GFT unsupervised kümeleme için önerilir, supervised için değil.

Nadir sınıf analizi:
  - Hepatosit (N=90):    GFT hafif avantajlı (+0.017)
  - NK T cell (N=121):   GFT dezavantajlı (-0.126)

Çalıştır:
    python experiments/05_supervised.py --tme data/tme.h5ad
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import h5py
from sklearn.preprocessing import LabelEncoder

from gft import build_graph, gft_embed, gft_denoise, supervised_metrics

SEED  = 42
K_EIG = 5
K_NN  = 20


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
    keep = set(uniq[cnts >= 5])
    mask = np.array([l in keep for l in labels])
    labels, pts = labels[mask], pts[mask]
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    return pts, y, le.classes_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tme', required=True)
    parser.add_argument('--out', default='results/05_supervised.json')
    args = parser.parse_args()

    np.random.seed(SEED)
    pts, y, classes = load_tme(args.tme)
    k = len(classes)

    print(f"Supervised Sınıflandırma | N={len(pts)} | k={k}")
    print("(Bu deney uzun sürebilir — RF 5-fold CV)")

    W   = build_graph(pts, k_nn=K_NN, method='cosine')
    emb = gft_embed(W, k_eig=K_EIG)
    Xh  = gft_denoise(pts, emb)

    results = {}
    print(f"\n  {'Yöntem':<15} | {'BAC':>8} | {'Macro-F1':>9}")
    print("  " + "-"*37)
    for name, e in [('PCA', pts), ('GFT', emb), ('Denoised', Xh)]:
        m = supervised_metrics(e, y, seed=SEED)
        results[name] = m
        print(f"  {name:<15} | {m['bac']:>8.4f} | {m['macro_f1']:>9.4f}")

    # Nadir sınıf analizi
    class_counts = np.array([(y == i).sum() for i in range(k)])
    rare_mask    = class_counts < 200
    rare_results = []

    if rare_mask.any():
        print("\n  Nadir Sınıflar (N<200):")
        pca_f1 = results['PCA']['per_class_f1']
        gft_f1 = results['GFT']['per_class_f1']
        for i in np.where(rare_mask)[0]:
            diff = gft_f1[i] - pca_f1[i]
            print(f"    {classes[i]:<25} N={class_counts[i]:>3}: "
                  f"PCA={pca_f1[i]:.3f} GFT={gft_f1[i]:.3f} Δ={diff:+.3f}")
            rare_results.append({
                'class': classes[i], 'n': int(class_counts[i]),
                'pca_f1': pca_f1[i], 'gft_f1': gft_f1[i], 'delta': diff
            })

    output = {'supervised': results, 'rare_class': rare_results,
              'classes': list(classes)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSonuçlar: {args.out}")


if __name__ == '__main__':
    main()
