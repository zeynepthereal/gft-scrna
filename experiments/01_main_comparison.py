"""
Deney 01 — Ana Karşılaştırma: GFT vs PCA vs Denoised

Her iki veri setinde (CMV kohort, TME kanser) şunları ölçer:
  - ARI (10 seed, K-Means)
  - Heterojenlik skoru
  - GFT/PCA oranı

Çalıştır:
    python experiments/01_main_comparison.py \
        --cmv data/cmv.h5ad \
        --tme data/tme.h5ad

Beklenen sonuçlar (seed=42, k_nn=20, cosine, k_eig=5):
    CMV Kohort : GFT 2.17x PCA
    TME Kanser : GFT 3.21x PCA
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import anndata as ad
import h5py
from sklearn.preprocessing import LabelEncoder

from gft import build_graph, gft_embed, gft_denoise
from gft import ari_multi_seed, heterogeneity_score, predict_gft_advantage

SEED   = 42
K_EIG  = 5
K_NN   = 20
METRIC = 'cosine'


def load_cmv(path: str):
    adata  = ad.read_h5ad(path)
    counts = adata.obs['predicted_AIFI_L2'].value_counts()
    adata  = adata[adata.obs['predicted_AIFI_L2'].isin(
        counts[counts >= 10].index)].copy()
    le  = LabelEncoder()
    y   = le.fit_transform(adata.obs['predicted_AIFI_L2'].values)
    pts = adata.obsm['X_pca_harmony'][:, :50]
    return pts, y, le.classes_


def load_tme(path: str):
    f     = h5py.File(path, 'r')
    pts   = f['obsm']['X_pca'][:][:, :50]
    cats  = [c.decode() if isinstance(c, bytes) else str(c)
             for c in f['obs']['cell_type']['categories']]
    codes = f['obs']['cell_type']['codes'][:]
    f.close()

    labels = np.array([cats[c] for c in codes if c >= 0])
    pts    = pts[codes >= 0]
    uniq, cnts = np.unique(labels, return_counts=True)
    keep   = set(uniq[cnts >= 10])
    mask   = np.array([l in keep for l in labels])
    labels, pts = labels[mask], pts[mask]

    le = LabelEncoder()
    y  = le.fit_transform(labels)
    return pts, y, le.classes_


def run(name, pts, y, classes):
    k   = len(classes)
    het = heterogeneity_score(pts, y)
    adv = predict_gft_advantage(pts, y)

    print(f"\n{'='*60}")
    print(f"  {name} | N={len(pts)} | k={k}")
    print(f"  Heterojenlik: {het:.2f} — {adv['reason']}")
    print(f"{'='*60}")

    W   = build_graph(pts, k_nn=K_NN, method=METRIC)
    emb = gft_embed(W, k_eig=K_EIG)
    Xh  = gft_denoise(pts, emb)

    results = {}
    for label, e in [('PCA', pts), ('GFT', emb), ('Denoised', Xh)]:
        mean, std = ari_multi_seed(e, y, k, n_seeds=10, seed=SEED)
        results[label] = {'mean': mean, 'std': std}
        ratio = mean / max(results['PCA']['mean'], 1e-6) if label != 'PCA' else 1.0
        ratio_str = f" ({ratio:.2f}x PCA)" if label != 'PCA' else ''
        print(f"  {label:<12}: ARI={mean:.4f} ± {std:.4f}{ratio_str}")

    return {
        'name': name, 'N': len(pts), 'k': k,
        'het': het, 'gft_recommended': adv['recommended'],
        'results': results, 'classes': list(classes)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmv', required=True)
    parser.add_argument('--tme', required=True)
    parser.add_argument('--out', default='results/01_main.json')
    args = parser.parse_args()

    np.random.seed(SEED)

    pts_cmv, y_cmv, cls_cmv = load_cmv(args.cmv)
    pts_tme, y_tme, cls_tme = load_tme(args.tme)

    all_results = [
        run('CMV Kohort', pts_cmv, y_cmv, cls_cmv),
        run('TME Kanser', pts_tme, y_tme, cls_tme),
    ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSonuçlar: {args.out}")


if __name__ == '__main__':
    main()
