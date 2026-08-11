"""
Deney 04 — Harmony Bağımlılık Testi

Hakem itirazı: "GFT başarısı Harmony PCA'ya mı bağlı?"

Test: Grafı 3 farklı kaynaktan kur:
  1. Harmony PCA (batch corrected)
  2. Ham PCA (batch correction yok)
  3. HVG → PCA (sıfırdan, Harmony'siz)

Bulgu: GFT ARI üç koşulda da 0.79-0.84 aralığında kaldı.
Harmony bağımlılığı yok.

Çalıştır:
    python experiments/04_harmony_independence.py --cmv data/cmv.h5ad
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import scipy.sparse
import anndata as ad
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from gft import build_graph, gft_embed, ari_multi_seed

SEED  = 42
K_EIG = 5
K_NN  = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmv', required=True)
    parser.add_argument('--out', default='results/04_harmony_independence.json')
    args = parser.parse_args()

    np.random.seed(SEED)

    adata = ad.read_h5ad(args.cmv)
    cnt   = adata.obs['predicted_AIFI_L2'].value_counts()
    adata = adata[adata.obs['predicted_AIFI_L2'].isin(cnt[cnt >= 10].index)].copy()
    le    = LabelEncoder()
    y     = le.fit_transform(adata.obs['predicted_AIFI_L2'].values)
    k     = len(le.classes_)

    # Üç başlangıç noktası
    pts_harmony = adata.obsm['X_pca_harmony'][:, :50]
    pts_raw_pca = adata.obsm['X_pca'][:, :50]

    # HVG → PCA (sıfırdan)
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X_norm = np.log1p(X / (X.sum(axis=1, keepdims=True) + 1e-9) * 1e4)
    gene_var = X_norm.var(axis=0)
    hvg_idx  = np.argsort(gene_var)[::-1][:2000]
    pts_hvg  = PCA(n_components=50, random_state=SEED).fit_transform(
        X_norm[:, hvg_idx])

    print("Harmony Bağımlılık Testi")
    print(f"{'Graf Kaynağı':<35} | {'PCA ARI':>9} | {'GFT ARI':>9} | {'GFT/PCA':>8}")
    print("-" * 68)

    results = []
    for name, pts in [
        ('Harmony PCA (batch corrected)', pts_harmony),
        ('Ham PCA (batch yok)',           pts_raw_pca),
        ('HVG → PCA (sıfırdan)',          pts_hvg),
    ]:
        W   = build_graph(pts, k_nn=K_NN, method='cosine')
        emb = gft_embed(W, k_eig=K_EIG)

        ari_pca, _ = ari_multi_seed(pts, y, k, n_seeds=5, seed=SEED)
        ari_gft, _ = ari_multi_seed(emb, y, k, n_seeds=5, seed=SEED)
        ratio       = ari_gft / max(ari_pca, 1e-6)

        print(f"  {name:<33} | {ari_pca:>9.4f} | {ari_gft:>9.4f} | {ratio:>7.2f}x")
        results.append({'source': name, 'ari_pca': ari_pca,
                        'ari_gft': ari_gft, 'ratio': ratio})

    # Sonuç yorumu
    ratios = [r['ratio'] for r in results]
    spread = max(ratios) - min(ratios)
    print(f"\n  GFT/PCA ratio spread: {spread:.3f} "
          f"({'Bağımsız ✓' if spread < 0.1 else 'Bağımlı ✗'})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'results': results, 'spread': spread}, f, indent=2)
    print(f"Sonuçlar: {args.out}")


if __name__ == '__main__':
    main()
