# GFT-scRNA: Graph Fourier Transform for Single-Cell RNA Sequencing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Tek cümle:** Heterojen tümör mikro-ortamı verisinde, Graf Fourier Dönüşümü (GFT) tabanlı spektral embedding, standart PCA'ya göre **2-3x daha iyi hücre tipi ayrışması** sağlar.

---

## Temel Bulgular

| Veri Seti | N | k | Het. Skoru | PCA ARI | GFT ARI | Oran |
|---|---|---|---|---|---|---|
| CMV Kohort (immün) | 9.325 | 8 | 9.1 | 0.387 | 0.839 | **2.17x** |
| TME Kanser (meme) | 12.493 | 9 | 18.4 | 0.290 | 0.931 | **3.21x** |

**Konfigürasyon:** Cosine kNN (k=20), k_eig=5, seed=42

### GFT Ne Zaman İşe Yarar?

```python
from gft import predict_gft_advantage

advice = predict_gft_advantage(pca_embedding, cell_labels)
# {'recommended': True, 'het_score': 9.1, 'reason': 'het=9.1>7, N=9325>=1000 → GFT önerilir'}
```

**Kural:** Heterojenlik skoru (inter/intra PCA mesafesi) > 7 **ve** N ≥ 1000 ise GFT önerilir.

---

## Kurulum

```bash
git clone https://github.com/<kullanici>/gft-scrna.git
cd gft-scrna
pip install -r requirements.txt
```

---

## Hızlı Başlangıç

```python
import anndata as ad
from gft import build_graph, gft_embed, predict_gft_advantage
from gft import ari_multi_seed, heterogeneity_score

# Verinizi yükleyin
adata = ad.read_h5ad('your_data.h5ad')
pca   = adata.obsm['X_pca'][:, :50]   # PCA embedding

# GFT uygun mu?
advice = predict_gft_advantage(pca, cell_labels)
print(advice['reason'])

# Graf inşa ve GFT
W   = build_graph(pca, k_nn=20, method='cosine')
emb = gft_embed(W, k_eig=5)

# Değerlendirme
ari_mean, ari_std = ari_multi_seed(emb, cell_labels, n_clusters)
print(f"GFT ARI: {ari_mean:.4f} ± {ari_std:.4f}")
```

---

## Deneyler

Her deney bağımsız çalıştırılabilir:

```bash
# Veriyi data/ klasörüne koyun (data/README.md'ye bakın)

# 01 — Ana karşılaştırma (GFT vs PCA vs Denoised)
python experiments/01_main_comparison.py \
    --cmv data/cmv.h5ad \
    --tme data/tme.h5ad

# 02 — N eşiği analizi
python experiments/02_n_sweep.py --cmv data/cmv.h5ad

# 03 — Graf topolojisi robustness
python experiments/03_robustness.py --tme data/tme.h5ad

# 04 — Harmony bağımlılık testi
python experiments/04_harmony_independence.py --cmv data/cmv.h5ad

# 05 — Supervised sınıflandırma (RF, 5-fold)
python experiments/05_supervised.py --tme data/tme.h5ad

# 06 — Adaptive k_eig analizi
python experiments/06_adaptive_keig.py \
    --cmv data/cmv.h5ad \
    --tme data/tme.h5ad
```

---

## Yöntem Özeti

### Graf Fourier Transform

1. **Graf İnşası:** Hücreler arasında Cosine kNN grafı kurulur (k=20)
2. **Normalize Laplacian:** L_sym = I - D^{-1/2} W D^{-1/2}
3. **Spektral Embedding:** İlk k=5 özvektör — biyolojik sinyal bu bileşenlerde yoğunlaşır
4. **Denoising (opsiyonel):** X̂ = U U^T X — teknik gürültü bastırma

### Neden Cosine kNN?

Yüksek boyutlu PCA uzayında Euclidean mesafe "küre" etkisinden muzdariptir.
Cosine benzerlik yön bilgisini kullandığından bu sorunu aşar ve
Gaussian kNN'e göre daha kararlı sonuçlar üretir.

### Heterojenlik Skoru

```
het_score = mean(inter-class centroid distance) / mean(intra-class variance)
```

Eşik: het > 7 → GFT avantajlı (5 veri seti üzerinde ampirik)

---

## Dikkat Edilmesi Gerekenler

- **N < 1000:** GFT çöküş gösterebilir (seyrek graf)  
- **Homojen veri:** Tek soy hattı subtipleri için PCA tercih edilmeli  
- **Supervised görev:** Supervised sınıflandırmada PCA daha iyi (Macro-F1: 0.973 vs 0.953)  
- **k_eig:** k=5 heuristic; veri setine göre 3-7 arasında sweep önerilir  

---

## Sonuçlar

Önceden hesaplanmış sonuçlar `results/` klasöründedir.

```
results/
├── 01_main.json          # Ana karşılaştırma
├── 02_n_sweep.json       # N eşiği
├── 03_robustness.json    # Robustness
├── 04_harmony.json       # Harmony bağımlılık
├── 05_supervised.json    # Supervised
└── 06_adaptive_keig.json # Adaptive k_eig
```

---

## Atıf

Eğer bu kodu kullanırsanız lütfen alıntı yapın (preprint yakında):

```bibtex
@misc{gft_scrna_2026,
  author = {...},
  title  = {Graph Fourier Transform for Heterogeneous scRNA-seq Cell Type Discovery},
  year   = {2026},
  url    = {https://github.com/<kullanici>/gft-scrna}
}
```

---

## Lisans

MIT License — bkz. [LICENSE](LICENSE)
