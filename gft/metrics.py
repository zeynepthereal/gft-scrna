"""
gft/metrics.py — Değerlendirme metrikleri

İçerik:
  - ari_multi_seed    : Çoklu seed ile kararlı ARI
  - heterogeneity_score : inter/intra class PCA mesafesi
  - supervised_metrics  : BAC ve Macro-F1 (RF, 5-fold CV)
  - predict_gft_advantage : Heterojenlik skoruna göre GFT önerisi
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                              f1_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def ari_multi_seed(emb: np.ndarray, y: np.ndarray, k: int,
                   n_seeds: int = 10, seed: int = 42) -> tuple[float, float]:
    """
    K-Means ARI'sini n_seeds farklı başlangıçta ölçer.

    Döndürür: (ortalama ARI, standart sapma)
    """
    aris = [
        adjusted_rand_score(
            y, KMeans(k, random_state=s, n_init=5).fit_predict(emb)
        )
        for s in range(seed, seed + n_seeds)
    ]
    return float(np.mean(aris)), float(np.std(aris))


def heterogeneity_score(pts: np.ndarray, y: np.ndarray) -> float:
    """
    Heterojenlik skoru: inter-class / intra-class PCA mesafesi.

    Yorumlama:
      > 7  → GFT büyük olasılıkla PCA'yı geçer
      < 7  → PCA daha uygun

    Bu eşik 5 veri seti üzerinde ampirik olarak belirlendi.
    """
    classes   = np.unique(y)
    centroids = np.array([pts[y == c].mean(axis=0) for c in classes])

    inter = np.mean([
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(len(classes))
        for j in range(i + 1, len(classes))
    ])
    intra = np.mean([pts[y == c].std(axis=0).mean() for c in classes])

    return float(inter / (intra + 1e-9))


def predict_gft_advantage(pts: np.ndarray, y: np.ndarray,
                           n_cells: int = None) -> dict:
    """
    GFT'nin bu veri setinde avantajlı olup olmadığını tahmin eder.

    Kriterler (ikisi de sağlanmalı):
      1. Heterojenlik skoru > 7
      2. N >= 1000

    Döndürür: {'recommended': bool, 'het_score': float, 'n': int, 'reason': str}
    """
    N   = n_cells or len(pts)
    het = heterogeneity_score(pts, y)

    if het > 7 and N >= 1000:
        reason = f"het={het:.1f}>7, N={N}>=1000 → GFT önerilir"
        rec = True
    elif het <= 7:
        reason = f"het={het:.1f}<=7 (homojen veri) → PCA önerilir"
        rec = False
    else:
        reason = f"N={N}<1000 (yetersiz örneklem) → PCA önerilir"
        rec = False

    return {'recommended': rec, 'het_score': het, 'n': N, 'reason': reason}


def supervised_metrics(emb: np.ndarray, y: np.ndarray,
                       n_splits: int = 5, seed: int = 42) -> dict:
    """
    Random Forest ile supervised sınıflandırma metrikleri.

    Döndürür: {'bac', 'macro_f1', 'per_class_f1'}

    NOT: Deneysel bulgularımızda supervised görevde PCA daha üstündür.
    GFT unsupervised kümeleme için önerilir.
    """
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rf     = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                     random_state=seed, n_jobs=-1)
    y_pred = cross_val_predict(rf, emb, y, cv=skf)

    return {
        'bac':          float(balanced_accuracy_score(y, y_pred)),
        'macro_f1':     float(f1_score(y, y_pred, average='macro')),
        'per_class_f1': f1_score(y, y_pred, average=None).tolist(),
    }
