"""
gft/graph.py — Graf inşa yöntemleri

Desteklenen yöntemler:
  - cosine_knn   : Cosine benzerlik tabanlı kNN (önerilen)
  - gaussian_knn : Gaussian kernel ağırlıklı kNN
  - adaptive_gaussian : Her nokta için ayrı sigma
  - mutual_knn   : Mutual kNN (simetrik komşuluk)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


def cosine_knn(pts: np.ndarray, k_nn: int = 20) -> csr_matrix:
    """
    Cosine benzerlik tabanlı kNN grafı.
    Yüksek boyutlu PCA uzayında Euclidean'dan daha tutarlı.

    Parametreler
    -----------
    pts   : (N, D) numpy array — hücre embedding matrisi
    k_nn  : komşu sayısı (varsayılan: 20)

    Döndürür
    --------
    W : (N, N) simetrik sparse ağırlık matrisi
    """
    N = len(pts)
    norms = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-10
    pts_n = pts / norms
    nbrs = NearestNeighbors(n_neighbors=k_nn, metric='cosine').fit(pts_n)
    dists, indices = nbrs.kneighbors(pts_n)

    rows, cols, vals = [], [], []
    for i in range(N):
        for ki, j in enumerate(indices[i]):
            if i == j:
                continue
            rows.append(i)
            cols.append(j)
            vals.append(float(max(1 - dists[i, ki], 0)))

    W = csr_matrix((vals, (rows, cols)), shape=(N, N))
    return (W + W.T) / 2


def gaussian_knn(pts: np.ndarray, k_nn: int = 15) -> csr_matrix:
    """
    Gaussian kernel ağırlıklı kNN grafı.
    Sigma = medyan komşu mesafesi (global).
    """
    N = len(pts)
    nbrs = NearestNeighbors(n_neighbors=k_nn).fit(pts)
    dists, indices = nbrs.kneighbors(pts)
    sigma = float(np.median(dists[:, 1:]))

    rows, cols, vals = [], [], []
    for i in range(N):
        for ki, j in enumerate(indices[i]):
            if i == j:
                continue
            w = float(np.exp(-dists[i, ki] ** 2 / (2 * sigma ** 2)))
            rows.append(i)
            cols.append(j)
            vals.append(w)

    W = csr_matrix((vals, (rows, cols)), shape=(N, N))
    return (W + W.T) / 2


def adaptive_gaussian(pts: np.ndarray, k_nn: int = 15) -> csr_matrix:
    """
    Adaptif Gaussian: her nokta için sigma = k-th komşu mesafesi.
    Yerel yoğunluğa duyarlı.
    """
    N = len(pts)
    nbrs = NearestNeighbors(n_neighbors=k_nn).fit(pts)
    dists, indices = nbrs.kneighbors(pts)
    sigma_i = dists[:, -1] + 1e-10  # her nokta için kendi sigma'sı

    rows, cols, vals = [], [], []
    for i in range(N):
        for ki, j in enumerate(indices[i]):
            if i == j:
                continue
            w = float(np.exp(-dists[i, ki] ** 2 / sigma_i[i] ** 2))
            rows.append(i)
            cols.append(j)
            vals.append(w)

    W = csr_matrix((vals, (rows, cols)), shape=(N, N))
    return (W + W.T) / 2


def mutual_knn(pts: np.ndarray, k_nn: int = 15) -> csr_matrix:
    """
    Mutual kNN: kenar sadece her iki yönde de komşu ise eklenir.
    Daha seyrek ama daha güvenilir bağlantılar.
    """
    N = len(pts)
    nbrs = NearestNeighbors(n_neighbors=k_nn).fit(pts)
    dists, indices = nbrs.kneighbors(pts)
    sigma = float(np.median(dists[:, 1:]))
    neighbor_sets = [set(indices[i]) for i in range(N)]

    rows, cols, vals = [], [], []
    for i in range(N):
        for ki, j in enumerate(indices[i]):
            if i == j:
                continue
            if i in neighbor_sets[j]:  # mutual kontrol
                w = float(np.exp(-dists[i, ki] ** 2 / (2 * sigma ** 2)))
                rows.append(i)
                cols.append(j)
                vals.append(w)

    W = csr_matrix((vals, (rows, cols)), shape=(N, N))
    return (W + W.T) / 2


def build_graph(pts: np.ndarray, k_nn: int = 20,
                method: str = 'cosine') -> csr_matrix:
    """
    Birleşik graf inşa fonksiyonu.

    Parametreler
    -----------
    pts    : (N, D) embedding matrisi
    k_nn   : komşu sayısı
    method : 'cosine' | 'gaussian' | 'adaptive' | 'mutual'
    """
    dispatch = {
        'cosine':   cosine_knn,
        'gaussian': gaussian_knn,
        'adaptive': adaptive_gaussian,
        'mutual':   mutual_knn,
    }
    if method not in dispatch:
        raise ValueError(f"Bilinmeyen metod: {method}. "
                         f"Seçenekler: {list(dispatch.keys())}")
    return dispatch[method](pts, k_nn=k_nn)
