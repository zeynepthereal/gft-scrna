"""
gft/embedding.py — GFT embedding ve denoising

Normalize Laplacian özvektörleri ile:
  1. Spektral embedding (boyut indirgeme)
  2. Inverse GFT denoising (X̂ = U U^T X)
  3. Adaptive k_eig seçimi (eigengap)
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh


def normalized_laplacian(W: csr_matrix) -> csr_matrix:
    """L_sym = I - D^{-1/2} W D^{-1/2}"""
    N = W.shape[0]
    d = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = diags(1.0 / np.sqrt(d + 1e-10))
    return eye(N, format='csr') - D_inv_sqrt @ W @ D_inv_sqrt


def gft_embed(W: csr_matrix, k_eig: int = 5,
              tol: float = 1e-4, maxiter: int = 10000,
              random_state: int = 42) -> np.ndarray:
    """
    Graf Fourier Transform — düşük frekanslı özvektörler.

    Parametreler
    -----------
    W            : (N, N) simetrik ağırlık matrisi
    k_eig        : özvektör sayısı (varsayılan: 5)
    tol          : ARPACK toleransı
    maxiter      : maksimum iterasyon
    random_state : eigsh'in v0 başlangıç vektörü için sabit tohum. eigsh varsayılan
                   olarak rastgele bir v0 kullanır; yakın/dejenere özdeğerlerde bu,
                   aynı W için çalıştırmalar arası tamamen farklı (ama matematiksel
                   olarak "geçerli") özvektör kombinasyonlarına yol açabilir. Sabit
                   bir v0 bu embedding'i tekrarlanabilir kılar.

    Döndürür
    --------
    U : (N, k_eig) spektral embedding matrisi
    """
    L = normalized_laplacian(W)
    rng = np.random.RandomState(random_state)
    v0 = rng.rand(L.shape[0])
    _, vecs = eigsh(L, k=k_eig + 1, which='SM', tol=tol, maxiter=maxiter, v0=v0)
    return vecs[:, 1:]  # trivial özvektörü at


def gft_denoise(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Inverse GFT denoising: X̂ = U U^T X

    Düşük frekanslı bileşenlere projeksiyon yaparak
    teknik gürültüyü (dropout) bastırır.

    Parametreler
    -----------
    X : (N, D) orijinal özellik matrisi (PCA veya ham)
    U : (N, k) GFT özvektörleri — gft_embed() çıktısı

    Döndürür
    --------
    X_hat : (N, D) denoised matris
    """
    return U @ (U.T @ X)


def get_eigenvalues(W: csr_matrix, k_max: int = 30) -> np.ndarray:
    """İlk k_max eigenvalue'yu döndür (λ₂'den başlar)."""
    L = normalized_laplacian(W)
    vals, _ = eigsh(L, k=k_max + 1, which='SM', tol=1e-3, maxiter=5000)
    return np.sort(vals)[1:]


def adaptive_k_eig(W: csr_matrix, k_max: int = 30,
                   method: str = 'eigengap') -> int:
    """
    Otomatik k_eig seçimi.

    Yöntemler:
      'eigengap' : En büyük λ_{i+1} - λ_i farkı
      'energy80' : Kümülatif spektral enerjinin %80'i
      'energy90' : Kümülatif spektral enerjinin %90'ı

    NOT: Deneysel bulgularımızda k=5 her iki yöntemden de
    üstün performans gösterdi. Bu fonksiyon araştırma amaçlıdır.
    """
    eigs = get_eigenvalues(W, k_max=k_max)

    if method == 'eigengap':
        gaps = np.diff(eigs)
        return int(np.argmax(gaps[:20]) + 1)
    elif method in ('energy80', 'energy90'):
        threshold = 0.8 if method == 'energy80' else 0.9
        cumsum = np.cumsum(eigs)
        return int(np.argmax(cumsum >= threshold * cumsum[-1]) + 1)
    else:
        raise ValueError(f"Bilinmeyen method: {method}")


def algebraic_connectivity(W: csr_matrix) -> float:
    """
    λ₂ — Fiedler değeri (Algebraic Connectivity).
    N büyüdükçe artar; GFT performansıyla korelasyon gösterir.
    """
    L = normalized_laplacian(W)
    vals, _ = eigsh(L, k=3, which='SM', tol=1e-3, maxiter=3000)
    return float(sorted(vals)[1])
