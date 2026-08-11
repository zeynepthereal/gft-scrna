"""
gft — Graph Fourier Transform for scRNA-seq Analysis

Temel kullanım:
    from gft.graph import build_graph
    from gft.embedding import gft_embed, gft_denoise
    from gft.metrics import ari_multi_seed, heterogeneity_score, predict_gft_advantage

    W   = build_graph(pca_embedding, k_nn=20, method='cosine')
    emb = gft_embed(W, k_eig=5)
"""

from .graph import build_graph
from .embedding import gft_embed, gft_denoise, adaptive_k_eig
from .metrics import (ari_multi_seed, heterogeneity_score,
                      predict_gft_advantage, supervised_metrics)

__version__ = "0.1.0"
__all__ = [
    "build_graph",
    "gft_embed",
    "gft_denoise",
    "adaptive_k_eig",
    "ari_multi_seed",
    "heterogeneity_score",
    "predict_gft_advantage",
    "supervised_metrics",
]
