from typing import Dict, List, Tuple
import numpy as np
from .config_loader import load_config
from .utils import cosine

Edge = Tuple[str, str, float]  # (gene_u, gene_v, weight)

def build_knn_edges(embeds: Dict[str, np.ndarray]) -> List[Edge]:
    cfg = load_config()
    params = cfg["ml"]["grn_model"]
    top_k = int(params.get("top_k", 3))
    min_cos = float(params.get("min_cosine", 0.25))

    genes = list(embeds.keys())
    X = np.stack([embeds[g] for g in genes], axis=0)
    # normalize once
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -1.0)  # ignore self
    edges: List[Edge] = []
    for i, gi in enumerate(genes):
        # top-k neighbors by cosine
        idx = np.argsort(-sims[i])[:top_k]
        for j in idx:
            w = float(sims[i, j])
            if w >= min_cos:
                gj = genes[j]
                # make undirected unique: order by name
                u, v = sorted([gi, gj])
                edges.append((u, v, w))

    # deduplicate undirected edges by smallest name pair
    uniq = {}
    for u, v, w in edges:
        key = (u, v)
        if key not in uniq or w > uniq[key]:
            uniq[key] = w
    return [(u, v, w) for (u, v), w in uniq.items()]
