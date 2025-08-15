import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

_GENE_RE = re.compile(r"^[A-Z0-9\-]{2,20}$")

def check_gene_symbols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["symbol_ok"] = df["gene_symbol"].apply(lambda s: bool(_GENE_RE.match(s)))
    return df

def embedding_stats(embeds: Dict[str, np.ndarray]) -> dict:
    if not embeds:
        return {"n": 0}
    mats = np.stack(list(embeds.values()), axis=0)
    norms = np.linalg.norm(mats, axis=1)
    return {
        "n": mats.shape[0],
        "dim": mats.shape[1],
        "mean_norm": float(norms.mean()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
    }

def nearest_neighbor_overlap(embeds: Dict[str, np.ndarray], topk: int = 3) -> List[Tuple[str, List[str]]]:
    if len(embeds) < 2:
        return []
    keys = list(embeds.keys())
    X = np.stack([embeds[k] for k in keys], axis=0)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sims = Xn @ Xn.T
    out = []
    for i, k in enumerate(keys):
        order = np.argsort(-sims[i])
        nn = [keys[j] for j in order if j != i][:topk]
        out.append((k, nn))
    return out

def summarize_validation(df: pd.DataFrame, embeds: Dict[str, np.ndarray]) -> str:
    sym_ok = df["symbol_ok"].sum()
    sym_bad = (~df["symbol_ok"]).sum()
    stats = embedding_stats(embeds)
    lines = [
        "=== Apollo: Validation Summary ===",
        f"Genes total: {len(df)} | Symbols OK: {sym_ok} | Symbols BAD: {sym_bad}",
        f"Embeddings -> n={stats.get('n',0)}, dim={stats.get('dim','?')}, "
        f"mean_norm={stats.get('mean_norm','?'):.3f} (min={stats.get('min_norm','?'):.3f}, max={stats.get('max_norm','?'):.3f})" if stats.get('n',0)>0 else "No embeddings.",
    ]
    if sym_bad > 0:
        bad = df.loc[~df["symbol_ok"], "gene_symbol"].tolist()
        lines.append(f"WARNING: {sym_bad} gene symbols look non-standard: {bad}")
    return "\n".join(lines)
