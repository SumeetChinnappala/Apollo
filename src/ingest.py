from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from .config_loader import load_config

def _stable_vec(text: str, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.standard_normal(dim).astype(np.float32)

def load_gene_table() -> pd.DataFrame:
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    csv_path = raw_dir / "sample_genes.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path} — add your CSV there.")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "gene_symbol" not in df.columns:
        raise ValueError("CSV must have a 'gene_symbol' column.")
    df["gene_symbol"] = df["gene_symbol"].astype(str).str.strip().str.upper()
    df["description"] = df.get("description", "").astype(str).fillna("")
    return df

def embed_genes(df: pd.DataFrame, dim: int = 64) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    embed_dim = dim
    out: Dict[str, np.ndarray] = {}
    for g, desc in zip(df["gene_symbol"], df["description"]):
        text = f"{g} :: {desc}" if isinstance(desc, str) and desc else g
        out[g] = _stable_vec(text, dim=embed_dim)
    return df.copy(), out
