from pathlib import Path
from typing import List, Tuple, Set
import pandas as pd
from .config_loader import load_config

Edge = Tuple[str, str, float]

def _load_ground_truth() -> Set[Tuple[str, str]]:
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    gt_path = raw_dir / "ground_truth_edges.csv"
    if not gt_path.exists():
        # allow the toy filename we created above
        gt_path = raw_dir / "ground_truth_edges.csv"
    if not gt_path.exists():
        # fallback to the toy file name we created: ground_truth_edges.csv
        raise FileNotFoundError("Missing ground truth CSV in data/raw/ground_truth_edges.csv")

    df = pd.read_csv(gt_path)
    df.columns = [c.strip().lower() for c in df.columns]
    req = {"gene_a", "gene_b"}
    if not req.issubset(df.columns):
        raise ValueError("Ground truth CSV must have columns gene_a,gene_b")

    pairs = set()
    for a, b in zip(df["gene_a"], df["gene_b"]):
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        u, v = sorted([a.strip().upper(), b.strip().upper()])
        pairs.add((u, v))
    return pairs

def evaluate(pred_edges: List[Edge]) -> str:
    gt = _load_ground_truth()
    pred_pairs = {(u, v) for (u, v, _w) in pred_edges}
    hits = pred_pairs & gt
    precision = (len(hits) / max(len(pred_pairs), 1))
    coverage = (len(hits) / max(len(gt), 1))
    lines = [
        "=== Apollo: Edge Evaluation ===",
        f"Predicted edges: {len(pred_pairs)} | Ground-truth edges: {len(gt)} | Hits: {len(hits)}",
        f"Precision: {precision:.3f} | GT coverage: {coverage:.3f}",
        f"Hit list: {sorted(list(hits))}" if hits else "Hit list: (none)"
    ]
    return "\n".join(lines)
