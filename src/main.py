from pathlib import Path
import csv
from .config_loader import load_config
from .ingest import load_gene_table, embed_genes
from .validate import check_gene_symbols, nearest_neighbor_overlap, summarize_validation
from .edges import build_knn_edges
from .evaluate import evaluate

def _save_edges(edges, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gene_u","gene_v","weight_cosine"])
        for u, v, wgt in sorted(edges, key=lambda x: -x[2]):
            w.writerow([u, v, f"{wgt:.6f}"])

def main():
    cfg = load_config()
    print(f"Apollo Initialized :: {cfg['project']['name']} v{cfg['project']['version']}")

    # Load & embed
    df = load_gene_table()
    df, embeds = embed_genes(df, dim=cfg["ml"]["grn_model"]["embed_dim"])

    # Validate inputs & embeddings
    df = check_gene_symbols(df)
    print(summarize_validation(df, embeds))
    nns = nearest_neighbor_overlap(embeds, topk=2)
    print("Nearest neighbors (top-2):")
    for gene, neighbors in nns:
        print(f"  {gene} -> {neighbors}")

    # Build candidate edges
    edges = build_knn_edges(embeds)
    print(f"Built {len(edges)} candidate edges.")

    # Evaluate vs ground-truth
    eval_summary = evaluate(edges)
    print(eval_summary)

    # Persist outputs
    proc_dir = Path(cfg["data"]["processed_dir"])
    _save_edges(edges, proc_dir / "predicted_edges.csv")
    (Path("reports")).mkdir(exist_ok=True)
    (Path("reports") / "summary.txt").write_text(eval_summary, encoding="utf-8")
    print("Saved: data/processed/predicted_edges.csv and reports/summary.txt")

if __name__ == "__main__":
    main()
