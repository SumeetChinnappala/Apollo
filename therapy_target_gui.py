# ===============================================================
# GRN Explorer (LDL GWAS) — Streamlit one-shot app
# - Upload GWAS associations or summary-stats (TSV / TSV.GZ)
# - Auto-detect gene / p-value / study columns
# - Build a lightweight GRN from gene×study signal correlations
# - Interactive 3D force-directed network (Plotly)
# - Plots: target landscape, hub bars, degree histogram
# - Model explainability: SHAP for a small tree model (fallback to permutation importance)
# ===============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# ---------- Page setup ----------
st.set_page_config(page_title="3D GRN Explorer — LDL GWAS", layout="wide")
st.title("3D GRN Explorer — LDL Cholesterol GWAS")
st.write("Upload a GWAS **associations** TSV or **summary-stats** TSV/TSV.GZ. "
         "This app builds a lightweight gene regulatory network by correlating gene signals across studies, "
         "then renders an interactive **3D force-directed** network with hub analytics and explainability.")

# ---------- Helpers ----------
def guess_first_match(candidates, columns):
    low = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in low:
            return low[name.lower()]
    return None

def safe_neglog10_p(series):
    x = pd.to_numeric(series, errors="coerce")
    x = x.clip(lower=1e-300)
    return -np.log10(x)

def first_gene_token(s):
    if pd.isna(s): return np.nan
    s = str(s)
    for splitter in ["|", ",", ";"]:
        if splitter in s:
            parts = [t for t in s.split(splitter) if t.strip()]
            return parts[0].strip() if parts else s.strip()
    return s.strip()

def spring_layout_3d(G, seed=42):
    # NetworkX spring_layout with 3D embedding (Fruchterman-Reingold)
    pos = nx.spring_layout(G, dim=3, seed=seed)
    # Convert to numpy arrays for plotting
    xyz = np.array([pos[n] for n in G.nodes()])
    return {n: xyz[i] for i, n in enumerate(G.nodes())}

def plotly_3d_graph(G, pos3d, node_color_attr=None, node_size_attr=None, title="3D GRN"):
    # Build edge coordinates
    xe, ye, ze = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos3d[u]
        x1, y1, z1 = pos3d[v]
        xe += [x0, x1, None]
        ye += [y0, y1, None]
        ze += [z0, z1, None]

    # Node positions
    xn = [pos3d[n][0] for n in G.nodes()]
    yn = [pos3d[n][1] for n in G.nodes()]
    zn = [pos3d[n][2] for n in G.nodes()]

    # Node colors/sizes
    if node_color_attr is None:
        colors = None
        showscale = False
    else:
        colors = [G.nodes[n].get(node_color_attr, 0.0) for n in G.nodes()]
        showscale = True

    if node_size_attr is None:
        sizes = [8] * G.number_of_nodes()
    else:
        base = np.array([G.nodes[n].get(node_size_attr, 1.0) for n in G.nodes()])
        # Normalize to reasonable bubble sizes
        if base.max() > 0:
            base = 6 + 10 * (base - base.min()) / (base.max() - base.min() + 1e-12)
        sizes = base.tolist()

    edge_trace = go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode='lines',
        line=dict(width=1),
        hoverinfo='none'
    )
    node_trace = go.Scatter3d(
        x=xn, y=yn, z=zn,
        mode='markers',
        marker=dict(size=sizes, color=colors, opacity=0.9, showscale=showscale),
        text=[str(n) for n in G.nodes()],
        hovertemplate="<b>%{text}</b><extra></extra>"
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=700
    )
    return fig

# ---------- Sidebar: upload & options ----------
with st.sidebar:
    st.header("Upload")
    file = st.file_uploader("Upload GWAS TSV / TSV.GZ", type=["tsv", "gz"], accept_multiple_files=False)
    st.caption("Tip: the **associations** TSV often includes gene+study columns directly "
               "and is smaller than full summary statistics.")

    st.header("Build Options")
    topN = st.slider("Max genes (ranked by aggregated signal)", 50, 600, 250, 25)
    corr_min = st.slider("Edge threshold (|Spearman ρ|)", 0.1, 1.0, 0.5, 0.05)
    max_edges = st.slider("Max edges (cap to avoid hairballs)", 100, 5000, 1200, 100)

if file is None:
    st.info("⬆️ Upload a file to begin. Works with GWAS **associations** TSV or **summary-stats** TSV.GZ.")
    st.stop()

# ---------- Read sample to detect columns ----------
try:
    head = pd.read_csv(file, sep="\t", nrows=8000, low_memory=False)
except Exception as e:
    st.error(f"Could not read TSV header: {e}")
    st.stop()

st.subheader("Detected columns (sample)")
with st.expander("Show"):
    st.write(list(head.columns))

gene_cands = [
    "mapped_gene", "mappedGenes", "mapped_genes", "gene", "genes",
    "nearest_gene", "nearest_genes", "gene_symbol", "gene_symbols", "hgnc_symbol"
]
pval_cands = ["p_value", "pvalue", "pval", "p", "pval_nominal"]
study_cands = ["study_accession", "study", "pmid", "pubmed_id", "efo_trait", "trait"]
variant_cands = ["rsid", "variant_id", "snp", "rs_id", "markername"]

gene_col_guess  = guess_first_match(gene_cands, head.columns)
pval_col_guess  = guess_first_match(pval_cands, head.columns)
study_col_guess = guess_first_match(study_cands, head.columns)
var_col_guess   = guess_first_match(variant_cands, head.columns)

col1, col2, col3, col4 = st.columns(4)
with col1:
    gene_col = st.selectbox("Gene column", options=list(head.columns), index=list(head.columns).index(gene_col_guess) if gene_col_guess in head.columns else 0)
with col2:
    pval_col = st.selectbox("P-value column", options=list(head.columns), index=list(head.columns).index(pval_col_guess) if pval_col_guess in head.columns else 0)
with col3:
    study_col = st.selectbox("Study column (grouping)", options=list(head.columns), index=list(head.columns).index(study_col_guess) if study_col_guess in head.columns else 0,
                              help="A column that identifies studies or cohorts (accession/PMID/trait). Used to build gene×study matrix.")
with col4:
    var_col = st.selectbox("Variant ID (optional)", options=["<None>"] + list(head.columns),
                           index=(list(head.columns).index(var_col_guess)+1) if var_col_guess in head.columns else 0)

# ---------- Load full file ----------
file.seek(0)
try:
    df = pd.read_csv(file, sep="\t", low_memory=True)
except Exception as e:
    st.error(f"Could not read full file: {e}")
    st.stop()

# ---------- Clean & aggregate per (gene, study) ----------
work = df[[gene_col, pval_col, study_col]].copy()
if var_col != "<None>" and var_col in df.columns:
    work[var_col] = df[var_col]

# gene cleaning (take first token for multi-genes)
work["gene"] = work[gene_col].apply(first_gene_token)
work["neglogp"] = safe_neglog10_p(work[pval_col])
work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["gene", "neglogp", study_col])

# Aggregate gene×study (take max neglogp per gene per study as the sentinel signal)
agg = (work.groupby(["gene", study_col])
            .agg(neglogp=("neglogp", "max"),
                 hits=(pval_col, "count"))
            .reset_index())

# Rank genes by overall strength to enforce topN gating
gene_strength = (agg.groupby("gene")
                    .agg(total_signal=("neglogp", "sum"),
                         studies=("neglogp", "count"))
                    .reset_index()
                    .sort_values("total_signal", ascending=False))
keep_genes = set(gene_strength.head(topN)["gene"])
agg = agg[agg["gene"].isin(keep_genes)].copy()

# ---------- Pivot to gene×study matrix and build GRN via correlations ----------
mat = agg.pivot_table(index="gene", columns=study_col, values="neglogp", fill_value=0.0)
genes = mat.index.tolist()

# Spearman correlation across studies (gene vs. gene)
corr = pd.DataFrame(np.nan, index=genes, columns=genes)
vals = mat.values
for i in range(len(genes)):
    rho, _ = spearmanr(vals[i, :], vals, axis=1)
    # spearmanr with a vector vs matrix returns vector of rhos
    corr.iloc[i, :] = rho

# Threshold edges by |rho| and cap total edges
edges = []
for i in range(len(genes)):
    for j in range(i+1, len(genes)):
        r = corr.iat[i, j]
        if pd.notna(r) and abs(r) >= corr_min:
            edges.append((genes[i], genes[j], float(r)))
# Sort by absolute strength and cap
edges = sorted(edges, key=lambda e: abs(e[2]), reverse=True)[:max_edges]

# ---------- Build graph and compute node features ----------
G = nx.Graph()
G.add_nodes_from(genes)
for u, v, w in edges:
    G.add_edge(u, v, weight=w, confidence=abs(w))

# Node features: degree, weighted degree, centralities
deg = dict(G.degree())
wdeg = {n: sum(abs(G[n][nbr]["weight"]) for nbr in G[n]) for n in G.nodes()}
btw = nx.betweenness_centrality(G, normalized=True) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
close = nx.closeness_centrality(G) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}

# Merge with gene-level scores (total_signal, studies)
features = pd.DataFrame({
    "gene": list(G.nodes()),
    "degree": [deg[g] for g in G.nodes()],
    "wdegree": [wdeg[g] for g in G.nodes()],
    "betweenness": [btw[g] for g in G.nodes()],
    "closeness": [close[g] for g in G.nodes()],
})

# Ensure both 'gene' columns are strings before merging
features['gene'] = features['gene'].astype(str)
gene_strength['gene'] = gene_strength['gene'].astype(str)

# Now merge
features = features.merge(
    gene_strength.rename(columns={"studies": "study_hits"}),
    on="gene",
    how="left"
)

# Define a simple target score for ranking (trans-dominance proxy)
features["trans_dominance_score"] = features["total_signal"] * np.maximum(features["degree"], 1)

# ---------- LAYOUT & 3D PLOT ----------
st.subheader("3D Force-Directed GRN")
pos3d = spring_layout_3d(G, seed=42)
# set node attributes used by plot
for _, row in features.iterrows():
    g = row["gene"]
    nx.set_node_attributes(G, {g: {"score": row["trans_dominance_score"], "degree": int(row["degree"])}})

fig3d = plotly_3d_graph(
    G, pos3d,
    node_color_attr="score",   # color by trans-dominance score
    node_size_attr="degree",   # size by degree (hubness)
    title="LDL GRN — force-directed (color=score, size=degree)"
)
st.plotly_chart(fig3d, use_container_width=True)

# ---------- Plots: landscape, hubs, degree histogram ----------
st.subheader("Target Landscape & Hubs")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    # Landscape: total_signal vs score
    import plotly.express as px
    fig_land = px.scatter(
        features, x="total_signal", y="trans_dominance_score",
        hover_name="gene", size="degree", title="Landscape: total_signal vs trans_dominance_score"
    )
    st.plotly_chart(fig_land, use_container_width=True)

with c2:
    top_hubs = features.sort_values("degree", ascending=False).head(20)
    fig_hubs = go.Figure(data=[go.Bar(x=top_hubs["gene"], y=top_hubs["degree"])])
    fig_hubs.update_layout(title="Top hubs (degree)", xaxis_tickangle=-45, height=350, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_hubs, use_container_width=True)

with c3:
    deg_hist = np.bincount(features["degree"].astype(int))
    fig_deg = go.Figure(data=[go.Bar(x=list(range(len(deg_hist))), y=deg_hist)])
    fig_deg.update_layout(title="Degree histogram", xaxis_title="Degree", height=350, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_deg, use_container_width=True)

# ---------- Explainability: SHAP (fallback to permutation importance) ----------
st.subheader("Explainability: What drives the trans-dominance score?")
X = features[["degree", "wdegree", "betweenness", "closeness", "total_signal", "study_hits"]].fillna(0.0)
y = features["trans_dominance_score"].values

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

explainer_mode = "shap"
try:
    import shap  # optional dependency
    try:
        # For tree models, TreeExplainer is efficient
        expl = shap.TreeExplainer(model)
        shap_values = expl.shap_values(X)
        st.write("SHAP summary (features ↑ increase score):")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap_fig = shap.summary_plot(shap_values, X, show=False)
        st.pyplot(shap_fig)
    except Exception as _inner:
        raise
except Exception:
    explainer_mode = "permutation"
    st.info("SHAP not available or failed — falling back to permutation importance.")
    pi = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
    fig_imp = go.Figure(data=[go.Bar(x=imp["feature"], y=imp["importance"])])
    fig_imp.update_layout(title="Permutation importance (approx. feature influence)", height=350, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

# ---------- Download ----------
st.subheader("Download Tables")
st.download_button("Download node features (CSV)", features.to_csv(index=False).encode("utf-8"), "node_features.csv", "text/csv")
edge_df = pd.DataFrame([(u, v, d["confidence"]) for u, v, d in G.edges(data=True)], columns=["source","target","edge_confidence"])
st.download_button("Download edges (CSV)", edge_df.to_csv(index=False).encode("utf-8"), "edges.csv", "text/csv")

with st.expander("Notes & Tips"):
    st.markdown(
        "- **GRN inference here is lightweight**: we correlate per-gene sentinel signals across studies to estimate functional co-regulation. "
        "For heavier methods (CLR/ARACNE/GENIE3), restrict to top genes and subsample studies.\n"
        "- Use **edge threshold** and **max edges** to avoid hairballs; start conservative and dial up as needed.\n"
        "- The **3D view** colors by trans-dominance score and sizes by degree to spotlight hubs that also carry strong evidence."
    )
