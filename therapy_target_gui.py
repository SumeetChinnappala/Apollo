import pandas as pd
import streamlit as st
import gzip

# === 1. CONFIGURATION ===
# Change this to the full path of your downloaded file
GWAS_FILE = r"C:\Users\Owner\Documents\startup\GCST90239001_buildGRCh38.gz"


# === 2. LOAD DATA ===
st.title("Therapy Target Finder - Cardiovascular & Metabolic Disorders")
st.write("Rank genes by trans-dominance score using GWAS summary statistics.")


with gzip.open(GWAS_FILE, 'rt') as f:
    df = pd.read_csv(f, sep="\t")

# Keep only relevant columns (adjust if column names differ)
columns_needed = ['variant_id', 'effect_allele', 'mapped_gene', 'p_value', 'beta']
df = df[columns_needed].dropna()

# === 3. PROCESSING ===
# Calculate -log10(p-value) for weighting
df['neg_log_p'] = -df['p_value'].apply(lambda x: pd.np.log10(x) * -1)

# Group by gene and calculate counts + average neg_log_p
gene_stats = df.groupby('mapped_gene').agg(
    variant_count=('variant_id', 'nunique'),
    mean_neg_log_p=('neg_log_p', 'mean')
).reset_index()

# Trans-dominance score = count * mean_neg_log_p
gene_stats['trans_dominance_score'] = gene_stats['variant_count'] * gene_stats['mean_neg_log_p']

# Sort by score
gene_stats = gene_stats.sort_values('trans_dominance_score', ascending=False)

# === 4. WEB GUI ===
min_score = st.slider("Minimum Trans-Dominance Score", 0.0, float(gene_stats['trans_dominance_score'].max()), 0.0)
search_gene = st.text_input("Search for a gene (case-insensitive)")

filtered = gene_stats[gene_stats['trans_dominance_score'] >= min_score]
if search_gene:
    filtered = filtered[filtered['mapped_gene'].str.contains(search_gene, case=False, na=False)]

st.dataframe(filtered)

# === 5. EXPORT ===
csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download results as CSV",
    data=csv,
    file_name="ranked_targets.csv",
    mime="text/csv",
)

# === 6. HOW TO RUN ===
st.write("**To run this app:** In Anaconda Prompt, navigate to this script's folder and run:")
st.code("streamlit run this_script.py")
