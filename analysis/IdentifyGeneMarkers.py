import torch
import numpy as np
import pandas as pd

def extract_gene_embeddings(model):
    model.eval()
    x_dict = {'gene': model.data['gene'].x}
    edge_index_dict = model.data.edge_index_dict
    with torch.no_grad():
        embeddings = model(x_dict, edge_index_dict)
    return embeddings['gene'].numpy()

def compute_importance_scores(embeddings):
    return np.linalg.norm(embeddings, axis=1)

def get_top_genes_per_cell_type(importance_scores, adata, n_genes=50):
    cell_types = adata.obs['cell_ontology_class'].unique()
    top_genes_per_cell_type = {}

    # Create a DataFrame containing the gene importance scores
    gene_scores_df = pd.DataFrame(
        {"gene": adata.var_names, "importance_score": importance_scores}
    )

    for cell_type in cell_types:
        # Filter genes expressed in the current cell type
        cell_type_adata = adata[adata.obs['cell_ontology_class'] == cell_type]

        # Get the gene indices that are expressed in the current cell type
        expressed_genes_indices = np.where(cell_type_adata.X > 0.5)

        # Filter the gene_scores_df for the expressed genes in the current cell type
        cell_type_gene_scores_df = gene_scores_df.loc[expressed_genes_indices[1]]

        # Sort the DataFrame by the importance scores in descending order
        cell_type_gene_scores_df = cell_type_gene_scores_df.sort_values(
            by="importance_score", ascending=False
        )

        # Get the top `n_genes` genes for the current cell type
        top_genes = cell_type_gene_scores_df.head(n_genes)["gene"].tolist()

        # Store the top genes for the current cell type
        top_genes_per_cell_type[cell_type] = top_genes

    return top_genes_per_cell_type

# Usage
# gene_embeddings = extract_gene_embeddings(model)
# gene_importance_scores = compute_importance_scores(gene_embeddings)
# top_50_genes_per_cell_type = get_top_genes_per_cell_type(gene_importance_scores, adata, n_genes=50)
