import sys
import torch
import numpy as np
import scanpy as sc
from torch_geometric.data import HeteroData


def downsample_adata(adata, frac=0.1):
    """
    Downsample the AnnData object by selecting a fraction of cells.

    Args:
        adata (AnnData): The AnnData object.
        frac (float): Fraction of cells to select.

    Returns:
        adata_downsampled (AnnData): The downsampled AnnData object.
    """
    n_samples = int(adata.n_obs * frac)
    indices = np.random.choice(adata.n_obs, size=n_samples, replace=False)
    return adata[indices].copy()


def check_normalized(adata, threshold=1.01):
    """
    Check if the AnnData object is normalized and normalize if needed.

    Args:
        adata (AnnData): The AnnData object.
        threshold (float): Maximum allowed value after normalization.

    Raises:
        ValueError: If the maximum value after normalization is above the threshold.
    """
    adata.layers["raw"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1, inplace=True)
    max_norm = adata.X.max()
    if max_norm > threshold:
        raise ValueError(f"Data is not normalized. Maximum value after normalization: {max_norm}")
    adata.X = adata.layers["raw"]


def construct_edge_index(adata):
    """
    Construct an edge index for a given AnnData object.

    Args:
        adata (AnnData): The AnnData object.

    Returns:
        edge_index (torch.Tensor): The edge index tensor.
    """
    cells, genes = np.nonzero(adata.X)
    edge_index = torch.tensor([cells, genes + adata.X.shape[0]], dtype=torch.long)
    return edge_index


def get_cell_type_annotations(adata):
    """
    Get cell type annotations from the AnnData object.

    Args:
        adata (AnnData): The AnnData object.

    Returns:
        cell_types (list): List of cell type annotations.
    """
    annotations = adata.obs["cell_ontology_class"]
    categories = annotations.cat.categories
    cell_types = [categories.get_loc(a) for a in annotations]
    return cell_types

def create_indices(num_cells, train_frac=0.8, val_frac=0.1):
    """
    Create train, validation, and test indices for an AnnData object.

    Args:
        num_cells (int): Number of cells in the dataset.
        train_frac (float): Fraction of cells for training.
        val_frac (float): Fraction of cells for validation.

    Returns:
        train_indices (numpy.ndarray): Array of training indices.
        val_indices (numpy.ndarray): Array of validation indices.
        test_indices (numpy.ndarray): Array of test indices.
    """
    indices = np.random.permutation(num_cells)
    
    train_size = int(num_cells * train_frac)
    val_size = int(num_cells * val_frac)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

def h5ad_to_pyg_data(adata, downsample=False):
    """
    Convert an AnnData object to PyTorch Geometric HeteroData objects for training, validation, and testing.

    Args:
        adata (AnnData): The AnnData object.
        downsample (bool): Whether to downsample the dataset.

    Returns:
        train_data (HeteroData), val_data (HeteroData), test_data (HeteroData): PyTorch Geometric HeteroData objects.
    """

    # Downsample the dataset if requested
    if downsample:
        adata = downsample_adata(adata, frac=0.1)
    print(f'AnnData Size (GB): {sys.getsizeof(adata) / (1024 ** 3)}')

    # Check if the data is normalized
    check_normalized(adata)

    # Create the train, validation, and test indices
    train_indices, val_indices, test_indices = create_indices(adata.n_obs)

    def create_hetero_data(indices):
        """
        Create a PyTorch Geometric HeteroData object from the given indices of an AnnData object.

        Args:
            indices (numpy.ndarray): Indices of the cells in the AnnData object.

        Returns:
            data (HeteroData): A PyTorch Geometric HeteroData object.
        """
        data = HeteroData()

        # Set the node features
        data['cell'].x = torch.tensor(adata[indices].X.toarray(), dtype=torch.float)  # Convert sparse matrix to dense numpy array
        data['gene'].x = torch.tensor(adata.var['n_cells'].values.reshape(-1, 1), dtype=torch.float)

        # Set the edge index
        data['cell', 'expr', 'gene'].edge_index = construct_edge_index(adata[indices])

        # Set the cell type annotations
        data['cell'].y = torch.tensor(get_cell_type_annotations(adata[indices]), dtype=torch.long)

        return data

    train_dataset = create_hetero_data(train_indices)
    val_dataset = create_hetero_data(val_indices)
    test_dataset = create_hetero_data(test_indices)

    return train_dataset, val_dataset, test_dataset