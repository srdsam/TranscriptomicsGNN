import torch
import scanpy as sc
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader

from data.helpers import downsample_adata, check_normalized

class CellGeneDataset(InMemoryDataset):
    def __init__(self, root, h5ad_file, threshold=1, transform=None, pre_transform=None, downsample=False):
        self.h5ad_file = h5ad_file
        self.threshold = threshold
        self.downsample = downsample
        super(CellGeneDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return [self.h5ad_file]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def get_num_cell_types(self):
        return len(self.data["cell"].y.unique().tolist())

    def process(self):
        adata = sc.read_h5ad(self.h5ad_file)
        if self.downsample:
            adata = downsample_adata(adata)

        # Check if the data is normalized
        check_normalized(adata)

        # Create a binary adjacency matrix based on the threshold (count must be higher than threshold, for relationship to matter)
        adj_matrix = (adata.X > self.threshold).astype(int).toarray()  # Convert to dense array

        # Define the number of cell and gene nodes
        n_cells, n_genes = adj_matrix.shape

        # Extract the edges from the adjacency matrix
        cell_nodes, gene_nodes = np.where(adj_matrix == 1)
        edges = torch.tensor(np.array([cell_nodes, gene_nodes]), dtype=torch.long)


        # Extract counts for the edges
        edge_counts = adata.X[cell_nodes, gene_nodes].A1
        edge_counts = torch.tensor(edge_counts, dtype=torch.float).view(-1, 1)

        # Create a PyTorch Geometric HeteroData graph
        graph = HeteroData()

        # Encode cell type labels as integers
        le = LabelEncoder()
        cell_type_int = le.fit_transform(adata.obs['cell_ontology_class'].values)

        # One-hot encode the cell type labels
        ohe = OneHotEncoder(sparse_output=False)
        cell_features = ohe.fit_transform(cell_type_int.reshape(-1, 1))

        # Prepare gene features (constant feature vector)
        gene_features = np.ones((n_genes, 1))

        # Set node features for cell and gene nodes
        graph['cell'].x = torch.tensor(cell_features, dtype=torch.float)
        graph['gene'].x = torch.tensor(gene_features, dtype=torch.float)

        # Set edge index and edge features for the ('cell', 'expresses', 'gene') relation
        graph['cell', 'expresses', 'gene'].edge_index = edges
        graph['cell', 'expresses', 'gene'].edge_attr = edge_counts

        # Set cell type labels for the cell nodes
        graph['cell'].y = torch.tensor(cell_type_int, dtype=torch.long)

        data, slices = self.collate([graph])
        torch.save((data, slices), self.processed_paths[0])

    # def __repr__(self):
    #     return '{}()'.format(self.__class__.__name__)
    
    # def __len__(self):
    #     return len(self.data)

    # def get(self, idx):
    #     return self.data[idx]

class CellGeneDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=32):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Usage
# dataset = CellGeneDataset(h5ad_file='path/to/your/h5ad/file', threshold=0.5)
# dataset = dataset.shuffle()
# train_dataset = dataset[:int(len(dataset) * 0.8)]
# val_dataset = dataset[int(len(dataset) * 0.8):]
# data_module = CellGeneDataModule(train_dataset, val_dataset)
