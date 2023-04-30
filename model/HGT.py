import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear, HGTConv

class HGT(pl.LightningModule):
    """
    HGT (Heterogeneous Graph Transformer) class implementation.
    This class inherits from PyTorch Lightning's LightningModule.
    It implements a graph neural network with the HGTConv layer for heterogeneous graphs.
    """
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        """
        Initialize the HGT model.

        Parameters:
        - hidden_channels (int): The number of hidden channels in the HGTConv layers.
        - out_channels (int): The number of output channels in the final linear layer.
        - num_heads (int): The number of attention heads in the HGTConv layers.
        - num_layers (int): The number of HGTConv layers in the model.
        """
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        # Define a dictionary of linear layers for each node type
        for node_type in ['cell', 'gene']:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        
        # Define a list of HGTConv layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass for the HGT model.

        Parameters:
        - x_dict (Dict[str, Tensor]): A dictionary of node features for each node type.
        - edge_index_dict (Dict[Tuple[str, str], Tensor]): A dictionary of edge indices for each pair of node types.

        Returns:
        - Tensor: The output of the HGT model for the target node type.
        """
        # Apply the initial linear layers to the input features of each node type
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        # Apply the HGTConv layers sequentially
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # Apply the final linear layer to the output of the last HGTConv layer
        return self.lin(x_dict[target])

    def training_step(self, batch, batch_idx):
        """
        Define the training step for the HGT model.

        Parameters:
        - batch: The current batch of data.
        - batch_idx (int): The index of the current batch.

        Returns:
        - Tensor: The training loss.
        """
        x_dict, edge_index_dict, y = batch
        y_hat = self(x_dict, edge_index_dict)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Define the validation step for the HGT model.

        Parameters:
        - batch: The current batch of data.
        - batch_idx (int): The index of the current batch.

        Returns:
        - Dict[str, Tensor]: The validation loss.
        """
        x_dict, edge_index_dict, y = batch
        y_hat = self(x_dict, edge_index_dict)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return {'val_loss': loss}

    def configure_optimizers(self):
        """
        Configure the optimizer for the HGT model.

        Returns:
        - Optimizer: The optimizer to be used for training the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
