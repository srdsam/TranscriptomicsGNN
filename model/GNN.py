import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from pytorch_lightning import LightningModule

class HeteroGNN(LightningModule):
    def __init__(self, hidden_channels, num_classes, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.gnn = GNN(hidden_channels, num_classes)
        self.loss_fn = F.cross_entropy

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)

    def training_step(self, batch, batch_idx):
        out = self(batch.x_dict, batch.edge_index_dict)
        mask = batch['cell'].train_mask
        loss = self.loss_fn(out['cell'][mask], batch['cell'].y[mask])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x_dict, batch.edge_index_dict)
        mask = batch['cell'].val_mask
        loss = self.loss_fn(out['cell'][mask], batch['cell'].y[mask])
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), num_classes)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {k: self.conv1(x, edge_index_dict).relu() for k, x in x_dict.items()}
        x_dict = {k: self.conv2(x, edge_index_dict) for k, x in x_dict.items()}
        return x_dict

# Usage
# model = HeteroGNN(hidden_channels=64, num_classes=num_cell_types, learning_rate=1e-3)
# model.gnn = to_hetero(model.gnn, dataset[0].metadata(), aggr='sum')
