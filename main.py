import yaml
import argparse
import scanpy as sc

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch_geometric.nn import to_hetero

from data.CellGeneDataset import CellGeneDataset, CellGeneDataModule
from model.HGT import HGT
from model.GNN import HeteroGNN
from analysis.IdentifyGeneMarkers import extract_gene_embeddings, compute_importance_scores, get_top_genes_per_cell_type


def load_config(config_path):
    """
    Load the configuration from the YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        config (dict): Dictionary containing the configuration parameters.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main(config_path):
    """
    Main function to train and evaluate the HGT model.

    Args:
        config_path (str): Path to the configuration file.
    """
    # Load configuration
    config = load_config(config_path)

    # Load Data
    file_path = config["data"]["input_path"]
    root_path = config["data"]["root_path"]
    downsample = config["data"]["downsample"]
    adata = sc.read_h5ad(file_path)
    dataset = CellGeneDataset(root=root_path, h5ad_file=file_path, threshold=0.5, downsample=downsample)
    dataset = dataset.shuffle()
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    val_dataset = dataset[int(len(dataset) * 0.8):]
    data_module = CellGeneDataModule(train_dataset, val_dataset)

    # Initialize Model
    out_channels = train_dataset.get_num_cell_types() # number of annotated cell types
    if config["model"]["architecture"] == 'HGT': 
        hidden_channels = config["model"]["hidden_channels"]
        num_layers = config["model"]["num_layers"]
        num_heads = config["model"]["num_heads"]
        model = HGT(hidden_channels, out_channels, num_heads, num_layers, train_dataset.metadata())
    elif config["model"]["architecture"] == 'GNN': 
        hidden_channels = config["model"]["hidden_channels"]
        model = HeteroGNN(hidden_channels=64, num_classes=out_channels, learning_rate=1e-3)
        model.gnn = to_hetero(model.gnn, dataset[0].metadata(), aggr='sum')
    else:
        raise ValueError("Please specify a model architecture")

    # Set up callbacks
    callbacks = []
    if config["training"]["early_stopping"]:
        early_stopping = EarlyStopping(monitor="val_loss", patience=config["training"]["patience"])
        callbacks.append(early_stopping)

    # Set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='HGT-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    callbacks.append(checkpoint_callback)

    # Set up logger
    wandb_logger = WandbLogger(project="graph-transcriptomics")
    
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        logger=[wandb_logger],
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Evaluate top genes per celltype
    gene_embeddings = extract_gene_embeddings(model)
    gene_importance_scores = compute_importance_scores(gene_embeddings)
    top_50_genes_per_cell_type = get_top_genes_per_cell_type(gene_importance_scores, adata, n_genes=50)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    # Run the main function with the specified configuration file
    main(args.config)
