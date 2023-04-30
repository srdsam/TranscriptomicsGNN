import yaml
import argparse
import scanpy as sc

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch_geometric.data.lightning import LightningDataset

from data.data import h5ad_to_pyg_data
from model.HGT import HGT


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

    DATA = config["data"]["input_path"]

    # Read the dataset using scanpy
    file = sc.read_h5ad(DATA)

    # Convert the dataset to PyTorch Geometric format
    train_dataset, val_dataset, test_dataset = h5ad_to_pyg_data(file, downsample=True)

    # Create LightningDataset for training, validation, and testing
    dataset = LightningDataset(
        train_dataset,
        test_dataset,
        val_dataset,
        # num_neighbors=config["data"]["num_neighbors"],
        # num_workers=config["data"]["num_workers"],
    )

    # Initialize the HGT model with the specified configuration
    hidden_channels = config["model"]["hidden_channels"]
    out_channels = len(train_dataset["cell"].y.unique())
    num_layers = config["model"]["num_layers"]
    num_heads = config["model"]["num_heads"]
    model = HGT(hidden_channels, out_channels, num_heads, num_layers, train_dataset.metadata())

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

    # Set up loggers
    wandb_logger = WandbLogger(project="graph-transcriptomics")
    
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        logger=[wandb_logger],
    )

    # Train the model
    trainer.fit(model, datamodule=dataset)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    # Run the main function with the specified configuration file
    main(args.config)
