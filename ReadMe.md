# Heterogeneous Graph Transformer for Single-Cell Data

This project demonstrates how to use a Heterogeneous Graph Transformer (HGT) model to analyze single-cell RNA-seq data. It is built using PyTorch, PyTorch Geometric, and PyTorch Lightning.

## Project Structure

The project is organized as follows:

- `main.py`: Main script to load the data, set up the model, and train the model.
- `data`: Class to load and preprocess the single-cell data using Scanpy and PyTorch.
- `model`: Definition of the models.
- `config`: Configuration file containing data paths, model settings, and training parameters.

## Setup

1. Create a virtual environment and activate it:

```bash
conda activate gnn
```
2. Install the required packages from `requirements.txt`.
3. Prepare the dataset by downloading it and placing it in the appropriate directory. Update the config.yaml file with the correct data path.

## Usage
Train the HGT model:
```
python main.py --config config.yaml
```

## Configuration
You can adjust the settings in the config.yaml file to customize the training process, including:

- Data input path
- Model hyperparameters (e.g., hidden channels, number of layers, number of heads)
- Training parameters (e.g., number of epochs, early stopping)

## Logging and Performance Tracking
This project uses PyTorch Lightning's built-in logging functionality to track the model's performance. By default, it logs to Weights & Biases. To view the logs, use the following commands:

Weights & Biases:
Create an account and set up the API key. The logged data will appear on your project's dashboard.
