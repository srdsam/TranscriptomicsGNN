#!/bin/bash

#SBATCH --job-name=training_gnn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load necessary modules
module load anaconda

# Activate your virtual environment
source /home/samuel.dsouza/.conda/envs/gnn

# Run your main.py with the configuration file as an argument
python main.py --config config/config.yaml

# Deactivate the virtual environment
deactivate
