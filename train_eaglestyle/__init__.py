"""
MTP Head Training Pipeline

A complete implementation of the MTP (Multi-Token Prediction) head training pipeline,
following the architecture and training strategy from pipeline.md.

Modules:
    - data_collection: Collect hidden states and tokens from base model
    - mtp_dataset: Load and prepare data for training
    - train_mtp_head: Main training script
    - mtphead_trainer: Custom trainer with diffusion loss
    - schedulers: Alpha scheduling for diffusion timesteps

Quick start:
    1. python data_collection.py --base_model_path ... --input_data_path ...
    2. python train_mtp_head.py --train_data_dir ... --output_dir ...
"""

__version__ = "0.1.0"
__author__ = "MTP Training Team"

from .mtp_dataset import (
    MTPHeadDataset,
    MTPHeadCollator,
    create_dataloaders,
)
from .mtphead_trainer import MDLMTrainer

__all__ = [
    "MTPHeadDataset",
    "MTPHeadCollator",
    "create_dataloaders",
    "MDLMTrainer",
]
