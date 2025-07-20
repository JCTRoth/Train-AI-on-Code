#!/usr/bin/env python3

import sys
import os
import torch
from torch.utils.data import DataLoader
import language_filter_lists
import data_loader
import logger
from objects import ClassDataset

# Configure logging
logger.config_logger()

# Load data
list_of_files = data_loader.load_dataset_as_list(
    input_dir="/home/jonas/Git/Train-AI-on-Code/test_data",
    removeToGetRelativePath="/home/jonas/Git",
    listOfFilePostFixes=language_filter_lists.csharp_postfixes
)

# Create dataset
dataset = ClassDataset(inputDataList=list_of_files)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1)

# Print dataset structure
print(f"Dataset length: {len(dataset)}")
print(f"First item type: {type(dataset[0])}")
print(f"First item keys: {dataset[0].keys()}")
print(f"First item shape: {dataset[0]['input_ids'].shape}")

# Print batch structure
for batch_idx, batch in enumerate(dataloader):
    print(f"\nBatch {batch_idx} type: {type(batch)}")
    print(f"Batch {batch_idx} keys: {batch.keys()}")
    for k, v in batch.items():
        print(f"Key: {k}, Type: {type(v)}, Shape: {v.shape}")
    
    if batch_idx >= 2:  # Just check the first 3 batches
        break
