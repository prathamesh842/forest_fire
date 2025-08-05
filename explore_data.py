"""
Exploratory Data Analysis for wildfire spread prediction dataset.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import WildfireDataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_dataset():
    """Perform exploratory data analysis on the wildfire dataset."""
    loader = WildfireDataLoader()
    
    # Get dataset information
    print("=" * 50)
    print("WILDFIRE DATASET EXPLORATION")
    print("=" * 50)
    
    info = loader.get_dataset_info()
    total_size = sum(data['total_size_mb'] for data in info.values())
    total_files = sum(data['num_files'] for data in info.values())
    
    print(f"Total Dataset Size: {total_size:.1f} MB")
    print(f"Total Files: {total_files}")
    print()
    
    for split, data in info.items():
        print(f"{split.upper()} SET:")
        print(f"  Files: {data['num_files']}")
        print(f"  Size: {data['total_size_mb']:.1f} MB")
        print(f"  Avg file size: {data['total_size_mb']/data['num_files']:.1f} MB")
        print()
    
    # Inspect data structure
    print("=" * 50)
    print("DATA STRUCTURE ANALYSIS")
    print("=" * 50)
    
    train_files = loader.get_file_paths("train")
    if train_files:
        print(f"Analyzing structure of: {train_files[0]}")
        example = loader.inspect_tfrecord(train_files[0], num_examples=3)
        
        # Try to understand the features
        print("\nFeature Analysis:")
        for key, feature in example.features.feature.items():
            if feature.HasField('bytes_list'):
                print(f"  {key}: bytes feature")
            elif feature.HasField('float_list'):
                values = list(feature.float_list.value)
                print(f"  {key}: float feature (shape: {len(values)})")
                if len(values) <= 10:
                    print(f"    Sample values: {values}")
                else:
                    print(f"    Sample values: {values[:5]}...{values[-5:]}")
            elif feature.HasField('int64_list'):
                values = list(feature.int64_list.value)
                print(f"  {key}: int64 feature (shape: {len(values)})")
                if len(values) <= 10:
                    print(f"    Sample values: {values}")
                else:
                    print(f"    Sample values: {values[:5]}...{values[-5:]}")

def visualize_data_distribution():
    """Visualize data distribution across splits."""
    loader = WildfireDataLoader()
    info = loader.get_dataset_info()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # File count distribution
    splits = list(info.keys())
    file_counts = [info[split]['num_files'] for split in splits]
    sizes = [info[split]['total_size_mb'] for split in splits]
    
    ax1.bar(splits, file_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Number of Files per Split')
    ax1.set_ylabel('Number of Files')
    
    # Size distribution
    ax2.bar(splits, sizes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Data Size per Split (MB)')
    ax2.set_ylabel('Size (MB)')
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Data distribution visualization saved as 'data_distribution.png'")

if __name__ == "__main__":
    explore_dataset()
    visualize_data_distribution()
