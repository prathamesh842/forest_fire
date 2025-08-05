"""
Data loader for wildfire spread prediction TFRecord files.
"""

import tensorflow as tf
import numpy as np
import os
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WildfireDataLoader:
    """Data loader for wildfire spread prediction dataset."""
    
    def __init__(self, data_dir: str = "archive"):
        self.data_dir = data_dir
        # Define the feature description for wildfire dataset
        self.feature_description = {
            'NDVI': tf.io.FixedLenFeature([64*64], tf.float32),
            'tmmn': tf.io.FixedLenFeature([64*64], tf.float32),
            'tmmx': tf.io.FixedLenFeature([64*64], tf.float32),
            'pr': tf.io.FixedLenFeature([64*64], tf.float32),
            'sph': tf.io.FixedLenFeature([64*64], tf.float32),
            'vs': tf.io.FixedLenFeature([64*64], tf.float32),
            'th': tf.io.FixedLenFeature([64*64], tf.float32),
            'pdsi': tf.io.FixedLenFeature([64*64], tf.float32),
            'erc': tf.io.FixedLenFeature([64*64], tf.float32),
            'elevation': tf.io.FixedLenFeature([64*64], tf.float32),
            'population': tf.io.FixedLenFeature([64*64], tf.float32),
            'FireMask': tf.io.FixedLenFeature([64*64], tf.float32),
            'PrevFireMask': tf.io.FixedLenFeature([64*64], tf.float32),
        }
        self.feature_names = ['NDVI', 'tmmn', 'tmmx', 'pr', 'sph', 'vs', 'th', 
                             'pdsi', 'erc', 'elevation', 'population', 'PrevFireMask']
        self.target_name = 'FireMask'
        
    def _parse_tfrecord_fn(self, example):
        """Parse a single TFRecord example."""
        # Parse the TFRecord
        parsed = tf.io.parse_single_example(example, self.feature_description)
        
        # Stack input features (12 channels: all except FireMask)
        input_features = []
        for feature_name in self.feature_names:
            feature = tf.reshape(parsed[feature_name], [64, 64, 1])
            input_features.append(feature)
        
        # Combine all input features into a single tensor (64, 64, 12)
        inputs = tf.concat(input_features, axis=-1)
        
        # Get target (FireMask) and reshape to (64, 64, 1)
        target = tf.reshape(parsed[self.target_name], [64, 64, 1])
        
        return inputs, target
    
    def inspect_tfrecord(self, file_path: str, num_examples: int = 1):
        """Inspect TFRecord file to understand data structure."""
        logger.info(f"Inspecting TFRecord file: {file_path}")
        
        dataset = tf.data.TFRecordDataset(file_path)
        
        for i, raw_record in enumerate(dataset.take(num_examples)):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            logger.info(f"Example {i + 1}:")
            for key, feature in example.features.feature.items():
                feature_type = None
                if feature.HasField('bytes_list'):
                    feature_type = 'bytes'
                    value_count = len(feature.bytes_list.value)
                elif feature.HasField('float_list'):
                    feature_type = 'float'
                    value_count = len(feature.float_list.value)
                elif feature.HasField('int64_list'):
                    feature_type = 'int64'
                    value_count = len(feature.int64_list.value)
                
                logger.info(f"  {key}: {feature_type} (count: {value_count})")
                
        return example
    
    def get_file_paths(self, split: str = "train") -> list:
        """Get file paths for a specific split."""
        pattern = f"next_day_wildfire_spread_{split}_*.tfrecord"
        files = []
        
        for filename in os.listdir(self.data_dir):
            if filename.startswith(f"next_day_wildfire_spread_{split}_"):
                files.append(os.path.join(self.data_dir, filename))
        
        files.sort()  # Ensure consistent ordering
        logger.info(f"Found {len(files)} files for {split} split")
        return files
    
    def create_dataset(self, split: str = "train", batch_size: int = 32, 
                      shuffle: bool = True, buffer_size: int = 1000):
        """Create a TensorFlow dataset for training/evaluation."""
        file_paths = self.get_file_paths(split)
        
        if not file_paths:
            raise ValueError(f"No files found for split: {split}")
        
        # Create dataset from multiple TFRecord files
        dataset = tf.data.TFRecordDataset(file_paths)
        
        # Parse the records
        dataset = dataset.map(self._parse_tfrecord_fn, 
                            num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle and split == "train":
            dataset = dataset.shuffle(buffer_size)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_dataset_info(self):
        """Get information about the dataset."""
        info = {}
        
        for split in ["train", "eval", "test"]:
            files = self.get_file_paths(split)
            total_size = sum(os.path.getsize(f) for f in files)
            info[split] = {
                "num_files": len(files),
                "total_size_mb": total_size / (1024 * 1024),
                "files": files
            }
        
        return info


if __name__ == "__main__":
    # Example usage
    loader = WildfireDataLoader()
    
    # Get dataset info
    info = loader.get_dataset_info()
    print("Dataset Information:")
    for split, data in info.items():
        print(f"{split.capitalize()}: {data['num_files']} files, "
              f"{data['total_size_mb']:.1f} MB")
    
    # Inspect a sample file
    train_files = loader.get_file_paths("train")
    if train_files:
        loader.inspect_tfrecord(train_files[0])
