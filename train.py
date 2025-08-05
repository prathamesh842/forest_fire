"""
Training script for wildfire spread prediction model.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_loader import WildfireDataLoader
from model import WildfireSpreadModel
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_training_history(history, save_path='training_history.png'):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training & validation accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training & validation loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot precision if available
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot recall if available
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Training history plot saved to {save_path}")

# Preprocessing is now handled in the data_loader.py _parse_tfrecord_fn method

def main():
    """Main training function."""
    logger.info("Starting wildfire spread prediction model training...")
    
    # Initialize data loader
    loader = WildfireDataLoader()
    
    # Get dataset info
    info = loader.get_dataset_info()
    logger.info("Dataset Information:")
    for split, data in info.items():
        logger.info(f"  {split}: {data['num_files']} files, {data['total_size_mb']:.1f} MB")
    
    # First, let's inspect the data structure
    train_files = loader.get_file_paths("train")
    if train_files:
        logger.info("Inspecting data structure...")
        example = loader.inspect_tfrecord(train_files[0], num_examples=1)
        
        # Determine input shape and number of classes based on data inspection
        # This is a placeholder - you'll need to adapt based on actual data
        input_shape = (64, 64, 3)  # Placeholder - adjust based on your data
        num_classes = 2  # Binary classification for fire spread (yes/no)
        
        logger.info(f"Using input shape: {input_shape}")
        logger.info(f"Number of classes: {num_classes}")
        
        # Create datasets
        batch_size = 32
        logger.info("Creating datasets...")
        
        try:
            train_dataset = loader.create_dataset("train", batch_size=batch_size, shuffle=True)
            val_dataset = loader.create_dataset("eval", batch_size=batch_size, shuffle=False)
            test_dataset = loader.create_dataset("test", batch_size=batch_size, shuffle=False)
            
            # Preprocess datasets
            train_dataset = preprocess_data(train_dataset)
            val_dataset = preprocess_data(val_dataset)
            test_dataset = preprocess_data(test_dataset)
            
            # Initialize model
            model = WildfireSpreadModel(input_shape=input_shape, num_classes=num_classes)
            
            # Train model
            logger.info("Starting model training...")
            history = model.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=50,
                model_type="unet",
                learning_rate=0.001
            )
            
            # Plot training history
            plot_training_history(history)
            
            # Evaluate on test set
            logger.info("Evaluating model on test set...")
            test_results = model.evaluate(test_dataset)
            
            # Save model
            model.save_model("wildfire_model_final.h5")
            
            logger.info("Training completed successfully!")
            logger.info(f"Final test results: {test_results}")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.info("This might be due to the data structure not matching our assumptions.")
            logger.info("Please run explore_data.py first to understand the data format.")
            
    else:
        logger.error("No training files found!")

if __name__ == "__main__":
    main()
