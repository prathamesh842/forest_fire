"""
Fast training script for wildfire U-Net with optimizations.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_loader import WildfireDataLoader
from model import build_unet_model
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed custom metrics
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Fixed Dice coefficient for segmentation evaluation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    y_pred_f = tf.keras.backend.round(y_pred_f)  # Binarize predictions
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    return tf.cond(
        tf.equal(union, 0),
        lambda: 1.0,  # Perfect score when both are empty
        lambda: (2. * intersection + smooth) / (union + smooth)
    )

def iou_metric(y_true, y_pred, smooth=1e-6):
    """Fixed IoU metric for segmentation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    y_pred_f = tf.keras.backend.round(y_pred_f)  # Binarize predictions
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return tf.cond(
        tf.equal(union, 0),
        lambda: 1.0,  # Perfect score when both are empty
        lambda: (intersection + smooth) / (union + smooth)
    )

def main():
    """Fast training with optimizations."""
    logger.info("Starting FAST wildfire U-Net training...")
    
    # Initialize data loader
    loader = WildfireDataLoader()
    
    # SPEED OPTIMIZATION 1: Larger batch size
    batch_size = 32  # Increased from 8
    
    # SPEED OPTIMIZATION 2: Limit dataset size for testing
    logger.info("Creating FAST datasets (limited size for testing)...")
    
    # Create datasets with optimizations
    train_dataset = loader.create_dataset("train", batch_size=batch_size, shuffle=True, buffer_size=500)
    val_dataset = loader.create_dataset("eval", batch_size=batch_size, shuffle=False)
    
    # SPEED OPTIMIZATION 3: Take only subset for quick training
    train_dataset = train_dataset.take(100)  # Only 100 batches = 3200 samples
    val_dataset = val_dataset.take(20)       # Only 20 batches = 640 samples
    
    # SPEED OPTIMIZATION 4: Prefetch and cache
    train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
    
    # Build smaller U-Net model for speed
    logger.info("Building FAST U-Net model...")
    model = build_unet_model(input_shape=(64, 64, 12), num_classes=1)
    
    # SPEED OPTIMIZATION 5: Higher learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # 10x higher
        loss='binary_crossentropy',
        metrics=['accuracy', dice_coefficient, iou_metric]
    )
    
    logger.info("Model compiled. Starting FAST training...")
    
    # SPEED OPTIMIZATION 6: Fewer epochs
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,  # Reduced patience
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,  # Reduced patience
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train with fewer epochs
    logger.info("Training for 10 epochs only...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,  # Much fewer epochs
        callbacks=callbacks,
        verbose=1
    )
    
    # Quick evaluation
    logger.info("Quick evaluation...")
    test_results = model.evaluate(val_dataset, verbose=1)
    
    metrics_names = model.metrics_names
    results_dict = dict(zip(metrics_names, test_results))
    
    logger.info("FAST Training Results:")
    for metric, value in results_dict.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save model
    model.save("wildfire_unet_fast.h5")
    logger.info("Fast model saved!")
    
    # Quick prediction test
    logger.info("Testing prediction...")
    for inputs, targets in val_dataset.take(1):
        predictions = model.predict(inputs[:1], verbose=0)
        logger.info(f"Prediction shape: {predictions.shape}")
        logger.info(f"Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
        break
    
    logger.info("FAST training completed successfully!")
    return model, history

if __name__ == "__main__":
    model, history = main()
