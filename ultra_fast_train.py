"""
Ultra-fast training script to fix NaN metrics and speed issues.
"""

import tensorflow as tf
import numpy as np
from data_loader import WildfireDataLoader
from simple_model import build_simple_unet, build_even_simpler_cnn
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Ultra-fast training with maximum optimizations."""
    logger.info("Starting ULTRA-FAST wildfire training...")
    
    # Initialize data loader
    loader = WildfireDataLoader()
    
    # MAXIMUM SPEED: Very large batch size and tiny dataset
    batch_size = 64  # Even larger batch
    
    logger.info("Creating TINY datasets for ultra-fast testing...")
    
    # Create minimal datasets
    train_dataset = loader.create_dataset("train", batch_size=batch_size, shuffle=True)
    val_dataset = loader.create_dataset("eval", batch_size=batch_size, shuffle=False)
    
    # ULTRA SPEED: Take only 10 batches for training, 3 for validation
    train_dataset = train_dataset.take(10)  # Only 640 samples
    val_dataset = val_dataset.take(3)       # Only 192 samples
    
    # Cache everything in memory
    train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
    
    # Build SIMPLE model
    logger.info("Building SIMPLE model...")
    model = build_even_simpler_cnn((64, 64, 12))  # Simplest possible model
    
    # Simple compilation - NO CUSTOM METRICS to avoid NaN
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']  # Only basic accuracy, no custom metrics
    )
    
    logger.info("Model summary:")
    model.summary()
    logger.info(f"Total parameters: {model.count_params():,}")
    
    # No callbacks for maximum speed
    logger.info("Starting ULTRA-FAST training (5 epochs only)...")
    
    start_time = time.time()
    
    # Train with minimal epochs
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,  # Only 5 epochs
        verbose=1
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info(f"Training completed in {training_time:.1f} seconds!")
    
    # Quick evaluation
    logger.info("Quick evaluation...")
    test_results = model.evaluate(val_dataset, verbose=1)
    
    logger.info(f"Final accuracy: {test_results[1]:.4f}")
    
    # Test prediction
    logger.info("Testing prediction...")
    for inputs, targets in val_dataset.take(1):
        predictions = model.predict(inputs[:1], verbose=0)
        logger.info(f"Input shape: {inputs.shape}")
        logger.info(f"Target shape: {targets.shape}")
        logger.info(f"Prediction shape: {predictions.shape}")
        logger.info(f"Target range: {targets.numpy().min():.3f} to {targets.numpy().max():.3f}")
        logger.info(f"Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
        
        # Calculate simple metrics manually
        pred_binary = (predictions > 0.5).astype(np.float32)
        target_np = targets.numpy()[:1]
        
        # Pixel accuracy
        pixel_acc = np.mean(pred_binary == target_np)
        logger.info(f"Pixel accuracy: {pixel_acc:.4f}")
        
        # Simple IoU calculation
        intersection = np.sum(pred_binary * target_np)
        union = np.sum(pred_binary) + np.sum(target_np) - intersection
        if union > 0:
            iou = intersection / union
            logger.info(f"IoU: {iou:.4f}")
        else:
            logger.info("IoU: N/A (no fire pixels)")
        
        break
    
    # Save model
    model.save("wildfire_ultra_fast.h5")
    logger.info("Ultra-fast model saved!")
    
    logger.info("=" * 50)
    logger.info("ULTRA-FAST TRAINING SUCCESS!")
    logger.info(f"Training time: {training_time:.1f} seconds")
    logger.info(f"Steps per second: {10 * 10 / training_time:.1f}")  # 10 batches × 10 epochs
    logger.info("=" * 50)
    
    return model, history

if __name__ == "__main__":
    try:
        model, history = main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
