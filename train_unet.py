"""
Complete training script for wildfire spread prediction using U-Net.
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

# Custom metrics for segmentation
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for segmentation evaluation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred, smooth=1e-6):
    """IoU (Intersection over Union) metric for segmentation."""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

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
    
    # Plot Dice coefficient
    if 'dice_coefficient' in history.history:
        axes[1, 0].plot(history.history['dice_coefficient'], label='Training Dice')
        axes[1, 0].plot(history.history['val_dice_coefficient'], label='Validation Dice')
        axes[1, 0].set_title('Dice Coefficient')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot IoU
    if 'iou_metric' in history.history:
        axes[1, 1].plot(history.history['iou_metric'], label='Training IoU')
        axes[1, 1].plot(history.history['val_iou_metric'], label='Validation IoU')
        axes[1, 1].set_title('IoU Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('IoU')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Training history plot saved to {save_path}")

def visualize_predictions(model, dataset, num_samples=3, save_path='predictions.png'):
    """Visualize model predictions."""
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i, (inputs, targets) in enumerate(dataset.take(num_samples)):
        # Get first sample from batch
        input_sample = inputs[0:1]  # Keep batch dimension
        target_sample = targets[0]
        
        # Make prediction
        prediction = model.predict(input_sample, verbose=0)[0]
        
        # Plot input (show first 3 channels as RGB)
        input_rgb = input_sample[0, :, :, :3]  # NDVI, tmmn, tmmx as RGB
        input_rgb = (input_rgb - input_rgb.min()) / (input_rgb.max() - input_rgb.min())  # Normalize
        axes[i, 0].imshow(input_rgb)
        axes[i, 0].set_title('Input (RGB: NDVI, tmmn, tmmx)')
        axes[i, 0].axis('off')
        
        # Plot PrevFireMask (channel 11)
        prev_fire = input_sample[0, :, :, 11]
        axes[i, 1].imshow(prev_fire, cmap='Reds', vmin=0, vmax=1)
        axes[i, 1].set_title('Previous Fire Mask')
        axes[i, 1].axis('off')
        
        # Plot ground truth
        axes[i, 2].imshow(target_sample[:, :, 0], cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_title('Ground Truth Fire Mask')
        axes[i, 2].axis('off')
        
        # Plot prediction
        axes[i, 3].imshow(prediction[:, :, 0], cmap='Reds', vmin=0, vmax=1)
        axes[i, 3].set_title('Predicted Fire Mask')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Prediction visualization saved to {save_path}")

def main():
    """Main training function."""
    logger.info("Starting wildfire spread prediction U-Net training...")
    
    # Initialize data loader
    loader = WildfireDataLoader()
    
    # Get dataset info
    info = loader.get_dataset_info()
    logger.info("Dataset Information:")
    for split, data in info.items():
        logger.info(f"  {split}: {data['num_files']} files, {data['total_size_mb']:.1f} MB")
    
    # Create datasets
    batch_size = 8  # Smaller batch size for 64x64x12 data
    logger.info("Creating datasets...")
    
    try:
        train_dataset = loader.create_dataset("train", batch_size=batch_size, shuffle=True)
        val_dataset = loader.create_dataset("eval", batch_size=batch_size, shuffle=False)
        test_dataset = loader.create_dataset("test", batch_size=batch_size, shuffle=False)
        
        # Build U-Net model
        logger.info("Building U-Net model...")
        model = build_unet_model(input_shape=(64, 64, 12), num_classes=1)
        
        # Compile model with segmentation-specific loss and metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', dice_coefficient, iou_metric]
        )
        
        # Print model summary
        model.summary()
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='wildfire_unet_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting model training...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=50,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_results = model.evaluate(test_dataset, verbose=1)
        
        # Print test results
        metrics_names = model.metrics_names
        results_dict = dict(zip(metrics_names, test_results))
        
        logger.info("Final Test Results:")
        for metric, value in results_dict.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Visualize predictions
        logger.info("Generating prediction visualizations...")
        visualize_predictions(model, test_dataset, num_samples=3)
        
        # Save final model
        model.save("wildfire_unet_final.h5")
        logger.info("Model saved as 'wildfire_unet_final.h5'")
        
        # Save predictions as numpy arrays
        logger.info("Saving prediction arrays...")
        predictions = []
        targets = []
        
        for inputs, target_batch in test_dataset.take(5):  # Save first 5 batches
            pred_batch = model.predict(inputs, verbose=0)
            predictions.append(pred_batch)
            targets.append(target_batch.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        np.save('wildfire_predictions.npy', predictions)
        np.save('wildfire_targets.npy', targets)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final test accuracy: {results_dict['accuracy']:.4f}")
        logger.info(f"Final test Dice score: {results_dict['dice_coefficient']:.4f}")
        logger.info(f"Final test IoU score: {results_dict['iou_metric']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.info("Please check your data format and file paths.")
        raise

if __name__ == "__main__":
    main()
