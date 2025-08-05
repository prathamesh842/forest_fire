"""
Simple and fast wildfire prediction model.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple


def build_simple_unet(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Build a much simpler and faster U-Net model."""
    inputs = layers.Input(shape=input_shape)

    # Much simpler encoder - fewer filters for speed
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D(2)(conv1)
    
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D(2)(conv2)
    
    # Bottleneck - very small
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    
    # Simple decoder
    up1 = layers.UpSampling2D(2)(conv3)
    up1 = layers.Conv2D(64, 3, activation='relu', padding='same')(up1)
    merge1 = layers.Add()([conv2, up1])  # Skip connection
    
    up2 = layers.UpSampling2D(2)(merge1)
    up2 = layers.Conv2D(32, 3, activation='relu', padding='same')(up2)
    merge2 = layers.Add()([conv1, up2])  # Skip connection
    
    # Output
    output = layers.Conv2D(1, 1, activation='sigmoid')(merge2)
    
    model = Model(inputs=inputs, outputs=output)
    return model


def build_even_simpler_cnn(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Build an extremely simple CNN for speed testing."""
    inputs = layers.Input(shape=input_shape)
    
    # Very simple architecture
    x = layers.Conv2D(16, 5, activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, 5, activation='relu', padding='same')(x)
    x = layers.Conv2D(16, 5, activation='relu', padding='same')(x)
    x = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model


if __name__ == "__main__":
    # Test both models
    simple_unet = build_simple_unet((64, 64, 12))
    simple_cnn = build_even_simpler_cnn((64, 64, 12))
    
    print("Simple U-Net:")
    simple_unet.summary()
    print(f"Parameters: {simple_unet.count_params():,}")
    
    print("\nEven Simpler CNN:")
    simple_cnn.summary()
    print(f"Parameters: {simple_cnn.count_params():,}")
