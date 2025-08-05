"""
Prediction script for wildfire spread prediction model.
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

class WildfirePredictor:
    """Wildfire spread predictor using trained model."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_single(self, data):
        """Make prediction on a single sample."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Ensure data has batch dimension
        if len(data.shape) == 3:  # Add batch dimension
            data = np.expand_dims(data, axis=0)
        
        prediction = self.model.predict(data)
        return prediction
    
    def predict_batch(self, data_batch):
        """Make predictions on a batch of samples."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        predictions = self.model.predict(data_batch)
        return predictions
    
    def predict_probability(self, data):
        """Get prediction probabilities."""
        predictions = self.predict_single(data)
        
        if predictions.shape[-1] == 1:  # Binary classification
            prob_fire = predictions[0][0]
            prob_no_fire = 1 - prob_fire
            return {"fire_spread": prob_fire, "no_fire_spread": prob_no_fire}
        else:  # Multi-class
            probs = predictions[0]
            return {f"class_{i}": prob for i, prob in enumerate(probs)}
    
    def visualize_prediction(self, data, prediction, save_path=None):
        """Visualize input data and prediction."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot input data (assuming it's an image)
        if len(data.shape) == 4:  # Batch dimension
            data_to_plot = data[0]
        else:
            data_to_plot = data
        
        if data_to_plot.shape[-1] == 3:  # RGB image
            axes[0].imshow(data_to_plot)
        elif data_to_plot.shape[-1] == 1:  # Grayscale
            axes[0].imshow(data_to_plot[:, :, 0], cmap='gray')
        else:  # Multi-channel, show first channel
            axes[0].imshow(data_to_plot[:, :, 0], cmap='viridis')
        
        axes[0].set_title('Input Data')
        axes[0].axis('off')
        
        # Plot prediction
        if prediction.shape[-1] == 1:  # Binary
            prob = prediction[0][0]
            axes[1].bar(['No Fire Spread', 'Fire Spread'], [1-prob, prob], 
                       color=['green', 'red'])
            axes[1].set_title(f'Prediction: {prob:.3f}')
        else:  # Multi-class
            probs = prediction[0]
            class_names = [f'Class {i}' for i in range(len(probs))]
            axes[1].bar(class_names, probs)
            axes[1].set_title('Class Probabilities')
        
        axes[1].set_ylabel('Probability')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction visualization saved to {save_path}")
        
        plt.show()

def evaluate_model_performance(model_path: str):
    """Evaluate model performance on test set."""
    logger.info("Evaluating model performance...")
    
    # Load data
    loader = WildfireDataLoader()
    test_dataset = loader.create_dataset("test", batch_size=32, shuffle=False)
    
    # Load model
    predictor = WildfirePredictor(model_path)
    
    # Evaluate
    results = predictor.model.evaluate(test_dataset, verbose=1)
    
    # Print results
    metrics_names = predictor.model.metrics_names
    results_dict = dict(zip(metrics_names, results))
    
    logger.info("Model Performance on Test Set:")
    for metric, value in results_dict.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return results_dict

def predict_on_new_data(model_path: str, data_path: str = None):
    """Make predictions on new data."""
    logger.info("Making predictions on new data...")
    
    predictor = WildfirePredictor(model_path)
    
    if data_path:
        # Load data from file
        # This would need to be implemented based on your data format
        logger.info(f"Loading data from {data_path}")
        # data = load_your_data(data_path)
    else:
        # Use test data as example
        loader = WildfireDataLoader()
        test_dataset = loader.create_dataset("test", batch_size=1, shuffle=False)
        
        # Get a sample
        for data_batch in test_dataset.take(1):
            # This assumes the data structure - you'll need to adapt
            if isinstance(data_batch, tuple):
                data, labels = data_batch
            else:
                data = data_batch
                labels = None
            
            # Make prediction
            prediction = predictor.predict_single(data[0])
            probs = predictor.predict_probability(data[0])
            
            logger.info("Prediction Results:")
            logger.info(f"  Raw prediction: {prediction}")
            logger.info(f"  Probabilities: {probs}")
            
            # Visualize if possible
            try:
                predictor.visualize_prediction(data, prediction, 'prediction_result.png')
            except Exception as e:
                logger.warning(f"Could not visualize prediction: {str(e)}")
            
            break

def main():
    """Main prediction function."""
    model_path = "wildfire_model_final.h5"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please train the model first using train.py")
        return
    
    try:
        # Evaluate model performance
        evaluate_model_performance(model_path)
        
        # Make predictions on sample data
        predict_on_new_data(model_path)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.info("Make sure the model is trained and data format is correct.")

if __name__ == "__main__":
    main()
