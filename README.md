# Wildfire Spread Prediction Model

This project implements a deep learning model for predicting wildfire spread using TensorFlow and the provided wildfire dataset.

## Dataset

The dataset contains TensorFlow Record files with wildfire data split into:
- **Training**: 15 files (~3.1GB total)
- **Evaluation**: 2 files (~400MB total)
- **Test**: 2 files (~360MB total)

## Project Structure

```
kaggle_fire/
├── archive/                    # TFRecord data files
├── requirements.txt           # Python dependencies
├── data_loader.py            # Data loading utilities
├── explore_data.py           # Exploratory data analysis
├── model.py                  # Model architectures
├── train.py                  # Training script
├── predict.py                # Prediction and evaluation
└── README.md                 # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Explore the data structure:
```bash
python explore_data.py
```

3. Train the model:
```bash
python train.py
```

4. Make predictions:
```bash
python predict.py
```

## Model Architectures

The project includes three model types:

### 1. CNN Model
- Convolutional layers for spatial feature extraction
- Batch normalization and dropout for regularization
- Global average pooling and dense layers
- Best for image-like spatial data

### 2. LSTM Model
- Long Short-Term Memory layers for temporal sequences
- Dropout and recurrent dropout for regularization
- Best for time-series data

### 3. Hybrid CNN-LSTM Model
- Combines spatial (CNN) and temporal (LSTM) branches
- Processes both spatial and temporal features
- Best for spatio-temporal data

## Features

- **Data Loading**: Efficient TFRecord parsing and dataset creation
- **Model Training**: Multiple architectures with callbacks and monitoring
- **Evaluation**: Comprehensive model performance metrics
- **Visualization**: Training history and prediction plots
- **Prediction**: Easy-to-use prediction interface

## Usage Examples

### Basic Training
```python
from data_loader import WildfireDataLoader
from model import WildfireSpreadModel

# Load data
loader = WildfireDataLoader()
train_ds = loader.create_dataset("train", batch_size=32)
val_ds = loader.create_dataset("eval", batch_size=32)

# Create and train model
model = WildfireSpreadModel(input_shape=(64, 64, 3), num_classes=2)
history = model.train(train_ds, val_ds, epochs=50, model_type="cnn")
```

### Making Predictions
```python
from predict import WildfirePredictor

# Load trained model
predictor = WildfirePredictor("wildfire_model_final.h5")

# Make prediction
prediction = predictor.predict_single(your_data)
probabilities = predictor.predict_probability(your_data)
```

## Model Performance

The model includes the following metrics:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive cases
- **Loss**: Training and validation loss

## Files Generated

During training and evaluation, the following files are created:
- `wildfire_cnn_best.h5`: Best model checkpoint
- `wildfire_model_final.h5`: Final trained model
- `training_history.png`: Training metrics visualization
- `data_distribution.png`: Dataset distribution plots
- `prediction_result.png`: Sample prediction visualization

## Customization

To adapt the model for your specific data:

1. **Data Structure**: Modify the `_parse_tfrecord_fn` in `data_loader.py` based on your TFRecord format
2. **Input Shape**: Adjust `input_shape` in model initialization
3. **Model Architecture**: Customize layers in `model.py`
4. **Preprocessing**: Add data preprocessing steps in `train.py`

## Troubleshooting

1. **Data Format Issues**: Run `explore_data.py` to understand your data structure
2. **Memory Issues**: Reduce batch size or use gradient accumulation
3. **Training Issues**: Adjust learning rate or model complexity

## Next Steps

1. Run data exploration to understand the actual data format
2. Customize the data parsing based on TFRecord structure
3. Adjust model architecture based on data dimensions
4. Fine-tune hyperparameters for optimal performance

## Requirements

- TensorFlow >= 2.12.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
