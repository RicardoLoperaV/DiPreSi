# CNN Model Producer for Plant Health Classification

This module provides a complete pipeline for training Convolutional Neural Networks (CNNs) to classify plant health status (Healthy vs Unhealthy) using Fourier magnitude images.

## Overview

The system automatically:
- Loads Fourier magnitude images from specified day folders
- Extracts labels from image filenames using regex patterns
- Splits data into training, validation, and test sets
- Trains a custom CNN architecture
- Evaluates model performance
- Generates visualizations and reports

## Image Naming Convention

Images must follow this regex pattern:
```
Sample_(?P<id>\d+)_(?P<estado>[a-zA-Z]+)_(?P<valor>\d+(?:\.\d+)?)_magnitude.png
```

Example filenames:
- `Sample_0_Healthy_1.0_magnitude.png` → Label: Healthy (1)
- `Sample_102_Unhealthy_0.0_magnitude.png` → Label: Unhealthy (0)

## Installation

Required packages:
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow
```

Note: If you don't have PyTorch installed:
```bash
# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (GPU support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

The easiest way to get started is using the `Quick_Start.py` script:

```python
# 1. Open Quick_Start.py
# 2. Modify the DAYS variable to choose which day(s) to train on:
DAYS = 0              # Single day (Day 0)
# or
DAYS = [0, 1, 2]      # Multiple days
# or
DAYS = list(range(16)) # All days (0-15)

# 3. Run the script
python Quick_Start.py
```

This will:
- Load and preprocess the data
- Train the CNN model
- Evaluate performance
- Save all results to `./CNN_Results/`

## Usage Examples

### Example 1: Train on a Single Day

```python
from Model_Productor import CNNModelProductor

# Initialize model producer for Day 0
model_producer = CNNModelProductor(
    days=0,  # Train on Day 0 only
    img_size=224,
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001
)

# Run complete pipeline
results = model_producer.run_full_pipeline(save_dir='./Results_Day0')
```

### Example 2: Train on Multiple Days

```python
# Train on Days 0, 1, 2, and 3
model_producer = CNNModelProductor(
    days=[0, 1, 2, 3],  # Multiple days
    img_size=224,
    batch_size=32,
    num_epochs=50
)

results = model_producer.run_full_pipeline(save_dir='./Results_Multi_Day')
```

### Example 3: Step-by-Step Training (Advanced)

```python
model_producer = CNNModelProductor(days=[5, 6, 7])

# Load data
train_dataset, val_dataset, test_dataset = model_producer.load_data()

# Build model
model = model_producer.build_model(dropout_rate=0.5)

# Train with early stopping
model_producer.train_model(early_stopping_patience=15)

# Evaluate
results = model_producer.evaluate_model()

# Save model
model_producer.save_model('./my_model.pth')

# Generate plots
model_producer.plot_training_history(save_path='./history.png')
model_producer.plot_confusion_matrix(results, save_path='./cm.png')
```

### Example 4: Load and Use a Saved Model

```python
# Create a new instance
model_producer = CNNModelProductor(days=0)

# Load data
model_producer.load_data()

# Load previously saved model
model_producer.load_model('./Results_Day0/plant_cnn_model.pth')

# Evaluate on test set
results = model_producer.evaluate_model()
```

## Configuration Parameters

### CNNModelProductor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | int or list | Required | Day number(s) to use for training |
| `root_dir` | str | (repo path) | Root directory containing Day_# folders |
| `img_size` | int | 224 | Image resize dimension (224x224) |
| `batch_size` | int | 32 | Training batch size |
| `test_size` | float | 0.2 | Proportion for test set (0.0-1.0) |
| `val_size` | float | 0.1 | Proportion of training data for validation |
| `random_state` | int | 42 | Random seed for reproducibility |
| `num_epochs` | int | 50 | Maximum number of training epochs |
| `learning_rate` | float | 0.001 | Learning rate for optimizer |
| `device` | torch.device | auto | Device for training (cuda/cpu) |

## CNN Architecture

The model uses a custom CNN architecture with:
- **3 Convolutional Blocks** with BatchNorm and Dropout
  - Block 1: 32 filters
  - Block 2: 64 filters
  - Block 3: 128 filters
- **Fully Connected Layers** with Dropout
  - FC1: 256 neurons
  - FC2: 128 neurons
  - Output: 2 classes (Healthy/Unhealthy)

Total parameters: ~6.7M (trainable)

## Data Augmentation

Training images are augmented with:
- Random horizontal and vertical flips
- Random rotation (±15 degrees)
- Color jitter (brightness and contrast)
- Normalization with ImageNet statistics

Test/validation images only undergo resizing and normalization.

## Output Files

Running the pipeline generates:

1. **plant_cnn_model.pth** - Trained model weights
2. **training_history.png** - Loss and accuracy curves
3. **confusion_matrix.png** - Visual confusion matrix
4. **test_predictions.csv** - Detailed predictions with probabilities
5. **metrics_summary.csv** - Performance metrics summary

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

## Directory Structure

```
Fourier imaging/
└── Magnitude/
    ├── Model_Productor.py      # Main module
    ├── Quick_Start.py           # Quick start script
    ├── Example_Usage.py         # Detailed examples
    ├── README.md                # This file
    └── Fourier_Images/
        ├── Day_0/
        │   ├── Sample_0_Healthy_1.0_magnitude.png
        │   ├── Sample_102_Unhealthy_0.0_magnitude.png
        │   └── ...
        ├── Day_1/
        ├── Day_2/
        └── ...
```

## Training Tips

### For Small Datasets (single day):
```python
CNNModelProductor(
    days=0,
    batch_size=16,          # Smaller batch size
    num_epochs=100,         # More epochs
    learning_rate=0.0005    # Lower learning rate
)
```

### For Large Datasets (multiple days):
```python
CNNModelProductor(
    days=list(range(16)),   # All days
    batch_size=64,          # Larger batch size
    num_epochs=50,          # Fewer epochs needed
    learning_rate=0.001     # Standard learning rate
)
```

### For Overfitting Issues:
- Increase dropout rate: `build_model(dropout_rate=0.6)`
- Use more data augmentation
- Add more training data (more days)
- Enable early stopping with patience

## GPU Usage

The model automatically uses GPU if available:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

For CPU-only training, explicitly set:
```python
model_producer = CNNModelProductor(
    days=0,
    device=torch.device('cpu')
)
```

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` (try 16 or 8)
- Reduce `img_size` (try 128 instead of 224)
- Use CPU instead of GPU

### Poor Performance
- Train on more days (more data)
- Increase `num_epochs`
- Adjust `learning_rate` (try 0.0005 or 0.0001)
- Check data quality and balance

### Import Errors
- Install required packages: `pip install torch torchvision pillow`
- For torchvision issues: reinstall with `pip install --upgrade torchvision`

## Advanced Features

### Custom Data Transforms
```python
import torchvision.transforms as transforms

custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Apply to dataset manually
dataset = PlantDataset(day_dir, transform=custom_transform)
```

### Access Raw Dataset
```python
from Model_Productor import PlantDataset

dataset = PlantDataset('path/to/Day_0')
print(f"Total samples: {len(dataset)}")
print(f"Labels: {dataset.get_labels()}")

# Get a sample
image, label, sample_id = dataset[0]
```

## Citation

If you use this code in your research, please cite:
```
Plant Health Classification using CNN on Fourier Magnitude Images
Repository: DiPreSi
```

## License

This code is part of the DiPreSi project.

## Contact

For issues or questions, please open an issue in the GitHub repository.
