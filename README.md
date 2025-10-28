# ELEC 475 Lab 2 - Pet Nose Localization with SnoutNet

A PyTorch implementation of a CNN-based model for predicting the (x, y) coordinates of a pet's nose from images using the Oxford-IIIT Pet Noses dataset.

## Project Structure

```
ELEC475_Lab2/
├── model.py              # SnoutNet CNN architecture
├── dataset.py            # Custom Dataset class and reality check
├── train.py              # Training script
├── test.py               # Evaluation script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── train-noses.txt      # Training annotations (not included)
├── test-noses.txt       # Test annotations (not included)
├── images-original/     # Image directory (not included)
│   └── images/
└── weights/             # Saved models (created during training)
    └── snoutnet.pt
```

## Dataset Structure

The project expects the following dataset structure:

```
images-original/
└── images/
    ├── beagle_145.jpg
    ├── cat_123.jpg
    └── ...

train-noses.txt    # Format: filename.jpg,"(x, y)"
test-noses.txt     # Format: filename.jpg,"(x, y)"
```

**Annotation Format Example:**
```
beagle_145.jpg,"(198, 304)"
persian_cat_22.jpg,"(156, 189)"
```

## Installation

1. **Clone the repository** (or download the files)

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install torch>=2.4.0 torchvision>=0.19.0 Pillow>=10.0.0 numpy>=1.24.0 matplotlib>=3.7.0
```

## Usage

### 1. Test the Model Architecture

Verify that the SnoutNet model works correctly:

```bash
python model.py
```

This will:
- Create a SnoutNet instance
- Feed a random tensor through the network
- Print output shapes and verify correctness

**Expected Output:**
```
============================================================
SnoutNet Model Test
============================================================

Model created successfully!
Total parameters: 179,906,562

Input shape: torch.Size([1, 3, 227, 227])
Output shape: torch.Size([1, 2])
Predicted coordinates (x, y): [0.1234, -0.5678]

✓ Test passed! Output shape is correct.
```

### 2. Test the Dataset

Perform a reality check on the dataset:

```bash
python dataset.py --visualize --num_samples 5
```

**Options:**
- `--train_file`: Path to training annotations (default: `train-noses.txt`)
- `--test_file`: Path to test annotations (default: `test-noses.txt`)
- `--img_dir`: Directory containing images (default: `images-original/images`)
- `--num_samples`: Number of samples to display (default: 5)
- `--visualize`: Show images with ground-truth points
- `--batch_size`: Batch size for DataLoader test (default: 4)

**Output:**
- Prints sample information (filenames, shapes, coordinates)
- Saves visualization to `dataset_reality_check.png`
- Tests DataLoader batching

### 3. Train the Model

Train SnoutNet on the training dataset:

```bash
python train.py --epochs 50 --batch_size 32 --lr 0.001
```

**With data augmentation:**
```bash
python train.py --epochs 100 --batch_size 32 --lr 0.001 --augment
```

**Training Arguments:**
- `--train_file`: Training annotations file (default: `train-noses.txt`)
- `--test_file`: Validation annotations file (default: `test-noses.txt`)
- `--img_dir`: Image directory (default: `images-original/images`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--augment`: Enable data augmentation
- `--weights_dir`: Directory to save weights (default: `weights/`)
- `--num_workers`: DataLoader workers (default: 0)

**Training Output:**
```
======================================================================
SnoutNet Training
======================================================================
Configuration:
  Epochs: 50
  Batch size: 32
  Learning rate: 0.001
  Augmentation: False
  Device: cuda
======================================================================

Using device: cuda

Loading datasets...
Loaded 5000 samples from train-noses.txt
Loaded 1000 samples from test-noses.txt
Training samples: 5000
Validation samples: 1000

Epoch [1/50] | Train Loss: 2841.234567 | Val Loss: 2456.789012 | Time: 45.23s
  → Best model saved (val_loss: 2456.789012)
Epoch [2/50] | Train Loss: 1823.456789 | Val Loss: 1678.901234 | Time: 43.87s
  → Best model saved (val_loss: 1678.901234)
...
```

**Saved Artifacts:**
- `weights/snoutnet.pt`: Best model checkpoint
- `weights/training_losses.csv`: Training/validation losses per epoch
- `weights/training_curve.png`: Loss curve plot

### 4. Test/Evaluate the Model

Evaluate the trained model on the test dataset:

```bash
python test.py --model_path weights/snoutnet.pt --visualize
```

**Testing Arguments:**
- `--model_path`: Path to trained model checkpoint (default: `weights/snoutnet.pt`)
- `--test_file`: Test annotations file (default: `test-noses.txt`)
- `--img_dir`: Image directory (default: `images-original/images`)
- `--output_csv`: Path to save results CSV (default: `test_results.csv`)
- `--visualize`: Show prediction visualizations
- `--num_vis_samples`: Number of samples to visualize (default: 4)
- `--batch_size`: Batch size for evaluation (default: 32)

**Evaluation Output:**
```
======================================================================
SnoutNet Testing
======================================================================
Using device: cuda

Loading model from weights/snoutnet.pt...
✓ Model loaded successfully
  Trained for 50 epochs
  Validation loss: 234.567890

Loading test dataset...
Loaded 1000 samples from test-noses.txt
Test samples: 1000

Evaluating model...

======================================================================
Evaluation Results
======================================================================
Total test samples: 1000

Distance Statistics (pixels):
  Min:      1.2345
  Max:      89.6789
  Mean:     12.3456
  Median:   10.2345
  Std Dev:  8.4567
======================================================================

Results saved to test_results.csv
Summary saved to test_results_summary.txt
Histogram saved to test_results_histogram.png
Visualization saved to test_results_predictions.png

✓ Testing complete!
```

**Saved Artifacts:**
- `test_results.csv`: Detailed results for each test sample
- `test_results_summary.txt`: Summary statistics
- `test_results_histogram.png`: Distribution of prediction errors
- `test_results_predictions.png`: Visual comparison of predictions vs ground truth

## Model Architecture

**SnoutNet** is a convolutional neural network with the following architecture:

### Input
- RGB image: `[batch_size, 3, 227, 227]`

### Architecture Layers

**Convolutional Blocks:**
1. Conv2d(3 → 64, kernel=3) → ReLU → MaxPool(2x2)
2. Conv2d(64 → 128, kernel=3) → ReLU → MaxPool(2x2)
3. Conv2d(128 → 256, kernel=3) → ReLU → MaxPool(2x2)

**Fully Connected Layers:**
1. Linear(173,056 → 1024) → ReLU
2. Linear(1024 → 1024) → ReLU
3. Linear(1024 → 2) [Output: (x, y)]

### Output
- Predicted nose coordinates: `[batch_size, 2]`

**Total Parameters:** ~180 million

## Training Details

- **Loss Function:** MSE (Mean Squared Error)
- **Optimizer:** Adam
- **Learning Rate:** 0.001 (default)
- **Task:** Regression (predicting continuous x, y coordinates)

**Optional Data Augmentation:**
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Normalization with ImageNet stats

## Evaluation Metrics

The model is evaluated using **Euclidean Distance** between predicted and ground-truth coordinates:

```
distance = sqrt((pred_x - true_x)² + (pred_y - true_y)²)
```

**Reported Statistics:**
- Minimum distance
- Maximum distance
- Mean distance
- Median distance
- Standard deviation

## Tips for Best Performance

1. **Start with baseline training:**
   ```bash
   python train.py --epochs 50 --batch_size 32
   ```

2. **If overfitting, try data augmentation:**
   ```bash
   python train.py --epochs 100 --batch_size 32 --augment
   ```

3. **Adjust learning rate if loss isn't decreasing:**
   ```bash
   python train.py --epochs 50 --lr 0.0001
   ```

4. **Use GPU for faster training** (PyTorch will automatically use CUDA if available)

5. **Monitor the training curve** in `weights/training_curve.png` to detect overfitting

## Troubleshooting

### Dataset not found
```
✗ Error: FileNotFoundError: train-noses.txt
```
**Solution:** Ensure your dataset files are in the correct location relative to the scripts.

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size:
```bash
python train.py --batch_size 16
```

### Images fail to load
```
Error loading image images-original/images/beagle_145.jpg
```
**Solution:** Verify the image directory path and that images exist:
```bash
python dataset.py --img_dir your/path/to/images
```

## File Descriptions

- **model.py**: Defines the `SnoutNet` class (CNN architecture)
- **dataset.py**: Defines `PetNoseDataset` class for loading images and annotations
- **train.py**: Training script with validation loop, checkpointing, and loss tracking
- **test.py**: Evaluation script that computes distances and generates visualizations
- **requirements.txt**: List of Python package dependencies

## Example Workflow

```bash
# 1. Test the model architecture
python model.py

# 2. Verify the dataset
python dataset.py --visualize

# 3. Train the model
python train.py --epochs 50 --batch_size 32 --augment

# 4. Evaluate the model
python test.py --model_path weights/snoutnet.pt --visualize

# 5. Review results
# - Check weights/training_curve.png for training progress
# - Check test_results_histogram.png for error distribution
# - Check test_results_predictions.png for visual comparison
```

## License

This project is created for educational purposes as part of ELEC 475 coursework.

## Author

Generated for ELEC 475 Lab 2 - 2025
