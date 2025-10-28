# Quick Start Guide - SnoutNet Pet Nose Localization

## Installation (One Command)

```bash
pip install torch>=2.4.0 torchvision>=0.19.0 Pillow>=10.0.0 numpy>=1.24.0 matplotlib>=3.7.0
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## Quick Test Commands

### 1. Test Model (Verify Architecture)
```bash
python model.py
```

### 2. Test Dataset (Reality Check)
```bash
python dataset.py --visualize --num_samples 5
```

### 3. Train Model (Quick Training)
```bash
python train.py --epochs 20 --batch_size 32 --lr 0.001
```

### 4. Train with Augmentation (Better Results)
```bash
python train.py --epochs 50 --batch_size 32 --lr 0.001 --augment
```

### 5. Evaluate Model
```bash
python test.py --model_path weights/snoutnet.pt --visualize
```

## Expected Dataset Structure

```
ELEC475_Lab2/
├── train-noses.txt
├── test-noses.txt
└── images-original/
    └── images/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Annotation Format

Each line in `train-noses.txt` and `test-noses.txt`:
```
filename.jpg,"(x, y)"
```

Example:
```
beagle_145.jpg,"(198, 304)"
persian_cat_22.jpg,"(156, 189)"
```

## Common Issues

### Issue: Files not found
**Solution:** Make sure you're in the project directory:
```bash
cd C:\Users\20sr91\ELEC475_Lab2
```

### Issue: CUDA out of memory
**Solution:** Reduce batch size:
```bash
python train.py --batch_size 16
```

### Issue: Dataset path incorrect
**Solution:** Specify full paths:
```bash
python train.py --img_dir "path/to/images" --train_file "path/to/train-noses.txt"
```

## Quick Workflow

```bash
# Complete workflow in 4 commands:

# 1. Verify everything works
python model.py

# 2. Check dataset
python dataset.py --visualize

# 3. Train (quick test)
python train.py --epochs 10 --batch_size 16

# 4. Evaluate
python test.py --visualize
```

## Output Files

After training:
- `weights/snoutnet.pt` - Best model checkpoint
- `weights/training_losses.csv` - Loss history
- `weights/training_curve.png` - Training plot

After testing:
- `test_results.csv` - All predictions
- `test_results_summary.txt` - Statistics
- `test_results_histogram.png` - Error distribution
- `test_results_predictions.png` - Visual results

## Minimum Working Example

```python
# Test in Python REPL
import torch
from model import SnoutNet

# Create model
model = SnoutNet()

# Create random input
x = torch.randn(1, 3, 227, 227)

# Forward pass
model.eval()
with torch.no_grad():
    output = model(x)

print(f"Output shape: {output.shape}")  # Should be torch.Size([1, 2])
print(f"Predicted (x, y): {output[0].tolist()}")
```

## Performance Tips

1. **Use GPU if available** (automatic with PyTorch)
2. **Start with small epochs** (10-20) to test, then increase
3. **Use data augmentation** for better generalization (`--augment`)
4. **Monitor training curve** to detect overfitting
5. **Adjust learning rate** if loss plateaus (`--lr 0.0001`)

## Need Help?

Check the full README.md for detailed documentation.

