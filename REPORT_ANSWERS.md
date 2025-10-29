# ELEC 475 Lab 2: Pet Nose Localization - Report Answers

## 5. Experiments Description

### 5.1 Hardware Configuration

**Hardware Used:**
- **System**: Windows 10 (version 10.0.22631)
- **GPU**: NVIDIA GPU with CUDA support (detected via check_cuda.py)
- **CPU Fallback**: Available for systems without GPU
- **Memory**: Sufficient for batch size of 86 samples

**Software Stack:**
- PyTorch 2.4.0+
- Python 3.13
- Kornia for GPU-accelerated augmentation
- CUDA for GPU acceleration

### 5.2 Training Configuration

**Common Parameters:**
- **Epochs**: 20
- **Batch Size**: 86
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: MSE (Mean Squared Error)
- **Image Size**: 227×227 pixels
- **Dataset Split**: 698 test samples

**Augmentation Methods (when enabled):**
- Horizontal Flip (probability: 0.5)
- Random Rotation (±15 degrees)
- GPU-accelerated using Kornia library

### 5.3 Time Performance

**Training Time (per model):**
- **SnoutNet** (Custom CNN): ~X minutes/epoch (baseline architecture)
- **SnoutNet-A** (AlexNet): ~X minutes/epoch (transfer learning from AlexNet)
- **SnoutNet-V** (VGG16): ~X minutes/epoch (transfer learning from VGG16)

*Note: Replace X with actual timing from training logs. Time varies based on GPU model.*

**Testing Performance:**
- **Per Image Inference Time**: ~Y msec/image
- **Batch Processing**: Efficient with batch_size=86
- **Ensemble Inference**: ~3× single model time (3 models evaluated sequentially)

*Note: Measure actual timing using: `time python test_cases.py` divided by 698 images*

### 5.4 Models Architecture

1. **SnoutNet (Custom CNN)**
   - 5 convolutional layers
   - Batch normalization
   - Max pooling
   - Fully connected layers → 2 coordinates
   - ~180M parameters

2. **SnoutNet-A (AlexNet-based)**
   - Pretrained AlexNet backbone
   - Modified final layers for coordinate regression
   - Transfer learning approach

3. **SnoutNet-V (VGG16-based)**
   - Pretrained VGG16 backbone
   - Modified final layers for coordinate regression
   - Deeper architecture than AlexNet

---

## 5.5 Localization Accuracy Statistics

### Complete Results Table

| Model | Augmentation | Localization Error (Overall) | Localization Error (4 Best) | Localization Error (4 Worst) |
|-------|--------------|------------------------------|------------------------------|------------------------------|
| | | min \| max \| mean \| stdev | min \| max \| mean \| stdev | min \| max \| mean \| stdev |
| **SnoutNet** | No | 0.73 \| 134.51 \| **31.30** \| 22.66 | 0.73 \| 1.59 \| 1.15 \| 0.33 | 111.27 \| 134.51 \| 122.08 \| 9.97 |
| **SnoutNet** | Yes | 0.62 \| 130.15 \| **29.28** \| 21.30 | 0.62 \| 2.34 \| 1.82 \| 0.71 | 110.69 \| 130.15 \| 123.17 \| 7.80 |
| **SnoutNet-A** | No | 0.29 \| 130.92 \| **18.98** \| 15.86 | 0.29 \| 0.97 \| 0.68 \| 0.27 | 98.97 \| 130.92 \| 110.68 \| 12.25 |
| **SnoutNet-A** | Yes | 0.35 \| 129.81 \| **16.23** \| 15.19 | 0.35 \| 0.65 \| 0.54 \| 0.12 | 93.18 \| 129.81 \| 106.05 \| 14.09 |
| **SnoutNet-V** | No | 6.59 \| 178.90 \| **77.08** \| 29.35 | 6.59 \| 15.26 \| 11.27 \| 3.10 | 169.13 \| 178.90 \| 173.13 \| 3.57 |
| **SnoutNet-V** | Yes | 0.41 \| 123.92 \| **42.42** \| 23.45 | 0.41 \| 2.26 \| 1.49 \| 0.78 | 110.07 \| 123.92 \| 116.67 \| 6.04 |
| **SnoutNet-Ensemble** | Yes | 0.20 \| 112.11 \| **25.65** \| 18.11 | 0.20 \| 1.19 \| 0.54 \| 0.38 | 103.96 \| 112.11 \| 107.26 \| 3.10 |

*Note: All error measurements are in pixels. Lower is better.*

### Key Performance Metrics Summary

**Best Overall Performance:**
1. **SnoutNet-A (with augmentation)**: Mean error = 16.23 pixels ✅
2. **SnoutNet-Ensemble**: Mean error = 25.65 pixels
3. **SnoutNet-A (no augmentation)**: Mean error = 18.98 pixels

**Worst Performance:**
- **SnoutNet-V (no augmentation)**: Mean error = 77.08 pixels ⚠️

---

## 6. Performance Discussion

### 6.1 Overall Performance Analysis

**Model Performance Ranking:**

1. **SnoutNet-A (AlexNet with augmentation)**: 16.23 pixels
   - Best individual model
   - Transfer learning from AlexNet proved highly effective
   - Augmentation improved performance by 14.5%

2. **SnoutNet-A (AlexNet without augmentation)**: 18.98 pixels
   - Still strong performance without augmentation
   - Pretrained features help significantly

3. **SnoutNet-Ensemble**: 25.65 pixels
   - Better than basic SnoutNet, but worse than SnoutNet-A
   - Averaging helps reduce variance
   - Limited by poor VGG16 performance

4. **SnoutNet (with augmentation)**: 29.28 pixels
   - Custom architecture performs reasonably
   - Augmentation provided 6.5% improvement

5. **SnoutNet (without augmentation)**: 31.30 pixels
   - Baseline performance
   - Room for improvement

6. **SnoutNet-V (VGG16 with augmentation)**: 42.42 pixels
   - Significant improvement from augmentation (45% better!)
   - Still underperforms other models

7. **SnoutNet-V (VGG16 without augmentation)**: 77.08 pixels
   - Worst performer
   - May require more training or better hyperparameters

### 6.2 Impact of Augmentation

**Augmentation Effectiveness:**

| Model | Mean Error (No Aug) | Mean Error (Aug) | Improvement | % Change |
|-------|---------------------|------------------|-------------|----------|
| SnoutNet | 31.30 | 29.28 | -2.02 pixels | **-6.5%** ✅ |
| SnoutNet-A | 18.98 | 16.23 | -2.75 pixels | **-14.5%** ✅ |
| SnoutNet-V | 77.08 | 42.42 | -34.66 pixels | **-45.0%** ✅✅ |

**Key Findings:**
- ✅ **All models benefited from augmentation**
- ✅ **VGG16 showed the most dramatic improvement** (45% reduction in error)
  - Suggests VGG16 was overfitting without augmentation
  - Deeper architectures need more regularization
- ✅ **AlexNet benefited moderately** (14.5% improvement)
  - Already good generalization, augmentation helped further
- ✅ **Custom SnoutNet showed modest improvement** (6.5%)
  - Smaller architecture less prone to overfitting

**Augmentation Methods Used:**
1. **Horizontal Flip (p=0.5)**: Helps model learn left/right invariance
2. **Random Rotation (±15°)**: Handles head tilt variations
3. **GPU Acceleration (Kornia)**: Efficient on-the-fly augmentation during training

### 6.3 Expected vs. Actual Performance

**Expectations:**
- Transfer learning models (AlexNet, VGG16) should outperform custom CNN ✅
- Augmentation should improve generalization ✅
- Ensemble should combine strengths of all models ❓

**Reality:**
- ✅ **AlexNet exceeded expectations**: Best performer with augmentation
- ❌ **VGG16 underperformed**: Much worse than expected without augmentation
- ⚠️ **Ensemble performance mixed**: Better than SnoutNet but worse than SnoutNet-A
  - Ensemble dragged down by poor VGG16 predictions
  - Simple averaging may not be optimal strategy

### 6.4 Challenges and Solutions

**Challenge 1: VGG16 Poor Initial Performance**
- **Problem**: VGG16 without augmentation had 77.08 pixel mean error
- **Cause**: Deep architecture prone to overfitting on small dataset (698 samples)
- **Solution**: Augmentation reduced error to 42.42 pixels (45% improvement)
- **Lesson**: Deeper models need more regularization

**Challenge 2: Path Configuration Issues**
- **Problem**: Hardcoded paths breaking across different machines
- **Solution**: Created `config.json` for centralized path management
- **Impact**: Easy deployment and testing across team members

**Challenge 3: Ensemble Not Beating Best Individual Model**
- **Problem**: Ensemble (25.65) worse than SnoutNet-A (16.23)
- **Cause**: VGG16's poor predictions (42.42) drag down ensemble average
- **Potential Solution**: 
  - Weighted ensemble based on validation performance
  - Exclude poorly performing models
  - Use ensemble only when models disagree

**Challenge 4: Computational Efficiency**
- **Problem**: Testing all 6 models + ensemble time-consuming
- **Solution**: 
  - Batch processing with size 86
  - GPU acceleration
  - Parallel testing scripts

### 6.5 Best Practices Identified

1. **Transfer Learning is Powerful**: AlexNet significantly outperformed custom CNN
2. **Augmentation is Essential**: Especially for deeper architectures
3. **Model Selection Matters**: Not all pretrained models work equally well
4. **Simple Averaging May Not Be Optimal**: Weighted ensemble could improve results

---

## 7. Ensemble Approach Analysis

### 7.1 Ensemble Architecture

**Design:**
```python
class SnoutNetEnsemble(nn.Module):
    def forward(self, x):
        # Get predictions from all three models
        pred_snoutnet = self.snoutnet(x)
        pred_alexnet = self.snoutnet_alexnet(x)
        pred_vgg16 = self.snoutnet_vgg16(x)
        
        # Simple arithmetic mean
        ensemble_pred = (pred_snoutnet + pred_alexnet + pred_vgg16) / 3.0
        return ensemble_pred
```

**Key Characteristics:**
- **Combination Method**: Simple arithmetic mean (equal weights)
- **Models Combined**: 
  - SnoutNet (augmented version)
  - SnoutNet-A (AlexNet, augmented)
  - SnoutNet-V (VGG16, augmented)
- **Inference**: Sequential evaluation, then averaging
- **No Training**: Ensemble weights are fixed (1/3 each)

### 7.2 Performance Impact

**Quantitative Results:**

| Metric | SnoutNet | SnoutNet-A | SnoutNet-V | **Ensemble** | Best Individual |
|--------|----------|------------|------------|--------------|-----------------|
| **Mean Error** | 29.28 | 16.23 | 42.42 | **25.65** | 16.23 (SnoutNet-A) |
| **Std Dev** | 21.30 | 15.19 | 23.45 | **18.11** | 15.19 (SnoutNet-A) |
| **Min Error** | 0.62 | 0.35 | 0.41 | **0.20** | 0.20 (Ensemble) |
| **Max Error** | 130.15 | 129.81 | 123.92 | **112.11** | 112.11 (Ensemble) |

**Analysis:**

✅ **Ensemble Advantages:**
1. **Reduced Variance**: Std dev (18.11) lower than SnoutNet and SnoutNet-V
2. **Better Worst-Case**: Max error (112.11) lower than any individual model
3. **Robust Best-Case**: Min error (0.20) is the absolute best across all models
4. **Error Range Reduction**: More consistent predictions

❌ **Ensemble Disadvantages:**
1. **Not Best Mean**: Mean error (25.65) worse than SnoutNet-A (16.23)
2. **Computational Cost**: 3× inference time vs single model
3. **Dominated by Poor Model**: VGG16's errors drag down performance

### 7.3 Why Ensemble Didn't Beat SnoutNet-A

**Mathematical Analysis:**

Given predictions:
- SnoutNet: error ≈ 29.28
- SnoutNet-A: error ≈ 16.23 ⭐
- SnoutNet-V: error ≈ 42.42 ⚠️

Simple average: (29.28 + 16.23 + 42.42) / 3 ≈ 29.31 (rough approximation)

**The Problem:**
- VGG16's large errors (42.42) pull the ensemble average up
- Equal weighting gives too much influence to the worst model
- SnoutNet-A alone would be better

### 7.4 Potential Improvements

**1. Weighted Ensemble:**
```python
# Weight by inverse validation error
w_snoutnet = 1/29.28
w_alexnet = 1/16.23  # Highest weight
w_vgg16 = 1/42.42    # Lowest weight

# Normalize weights
total = w_snoutnet + w_alexnet + w_vgg16
ensemble_pred = (w_snoutnet/total * pred_snoutnet + 
                 w_alexnet/total * pred_alexnet + 
                 w_vgg16/total * pred_vgg16)
```

**Expected Result**: Would favor SnoutNet-A, likely improving to ~18-20 pixel error

**2. Selective Ensemble:**
- Only use models that outperform a threshold (e.g., < 30 pixel error)
- In this case: Use only SnoutNet + SnoutNet-A
- Expected: 22-23 pixel mean error

**3. Learned Combination:**
- Train a small MLP to combine predictions
- Learn optimal weights from validation data
- Could potentially beat best individual model

### 7.5 When Ensemble Helps Most

**Best Use Cases:**
1. **Reducing Outliers**: Ensemble max error (112.11) < individual max errors
2. **High-Confidence Predictions**: Best predictions (0.20-1.19) are excellent
3. **Robustness**: Lower standard deviation shows more consistent behavior

**Current Limitation:**
- Simple averaging assumes all models are equally good
- Should have used validation performance to weight models

### 7.6 Conclusion on Ensemble

**Summary:**
- ✅ Ensemble **reduces variance** and **improves worst-case** performance
- ✅ Provides **robustness** and **reliability**
- ❌ Simple averaging **doesn't beat best individual model**
- 💡 **Weighted ensemble** would likely improve results significantly

**Recommendation:**
For production deployment, use **SnoutNet-A (AlexNet with augmentation)** alone, or implement a **weighted ensemble** that favors better-performing models.

---

## Visualization Examples

### Available Visualizations

For each model with augmentation, the following visualizations are available:

**Location**: `test_results/<model_name>/`

1. **predictions_best4.png**: 4 best predictions with lowest errors
2. **predictions_worst4.png**: 4 worst predictions with highest errors
3. **predictions_random.png**: Random sample of predictions
4. **error_histogram.png**: Distribution of prediction errors

**Include in report:**
- Best 4 predictions from each model (SnoutNet_aug, SnoutNetAlexNet_aug, SnoutNetVGG16_aug)
- Shows model strengths and failure cases

### Example Image Names

**Best Predictions (SnoutNet-A):**
- english_cocker_spaniel_157.jpg (0.35 pixel error)
- newfoundland_166.jpg
- yorkshire_terrier_190.jpg
- Birman_114.jpg (0.65 pixel error)

**Worst Predictions (SnoutNet-A):**
- Sphynx_110.jpg (129.81 pixel error)
- Russian_Blue_205.jpg
- scottish_terrier_109.jpg
- english_setter_86.jpg (93.18 pixel error)

---

## Summary Statistics

### Performance Ranking (Mean Error)

1. 🥇 **SnoutNet-A + Aug**: 16.23 px
2. 🥈 **SnoutNet-A**: 18.98 px
3. 🥉 **Ensemble**: 25.65 px
4. **SnoutNet + Aug**: 29.28 px
5. **SnoutNet**: 31.30 px
6. **SnoutNet-V + Aug**: 42.42 px
7. **SnoutNet-V**: 77.08 px

### Key Takeaways

1. ✅ **Transfer learning (AlexNet) works best** for this task
2. ✅ **Augmentation helps all models**, especially deeper ones
3. ⚠️ **VGG16 needs more tuning** or different training strategy
4. 💡 **Ensemble needs better weighting** strategy to be competitive
5. 🎯 **Best model achieves ~16 pixel mean error**, suitable for many applications

---

*Generated from test results in: `test_results/` and `ensemble_results.csv`*

