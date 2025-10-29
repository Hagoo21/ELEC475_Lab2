================================================================================
MODEL WEIGHTS FOLDER
================================================================================

This folder should contain the trained model weights for the ensemble.

REQUIRED FILES:
---------------

1. snoutnet.pt
   - SnoutNet (Custom CNN) model weights
   - Train using: train.py or train_cases.py

2. snoutnet_alexnet.pt
   - SnoutNet-A (AlexNet-based) model weights
   - Train using the AlexNet training script

3. snoutnet_vgg16.pt
   - SnoutNet-V (VGG16-based) model weights
   - Train using the VGG16 training script

================================================================================

FILE FORMAT:
------------

Each .pt file should be a PyTorch checkpoint dictionary containing:
  - 'model_state_dict': The model's state dictionary
  - 'epoch': Training epoch (optional)
  - 'val_loss': Validation loss (optional)
  - 'optimizer_state_dict': Optimizer state (optional)

Example structure:
    {
        'epoch': 50,
        'model_state_dict': ...,
        'optimizer_state_dict': ...,
        'train_loss': 45.234,
        'val_loss': 42.567
    }

================================================================================

HOW TO GET WEIGHTS:
-------------------

Option 1: Train from scratch
    python train.py            # For SnoutNet
    python train_cases.py      # For different configurations
    
    Copy the resulting .pt files to this folder and rename them:
    - weights/run_XXXXX/snoutnet.pt -> model weights/snoutnet.pt
    - (similar for AlexNet and VGG16)

Option 2: Use pretrained weights
    If you have pretrained weights, place them here with the correct names

Option 3: Test without weights
    The ensemble will still initialize but use random weights
    (for testing the ensemble structure only)

================================================================================

USAGE:
------

Once weights are placed here, run the ensemble:
    python visualize_ensemble.py

Or use programmatically:
    from ensemble_model import SnoutNetEnsemble
    
    ensemble = SnoutNetEnsemble(
        snoutnet_path='model weights/snoutnet.pt',
        alexnet_path='model weights/snoutnet_alexnet.pt',
        vgg16_path='model weights/snoutnet_vgg16.pt'
    )

================================================================================

NOTES:
------

- File names must match exactly (case-sensitive on some systems)
- All three models should use input size 227x227
- Each model outputs [batch, 2] coordinates
- The ensemble averages predictions across all three models

================================================================================

