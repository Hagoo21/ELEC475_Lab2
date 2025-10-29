"""
Debug script to check if ensemble models are loading correctly.
"""

import os
import torch
from ensemble_model import SnoutNetEnsemble
from dataset import PetNoseDataset

def debug_weights():
    print("=" * 70)
    print("DEBUG: Checking Model Weights")
    print("=" * 70)
    
    # Check if weight files exist (using absolute paths)
    base_path = r'C:/Users/20mmz2/ELEC475_Lab2'
    snoutnet_path = os.path.join(base_path, 'model_weights', 'SnoutNet.pt')
    alexnet_path = os.path.join(base_path, 'model_weights', 'SnoutNetAlexNet.pt')
    vgg16_path = os.path.join(base_path, 'model_weights', 'SnoutNetVGG16.pt')
    
    print(f"\nChecking weight files:")
    print(f"  SnoutNet:  {snoutnet_path} - {'✓ EXISTS' if os.path.exists(snoutnet_path) else '✗ MISSING'}")
    print(f"  AlexNet:   {alexnet_path} - {'✓ EXISTS' if os.path.exists(alexnet_path) else '✗ MISSING'}")
    print(f"  VGG16:     {vgg16_path} - {'✓ EXISTS' if os.path.exists(vgg16_path) else '✗ MISSING'}")
    
    # Check checkpoint structure
    if os.path.exists(snoutnet_path):
        print(f"\n\nInspecting {snoutnet_path}:")
        checkpoint = torch.load(snoutnet_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            print(f"  Type: Dictionary")
            print(f"  Keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                print(f"  ✓ Has 'model_state_dict' key")
                state_dict = checkpoint['model_state_dict']
            else:
                print(f"  ✗ No 'model_state_dict' key - assuming checkpoint IS the state_dict")
                state_dict = checkpoint
            
            # Check first few weights
            print(f"\n  Sample weights (first 5 values of first layer):")
            first_key = list(state_dict.keys())[0]
            first_weights = state_dict[first_key].flatten()[:5]
            print(f"    {first_key}: {first_weights}")
            
            # Check if weights are zeros
            all_zeros = all(torch.allclose(param, torch.zeros_like(param)) for param in state_dict.values())
            if all_zeros:
                print(f"  ⚠️  WARNING: All weights are zeros!")
            else:
                print(f"  ✓ Weights are non-zero")
        else:
            print(f"  Type: {type(checkpoint)}")
    
    # Load ensemble and test
    print(f"\n\nLoading ensemble:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensemble = SnoutNetEnsemble(
        snoutnet_path=snoutnet_path,
        alexnet_path=alexnet_path,
        vgg16_path=vgg16_path,
        device=device
    )
    
    # Test with a real image
    print(f"\n\nTesting with real image:")
    test_file = os.path.join(base_path, 'oxford-iiit-pet-noses', 'test_noses.txt')
    img_dir = os.path.join(base_path, 'oxford-iiit-pet-noses', 'images-original', 'images')
    
    dataset = PetNoseDataset(
        annotations_file=test_file,
        img_dir=img_dir,
        target_size=(227, 227)
    )
    
    # Get first test image
    image, label = dataset[0]
    filename = dataset.get_filename(0)
    
    print(f"\n  Image: {filename}")
    print(f"  Ground truth: {label.numpy()}")
    print(f"  Image shape: {image.shape}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Make predictions
    image_batch = image.unsqueeze(0).to(device)
    
    ensemble.eval()
    with torch.no_grad():
        predictions = ensemble.get_individual_predictions(image_batch)
    
    print(f"\n  Predictions:")
    for model_name, pred in predictions.items():
        pred_np = pred[0].cpu().numpy()
        print(f"    {model_name:12s}: [{pred_np[0]:8.2f}, {pred_np[1]:8.2f}]")
    
    # Check if predictions are reasonable
    print(f"\n\nDiagnosis:")
    ensemble_pred = predictions['ensemble'][0].cpu().numpy()
    
    if abs(ensemble_pred[0]) < 1.0 and abs(ensemble_pred[1]) < 1.0:
        print(f"  ⚠️  PROBLEM: Predictions are near zero!")
        print(f"  Possible causes:")
        print(f"    1. Weights didn't load correctly")
        print(f"    2. Models are in wrong mode (training vs eval)")
        print(f"    3. Checkpoint format mismatch")
        print(f"    4. Models were trained on different input format")
    elif ensemble_pred[0] < 0 or ensemble_pred[1] < 0:
        print(f"  ⚠️  PROBLEM: Predictions are negative!")
        print(f"  This is unusual for pixel coordinates")
    elif ensemble_pred[0] > 227 or ensemble_pred[1] > 227:
        print(f"  ⚠️  WARNING: Predictions are outside image bounds")
        print(f"  But this might be okay")
    else:
        print(f"  ✓ Predictions look reasonable!")
    
    print("=" * 70)

if __name__ == "__main__":
    debug_weights()

