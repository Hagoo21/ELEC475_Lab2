"""
Visualization script for SnoutNet models

This script loads a trained model and generates visualizations of predictions
on the test dataset.
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

from dataset import PetNoseDataset


# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

def load_config():
    """Load configuration from config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

# Load config
CONFIG = load_config()
BASE_PATH = os.path.dirname(os.path.abspath(__file__)) if CONFIG['base_path'] == '.' else CONFIG['base_path']

# Construct default paths from config (normalize for cross-platform compatibility)
DEFAULT_TEST_FILE = os.path.normpath(os.path.join(BASE_PATH, CONFIG['paths']['test_file']))
DEFAULT_IMG_DIR = os.path.normpath(os.path.join(BASE_PATH, CONFIG['paths']['img_dir']))
DEFAULT_MODEL_WEIGHTS_DIR = os.path.normpath(os.path.join(BASE_PATH, CONFIG['paths']['model_weights_dir']))
DEFAULT_BATCH_SIZE = CONFIG['testing']['batch_size']


def compute_euclidean_distance(pred_coords, true_coords):
    """Compute Euclidean distance between predicted and true coordinates."""
    distances = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=1))
    return distances


def evaluate_model(model, dataloader, device):
    """Evaluate model on test set and return predictions and distances."""
    model.eval()
    all_predictions = []
    all_ground_truths = []
    all_distances = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            distances = compute_euclidean_distance(outputs, labels)
            
            all_predictions.append(outputs.cpu().numpy())
            all_ground_truths.append(labels.cpu().numpy())
            all_distances.append(distances.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    ground_truths = np.vstack(all_ground_truths)
    distances = np.concatenate(all_distances)
    
    return predictions, ground_truths, distances


def visualize_predictions(dataset, predictions, ground_truths, distances, 
                         indices, save_path, title_prefix=''):
    """Visualize predictions for specific indices."""
    num_samples = len(indices)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        filename = dataset.get_filename(idx)
        img_path = os.path.join(dataset.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((227, 227))
        
        pred_x, pred_y = predictions[idx]
        true_x, true_y = ground_truths[idx]
        dist = distances[idx]
        
        axes[i].imshow(image)
        axes[i].scatter([true_x], [true_y], c='green', s=150, marker='o', 
                        label='Ground Truth', linewidths=2, edgecolors='white')
        axes[i].scatter([pred_x], [pred_y], c='red', s=150, marker='x', 
                        label='Prediction', linewidths=3)
        axes[i].plot([true_x, pred_x], [true_y, pred_y], 'y--', linewidth=2, alpha=0.7)
        
        axes[i].set_title(f'{filename}\nDistance: {dist:.2f} px', fontsize=9)
        axes[i].legend(loc='upper right', fontsize=7)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ {title_prefix} visualization saved to {save_path}")
    plt.close()


def visualize_ensemble_predictions(ensemble, dataset, num_samples=4, save_path='ensemble_predictions.png'):
    """Visualize ensemble predictions with individual model predictions."""
    from visualize_ensemble import visualize_ensemble_predictions as vis_ensemble
    
    # Randomly select samples
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    vis_ensemble(
        ensemble=ensemble,
        dataset=dataset,
        num_samples=num_samples,
        save_path=save_path,
        indices=indices
    )


def main():
    parser = argparse.ArgumentParser(description='Visualize SnoutNet model predictions')
    parser.add_argument('--model', type=str, default='snoutnet',
                        choices=['snoutnet', 'alexnet', 'vgg16', 'ensemble'],
                        help='Model architecture to visualize')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights file (.pt)')
    parser.add_argument('--weights-dir', type=str, default=None,
                        help='Directory containing model weights (for ensemble)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--num-samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--show-best-worst', action='store_true',
                        help='Show best and worst predictions in addition to random')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Visualizing {args.model.upper()} Model")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = PetNoseDataset(
        annotations_file=DEFAULT_TEST_FILE,
        img_dir=DEFAULT_IMG_DIR,
        target_size=(227, 227)
    )
    print(f"Test samples: {len(test_dataset)}")
    
    if args.model == 'ensemble':
        # Handle ensemble visualization
        from ensemble_model import SnoutNetEnsemble
        
        weights_dir = args.weights_dir if args.weights_dir else DEFAULT_MODEL_WEIGHTS_DIR
        
        snoutnet_path = os.path.join(weights_dir, 'SnoutNet.pt')
        alexnet_path = os.path.join(weights_dir, 'SnoutNetAlexNet.pt')
        vgg16_path = os.path.join(weights_dir, 'SnoutNetVGG16.pt')
        
        print("\nLoading ensemble model...")
        ensemble = SnoutNetEnsemble(
            snoutnet_path=snoutnet_path,
            alexnet_path=alexnet_path,
            vgg16_path=vgg16_path,
            device=device
        )
        
        # Generate ensemble visualization
        save_path = os.path.join(args.output_dir, 'ensemble_predictions.png')
        visualize_ensemble_predictions(ensemble, test_dataset, args.num_samples, save_path)
        
    else:
        # Handle individual model visualization
        if args.weights is None:
            # Use default weights path
            if args.model == 'snoutnet':
                model_file = 'SnoutNet.pt'
                from model import SnoutNet
                model = SnoutNet()
                model_name = 'SnoutNet'
            elif args.model == 'alexnet':
                model_file = 'SnoutNetAlexNet.pt'
                from snoutnet_alexnet import SnoutNetAlexNet
                model = SnoutNetAlexNet(pretrained=False)
                model_name = 'SnoutNetAlexNet'
            elif args.model == 'vgg16':
                model_file = 'SnoutNetVGG16.pt'
                from snoutnet_vgg16 import SnoutNetVGG16
                model = SnoutNetVGG16(pretrained=False)
                model_name = 'SnoutNetVGG16'
            
            weights_path = os.path.join(DEFAULT_MODEL_WEIGHTS_DIR, model_file)
        else:
            weights_path = args.weights
            if args.model == 'snoutnet':
                from model import SnoutNet
                model = SnoutNet()
                model_name = 'SnoutNet'
            elif args.model == 'alexnet':
                from snoutnet_alexnet import SnoutNetAlexNet
                model = SnoutNetAlexNet(pretrained=False)
                model_name = 'SnoutNetAlexNet'
            elif args.model == 'vgg16':
                from snoutnet_vgg16 import SnoutNetVGG16
                model = SnoutNetVGG16(pretrained=False)
                model_name = 'SnoutNetVGG16'
        
        print(f"\nLoading model from {weights_path}...")
        model = model.to(device)
        
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded (Epoch {checkpoint.get('epoch', 'N/A')}, "
                  f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f})")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return
        
        # Evaluate model to get predictions
        print("\nEvaluating model on test set...")
        test_loader = DataLoader(test_dataset, batch_size=DEFAULT_BATCH_SIZE, 
                                shuffle=False, num_workers=0)
        predictions, ground_truths, distances = evaluate_model(model, test_loader, device)
        
        # Statistics
        mean_dist = np.mean(distances)
        print(f"Mean distance: {mean_dist:.4f} px")
        
        # Random samples visualization
        print(f"\nGenerating visualizations...")
        random_indices = np.random.choice(len(test_dataset), 
                                         size=min(args.num_samples, len(test_dataset)), 
                                         replace=False)
        save_path = os.path.join(args.output_dir, f'{model_name}_predictions_random.png')
        visualize_predictions(test_dataset, predictions, ground_truths, distances,
                            random_indices, save_path, "Random samples")
        
        if args.show_best_worst:
            # Best 4 predictions
            sorted_indices = np.argsort(distances)
            best_4_indices = sorted_indices[:4]
            save_path = os.path.join(args.output_dir, f'{model_name}_predictions_best4.png')
            visualize_predictions(test_dataset, predictions, ground_truths, distances,
                                best_4_indices, save_path, "Best 4")
            
            # Worst 4 predictions
            worst_4_indices = sorted_indices[-4:]
            save_path = os.path.join(args.output_dir, f'{model_name}_predictions_worst4.png')
            visualize_predictions(test_dataset, predictions, ground_truths, distances,
                                worst_4_indices, save_path, "Worst 4")
    
    print(f"\n✓ All visualizations saved to {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

