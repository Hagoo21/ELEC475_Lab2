import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

from ensemble_model import SnoutNetEnsemble
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
BASE_PATH = CONFIG['base_path']


def compute_euclidean_distance(pred_coords, true_coords):
    """
    Compute the Euclidean distance between predicted and true coordinates.
    
    Args:
        pred_coords: Predicted coordinates [batch, 2]
        true_coords: Ground truth coordinates [batch, 2]
    
    Returns:
        Distances for each sample [batch]
    """
    distances = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=1))
    return distances


def visualize_ensemble_predictions(ensemble, dataset, num_samples=4, 
                                   save_path='ensemble_predictions.png',
                                   indices=None):
    """
    Visualize ensemble predictions vs ground truth for test images.
    
    Args:
        ensemble: SnoutNetEnsemble model
        dataset: Test dataset
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
        indices: Optional list of specific indices to visualize
    """
    print("=" * 70)
    print(f"Visualizing Ensemble Predictions")
    print("=" * 70)
    
    ensemble.eval()
    device = ensemble.device
    
    # Select indices to visualize
    if indices is None:
        # Randomly select samples
        indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    else:
        indices = indices[:num_samples]
    
    num_samples = len(indices)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    print(f"\nProcessing {num_samples} images...")
    
    for i, idx in enumerate(indices):
        # Get image and label
        image_tensor, label_tensor = dataset[idx]
        filename = dataset.get_filename(idx)
        
        # Add batch dimension and move to device
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        # Get ensemble prediction
        with torch.no_grad():
            pred = ensemble(image_batch)
        
        # Get individual model predictions for display
        individual_preds = ensemble.get_individual_predictions(image_batch)
        
        # DEBUG: Print raw predictions
        if i == 0:
            print(f"\nDEBUG - Raw predictions for first image:")
            print(f"  SnoutNet:  {individual_preds['snoutnet'][0].cpu().numpy()}")
            print(f"  AlexNet:   {individual_preds['alexnet'][0].cpu().numpy()}")
            print(f"  VGG16:     {individual_preds['vgg16'][0].cpu().numpy()}")
            print(f"  Ensemble:  {individual_preds['ensemble'][0].cpu().numpy()}")
            print(f"  True:      {label_tensor.numpy()}")
        
        # Convert to numpy
        pred_x, pred_y = pred[0].cpu().numpy()
        true_x, true_y = label_tensor.numpy()
        
        # Compute distance
        distance = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        
        # Load and display image
        img_path = os.path.join(dataset.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((227, 227))
        
        # Plot image
        axes[i].imshow(image)
        
        # Plot ground truth (green circle)
        axes[i].scatter([true_x], [true_y], c='green', s=200, marker='o', 
                       label='Ground Truth', linewidths=2.5, edgecolors='white', zorder=5)
        
        # Plot ensemble prediction (red X)
        axes[i].scatter([pred_x], [pred_y], c='red', s=200, marker='x', 
                       label='Ensemble', linewidths=3.5, zorder=5)
        
        # Plot individual model predictions (smaller, transparent)
        snoutnet_pred = individual_preds['snoutnet'][0].cpu().numpy()
        alexnet_pred = individual_preds['alexnet'][0].cpu().numpy()
        vgg16_pred = individual_preds['vgg16'][0].cpu().numpy()
        
        axes[i].scatter([snoutnet_pred[0]], [snoutnet_pred[1]], c='blue', s=80, 
                       marker='s', alpha=0.6, label='SnoutNet', edgecolors='white', 
                       linewidths=1, zorder=4)
        axes[i].scatter([alexnet_pred[0]], [alexnet_pred[1]], c='orange', s=80, 
                       marker='^', alpha=0.6, label='AlexNet', edgecolors='white', 
                       linewidths=1, zorder=4)
        axes[i].scatter([vgg16_pred[0]], [vgg16_pred[1]], c='purple', s=80, 
                       marker='D', alpha=0.6, label='VGG16', edgecolors='white', 
                       linewidths=1, zorder=4)
        
        # Draw line from ground truth to ensemble prediction
        axes[i].plot([true_x, pred_x], [true_y, pred_y], 'y--', 
                    linewidth=2.5, alpha=0.8, zorder=3)
        
        # Set title with filename and error
        axes[i].set_title(f'{filename}\nEnsemble Error: {distance:.2f} px', 
                         fontsize=10, fontweight='bold')
        
        # Add legend
        axes[i].legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        # Set axis limits to show full image range
        axes[i].set_xlim(-10, 237)
        axes[i].set_ylim(237, -10)  # Inverted for image coordinates
        axes[i].axis('off')
        
        print(f"  {i+1}. {filename:30s} - Error: {distance:.2f} px")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    print("=" * 70)
    
    return fig


def evaluate_ensemble(ensemble, dataset, batch_size=32):
    """
    Evaluate the ensemble model on the entire test dataset.
    
    Args:
        ensemble: SnoutNetEnsemble model
        dataset: Test dataset
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 70)
    print("Evaluating Ensemble Model")
    print("=" * 70)
    
    ensemble.eval()
    device = ensemble.device
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_predictions = []
    all_ground_truths = []
    all_distances = []
    
    print(f"Processing {len(dataset)} samples...")
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get ensemble predictions
            outputs = ensemble(images)
            
            # Compute distances
            distances = compute_euclidean_distance(outputs, labels)
            
            all_predictions.append(outputs.cpu().numpy())
            all_ground_truths.append(labels.cpu().numpy())
            all_distances.append(distances.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.vstack(all_predictions)
    ground_truths = np.vstack(all_ground_truths)
    distances = np.concatenate(all_distances)
    
    # Compute statistics
    metrics = {
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'std_distance': np.std(distances),
        'num_samples': len(distances)
    }
    
    print(f"\nResults:")
    print(f"  Samples:  {metrics['num_samples']}")
    print(f"  Min:      {metrics['min_distance']:.4f} px")
    print(f"  Max:      {metrics['max_distance']:.4f} px")
    print(f"  Mean:     {metrics['mean_distance']:.4f} px")
    print(f"  Median:   {metrics['median_distance']:.4f} px")
    print(f"  Std Dev:  {metrics['std_distance']:.4f} px")
    print("=" * 70)
    
    return metrics, predictions, ground_truths, distances


def main():
    """
    Main function to run ensemble visualization.
    Assumes weight files are stored in the 'model_weights' folder.
    Paths are loaded from config.json
    """
    print("\n" + "=" * 70)
    print("SnoutNet Ensemble Visualization")
    print("=" * 70)
    
    # Default weight file paths from config
    model_weights_dir = os.path.join(BASE_PATH, CONFIG['paths']['model_weights_dir'])
    snoutnet_path = os.path.join(model_weights_dir, 'SnoutNet.pt')
    alexnet_path = os.path.join(model_weights_dir, 'SnoutNetAlexNet.pt')
    vgg16_path = os.path.join(model_weights_dir, 'SnoutNetVGG16.pt')
    
    # Test dataset paths from config
    test_file = os.path.join(BASE_PATH, CONFIG['paths']['test_file'])
    img_dir = os.path.join(BASE_PATH, CONFIG['paths']['img_dir'])
    
    print(f"\nWeight files:")
    print(f"  SnoutNet:  {snoutnet_path}")
    print(f"  AlexNet:   {alexnet_path}")
    print(f"  VGG16:     {vgg16_path}")
    print(f"\nTest data:")
    print(f"  Annotations: {test_file}")
    print(f"  Images:      {img_dir}")
    print()
    
    # Check if weight files exist
    weights_exist = all([
        os.path.exists(snoutnet_path),
        os.path.exists(alexnet_path),
        os.path.exists(vgg16_path)
    ])
    
    if not weights_exist:
        print("⚠️  Warning: Some weight files are missing!")
        print("   The ensemble will use random initialization for missing models.")
        print("   To use trained models, place .pt files in the 'model_weights' folder:")
        print(f"   - {snoutnet_path}")
        print(f"   - {alexnet_path}")
        print(f"   - {vgg16_path}")
        print()
    
    # Load ensemble model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()
    
    ensemble = SnoutNetEnsemble(
        snoutnet_path=snoutnet_path,
        alexnet_path=alexnet_path,
        vgg16_path=vgg16_path,
        device=device
    )
    
    # Load test dataset
    print(f"\nLoading test dataset...")
    test_dataset = PetNoseDataset(
        annotations_file=test_file,
        img_dir=img_dir,
        target_size=(227, 227)
    )
    
    # Visualize predictions for 4 random test images
    print()
    visualize_ensemble_predictions(
        ensemble=ensemble,
        dataset=test_dataset,
        num_samples=4,
        save_path='ensemble_predictions.png'
    )
    
    # Optional: Evaluate on entire test set
    print("\nPerforming full evaluation...")
    metrics, predictions, ground_truths, distances = evaluate_ensemble(
        ensemble=ensemble,
        dataset=test_dataset,
        batch_size=32
    )
    
    # Save results to CSV
    csv_path = 'ensemble_results.csv'
    with open(csv_path, 'w') as f:
        f.write('filename,pred_x,pred_y,true_x,true_y,distance\n')
        for i in range(len(test_dataset)):
            filename = test_dataset.get_filename(i)
            pred_x, pred_y = predictions[i]
            true_x, true_y = ground_truths[i]
            dist = distances[i]
            f.write(f'{filename},{pred_x:.4f},{pred_y:.4f},{true_x:.4f},{true_y:.4f},{dist:.4f}\n')
    print(f"\n✓ Results saved to {csv_path}")
    
    # Create histogram of errors
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(metrics['mean_distance'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {metrics['mean_distance']:.2f} px")
    plt.axvline(metrics['median_distance'], color='green', linestyle='--', 
                linewidth=2, label=f"Median: {metrics['median_distance']:.2f} px")
    plt.xlabel('Euclidean Distance (pixels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Ensemble Prediction Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    hist_path = 'ensemble_error_histogram.png'
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"✓ Error histogram saved to {hist_path}")
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()

