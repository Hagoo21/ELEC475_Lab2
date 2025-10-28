"""
SnoutNet Testing Script
ELEC 475 Lab 2

This script evaluates a trained SnoutNet model on the test dataset.

Features:
- Loads trained model checkpoint
- Computes Euclidean distance between predicted and ground-truth coordinates
- Outputs statistics: min, max, mean, std of distances
- Saves results to CSV
- Optional visualization of predictions

Author: Generated for ELEC 475 Lab 2
Date: 2025
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

from model import SnoutNet
from dataset import PetNoseDataset


def compute_euclidean_distance(pred_coords, true_coords):
    """
    Compute Euclidean distance between predicted and true coordinates.
    
    Args:
        pred_coords (torch.Tensor): Predicted (x, y) coordinates
        true_coords (torch.Tensor): Ground-truth (x, y) coordinates
        
    Returns:
        torch.Tensor: Euclidean distances
    """
    # pred_coords and true_coords are of shape [batch_size, 2]
    distances = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=1))
    return distances


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The trained neural network model
        dataloader: DataLoader for test data
        device: Device to evaluate on (cuda or cpu)
        
    Returns:
        tuple: (predictions, ground_truths, distances)
            - predictions: List of predicted (x, y) coordinates
            - ground_truths: List of ground-truth (x, y) coordinates
            - distances: List of Euclidean distances
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    all_distances = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute distances
            distances = compute_euclidean_distance(outputs, labels)
            
            # Store results
            all_predictions.append(outputs.cpu().numpy())
            all_ground_truths.append(labels.cpu().numpy())
            all_distances.append(distances.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.vstack(all_predictions)
    ground_truths = np.vstack(all_ground_truths)
    distances = np.concatenate(all_distances)
    
    return predictions, ground_truths, distances


def save_results_to_csv(filenames, predictions, ground_truths, distances, save_path='test_results.csv'):
    """
    Save evaluation results to a CSV file.
    
    Args:
        filenames (list): List of image filenames
        predictions (np.ndarray): Predicted coordinates
        ground_truths (np.ndarray): Ground-truth coordinates
        distances (np.ndarray): Euclidean distances
        save_path (str): Path to save the CSV file
    """
    with open(save_path, 'w') as f:
        f.write('filename,pred_x,pred_y,true_x,true_y,distance\n')
        for i, filename in enumerate(filenames):
            pred_x, pred_y = predictions[i]
            true_x, true_y = ground_truths[i]
            dist = distances[i]
            f.write(f'{filename},{pred_x:.4f},{pred_y:.4f},{true_x:.4f},{true_y:.4f},{dist:.4f}\n')
    print(f"Results saved to {save_path}")


def visualize_predictions(dataset, predictions, ground_truths, distances, num_samples=4, save_path='predictions_visualization.png'):
    """
    Visualize predictions by showing images with ground-truth and predicted points.
    
    Args:
        dataset (PetNoseDataset): Dataset instance
        predictions (np.ndarray): Predicted coordinates
        ground_truths (np.ndarray): Ground-truth coordinates
        distances (np.ndarray): Euclidean distances
        num_samples (int): Number of samples to visualize
        save_path (str): Path to save the visualization
    """
    # Select samples to visualize: include best, worst, and some random
    num_samples = min(num_samples, len(dataset))
    
    # Get indices of best and worst predictions
    best_idx = np.argmin(distances)
    worst_idx = np.argmax(distances)
    
    # Get some random indices
    random_indices = np.random.choice(len(dataset), size=max(0, num_samples - 2), replace=False)
    
    # Combine indices
    indices = [best_idx, worst_idx]
    indices.extend(random_indices.tolist())
    indices = indices[:num_samples]
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Load image (without transforms, just for visualization)
        filename = dataset.get_filename(idx)
        img_path = os.path.join(dataset.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = image.resize((227, 227))
        
        # Get coordinates
        pred_x, pred_y = predictions[idx]
        true_x, true_y = ground_truths[idx]
        dist = distances[idx]
        
        # Plot
        axes[i].imshow(image)
        axes[i].scatter([true_x], [true_y], c='green', s=150, marker='o', 
                        label='Ground Truth', linewidths=2, edgecolors='white')
        axes[i].scatter([pred_x], [pred_y], c='red', s=150, marker='x', 
                        label='Prediction', linewidths=3)
        
        # Draw line between predicted and true
        axes[i].plot([true_x, pred_x], [true_y, pred_y], 'y--', linewidth=2, alpha=0.7)
        
        axes[i].set_title(f'{filename}\nDistance: {dist:.2f} px', fontsize=9)
        axes[i].legend(loc='upper right', fontsize=7)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()


def test(args):
    """
    Main testing function.
    
    Args:
        args: Command-line arguments
    """
    print("=" * 70)
    print("SnoutNet Testing")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = SnoutNet().to(device)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully")
        if 'epoch' in checkpoint:
            print(f"  Trained for {checkpoint['epoch']} epochs")
        if 'val_loss' in checkpoint:
            print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    except FileNotFoundError:
        print(f"✗ Error: Model file not found at {args.model_path}")
        print("Please train the model first using train.py")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load dataset
    print(f"\nLoading test dataset...")
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
    ])
    
    test_dataset = PetNoseDataset(
        annotations_file=args.test_file,
        img_dir=args.img_dir,
        transform=transform,
        target_size=(227, 227)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate model
    print(f"\nEvaluating model...")
    predictions, ground_truths, distances = evaluate_model(model, test_loader, device)
    
    # Compute statistics
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    median_dist = np.median(distances)
    
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Total test samples: {len(distances)}")
    print(f"\nDistance Statistics (pixels):")
    print(f"  Min:      {min_dist:.4f}")
    print(f"  Max:      {max_dist:.4f}")
    print(f"  Mean:     {mean_dist:.4f}")
    print(f"  Median:   {median_dist:.4f}")
    print(f"  Std Dev:  {std_dist:.4f}")
    print("=" * 70)
    
    # Get filenames
    filenames = [test_dataset.get_filename(i) for i in range(len(test_dataset))]
    
    # Save results to CSV
    csv_path = args.output_csv
    save_results_to_csv(filenames, predictions, ground_truths, distances, csv_path)
    
    # Save summary statistics
    summary_path = args.output_csv.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SnoutNet Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test samples: {len(distances)}\n")
        f.write(f"\nDistance Statistics (pixels):\n")
        f.write(f"  Min:      {min_dist:.4f}\n")
        f.write(f"  Max:      {max_dist:.4f}\n")
        f.write(f"  Mean:     {mean_dist:.4f}\n")
        f.write(f"  Median:   {median_dist:.4f}\n")
        f.write(f"  Std Dev:  {std_dist:.4f}\n")
    print(f"Summary saved to {summary_path}")
    
    # Plot histogram of distances
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.2f}')
    plt.axvline(median_dist, color='green', linestyle='--', linewidth=2, label=f'Median: {median_dist:.2f}')
    plt.xlabel('Euclidean Distance (pixels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = args.output_csv.replace('.csv', '_histogram.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"Histogram saved to {hist_path}")
    
    # Visualize predictions if requested
    if args.visualize:
        print(f"\nGenerating visualizations...")
        vis_path = args.output_csv.replace('.csv', '_predictions.png')
        visualize_predictions(test_dataset, predictions, ground_truths, distances, 
                            num_samples=args.num_vis_samples, save_path=vis_path)
    
    print("\n✓ Testing complete!")


def main():
    """Parse command-line arguments and start testing."""
    parser = argparse.ArgumentParser(description='Test SnoutNet for pet nose localization')
    
    # Model and dataset arguments
    parser.add_argument('--model_path', type=str, default='weights/snoutnet.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_file', type=str, default='test-noses.txt',
                        help='Path to test annotations file')
    parser.add_argument('--img_dir', type=str, default='images-original/images',
                        help='Directory containing images')
    
    # Output arguments
    parser.add_argument('--output_csv', type=str, default='test_results.csv',
                        help='Path to save results CSV')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions on sample images')
    parser.add_argument('--num_vis_samples', type=int, default=4,
                        help='Number of samples to visualize (default: 4)')
    
    # Other settings
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    
    args = parser.parse_args()
    
    # Start testing
    test(args)


if __name__ == "__main__":
    main()

