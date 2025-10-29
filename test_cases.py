"""
Testing Cases for SnoutNet Models

This file contains 6 testing scenarios:
1. SnoutNet on regular dataset (SnoutNet.pt)
2. SnoutNet on augmented dataset (SnoutNet_aug.pt)
3. SnoutNetAlexNet on regular dataset (SnoutNetAlexNet.pt)
4. SnoutNetAlexNet on augmented dataset (SnoutNetAlexNet_aug.pt)
5. SnoutNetVGG16 on regular dataset (SnoutNetVGG16.pt)
6. SnoutNetVGG16 on augmented dataset (SnoutNetVGG16_aug.pt)

Instructions:
- Uncomment ONE case at a time to run it
- Comment out the other cases
- Each case is clearly marked with a header
- Ensure all model weights are in the model_weights/ folder
- Update base_path in config.json if needed
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

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


# ============================================================================
# HELPER FUNCTIONS (USED BY ALL CASES)
# ============================================================================

def compute_euclidean_distance(pred_coords, true_coords):
    """Compute Euclidean distance between predicted and true coordinates."""
    distances = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=1))
    return distances


def evaluate_model(model, dataloader, device):
    """Evaluate a model on the test set."""
    model.eval()
    all_predictions = []
    all_ground_truths = []
    all_distances = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating', unit='batch')
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            distances = compute_euclidean_distance(outputs, labels)
            
            all_predictions.append(outputs.cpu().numpy())
            all_ground_truths.append(labels.cpu().numpy())
            all_distances.append(distances.cpu().numpy())
            
            avg_dist = distances.mean().item()
            progress_bar.set_postfix({'avg_dist': f'{avg_dist:.2f}px'})
    
    predictions = np.vstack(all_predictions)
    ground_truths = np.vstack(all_ground_truths)
    distances = np.concatenate(all_distances)
    
    return predictions, ground_truths, distances


def save_results_to_csv(filenames, predictions, ground_truths, distances, save_path='test_results.csv'):
    """Save test results to CSV file."""
    with open(save_path, 'w') as f:
        f.write('filename,pred_x,pred_y,true_x,true_y,distance\n')
        for i, filename in enumerate(filenames):
            pred_x, pred_y = predictions[i]
            true_x, true_y = ground_truths[i]
            dist = distances[i]
            f.write(f'{filename},{pred_x:.4f},{pred_y:.4f},{true_x:.4f},{true_y:.4f},{dist:.4f}\n')
    print(f"Results saved to {save_path}")


def visualize_predictions(dataset, predictions, ground_truths, distances, num_samples=4, save_path='predictions_visualization.png', indices=None):
    """Visualize predictions. If indices are provided, use them; otherwise show best, worst, and random."""
    num_samples = min(num_samples, len(dataset))
    
    if indices is None:
        # Default behavior: show best, worst, and random
        best_idx = np.argmin(distances)
        worst_idx = np.argmax(distances)
        random_indices = np.random.choice(len(dataset), size=max(0, num_samples - 2), replace=False)
        
        indices = [best_idx, worst_idx]
        indices.extend(random_indices.tolist())
        indices = indices[:num_samples]
    else:
        # Use provided indices
        indices = list(indices[:num_samples])
    
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
    print(f"Visualization saved to {save_path}")
    plt.close()


def test_model(model, model_path, model_name, test_file, img_dir, output_dir, 
               batch_size=86, visualize=True, num_vis_samples=4):
    """
    Generic testing function that works for any model.
    
    Args:
        model: The model instance to test
        model_path: Path to the model weights file
        model_name: Name of the model (for output files)
        test_file: Path to test annotations file
        img_dir: Directory containing test images
        output_dir: Directory to save results (base directory)
        batch_size: Batch size for testing
        visualize: Whether to generate visualizations
        num_vis_samples: Number of samples to visualize
    """
    # Create a subdirectory for this specific model
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"{model_name} Testing")
    print("=" * 70)
    print(f"Output directory: {model_output_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️  Using CPU")
    print(f"Device: {device}")
    
    print(f"\nLoading model from {model_path}...")
    model = model.to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded (Epoch {checkpoint.get('epoch', 'N/A')}, Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f})")
    except FileNotFoundError:
        print(f"✗ Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print(f"\nLoading test dataset...")
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
    ])
    
    test_dataset = PetNoseDataset(
        annotations_file=test_file,
        img_dir=img_dir,
        transform=transform,
        target_size=(227, 227)
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Test samples: {len(test_dataset)}")
    
    print(f"\nEvaluating model...")
    predictions, ground_truths, distances = evaluate_model(model, test_loader, device)
    
    # Overall statistics
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    median_dist = np.median(distances)
    
    # Get indices of 4 best and 4 worst predictions
    sorted_indices = np.argsort(distances)
    best_4_indices = sorted_indices[:4]
    worst_4_indices = sorted_indices[-4:]
    
    # Statistics for 4 best predictions
    best_4_distances = distances[best_4_indices]
    best_4_min = np.min(best_4_distances)
    best_4_max = np.max(best_4_distances)
    best_4_mean = np.mean(best_4_distances)
    best_4_std = np.std(best_4_distances)
    
    # Statistics for 4 worst predictions
    worst_4_distances = distances[worst_4_indices]
    worst_4_min = np.min(worst_4_distances)
    worst_4_max = np.max(worst_4_distances)
    worst_4_mean = np.mean(worst_4_distances)
    worst_4_std = np.std(worst_4_distances)
    
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Overall ({len(distances)} samples):")
    print(f"  Min: {min_dist:.4f} | Max: {max_dist:.4f} | Mean: {mean_dist:.4f} | Std: {std_dist:.4f}")
    print(f"\n4 Best Predictions:")
    print(f"  Min: {best_4_min:.4f} | Max: {best_4_max:.4f} | Mean: {best_4_mean:.4f} | Std: {best_4_std:.4f}")
    print(f"\n4 Worst Predictions:")
    print(f"  Min: {worst_4_min:.4f} | Max: {worst_4_max:.4f} | Mean: {worst_4_mean:.4f} | Std: {worst_4_std:.4f}")
    print("=" * 70)
    
    # Save results
    filenames = [test_dataset.get_filename(i) for i in range(len(test_dataset))]
    csv_path = os.path.join(model_output_dir, 'results.csv')
    save_results_to_csv(filenames, predictions, ground_truths, distances, csv_path)
    
    # Save comprehensive summary
    summary_path = os.path.join(model_output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{model_name} Evaluation Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Samples: {len(distances)}\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Min:    {min_dist:.4f}\n")
        f.write(f"  Max:    {max_dist:.4f}\n")
        f.write(f"  Mean:   {mean_dist:.4f}\n")
        f.write(f"  Median: {median_dist:.4f}\n")
        f.write(f"  Std:    {std_dist:.4f}\n\n")
        
        f.write("4 BEST PREDICTIONS:\n")
        f.write(f"  Min:  {best_4_min:.4f}\n")
        f.write(f"  Max:  {best_4_max:.4f}\n")
        f.write(f"  Mean: {best_4_mean:.4f}\n")
        f.write(f"  Std:  {best_4_std:.4f}\n")
        f.write(f"  Images: {', '.join([filenames[i] for i in best_4_indices])}\n\n")
        
        f.write("4 WORST PREDICTIONS:\n")
        f.write(f"  Min:  {worst_4_min:.4f}\n")
        f.write(f"  Max:  {worst_4_max:.4f}\n")
        f.write(f"  Mean: {worst_4_mean:.4f}\n")
        f.write(f"  Std:  {worst_4_std:.4f}\n")
        f.write(f"  Images: {', '.join([filenames[i] for i in worst_4_indices])}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TABLE FORMAT (for copy-paste):\n")
        f.write("=" * 70 + "\n")
        f.write(f"Overall: {min_dist:.4f} | {max_dist:.4f} | {mean_dist:.4f} | {std_dist:.4f}\n")
        f.write(f"4 Best:  {best_4_min:.4f} | {best_4_max:.4f} | {best_4_mean:.4f} | {best_4_std:.4f}\n")
        f.write(f"4 Worst: {worst_4_min:.4f} | {worst_4_max:.4f} | {worst_4_mean:.4f} | {worst_4_std:.4f}\n")
    print(f"Summary saved to {summary_path}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(mean_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dist:.2f}')
    plt.axvline(median_dist, color='green', linestyle='--', linewidth=2, label=f'Median: {median_dist:.2f}')
    plt.xlabel('Euclidean Distance (pixels)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{model_name} - Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(model_output_dir, 'error_histogram.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"Histogram saved to {hist_path}")
    plt.close()
    
    # Visualize predictions
    if visualize:
        print(f"\nGenerating visualizations...")
        
        # Random sample visualization
        vis_path = os.path.join(model_output_dir, 'predictions_random.png')
        visualize_predictions(test_dataset, predictions, ground_truths, distances, 
                            num_samples=num_vis_samples, save_path=vis_path)
        
        # Best 4 predictions visualization
        best_vis_path = os.path.join(model_output_dir, 'predictions_best4.png')
        visualize_predictions(test_dataset, predictions, ground_truths, distances, 
                            num_samples=4, save_path=best_vis_path, indices=best_4_indices)
        print(f"  4 Best predictions visualization saved to {best_vis_path}")
        
        # Worst 4 predictions visualization
        worst_vis_path = os.path.join(model_output_dir, 'predictions_worst4.png')
        visualize_predictions(test_dataset, predictions, ground_truths, distances, 
                            num_samples=4, save_path=worst_vis_path, indices=worst_4_indices)
        print(f"  4 Worst predictions visualization saved to {worst_vis_path}")
    
    print("\n✓ Testing complete!")
    
    # Return statistics for potential aggregation
    return {
        'model_name': model_name,
        'overall': {'min': min_dist, 'max': max_dist, 'mean': mean_dist, 'std': std_dist},
        'best_4': {'min': best_4_min, 'max': best_4_max, 'mean': best_4_mean, 'std': best_4_std},
        'worst_4': {'min': worst_4_min, 'max': worst_4_max, 'mean': worst_4_mean, 'std': worst_4_std}
    }


def create_combined_table(results_list, save_path):
    """
    Create a combined table from multiple test results.
    
    Args:
        results_list: List of result dictionaries from test_model
        save_path: Path to save the combined table
    """
    with open(save_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("COMBINED RESULTS TABLE - ALL MODELS\n")
        f.write("=" * 120 + "\n\n")
        
        # Header
        f.write(f"{'Model':<25} | {'Localization Error (Overall)':<35} | {'4 Best':<35} | {'4 Worst':<35}\n")
        f.write(f"{'':<25} | {'min':<8} {'max':<8} {'mean':<8} {'std':<8} | {'min':<8} {'max':<8} {'mean':<8} {'std':<8} | {'min':<8} {'max':<8} {'mean':<8} {'std':<8}\n")
        f.write("-" * 120 + "\n")
        
        # Data rows
        for result in results_list:
            model = result['model_name']
            overall = result['overall']
            best = result['best_4']
            worst = result['worst_4']
            
            f.write(f"{model:<25} | "
                   f"{overall['min']:<8.4f} {overall['max']:<8.4f} {overall['mean']:<8.4f} {overall['std']:<8.4f} | "
                   f"{best['min']:<8.4f} {best['max']:<8.4f} {best['mean']:<8.4f} {best['std']:<8.4f} | "
                   f"{worst['min']:<8.4f} {worst['max']:<8.4f} {worst['mean']:<8.4f} {worst['std']:<8.4f}\n")
        
        f.write("=" * 120 + "\n")
    
    print(f"\n✓ Combined table saved to {save_path}")


# ============================================================================
# DATA PATHS (SHARED BY ALL CASES)
# ============================================================================

# Construct absolute paths from config
TEST_FILE = os.path.join(BASE_PATH, CONFIG['paths']['test_file'])
IMG_DIR = os.path.join(BASE_PATH, CONFIG['paths']['img_dir'])
MODEL_WEIGHTS_DIR = os.path.join(BASE_PATH, CONFIG['paths']['model_weights_dir'])
OUTPUT_DIR = os.path.join(BASE_PATH, CONFIG['paths']['test_results_dir'])

# Testing parameters from config
BATCH_SIZE = CONFIG['testing']['batch_size']
NUM_VIS_SAMPLES = CONFIG['testing']['num_vis_samples']


# ============================================================================
# HOW TO USE:
# ============================================================================
# 1. To test a single model: Uncomment ONE case below
# 2. To test all models and create combined table: Uncomment the "RUN ALL" section at the bottom
# 3. Each test creates a folder in test_results/ with all outputs
# ============================================================================


# ============================================================================
# CASE 1: SnoutNet - Regular Dataset (NO AUGMENTATION)
# ============================================================================
# UNCOMMENT TO RUN:
# from model import SnoutNet
# 
# model = SnoutNet()
# results_1 = test_model(
#     model=model,
#     model_path=os.path.join(MODEL_WEIGHTS_DIR, 'SnoutNet.pt'),
#     model_name='SnoutNet',
#     test_file=TEST_FILE,
#     img_dir=IMG_DIR,
#     output_dir=OUTPUT_DIR,
#     batch_size=BATCH_SIZE,
#     visualize=True,
#     num_vis_samples=NUM_VIS_SAMPLES
# )


# ============================================================================
# CASE 2: SnoutNet - Augmented Dataset
# ============================================================================
# UNCOMMENT TO RUN:
# from model import SnoutNet
# 
# model = SnoutNet()
# results_2 = test_model(
#     model=model,
#     model_path=os.path.join(MODEL_WEIGHTS_DIR, 'SnoutNet_aug.pt'),
#     model_name='SnoutNet_aug',
#     test_file=TEST_FILE,
#     img_dir=IMG_DIR,
#     output_dir=OUTPUT_DIR,
#     batch_size=BATCH_SIZE,
#     visualize=True,
#     num_vis_samples=NUM_VIS_SAMPLES
# )


# ============================================================================
# CASE 3: SnoutNetAlexNet - Regular Dataset (NO AUGMENTATION)
# ============================================================================
# UNCOMMENT TO RUN:
# from snoutnet_alexnet import SnoutNetAlexNet
# 
# model = SnoutNetAlexNet(pretrained=False)  # pretrained=False for testing
# results_3 = test_model(
#     model=model,
#     model_path=os.path.join(MODEL_WEIGHTS_DIR, 'SnoutNetAlexNet.pt'),
#     model_name='SnoutNetAlexNet',
#     test_file=TEST_FILE,
#     img_dir=IMG_DIR,
#     output_dir=OUTPUT_DIR,
#     batch_size=BATCH_SIZE,
#     visualize=True,
#     num_vis_samples=NUM_VIS_SAMPLES
# )


# ============================================================================
# CASE 4: SnoutNetAlexNet - Augmented Dataset
# ============================================================================
# UNCOMMENT TO RUN:
# from snoutnet_alexnet import SnoutNetAlexNet
# 
# model = SnoutNetAlexNet(pretrained=False)  # pretrained=False for testing
# results_4 = test_model(
#     model=model,
#     model_path=os.path.join(MODEL_WEIGHTS_DIR, 'SnoutNetAlexNet_aug.pt'),
#     model_name='SnoutNetAlexNet_aug',
#     test_file=TEST_FILE,
#     img_dir=IMG_DIR,
#     output_dir=OUTPUT_DIR,
#     batch_size=BATCH_SIZE,
#     visualize=True,
#     num_vis_samples=NUM_VIS_SAMPLES
# )


# ============================================================================
# CASE 5: SnoutNetVGG16 - Regular Dataset (NO AUGMENTATION)
# ============================================================================
# UNCOMMENT TO RUN:
# from snoutnet_vgg16 import SnoutNetVGG16
# 
# model = SnoutNetVGG16(pretrained=False)  # pretrained=False for testing
# results_5 = test_model(
#     model=model,
#     model_path=os.path.join(MODEL_WEIGHTS_DIR, 'SnoutNetVGG16.pt'),
#     model_name='SnoutNetVGG16',
#     test_file=TEST_FILE,
#     img_dir=IMG_DIR,
#     output_dir=OUTPUT_DIR,
#     batch_size=BATCH_SIZE,
#     visualize=True,
#     num_vis_samples=NUM_VIS_SAMPLES
# )


# ============================================================================
# CASE 6: SnoutNetVGG16 - Augmented Dataset
# ============================================================================
# UNCOMMENT TO RUN:
# from snoutnet_vgg16 import SnoutNetVGG16
# 
# model = SnoutNetVGG16(pretrained=False)  # pretrained=False for testing
# results_6 = test_model(
#     model=model,
#     model_path=os.path.join(MODEL_WEIGHTS_DIR, 'SnoutNetVGG16_aug.pt'),
#     model_name='SnoutNetVGG16_aug',
#     test_file=TEST_FILE,
#     img_dir=IMG_DIR,
#     output_dir=OUTPUT_DIR,
#     batch_size=BATCH_SIZE,
#     visualize=True,
#     num_vis_samples=NUM_VIS_SAMPLES
# )


# ============================================================================
# RUN ALL MODELS (Optional - for generating combined table)
# ============================================================================
# Uncomment this section to test all models and create a combined results table
# Make sure to comment out individual cases above if using this section
# 
# # Store all results
# all_results = []
# 
# # Test all 6 models
# from model import SnoutNet
# from snoutnet_alexnet import SnoutNetAlexNet
# from snoutnet_vgg16 import SnoutNetVGG16
# 
# models_to_test = [
#     (SnoutNet(), 'SnoutNet.pt', 'SnoutNet'),
#     (SnoutNet(), 'SnoutNet_aug.pt', 'SnoutNet_aug'),
#     (SnoutNetAlexNet(pretrained=False), 'SnoutNetAlexNet.pt', 'SnoutNetAlexNet'),
#     (SnoutNetAlexNet(pretrained=False), 'SnoutNetAlexNet_aug.pt', 'SnoutNetAlexNet_aug'),
#     (SnoutNetVGG16(pretrained=False), 'SnoutNetVGG16.pt', 'SnoutNetVGG16'),
#     (SnoutNetVGG16(pretrained=False), 'SnoutNetVGG16_aug.pt', 'SnoutNetVGG16_aug'),
# ]
# 
# for model, weight_file, model_name in models_to_test:
#     print(f"\n{'='*70}")
#     print(f"Testing {model_name}")
#     print(f"{'='*70}")
#     
#     result = test_model(
#         model=model,
#         model_path=os.path.join(MODEL_WEIGHTS_DIR, weight_file),
#         model_name=model_name,
#         test_file=TEST_FILE,
#         img_dir=IMG_DIR,
#         output_dir=OUTPUT_DIR,
#         batch_size=BATCH_SIZE,
#         visualize=True,
#         num_vis_samples=NUM_VIS_SAMPLES
#     )
#     all_results.append(result)
# 
# # Create combined table
# combined_table_path = os.path.join(OUTPUT_DIR, 'COMBINED_RESULTS_TABLE.txt')
# create_combined_table(all_results, combined_table_path)
# 
# print("\n" + "=" * 70)
# print("✓ ALL TESTING COMPLETE!")
# print(f"✓ Combined results table: {combined_table_path}")
# print("=" * 70)

