import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from model import SnoutNet
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

# Construct default paths from config (normalize for cross-platform compatibility)
DEFAULT_MODEL_PATH = os.path.normpath(os.path.join(BASE_PATH, CONFIG['paths']['weights_dir'], 'snoutnet.pt'))
DEFAULT_TEST_FILE = os.path.normpath(os.path.join(BASE_PATH, CONFIG['paths']['test_file']))
DEFAULT_IMG_DIR = os.path.normpath(os.path.join(BASE_PATH, CONFIG['paths']['img_dir']))
DEFAULT_BATCH_SIZE = CONFIG['testing']['batch_size']
DEFAULT_NUM_VIS_SAMPLES = CONFIG['testing']['num_vis_samples']


def compute_euclidean_distance(pred_coords, true_coords):
    distances = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=1))
    return distances


def evaluate_model(model, dataloader, device):
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
    with open(save_path, 'w') as f:
        f.write('filename,pred_x,pred_y,true_x,true_y,distance\n')
        for i, filename in enumerate(filenames):
            pred_x, pred_y = predictions[i]
            true_x, true_y = ground_truths[i]
            dist = distances[i]
            f.write(f'{filename},{pred_x:.4f},{pred_y:.4f},{true_x:.4f},{true_y:.4f},{dist:.4f}\n')
    print(f"Results saved to {save_path}")


def visualize_predictions(dataset, predictions, ground_truths, distances, num_samples=4, save_path='predictions_visualization.png'):
    num_samples = min(num_samples, len(dataset))
    
    best_idx = np.argmin(distances)
    worst_idx = np.argmax(distances)
    random_indices = np.random.choice(len(dataset), size=max(0, num_samples - 2), replace=False)
    
    indices = [best_idx, worst_idx]
    indices.extend(random_indices.tolist())
    indices = indices[:num_samples]
    
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
    plt.show()


def test(model_path=None,
         test_file=None,
         img_dir=None,
         output_csv='test_results.csv',
         batch_size=None,
         visualize=True,
         num_vis_samples=None):
    
    # Use config defaults if not specified
    model_path = model_path or DEFAULT_MODEL_PATH
    test_file = test_file or DEFAULT_TEST_FILE
    img_dir = img_dir or DEFAULT_IMG_DIR
    batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
    num_vis_samples = num_vis_samples if num_vis_samples is not None else DEFAULT_NUM_VIS_SAMPLES
    
    print("=" * 70)
    print("SnoutNet Testing")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"⚠️  Using CPU")
    print(f"Device: {device}")
    
    print(f"\nLoading model from {model_path}...")
    model = SnoutNet().to(device)
    
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
    
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    median_dist = np.median(distances)
    
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Samples: {len(distances)} | Min: {min_dist:.4f} | Max: {max_dist:.4f}")
    print(f"Mean: {mean_dist:.4f} | Median: {median_dist:.4f} | Std: {std_dist:.4f}")
    print("=" * 70)
    
    filenames = [test_dataset.get_filename(i) for i in range(len(test_dataset))]
    save_results_to_csv(filenames, predictions, ground_truths, distances, output_csv)
    
    summary_path = output_csv.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"SnoutNet Evaluation\nModel: {model_path}\nSamples: {len(distances)}\n")
        f.write(f"Min: {min_dist:.4f} | Max: {max_dist:.4f}\n")
        f.write(f"Mean: {mean_dist:.4f} | Median: {median_dist:.4f} | Std: {std_dist:.4f}\n")
    print(f"Summary saved to {summary_path}")
    
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
    hist_path = output_csv.replace('.csv', '_histogram.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"Histogram saved to {hist_path}")
    
    if visualize:
        print(f"\nGenerating visualizations...")
        vis_path = output_csv.replace('.csv', '_predictions.png')
        visualize_predictions(test_dataset, predictions, ground_truths, distances, 
                            num_samples=num_vis_samples, save_path=vis_path)
    
    print("\n✓ Testing complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test SnoutNet models')
    parser.add_argument('--model', type=str, default='snoutnet',
                        choices=['snoutnet', 'alexnet', 'vgg16', 'ensemble'],
                        help='Model architecture to test')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights file (.pt)')
    parser.add_argument('--weights-dir', type=str, default=None,
                        help='Directory containing model weights (for ensemble)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save test results')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for testing')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualizations')
    
    args = parser.parse_args()
    
    # Use test_cases functions for consistent testing
    from test_cases import test_model, TEST_FILE, IMG_DIR, MODEL_WEIGHTS_DIR, OUTPUT_DIR, BATCH_SIZE, NUM_VIS_SAMPLES
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE
    
    if args.model == 'ensemble':
        # Handle ensemble testing separately
        weights_dir = args.weights_dir if args.weights_dir else MODEL_WEIGHTS_DIR
        
        import os
        from ensemble_model import SnoutNetEnsemble
        import torch
        import numpy as np
        from torch.utils.data import DataLoader
        
        print(f"\n{'='*70}")
        print(f"Testing Ensemble Model")
        print(f"{'='*70}\n")
        
        # Load ensemble
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        snoutnet_path = os.path.join(weights_dir, 'SnoutNet.pt')
        alexnet_path = os.path.join(weights_dir, 'SnoutNetAlexNet.pt')
        vgg16_path = os.path.join(weights_dir, 'SnoutNetVGG16.pt')
        
        ensemble = SnoutNetEnsemble(
            snoutnet_path=snoutnet_path,
            alexnet_path=alexnet_path,
            vgg16_path=vgg16_path,
            device=device
        )
        
        # Load test dataset
        from dataset import PetNoseDataset
        test_dataset = PetNoseDataset(
            annotations_file=TEST_FILE,
            img_dir=IMG_DIR,
            target_size=(227, 227)
        )
        
        # Evaluate ensemble
        from visualize_ensemble import evaluate_ensemble
        metrics, predictions, ground_truths, distances = evaluate_ensemble(
            ensemble=ensemble,
            dataset=test_dataset,
            batch_size=batch_size
        )
        
        # Save results
        import matplotlib.pyplot as plt
        ensemble_output_dir = os.path.join(output_dir, 'Ensemble')
        os.makedirs(ensemble_output_dir, exist_ok=True)
        
        csv_path = os.path.join(ensemble_output_dir, 'results.csv')
        with open(csv_path, 'w') as f:
            f.write('filename,pred_x,pred_y,true_x,true_y,distance\n')
            for i in range(len(test_dataset)):
                filename = test_dataset.get_filename(i)
                pred_x, pred_y = predictions[i]
                true_x, true_y = ground_truths[i]
                dist = distances[i]
                f.write(f'{filename},{pred_x:.4f},{pred_y:.4f},{true_x:.4f},{true_y:.4f},{dist:.4f}\n')
        print(f"✓ Results saved to {csv_path}")
        
        # Save summary
        summary_path = os.path.join(ensemble_output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Ensemble Model Evaluation\n")
            f.write(f"Samples: {metrics['num_samples']}\n")
            f.write(f"Min: {metrics['min_distance']:.4f}\n")
            f.write(f"Max: {metrics['max_distance']:.4f}\n")
            f.write(f"Mean: {metrics['mean_distance']:.4f}\n")
            f.write(f"Median: {metrics['median_distance']:.4f}\n")
            f.write(f"Std: {metrics['std_distance']:.4f}\n")
        print(f"✓ Summary saved to {summary_path}")
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(metrics['mean_distance'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {metrics['mean_distance']:.2f}")
        plt.axvline(metrics['median_distance'], color='green', linestyle='--', linewidth=2,
                   label=f"Median: {metrics['median_distance']:.2f}")
        plt.xlabel('Euclidean Distance (pixels)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Ensemble - Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_path = os.path.join(ensemble_output_dir, 'error_histogram.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        print(f"✓ Histogram saved to {hist_path}")
        plt.close()
        
        print("\n✓ Testing complete!")
        
    else:
        # Handle individual model testing
        if args.weights is None:
            # Use default weights path
            if args.model == 'snoutnet':
                model_file = 'SnoutNet.pt'
            elif args.model == 'alexnet':
                model_file = 'SnoutNetAlexNet.pt'
            elif args.model == 'vgg16':
                model_file = 'SnoutNetVGG16.pt'
            
            weights_path = os.path.join(MODEL_WEIGHTS_DIR, model_file)
        else:
            weights_path = args.weights
        
        # Import and create model
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
        
        print(f"\n{'='*70}")
        print(f"Testing {model_name}")
        print(f"Weights: {weights_path}")
        print(f"{'='*70}\n")
        
        # Run test
        test_model(
            model=model,
            model_path=weights_path,
            model_name=model_name,
            test_file=TEST_FILE,
            img_dir=IMG_DIR,
            output_dir=output_dir,
            batch_size=batch_size,
            visualize=not args.no_visualize,
            num_vis_samples=NUM_VIS_SAMPLES
        )

