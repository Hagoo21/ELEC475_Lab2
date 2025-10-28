import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from model import SnoutNet
from dataset import PetNoseDataset


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


def test(model_path='weights/snoutnet.pt',
         test_file='test-noses.txt',
         img_dir='images-original/images',
         output_csv='test_results.csv',
         batch_size=86,
         visualize=True,
         num_vis_samples=4):
    
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
    test()

