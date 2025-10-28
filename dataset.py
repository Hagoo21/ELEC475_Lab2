"""
Pet Nose Dataset - Oxford-IIIT Pet Noses (Reannotated)
ELEC 475 Lab 2

This module provides a custom PyTorch Dataset class for loading pet images
and their corresponding nose coordinates.

Dataset structure:
    images-original/images/    # Directory containing all images
    train-noses.txt            # Training annotations
    test-noses.txt             # Test annotations

Annotation format (per line):
    filename.jpg,"(x, y)"

Author: Generated for ELEC 475 Lab 2
Date: 2025
"""

import os
import ast
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PetNoseDataset(Dataset):
    """
    Custom Dataset for Oxford-IIIT Pet Noses dataset.
    
    Loads images and their corresponding nose coordinates for training/testing
    a pet nose localization model.
    """
    
    def __init__(self, annotations_file: str, img_dir: str, transform=None, target_size=(227, 227)):
        """
        Initialize the dataset.
        
        Args:
            annotations_file (str): Path to the annotations text file (e.g., train-noses.txt)
            img_dir (str): Directory containing the images (e.g., images-original/images/)
            transform (callable, optional): Optional transform to be applied on images
            target_size (tuple): Target size for resizing images (width, height)
        """
        self.img_dir = img_dir
        self.target_size = target_size
        self.annotations = []
        
        # Read annotations file
        with open(annotations_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse line: filename.jpg,"(x, y)"
                parts = line.split(',', 1)
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line: {line}")
                    continue
                    
                filename = parts[0].strip()
                coords_str = parts[1].strip().strip('"')
                
                try:
                    # Parse coordinates tuple string "(x, y)" to actual values
                    coords = ast.literal_eval(coords_str)
                    x, y = float(coords[0]), float(coords[1])
                    self.annotations.append((filename, x, y))
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Failed to parse coordinates for {filename}: {coords_str}")
                    continue
        
        print(f"Loaded {len(self.annotations)} samples from {annotations_file}")
        
        # Default transform: Convert PIL Image to Tensor
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image_tensor, label_tensor)
                - image_tensor: torch.Tensor of shape [3, 227, 227]
                - label_tensor: torch.Tensor of shape [2] containing (x, y) coordinates
        """
        filename, x, y = self.annotations[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            original_size = image.size  # (width, height)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
        
        # Scale coordinates to match resized image
        scale_x = self.target_size[0] / original_size[0]
        scale_y = self.target_size[1] / original_size[1]
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        
        # Apply transformations to image
        image_tensor = self.transform(image)
        
        # Create label tensor
        label_tensor = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)
        
        return image_tensor, label_tensor
    
    def get_filename(self, idx: int) -> str:
        """Get the filename for a given index."""
        return self.annotations[idx][0]
    
    def get_original_coords(self, idx: int) -> Tuple[float, float]:
        """Get the original (unscaled) coordinates for a given index."""
        return self.annotations[idx][1], self.annotations[idx][2]


# ======== Reality Check Script ======== #
def reality_check(dataset: PetNoseDataset, num_samples: int = 5, visualize: bool = True):
    """
    Perform a reality check on the dataset by iterating through samples
    and optionally visualizing them.
    
    Args:
        dataset: PetNoseDataset instance
        num_samples: Number of samples to check
        visualize: Whether to display images with ground truth points
    """
    print("=" * 70)
    print("Dataset Reality Check")
    print("=" * 70)
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Checking first {num_samples} samples...\n")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    if visualize:
        fig, axes = plt.subplots(1, min(num_samples, 5), figsize=(15, 3))
        if num_samples == 1:
            axes = [axes]
    
    for i, (image_tensor, label_tensor) in enumerate(dataloader):
        if i >= num_samples:
            break
        
        filename = dataset.get_filename(i)
        x, y = label_tensor[0][0].item(), label_tensor[0][1].item()
        
        print(f"Sample {i + 1}:")
        print(f"  Filename: {filename}")
        print(f"  Image tensor shape: {image_tensor.shape}")
        print(f"  Label tensor shape: {label_tensor.shape}")
        print(f"  Nose coordinates (x, y): ({x:.2f}, {y:.2f})")
        print()
        
        if visualize and i < 5:
            # Convert tensor to numpy for visualization
            # Tensor is in [C, H, W] format, need [H, W, C] for matplotlib
            img_np = image_tensor[0].permute(1, 2, 0).numpy()
            
            axes[i].imshow(img_np)
            axes[i].scatter([x], [y], c='red', s=100, marker='x', linewidths=3)
            axes[i].set_title(f"{filename}\n({x:.1f}, {y:.1f})", fontsize=8)
            axes[i].axis('off')
    
    if visualize:
        plt.tight_layout()
        plt.savefig('dataset_reality_check.png', dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to 'dataset_reality_check.png'")
        plt.show()
    
    print("=" * 70)
    print("✓ Reality check complete!")
    print("=" * 70)


# ======== Main Test Code ======== #
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Pet Nose Dataset')
    parser.add_argument('--train_file', type=str, default='train-noses.txt',
                        help='Path to training annotations file')
    parser.add_argument('--test_file', type=str, default='test-noses.txt',
                        help='Path to test annotations file')
    parser.add_argument('--img_dir', type=str, default='images-original/images',
                        help='Directory containing images')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to display in reality check')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for DataLoader test')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize samples with matplotlib')
    
    args = parser.parse_args()
    
    # Test train dataset
    print("\n" + "=" * 70)
    print("Testing Training Dataset")
    print("=" * 70)
    
    try:
        train_dataset = PetNoseDataset(
            annotations_file=args.train_file,
            img_dir=args.img_dir,
            target_size=(227, 227)
        )
        
        # Reality check
        reality_check(train_dataset, num_samples=args.num_samples, visualize=args.visualize)
        
        # Test DataLoader
        print("\n" + "=" * 70)
        print("Testing DataLoader with Batching")
        print("=" * 70)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        print(f"DataLoader created with batch_size={args.batch_size}")
        print(f"Total batches: {len(train_loader)}")
        
        # Get first batch
        images, labels = next(iter(train_loader))
        print(f"\nFirst batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Sample coordinates:\n{labels}")
        
        print("\n✓ DataLoader test passed!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure the following files/directories exist:")
        print(f"  - {args.train_file}")
        print(f"  - {args.img_dir}")
        print("\nExample dataset structure:")
        print("  ELEC475_Lab2/")
        print("  ├── train-noses.txt")
        print("  ├── test-noses.txt")
        print("  └── images-original/")
        print("      └── images/")
        print("          ├── image1.jpg")
        print("          ├── image2.jpg")
        print("          └── ...")
    
    # Test test dataset
    print("\n" + "=" * 70)
    print("Testing Test Dataset")
    print("=" * 70)
    
    try:
        test_dataset = PetNoseDataset(
            annotations_file=args.test_file,
            img_dir=args.img_dir,
            target_size=(227, 227)
        )
        print(f"✓ Test dataset loaded successfully: {len(test_dataset)} samples")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")

