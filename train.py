"""
SnoutNet Training Script
ELEC 475 Lab 2

This script trains the SnoutNet model for pet nose localization using the
Oxford-IIIT Pet Noses dataset.

Features:
- Training and validation loops
- MSE loss for coordinate regression
- Adam optimizer
- Model checkpointing (saves best model)
- Loss tracking and visualization
- Command-line arguments for hyperparameters
- Optional data augmentation

Author: Generated for ELEC 475 Lab 2
Date: 2025
"""

import os
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import SnoutNet
from dataset import PetNoseDataset


def get_augmented_transform(target_size=(227, 227)):
    """
    Create a transform pipeline with data augmentation.
    
    Args:
        target_size (tuple): Target image size (width, height)
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_standard_transform(target_size=(227, 227)):
    """
    Create a standard transform pipeline without augmentation.
    
    Args:
        target_size (tuple): Target image size (width, height)
        
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda or cpu)
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
    
    avg_loss = running_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Validate the model on validation/test data.
    
    Args:
        model: The neural network model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cuda or cpu)
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            num_batches += 1
    
    avg_loss = running_loss / num_batches
    return avg_loss


def plot_losses(train_losses, val_losses, save_path='training_curve.png'):
    """
    Plot and save training and validation loss curves.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")


def save_losses_to_csv(train_losses, val_losses, save_path='training_losses.csv'):
    """
    Save training and validation losses to a CSV file.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_path (str): Path to save the CSV file
    """
    with open(save_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f}\n')
    print(f"Losses saved to {save_path}")


def train(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments containing hyperparameters
    """
    print("=" * 70)
    print("SnoutNet Training")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Augmentation: {args.augment}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create transforms
    if args.augment:
        print("Using data augmentation for training")
        train_transform = get_augmented_transform()
    else:
        train_transform = get_standard_transform()
    
    val_transform = get_standard_transform()
    
    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = PetNoseDataset(
        annotations_file=args.train_file,
        img_dir=args.img_dir,
        transform=train_transform,
        target_size=(227, 227)
    )
    
    val_dataset = PetNoseDataset(
        annotations_file=args.test_file,
        img_dir=args.img_dir,
        transform=val_transform,
        target_size=(227, 227)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model, loss function, and optimizer
    print(f"\nInitializing model...")
    model = SnoutNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create directory for saving weights
    os.makedirs(args.weights_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.weights_dir, 'snoutnet.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  → Best model saved (val_loss: {val_loss:.6f})")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {checkpoint_path}")
    
    # Save losses to CSV
    csv_path = os.path.join(args.weights_dir, 'training_losses.csv')
    save_losses_to_csv(train_losses, val_losses, csv_path)
    
    # Plot loss curves
    plot_path = os.path.join(args.weights_dir, 'training_curve.png')
    plot_losses(train_losses, val_losses, plot_path)
    
    print("\n✓ Training artifacts saved successfully!")


def main():
    """Parse command-line arguments and start training."""
    parser = argparse.ArgumentParser(description='Train SnoutNet for pet nose localization')
    
    # Dataset arguments
    parser.add_argument('--train_file', type=str, default='train-noses.txt',
                        help='Path to training annotations file')
    parser.add_argument('--test_file', type=str, default='test-noses.txt',
                        help='Path to test annotations file (used for validation)')
    parser.add_argument('--img_dir', type=str, default='images-original/images',
                        help='Directory containing images')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    
    # Data augmentation
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation during training')
    
    # Other settings
    parser.add_argument('--weights_dir', type=str, default='weights',
                        help='Directory to save model weights (default: weights/)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers (default: 0)')
    
    args = parser.parse_args()
    
    # Start training
    train(args)


if __name__ == "__main__":
    main()

