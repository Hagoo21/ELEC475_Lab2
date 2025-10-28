import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import SnoutNet
from dataset import PetNoseDataset


def get_augmented_transform(target_size=(227, 227)):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_standard_transform(target_size=(227, 227)):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
    
    return running_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            num_batches += 1
    
    return running_loss / num_batches


def plot_losses(train_losses, val_losses, save_path='training_curve.png'):
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
    with open(save_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f}\n')
    print(f"Losses saved to {save_path}")


def train(train_file='train-noses.txt', 
          test_file='test-noses.txt',
          img_dir='images-original/images',
          epochs=50,
          batch_size=86,
          lr=0.001,
          augment=False,
          weights_dir='weights',
          num_workers=0):
    
    print("=" * 70)
    print("SnoutNet Training")
    print("=" * 70)
    print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr} | Augment: {augment}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    train_transform = get_augmented_transform() if augment else get_standard_transform()
    val_transform = get_standard_transform()
    
    print(f"\nLoading datasets...")
    train_dataset = PetNoseDataset(
        annotations_file=train_file,
        img_dir=img_dir,
        transform=train_transform,
        target_size=(227, 227)
    )
    
    val_dataset = PetNoseDataset(
        annotations_file=test_file,
        img_dir=img_dir,
        transform=val_transform,
        target_size=(227, 227)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Training: {len(train_dataset)} samples | Validation: {len(val_dataset)} samples")
    
    print(f"\nInitializing model...")
    model = SnoutNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs(weights_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch}/{epochs}] | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(weights_dir, 'snoutnet.pt')
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
    print(f"Total time: {total_time / 60:.2f} minutes | Best val loss: {best_val_loss:.6f}")
    
    csv_path = os.path.join(weights_dir, 'training_losses.csv')
    save_losses_to_csv(train_losses, val_losses, csv_path)
    
    plot_path = os.path.join(weights_dir, 'training_curve.png')
    plot_losses(train_losses, val_losses, plot_path)


if __name__ == "__main__":
    train()

