import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SnoutNet
from dataset import PetNoseDataset
from augmentations import KorniaAugmentation


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

# Construct default paths from config
DEFAULT_TRAIN_FILE = os.path.join(BASE_PATH, CONFIG['paths']['train_file'])
DEFAULT_TEST_FILE = os.path.join(BASE_PATH, CONFIG['paths']['test_file'])
DEFAULT_IMG_DIR = os.path.join(BASE_PATH, CONFIG['paths']['img_dir'])
DEFAULT_WEIGHTS_DIR = os.path.join(BASE_PATH, CONFIG['paths']['weights_dir'])
DEFAULT_EPOCHS = CONFIG['training']['epochs']
DEFAULT_BATCH_SIZE = CONFIG['training']['batch_size']
DEFAULT_LR = CONFIG['training']['learning_rate']
DEFAULT_NUM_WORKERS = CONFIG['training']['num_workers']


def train_one_epoch(model, dataloader, criterion, optimizer, device, augmentor=None):
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        if augmentor is not None:
            images, labels = augmentor(images, labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation', leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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


def train(train_file=None, 
          test_file=None,
          img_dir=None,
          epochs=None,
          batch_size=None,
          lr=None,
          augment=False,
          weights_dir=None,
          num_workers=None):
    
    # Use config defaults if not specified
    train_file = train_file or DEFAULT_TRAIN_FILE
    test_file = test_file or DEFAULT_TEST_FILE
    img_dir = img_dir or DEFAULT_IMG_DIR
    epochs = epochs if epochs is not None else DEFAULT_EPOCHS
    batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
    lr = lr if lr is not None else DEFAULT_LR
    weights_dir = weights_dir or DEFAULT_WEIGHTS_DIR
    num_workers = num_workers if num_workers is not None else DEFAULT_NUM_WORKERS
    
    # Create timestamped output folder for this training run
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(weights_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 70)
    print("SnoutNet Training")
    print("=" * 70)
    print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr} | Augment: {augment}")
    print(f"Output folder: {run_dir}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"\n🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"\n⚠️  Using CPU (Training will be slow)")
    print(f"Device: {device}")
    
    if augment:
        print("\n✓ GPU-accelerated augmentation enabled (Kornia)")
        augmentor = KorniaAugmentation(p_hflip=0.5, rotation_degrees=15).to(device)
    else:
        print("\n✗ Augmentation disabled")
        augmentor = None
    
    print(f"\nLoading datasets...")
    train_dataset = PetNoseDataset(
        annotations_file=train_file,
        img_dir=img_dir,
        target_size=(227, 227)
    )
    
    val_dataset = PetNoseDataset(
        annotations_file=test_file,
        img_dir=img_dir,
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
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, augmentor)
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
            checkpoint_path = os.path.join(run_dir, 'snoutnet.pt')
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
    
    csv_path = os.path.join(run_dir, 'training_losses.csv')
    save_losses_to_csv(train_losses, val_losses, csv_path)
    
    plot_path = os.path.join(run_dir, 'training_curve.png')
    plot_losses(train_losses, val_losses, plot_path)


if __name__ == "__main__":
    train()

