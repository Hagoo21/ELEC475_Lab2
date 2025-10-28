import os
import ast
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PetNoseDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transform=None, target_size=(227, 227)):
        self.img_dir = img_dir
        self.target_size = target_size
        self.annotations = []
        
        with open(annotations_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(',', 1)
                if len(parts) != 2:
                    continue
                    
                filename = parts[0].strip()
                coords_str = parts[1].strip().strip('"')
                
                try:
                    coords = ast.literal_eval(coords_str)
                    x, y = float(coords[0]), float(coords[1])
                    self.annotations.append((filename, x, y))
                except (ValueError, SyntaxError):
                    continue
        
        print(f"Loaded {len(self.annotations)} samples from {annotations_file}")
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename, x, y = self.annotations[idx]
        img_path = os.path.join(self.img_dir, filename)
        
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        scale_x = self.target_size[0] / original_size[0]
        scale_y = self.target_size[1] / original_size[1]
        scaled_x = x * scale_x
        scaled_y = y * scale_y
        
        image_tensor = self.transform(image)
        label_tensor = torch.tensor([scaled_x, scaled_y], dtype=torch.float32)
        
        return image_tensor, label_tensor
    
    def get_filename(self, idx: int) -> str:
        return self.annotations[idx][0]
    
    def get_original_coords(self, idx: int) -> Tuple[float, float]:
        return self.annotations[idx][1], self.annotations[idx][2]


def reality_check(dataset: PetNoseDataset, num_samples: int = 5, visualize: bool = True):
    print("=" * 70)
    print(f"Dataset Reality Check - {len(dataset)} samples")
    print("=" * 70)
    
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
        
        print(f"Sample {i + 1}: {filename} | Shape: {image_tensor.shape} | Coords: ({x:.2f}, {y:.2f})")
        
        if visualize and i < 5:
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


# ======== Main Test Code ======== #
if __name__ == "__main__":
    train_file = r'C:\Users\20sr91\ELEC475_Lab2\oxford-iiit-pet-noses\train_noses.txt'
    test_file = r'C:\Users\20sr91\ELEC475_Lab2\oxford-iiit-pet-noses\test_noses.txt'
    img_dir = r'C:\Users\20sr91\ELEC475_Lab2\oxford-iiit-pet-noses\images-original\images'
    
    print("\n" + "=" * 70)
    print("Testing Training Dataset")
    print("=" * 70)
    
    try:
        train_dataset = PetNoseDataset(
            annotations_file=train_file,
            img_dir=img_dir,
            target_size=(227, 227)
        )
        
        reality_check(train_dataset, num_samples=5, visualize=True)
        
        print("\n" + "=" * 70)
        print("Testing DataLoader")
        print("=" * 70)
        
        train_loader = DataLoader(train_dataset, batch_size=86, shuffle=True, num_workers=0)
        
        print(f"DataLoader created with batch_size=86")
        print(f"Total batches: {len(train_loader)}")
        
        images, labels = next(iter(train_loader))
        print(f"\nFirst batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        print("\n✓ DataLoader test passed!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print("Testing Test Dataset")
    print("=" * 70)
    
    try:
        test_dataset = PetNoseDataset(
            annotations_file=test_file,
            img_dir=img_dir,
            target_size=(227, 227)
        )
        print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")

