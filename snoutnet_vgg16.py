import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class SnoutNetVGG16(nn.Module):
    """
    SnoutNet based on pretrained VGG16 for 2D coordinate regression.
    Adapts VGG16's classifier to output (x, y) coordinates.
    """
    def __init__(self, pretrained=True):
        super(SnoutNetVGG16, self).__init__()
        
        # Load pretrained VGG16
        if pretrained:
            weights = VGG16_Weights.IMAGENET1K_V1
            self.vgg16 = vgg16(weights=weights)
        else:
            self.vgg16 = vgg16(weights=None)
        
        # Get the number of features from VGG16's classifier
        # VGG16 features output: [batch, 512, 7, 7] -> flattened to 25088
        num_features = self.vgg16.classifier[0].in_features
        
        # Replace the classifier for regression (2D coordinate output)
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2)  # Output: (x, y) coordinates
        )
    
    def forward(self, x):
        """
        Forward pass through VGG16.
        Features are extracted, flattened, then passed through the modified classifier.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 224, 224] or [batch_size, 3, 227, 227]
        
        Returns:
            Tensor of shape [batch_size, 2] representing (x, y) coordinates
        """
        # VGG16 handles feature extraction and flattening internally
        x = self.vgg16(x)
        return x


if __name__ == "__main__":
    print("=" * 70)
    print("SnoutNet-VGG16 Model Test")
    print("=" * 70)
    
    # Create model
    print("Loading pretrained VGG16...")
    model = SnoutNetVGG16(pretrained=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with random input (VGG16 expects 224x224, but can handle 227x227)
    input_tensor = torch.randn(1, 3, 227, 227)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Predicted (x, y): {output[0].tolist()}")
    
    # Verify output shape
    assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"
    print("\n✓ Test passed! Output shape is [1, 2]")
    
    # Test with batch
    print("\n" + "-" * 70)
    print("Testing with batch size 4...")
    batch_input = torch.randn(4, 3, 227, 227)
    with torch.no_grad():
        batch_output = model(batch_input)
    print(f"Batch input shape: {batch_input.shape}")
    print(f"Batch output shape: {batch_output.shape}")
    assert batch_output.shape == (4, 2), f"Expected (4, 2), got {batch_output.shape}"
    print("✓ Batch test passed!")
    
    print("=" * 70)

