import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


class SnoutNetAlexNet(nn.Module):
    """
    SnoutNet based on pretrained AlexNet for 2D coordinate regression.
    Adapts AlexNet's classifier to output (x, y) coordinates.
    """
    def __init__(self, pretrained=True):
        super(SnoutNetAlexNet, self).__init__()
        
        # Load pretrained AlexNet
        if pretrained:
            weights = AlexNet_Weights.IMAGENET1K_V1
            self.alexnet = alexnet(weights=weights)
        else:
            self.alexnet = alexnet(weights=None)
        
        # Get the number of features from the last conv layer
        # AlexNet features output: [batch, 256, 6, 6]
        num_features = self.alexnet.classifier[1].in_features
        
        # Replace the classifier for regression (2D coordinate output)
        self.alexnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2)  # Output: (x, y) coordinates
        )
    
    def forward(self, x):
        """
        Forward pass through AlexNet.
        The features are automatically flattened by AlexNet's avgpool + flatten,
        then passed through the modified classifier.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 227, 227]
        
        Returns:
            Tensor of shape [batch_size, 2] representing (x, y) coordinates
        """
        # AlexNet handles feature extraction and flattening internally
        x = self.alexnet(x)
        return x


if __name__ == "__main__":
    print("=" * 70)
    print("SnoutNet-AlexNet Model Test")
    print("=" * 70)
    
    # Create model
    print("Loading pretrained AlexNet...")
    model = SnoutNetAlexNet(pretrained=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with random input (AlexNet expects 224x224, but can handle 227x227)
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

