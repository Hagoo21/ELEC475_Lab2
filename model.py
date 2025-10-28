"""
SnoutNet Model - Pet Nose Localization
ELEC 475 Lab 2

This module defines a CNN architecture for predicting the (x, y) coordinates
of a pet's nose from an RGB image.

Author: Generated for ELEC 475 Lab 2
Date: 2025
"""

import torch
import torch.nn as nn


class SnoutNet(nn.Module):
    """
    SnoutNet - A CNN model for localizing pet noses in images.
    
    Architecture:
    - 3 Convolutional blocks (Conv2d -> ReLU -> MaxPool2d)
    - 3 Fully Connected layers
    - Output: 2D coordinates (x, y)
    
    Input:  RGB image of shape [batch_size, 3, 227, 227]
    Output: (x, y) coordinates of the nose, shape [batch_size, 2]
    """

    def __init__(self):
        super(SnoutNet, self).__init__()

        # ======== Convolutional Layers ======== #
        # Conv1: 3x3 kernel, input channels = 3, output channels = 64
        # Input: [batch, 3, 227, 227] -> Output: [batch, 64, 225, 225]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=0)
        
        # Conv2: 3x3 kernel, input channels = 64, output channels = 128
        # Input: [batch, 64, 112, 112] -> Output: [batch, 128, 110, 110]
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        
        # Conv3: 3x3 kernel, input channels = 128, output channels = 256
        # Input: [batch, 128, 55, 55] -> Output: [batch, 256, 53, 53]
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)

        # MaxPooling layer (2x2, stride 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ======== Fully Connected Layers ======== #
        # After conv3 + pooling: [batch, 256, 26, 26] -> Flattened: 173,056
        self.fc1 = nn.Linear(26 * 26 * 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)  # Output 2 coordinates: (x, y)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 227, 227]
            
        Returns:
            torch.Tensor: Predicted (x, y) coordinates of shape [batch_size, 2]
        """
        # Convolutional block 1: Conv -> ReLU -> Pool
        # [batch, 3, 227, 227] -> [batch, 64, 225, 225] -> [batch, 64, 112, 112]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convolutional block 2: Conv -> ReLU -> Pool
        # [batch, 64, 112, 112] -> [batch, 128, 110, 110] -> [batch, 128, 55, 55]
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Convolutional block 3: Conv -> ReLU -> Pool
        # [batch, 128, 55, 55] -> [batch, 256, 53, 53] -> [batch, 256, 26, 26]
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten for fully connected layers
        # [batch, 256, 26, 26] -> [batch, 173056]
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers with ReLU activations
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        # Final output layer (no activation for regression)
        x = self.fc3(x)

        return x


# ======== Test Code ======== #
if __name__ == "__main__":
    print("=" * 60)
    print("SnoutNet Model Test")
    print("=" * 60)
    
    # Create model instance
    model = SnoutNet()
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create random input tensor
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 227, 227)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Forward pass
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Predicted coordinates (x, y): {output[0].tolist()}")
    
    # Verify shapes
    assert output.shape == (batch_size, 2), f"Expected output shape ({batch_size}, 2), got {output.shape}"
    print("\n✓ Test passed! Output shape is correct.")
    
    # Test with batch size > 1
    print(f"\n{'-' * 60}")
    print("Testing with batch_size = 4")
    batch_input = torch.randn(4, 3, 227, 227)
    with torch.no_grad():
        batch_output = model(batch_input)
    print(f"Input shape: {batch_input.shape}")
    print(f"Output shape: {batch_output.shape}")
    print(f"Batch predictions:\n{batch_output}")
    
    print("\n✓ All tests passed!")
    print("=" * 60)