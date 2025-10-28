import torch
import torch.nn as nn


class SnoutNet(nn.Module):
    def __init__(self):
        super(SnoutNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(26 * 26 * 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


if __name__ == "__main__":
    print("SnoutNet Model Test")
    print("=" * 60)
    
    model = SnoutNet()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    input_tensor = torch.randn(1, 3, 227, 227)
    print(f"Input shape: {input_tensor.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Predicted (x, y): {output[0].tolist()}")
    
    assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"
    print("✓ Test passed!")
    print("=" * 60)