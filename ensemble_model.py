import os
import torch
import torch.nn as nn

from model import SnoutNet
from snoutnet_alexnet import SnoutNetAlexNet
from snoutnet_vgg16 import SnoutNetVGG16


class SnoutNetEnsemble(nn.Module):
    """
    Ensemble model that averages predictions from three pretrained models:
    - SnoutNet (custom CNN)
    - SnoutNet-A (AlexNet-based)
    - SnoutNet-V (VGG16-based)
    
    Each model outputs [batch, 2] coordinate predictions, and the ensemble
    returns the mean prediction across all three models.
    """
    def __init__(self, 
                 snoutnet_path=None, 
                 alexnet_path=None, 
                 vgg16_path=None,
                 device='cuda'):
        super(SnoutNetEnsemble, self).__init__()
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize the three models
        print("Initializing ensemble models...")
        self.snoutnet = SnoutNet().to(self.device)
        self.snoutnet_alexnet = SnoutNetAlexNet(pretrained=False).to(self.device)
        self.snoutnet_vgg16 = SnoutNetVGG16(pretrained=False).to(self.device)
        
        # Load weights if paths are provided
        if snoutnet_path:
            self.load_model_weights(self.snoutnet, snoutnet_path, "SnoutNet")
        
        if alexnet_path:
            self.load_model_weights(self.snoutnet_alexnet, alexnet_path, "SnoutNet-A (AlexNet)")
        
        if vgg16_path:
            self.load_model_weights(self.snoutnet_vgg16, vgg16_path, "SnoutNet-V (VGG16)")
        
        # Set all models to evaluation mode
        self.snoutnet.eval()
        self.snoutnet_alexnet.eval()
        self.snoutnet_vgg16.eval()
        
        print("✓ Ensemble model initialized")
    
    def load_model_weights(self, model, weight_path, model_name):
        """
        Load weights from a .pt checkpoint file.
        
        Args:
            model: The model to load weights into
            weight_path: Path to the .pt file
            model_name: Name of the model (for logging)
        """
        if not os.path.exists(weight_path):
            print(f"⚠️  Warning: Weights file not found at {weight_path}")
            print(f"   {model_name} will use random initialization")
            return
        
        try:
            checkpoint = torch.load(weight_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'N/A')
                val_loss = checkpoint.get('val_loss', 'N/A')
                print(f"✓ {model_name} loaded from {weight_path}")
                print(f"  Epoch: {epoch}, Val Loss: {val_loss}")
            else:
                # Assume the checkpoint is just the state dict
                model.load_state_dict(checkpoint)
                print(f"✓ {model_name} loaded from {weight_path}")
        
        except Exception as e:
            print(f"✗ Error loading {model_name} from {weight_path}: {e}")
            print(f"  {model_name} will use random initialization")
    
    def forward(self, x):
        """
        Forward pass through the ensemble.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 227, 227]
        
        Returns:
            Tensor of shape [batch_size, 2] representing averaged (x, y) coordinates
        """
        # Get predictions from all three models
        with torch.no_grad():
            pred_snoutnet = self.snoutnet(x)
            pred_alexnet = self.snoutnet_alexnet(x)
            pred_vgg16 = self.snoutnet_vgg16(x)
        
        # Average the predictions
        ensemble_pred = (pred_snoutnet + pred_alexnet + pred_vgg16) / 3.0
        
        return ensemble_pred
    
    def get_individual_predictions(self, x):
        """
        Get predictions from each model individually (for analysis).
        
        Args:
            x: Input tensor of shape [batch_size, 3, 227, 227]
        
        Returns:
            Dictionary with predictions from each model and the ensemble average
        """
        with torch.no_grad():
            pred_snoutnet = self.snoutnet(x)
            pred_alexnet = self.snoutnet_alexnet(x)
            pred_vgg16 = self.snoutnet_vgg16(x)
            ensemble_pred = (pred_snoutnet + pred_alexnet + pred_vgg16) / 3.0
        
        return {
            'snoutnet': pred_snoutnet,
            'alexnet': pred_alexnet,
            'vgg16': pred_vgg16,
            'ensemble': ensemble_pred
        }


def test_ensemble():
    """
    Simple test to verify the ensemble model works correctly.
    """
    print("=" * 70)
    print("Testing SnoutNetEnsemble")
    print("=" * 70)
    
    # Create ensemble without loading weights (for testing structure)
    ensemble = SnoutNetEnsemble()
    
    # Count parameters
    total_params = sum(p.numel() for p in ensemble.parameters())
    print(f"\nTotal ensemble parameters: {total_params:,}")
    
    # Test with random input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 227, 227)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # Test forward pass
    ensemble.eval()
    output = ensemble(input_tensor)
    
    print(f"Ensemble output shape: {output.shape}")
    print(f"Sample predictions:\n{output}")
    
    # Test individual predictions
    print("\nTesting individual predictions...")
    individual_preds = ensemble.get_individual_predictions(input_tensor)
    
    for model_name, pred in individual_preds.items():
        print(f"{model_name:12s}: shape {pred.shape}, first sample: {pred[0].tolist()}")
    
    # Verify output shape
    assert output.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {output.shape}"
    print("\n✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_ensemble()

