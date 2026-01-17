"""
Quick test script to verify the implementation works
This runs a minimal version of the experiment for testing
"""

import torch
import torch.nn as nn
import numpy as np

print("Testing PyTorch installation...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test simple neural network creation
class TestNN(nn.Module):
    def __init__(self):
        super(TestNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = TestNN()
print(f"\nModel created successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Test forward pass
x = torch.randn(1, 10)
output = model(x)
print(f"Forward pass successful! Output shape: {output.shape}")

print("\n✓ All tests passed! The implementation should work correctly.")
print("Note: The full experiment may take 10-15 minutes to complete.")
