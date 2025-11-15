import torch
import torch_directml

dml = torch_directml.device()
print("Using device:", dml)

# Simple test tensor
x = torch.randn(3, 3, device=dml)
y = torch.randn(3, 3, device=dml)
z = x @ y
print("z:", z)