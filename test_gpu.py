import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Built with MPS: {torch.backends.mps.is_built()}")