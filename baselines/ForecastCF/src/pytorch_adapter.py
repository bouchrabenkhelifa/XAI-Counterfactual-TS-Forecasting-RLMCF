#!/usr/bin/env python
# coding: utf-8
"""
PyTorch adapter for ForecastCF to work with PyTorch models (e.g., iTransformer).
Converts between TensorFlow and PyTorch formats.
"""

import numpy as np
import torch

# Import TensorFlow only if available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available, using numpy fallback")


class PyTorchModelWrapper:
    """
    Wrapper to make PyTorch forecasting models compatible with ForecastCF's TensorFlow interface.
    """
    
    def __init__(self, pytorch_model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            pytorch_model: A PyTorch model with a forward() method
            device: 'cuda' or 'cpu'
        """
        self.model = pytorch_model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, x):
        """
        TensorFlow-compatible predict method.
        
        Args:
            x: numpy array or TensorFlow tensor of shape [B, L, N]
        
        Returns:
            numpy array of shape [B, H, N]
        """
        # Convert to numpy if TensorFlow tensor
        if TF_AVAILABLE:
            if isinstance(x, tf.Tensor):
                x_np = x.numpy()
            elif isinstance(x, tf.Variable):
                x_np = x.numpy()
            else:
                x_np = np.array(x)
        else:
            x_np = np.array(x)
        
        # Convert to PyTorch tensor
        x_torch = torch.FloatTensor(x_np).to(self.device)
        
        # Create dummy time features (zeros) - iTransformer expects them
        batch_size, seq_len, n_features = x_torch.shape
        x_mark_enc = torch.zeros(batch_size, seq_len, 4).to(self.device)  # 4 time features
        x_dec = torch.zeros(batch_size, seq_len // 2, n_features).to(self.device)
        x_mark_dec = torch.zeros(batch_size, seq_len // 2, 4).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(x_torch, x_mark_enc, x_dec, x_mark_dec)
            
            # Handle tuple output (model might return attention)
            if isinstance(output, tuple):
                output = output[0]
        
        # Convert back to numpy
        return output.cpu().numpy()
    
    def __call__(self, x):
        """Allow calling the wrapper like a function."""
        return self.predict(x)
