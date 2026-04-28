#!/usr/bin/env python
# coding: utf-8
"""
PyTorch adapter for ForecastCF to work with PyTorch models (e.g., iTransformer).
Converts between TensorFlow and PyTorch formats with gradient support.
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
    Uses tf.py_function with custom gradients for optimization.
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
        self.model.eval()  # Keep in eval mode but allow gradients
    
    def _pytorch_forward(self, x_np):
        """
        Internal PyTorch forward pass with gradient computation.
        
        Args:
            x_np: numpy array of shape [B, L, N]
        
        Returns:
            output_np: numpy array of shape [B, H, N]
            grad_fn: function to compute gradients
        """
        # Convert to PyTorch tensor with gradient tracking
        x_torch = torch.FloatTensor(x_np).to(self.device)
        x_torch.requires_grad = True
        
        # Create dummy time features
        batch_size, seq_len, n_features = x_torch.shape
        x_mark_enc = torch.zeros(batch_size, seq_len, 4).to(self.device)
        x_dec = torch.zeros(batch_size, seq_len // 2, n_features).to(self.device)
        x_mark_dec = torch.zeros(batch_size, seq_len // 2, 4).to(self.device)
        
        # Forward pass WITH gradients
        output = self.model(x_torch, x_mark_enc, x_dec, x_mark_dec)
        
        # Handle tuple output
        if isinstance(output, tuple):
            output = output[0]
        
        return output, x_torch
    
    def predict(self, x):
        """
        TensorFlow-compatible predict method with gradient support.
        
        Args:
            x: numpy array or TensorFlow tensor/variable of shape [B, L, N]
        
        Returns:
            TensorFlow tensor of shape [B, H, N]
        """
        # Convert to numpy if TensorFlow tensor
        if TF_AVAILABLE:
            if isinstance(x, (tf.Tensor, tf.Variable)):
                x_np = x.numpy()
            else:
                x_np = np.array(x)
        else:
            x_np = np.array(x)
        
        # Define custom gradient function
        @tf.custom_gradient
        def pytorch_predict_with_grad(x_input):
            # Forward pass
            output_torch, x_torch = self._pytorch_forward(x_input.numpy())
            output_np = output_torch.detach().cpu().numpy()
            
            def grad_fn(dy):
                """
                Compute gradients using PyTorch autograd.
                dy: gradient from TensorFlow (shape [B, H, N])
                """
                # Convert TensorFlow gradient to PyTorch
                dy_torch = torch.FloatTensor(dy.numpy()).to(self.device)
                
                # Compute gradients
                output_torch.backward(dy_torch, retain_graph=False)
                
                # Get input gradients
                if x_torch.grad is not None:
                    dx_np = x_torch.grad.cpu().numpy()
                else:
                    dx_np = np.zeros_like(x_input.numpy())
                
                return tf.constant(dx_np, dtype=tf.float32)
            
            return tf.constant(output_np, dtype=tf.float32), grad_fn
        
        # Use custom gradient function
        if TF_AVAILABLE and isinstance(x, (tf.Tensor, tf.Variable)):
            return pytorch_predict_with_grad(x)
        else:
            # Fallback without gradients
            output_torch, _ = self._pytorch_forward(x_np)
            return output_torch.detach().cpu().numpy()
    
    def __call__(self, x):
        """Allow calling the wrapper like a function."""
        return self.predict(x)
