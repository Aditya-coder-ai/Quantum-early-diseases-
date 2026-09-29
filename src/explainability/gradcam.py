"""
Part 8: Grad-CAM (Gradient-weighted Class Activation Mapping).
Provides spatial feature attribution for convolutional vision architectures.
Strictly inspects model layers, rejecting non-convolutional tabular models.
"""
from __future__ import annotations

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union

from src.explainability.router import UnsupportedModalityError
from src.explainability.schemas import GradCAMExplanation

logger = logging.getLogger(__name__)


class GradCAMExplainer:
    """
    Grad-CAM implementation for PyTorch convolutional vision models.
    Validates presence of 2D convolutional layers before computing spatial heatmaps.
    """
    def __init__(
        self,
        model: nn.Module,
        target_layer_name: Optional[str] = None
    ):
        self.model = model
        self.target_layer_name = target_layer_name
        self.target_layer = None
        self.gradients = None
        self.activations = None

        self._inspect_and_bind_target_layer()

    def _inspect_and_bind_target_layer(self) -> None:
        """Finds and attaches forward/backward hooks to the target convolutional layer."""
        conv_modules = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                conv_modules[name] = module

        if not conv_modules:
            raise UnsupportedModalityError(
                "Model contains no convolutional layers. Grad-CAM cannot be applied to "
                "purely tabular linear models, autoencoders with dense layers, or quantum circuits."
            )

        if self.target_layer_name is not None:
            if self.target_layer_name not in conv_modules:
                raise ValueError(
                    f"Target layer '{self.target_layer_name}' not found among convolutional modules: "
                    f"{list(conv_modules.keys())}"
                )
            self.target_layer = conv_modules[self.target_layer_name]
        else:
            # Default to the last convolutional layer
            last_name, last_module = list(conv_modules.items())[-1]
            self.target_layer_name = last_name
            self.target_layer = last_module

        # Register PyTorch hooks
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        logger.info(f"[Grad-CAM] Bound successfully to convolutional layer '{self.target_layer_name}'.")

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 0
    ) -> Tuple[np.ndarray, GradCAMExplanation]:
        """
        Computes the class-discriminative spatial activation heatmap.
        
        Args:
            input_tensor: Tensor of shape (1, C, H, W)
            target_class: Index of class to explain
            
        Returns:
            Tuple of (2D numpy heatmap in [0, 1], GradCAMExplanation contract)
        """
        if input_tensor.dim() != 4:
            raise ValueError(f"Expected 4D image tensor (1, C, H, W), got shape {input_tensor.shape}.")

        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        if output.dim() == 1:
            score = output[target_class]
        else:
            score = output[0, target_class]

        # Backward pass for gradients
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture activations or gradients from target layer.")

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: (1, K, H', W')
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H', W')
        cam = torch.relu(cam)  # Discard negative influences

        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0.0, 1.0]
        max_val = np.max(cam_np)
        min_val = np.min(cam_np)
        if max_val > min_val:
            heatmap = (cam_np - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(cam_np)

        explanation = GradCAMExplanation(
            target_layer=self.target_layer_name,
            input_shape=tuple(input_tensor.shape),
            heatmap_shape=tuple(heatmap.shape),
            heatmap_min=float(np.min(heatmap)),
            heatmap_max=float(np.max(heatmap)),
            notes=f"Grad-CAM generated for class {target_class} at layer '{self.target_layer_name}'."
        )

        return heatmap, explanation


def create_synthetic_cnn(in_channels: int = 1, num_classes: int = 2) -> nn.Module:
    """Helper factory generating a minimal CNN for Grad-CAM test verification."""
    class MiniConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(8 * 4 * 4, num_classes)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = x.flatten(1)
            return self.fc(x)

    return MiniConvNet()
