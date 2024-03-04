import torch
import torch.nn as nn
import tinycudann as tcnn

from abc import abstractmethod


class ReflectionField(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    @abstractmethod
    def forward(self, positions, directions, normals, albedo, roughness) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            
        Returns:
            
        """