from typing import Any, Optional, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np


def default_init(scale=1.0):
    """Default kernel initializer - Xavier uniform with scaling."""
    def init_fn(tensor):
        return nn.init.xavier_uniform_(tensor, gain=scale)
    return init_fn


class FourierFeatures(nn.Module):
    """Fourier features for timestep embedding."""
    
    def __init__(self, output_size: int = 64, learnable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        
        if learnable:
            self.kernel = nn.Parameter(torch.randn(output_size // 2, 1) * 0.2)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.learnable:
            f = 2 * np.pi * x @ self.kernel.T
        else:
            half_dim = self.output_size // 2
            f = np.log(10000) / (half_dim - 1)
            f = torch.exp(torch.arange(half_dim, device=x.device) * -f)
            f = x * f
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class MLP(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(
        self,
        hidden_dims: Sequence[int],
        activation: str = 'gelu',
        activate_final: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                continue
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            default_init(1.0)(layers[-1].weight)
            
        self.layers = nn.ModuleList(layers)
        
        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dims[i+1]) 
                for i in range(len(hidden_dims) - 1)
            ])
        
    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activation(x)
                if self.layer_norm:
                    x = self.layer_norms[i](x)
        return x


class Value(nn.Module):
    """Value/critic network with ensemble support."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        num_ensembles: int = 2,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_ensembles = num_ensembles
        self.encoder = encoder
        
        # Create ensemble of value networks
        self.value_nets = nn.ModuleList([
            MLP([input_dim] + list(hidden_dims) + [1], 
                activate_final=False, 
                layer_norm=layer_norm)
            for _ in range(num_ensembles)
        ])
        
    def forward(self, observations: Tensor, actions: Optional[Tensor] = None) -> Tensor:
        """Return values or critic values.
        
        Args:
            observations: Observations [batch_size, obs_dim]
            actions: Actions [batch_size, action_dim] (optional)
            
        Returns:
            values: [num_ensembles, batch_size]
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
            
        if actions is not None:
            inputs = torch.cat([inputs, actions], dim=-1)
        
        # Compute ensemble values
        values = torch.stack([net(inputs).squeeze(-1) for net in self.value_nets], dim=0)
        return values

class ActorVectorField(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = False,
        encoder: Optional[nn.Module] = None,
        use_time: bool = True,
        use_fourier_features: bool = False,
        fourier_feature_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_time = use_time
        self.use_fourier_features = use_fourier_features and use_time

        if self.use_time:
            time_dim = fourier_feature_dim if self.use_fourier_features else 1
        else:
            time_dim = 0

        input_dim = obs_dim + action_dim + time_dim
        self.mlp = MLP(
            [input_dim] + list(hidden_dims) + [action_dim],
            activate_final=False,
            layer_norm=layer_norm,
        )

        if self.use_fourier_features:
            self.ff = FourierFeatures(fourier_feature_dim)

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        times: Optional[Tensor] = None,
        is_encoded: bool = False,
    ) -> Tensor:
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        if self.use_time and times is not None:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = torch.cat([observations, actions, times], dim=-1)
        else:
            inputs = torch.cat([observations, actions], dim=-1)

        return self.mlp(inputs)