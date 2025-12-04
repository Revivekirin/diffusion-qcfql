from typing import Any, Optional, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


def default_init(scale=1.0):
    """Default kernel initializer - Xavier uniform with scaling."""
    def init_fn(tensor):
        return nn.init.xavier_uniform_(tensor, gain=scale)
    return init_fn


class Actor(nn.Module):
    """Gaussian actor network."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = False,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        tanh_squash: bool = False,
        state_dependent_std: bool = False,
        const_std: bool = True,
        final_fc_init_scale: float = 1e-2,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash
        self.state_dependent_std = state_dependent_std
        self.const_std = const_std
        self.encoder = encoder
        
        # Actor network
        self.actor_net = MLP(
            [obs_dim] + list(hidden_dims),
            activate_final=True,
            layer_norm=layer_norm
        )
        
        # Mean network
        self.mean_net = nn.Linear(hidden_dims[-1], action_dim)
        default_init(final_fc_init_scale)(self.mean_net.weight)
        
        # Std network
        if state_dependent_std:
            self.log_std_net = nn.Linear(hidden_dims[-1], action_dim)
            default_init(final_fc_init_scale)(self.log_std_net.weight)
        elif not const_std:
            self.log_stds = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, observations: Tensor, temperature: float = 1.0):
        """Return action distribution."""
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
            
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = torch.zeros_like(means)
            else:
                log_stds = self.log_stds.expand_as(means)
        
        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        stds = torch.exp(log_stds) * temperature
        
        # Create distribution
        dist = Normal(means, stds)
        
        if self.tanh_squash:
            dist = TransformedDistribution(dist, [TanhTransform(cache_size=1)])
        
        return dist
    
    def get_action_and_log_prob(self, observations: Tensor, temperature: float = 1.0):
        """Sample action and compute log probability."""
        dist = self.forward(observations, temperature)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob



class LogAlpha(nn.Module):
    """Learnable temperature parameter."""
    
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(np.log(init_value)))
    
    def forward(self):
        return torch.exp(self.log_alpha)



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