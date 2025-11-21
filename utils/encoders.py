import functools
from typing import Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.networks import MLP  



class ResnetStack(nn.Module):
    """ResNet stack module."""
    
    def __init__(self, num_features: int, num_blocks: int, max_pooling: bool = True):
        super().__init__()
        self.max_pooling = max_pooling
        
        self.conv_in = nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv_in.weight)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.ModuleList([
                nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            ])
            nn.init.xavier_uniform_(block[0].weight)
            nn.init.xavier_uniform_(block[1].weight)
            self.blocks.append(block)
            
    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv_in(x)
        
        if self.max_pooling:
            conv_out = F.max_pool2d(conv_out, kernel_size=3, stride=2, padding=1)
            
        for block in self.blocks:
            block_input = conv_out
            conv_out = F.relu(conv_out)
            conv_out = block[0](conv_out)
            conv_out = F.relu(conv_out)
            conv_out = block[1](conv_out)
            conv_out = conv_out + block_input
            
        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""
    
    def __init__(
        self,
        width: int = 1,
        stack_sizes: Tuple[int, ...] = (16, 32, 32),
        num_blocks: int = 2,
        dropout_rate: Optional[float] = None,
        mlp_hidden_dims: Sequence[int] = (512,),
        layer_norm: bool = False,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layer_norm_flag = layer_norm
        
        self.stack_blocks = nn.ModuleList([
            ResnetStack(
                num_features=stack_sizes[i] * width,
                num_blocks=num_blocks,
            )
            for i in range(len(stack_sizes))
        ])
        
        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)
            
        if layer_norm:
            self.layer_norm = nn.LayerNorm(stack_sizes[-1] * width)
            
        # Calculate flattened size (assuming 64x64 input after 3 stacks with stride 2)
        # This is approximate - adjust based on actual input size
        self.flatten_size = stack_sizes[-1] * width * 8 * 8  # For 64x64 input
        
        mlp_dims = [self.flatten_size] + list(mlp_hidden_dims)
        self.mlp = MLP(mlp_dims, activate_final=True, layer_norm=layer_norm)
        
    def forward(self, x: Tensor, train: bool = True) -> Tensor:
        """
        Args:
            x: Images [batch_size, height, width, channels] or 
               [batch_size, channels, height, width]
        """
        # Handle both (H,W,C) and (C,H,W) formats
        if x.shape[-1] == 3:  # (H,W,C) format
            x = x.permute(0, 3, 1, 2)  # Convert to (C,H,W)
            
        x = x.float() / 255.0
        
        conv_out = x
        for stack_block in self.stack_blocks:
            conv_out = stack_block(conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out) if train else conv_out
                
        conv_out = F.relu(conv_out)
        if self.layer_norm_flag:
            # Apply layer norm on channel dimension
            b, c, h, w = conv_out.shape
            conv_out = conv_out.permute(0, 2, 3, 1).contiguous()
            conv_out = self.layer_norm(conv_out)
            conv_out = conv_out.permute(0, 3, 1, 2).contiguous()
            
        out = conv_out.flatten(1)
        out = self.mlp(out)
        
        return out


encoder_modules = {
    "impala": ImpalaEncoder,
    "impala_debug": functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    "impala_small": functools.partial(ImpalaEncoder, num_blocks=1),
    "impala_large": functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}
