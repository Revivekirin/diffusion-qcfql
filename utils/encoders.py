import functools
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.networks import MLP  


class ResnetStack(nn.Module):

    def __init__(self, num_features: int, num_blocks: int, max_pooling: bool = True):
        super().__init__()
        self.max_pooling = max_pooling

        self.conv_in = nn.Conv2d(
            in_channels=None,  # 실제 in_channels는 forward에서 설정
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # in_channels를 런타임에 맞추기 위해 trick
        self._conv_in_built = False

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def _build_conv_in(self, x):
        if not self._conv_in_built:
            in_ch = x.shape[1]
            # PyTorch에서 in_channels 를 동적으로 바꿔주기 위해 재생성
            self.conv_in = nn.Conv2d(
                in_channels=in_ch,
                out_channels=self.conv_in.out_channels,
                kernel_size=self.conv_in.kernel_size,
                stride=self.conv_in.stride,
                padding=self.conv_in.padding,
            ).to(x.device)
            self._conv_in_built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        self._build_conv_in(x)
        conv_out = self.conv_in(x)

        if self.max_pooling:
            conv_out = F.max_pool2d(conv_out, kernel_size=3, stride=2, padding=1)

        for block in self.blocks:
            residual = conv_out
            out = block(conv_out)
            conv_out = out + residual

        return conv_out


class ImpalaEncoder(nn.Module):

    def __init__(
        self,
        width: int = 1,
        stack_sizes: Tuple[int, ...] = (16, 32, 32),
        num_blocks: int = 2,
        dropout_rate: float | None = None,
        mlp_hidden_dims: Sequence[int] = (512,),
        layer_norm: bool = False,
    ):
        super().__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.mlp_hidden_dims = mlp_hidden_dims
        self.layer_norm_flag = layer_norm

        # ResNet stacks
        stacks = []
        in_channels = None
        for i, s in enumerate(stack_sizes):
            # max_pooling=True for all stacks (원 코드와 동일)
            stacks.append(ResnetStack(num_features=s * width, num_blocks=num_blocks, max_pooling=True))
        self.stack_blocks = nn.ModuleList(stacks)

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        # conv 출력 flatten 후 MLP
        # conv 출력 크기를 알 수 없으니 forward에서 lazy init 방식으로 처리
        self._mlp_built = False
        self.mlp = None
        self.layer_norm = nn.LayerNorm if layer_norm else None

    def _build_mlp(self, x_flat: torch.Tensor):
        if not self._mlp_built:
            # x_flat: [B, C_flat]
            in_dim = x_flat.shape[-1]
            hidden_dims = list(self.mlp_hidden_dims)
            # Flax MLP는 hidden_dims로 MLP 만든 뒤 activate_final=True 였음
            self.mlp = MLP(hidden_dims, activate_final=True, layer_norm=self.layer_norm_flag, input_dim=in_dim).to(
                x_flat.device
            )
            self._mlp_built = True

    def forward(self, x: torch.Tensor, cond_var=None) -> torch.Tensor:
        # 원 Flax: x.astype(jnp.float32) / 255.
        # PyTorch: assume x is uint8 or float; cast & normalize
        x = x.float() / 255.0  # [B, H, W, C] 또는 [B, C, H, W] 가능성

        # Flax는 채널 마지막, PyTorch는 conv2d가 채널 첫 번째를 기대
        if x.dim() == 4 and x.shape[1] != 1 and x.shape[1] != 3:
            # 아마 [B, H, W, C]라고 가정
            x = x.permute(0, 3, 1, 2)  # -> [B, C, H, W]

        conv_out = x
        for stack in self.stack_blocks:
            conv_out = stack(conv_out)
            if self.dropout is not None and self.training:
                conv_out = self.dropout(conv_out)

        conv_out = F.relu(conv_out)
        if self.layer_norm_flag:
            # LayerNorm over channel+spatial
            # flatten spatial, do LN, reshape
            B, C, H, W = conv_out.shape
            conv_out_flat = conv_out.view(B, -1)
            ln = nn.LayerNorm(conv_out_flat.shape[-1]).to(conv_out.device)
            conv_out_flat = ln(conv_out_flat)
            conv_out = conv_out_flat.view(B, C, H, W)

        # flatten conv_out to [B, -1]
        out = conv_out.flatten(1)

        # build & apply MLP
        self._build_mlp(out)
        out = self.mlp(out)  # [B, hidden_dim]

        return out


encoder_modules = {
    "impala": ImpalaEncoder,
    "impala_debug": functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    "impala_small": functools.partial(ImpalaEncoder, num_blocks=1),
    "impala_large": functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}
