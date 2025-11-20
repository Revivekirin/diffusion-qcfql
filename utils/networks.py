from typing import Any, Optional, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_init_linear(layer: nn.Linear, scale: float = 1.0):
    """
    Rough equivalent of Flax variance_scaling('fan_avg', 'uniform').
    여기서는 Xavier uniform에 scale만 곱해준다.
    """
    nn.init.xavier_uniform_(layer.weight)
    layer.weight.data.mul_(scale)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class FourierFeatures(nn.Module):
    """Fourier-like features for time embedding."""

    def __init__(self, output_size: int = 64, learnable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        if learnable:
            # [output_size//2, input_dim] 은 forward에서 결정해야 하지만,
            # PyTorch에선 미리 input_dim을 모른다 → lazy 방식 사용
            self.weight = None
            self._built = False
        else:
            self.register_buffer("dummy", torch.zeros(1))  # device 관리를 위한 더미

    def _build(self, x: torch.Tensor):
        if self.learnable and not self._built:
            in_dim = x.shape[-1]
            self.weight = nn.Parameter(
                torch.randn(self.output_size // 2, in_dim) * 0.2
            )  # init std=0.2
            self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        if self.learnable:
            self._build(x)
            f = 2 * math.pi * x @ self.weight.t()  # [..., out//2]
        else:
            half_dim = self.output_size // 2
            device = x.device
            # Flax 코드와 동일한 방식
            freq = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
            freq = torch.exp(torch.arange(half_dim, device=device) * -freq)  # [half_dim]
            f = x * freq  # broadcasting: [..., D] * [half_dim]? 여기서는 D=1 가정이 강함
            # 필요하면 times의 마지막 dim=1 형태로 맞춰 사용

        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class Identity(nn.Module):
    def forward(self, x):
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron .
      - hidden_dims: List[int]
      - activations: 기본 gelu
      - activate_final: 마지막 layer에도 activation 적용 여부
      - layer_norm: 각 activation 뒤에 LayerNorm
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Any = F.gelu,
        activate_final: bool = False,
        layer_norm: bool = False,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        self.activations = activations
        self.activate_final = activate_final
        self.layer_norm_flag = layer_norm

        self.input_dim = input_dim
        self._built = False

        self.layers = nn.ModuleList()
        self.lns = nn.ModuleList() if layer_norm else None

    def _build(self, x: torch.Tensor):
        if self._built:
            return
        in_dim = x.shape[-1] if self.input_dim is None else self.input_dim
        dims = [in_dim] + self.hidden_dims
        for i in range(len(self.hidden_dims)):
            linear = nn.Linear(dims[i], dims[i + 1])
            default_init_linear(linear)
            self.layers.append(linear)
            if self.layer_norm_flag:
                self.lns.append(nn.LayerNorm(dims[i + 1]))
        self._built = True

        self.to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._build(x)
        h = x
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            is_last = (i == n_layers - 1)
            if (not is_last) or self.activate_final:
                h = self.activations(h)
                if self.layer_norm_flag:
                    h = self.lns[i](h)
        return h


class LogParam(nn.Module):
    """Scalar parameter module with log scale"""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.log_value = nn.Parameter(torch.log(torch.tensor(init_value, dtype=torch.float32)))

    def forward(self):
        return torch.exp(self.log_value)


class Value(nn.Module):
    """
    Value/critic network.
    - JAX 버전과 동일하게 num_ensembles>1이면 ensemble axis가 앞에 오도록 [num_qs, B] 반환.
    - encoder가 있으면 encoder(observations)를 먼저 적용.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        num_ensembles: int = 2,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        self.layer_norm_flag = layer_norm
        self.num_ensembles = num_ensembles
        self.encoder = encoder

        if num_ensembles <= 1:
            self.nets = nn.ModuleList(
                [
                    MLP(
                        list(hidden_dims) + [1],
                        activate_final=False,
                        layer_norm=self.layer_norm_flag,
                    )
                ]
            )
        else:
            nets = []
            for _ in range(num_ensembles):
                nets.append(
                    MLP(
                        list(hidden_dims) + [1],
                        activate_final=False,
                        layer_norm=self.layer_norm_flag,
                    )
                )
            self.nets = nn.ModuleList(nets)

    def forward(self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            observations: [B, obs_dim] or [...]
            actions: [B, act_dim] or None
        Returns:
            v: [num_ensembles, B] (JAX Value와 동일한 axis 구조)
        """
        if self.encoder is not None:
            obs_enc = self.encoder(observations)
        else:
            obs_enc = observations

        if actions is not None:
            x = torch.cat([obs_enc, actions], dim=-1)
        else:
            x = obs_enc

        # 각 ensemble 별로 scalar value 출력
        vs = []
        for net in self.nets:
            v = net(x)  # [B, 1]
            v = v.squeeze(-1)  # [B]
            vs.append(v)
        v_all = torch.stack(vs, dim=0)  # [num_ensembles, B]
        return v_all


class ActorVectorField(nn.Module):
    """
    Actor vector field network for flow matching.

    Args:
        hidden_dims: list of hidden layer sizes
        action_dim: output action dimension
        layer_norm: whether to apply LN
        encoder: optional encoder for observations
        use_fourier_features: apply FourierFeatures to times
        fourier_feature_dim: output dim of FourierFeatures
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        action_dim: int,
        layer_norm: bool = False,
        encoder: Optional[nn.Module] = None,
        use_fourier_features: bool = False,
        fourier_feature_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dims = list(hidden_dims)
        self.action_dim = action_dim
        self.layer_norm_flag = layer_norm
        self.encoder = encoder
        self.use_fourier_features = use_fourier_features
        self.fourier_feature_dim = fourier_feature_dim

        # MLP의 입력 차원은 forward에서 lazy build
        self.ff = FourierFeatures(fourier_feature_dim) if use_fourier_features else None
        self.mlp = None
        self._built = False

    def _build(self, obs: torch.Tensor, actions: torch.Tensor, times: Optional[torch.Tensor]):
        if self._built:
            return
        # obs: [B, obs_dim] (encoded 여부 상관X)
        in_dim = obs.shape[-1] + actions.shape[-1]
        if times is not None:
            if self.use_fourier_features:
                in_dim += self.fourier_feature_dim
            else:
                in_dim += times.shape[-1]
        self.mlp = MLP(
            hidden_dims=list(self.hidden_dims) + [self.action_dim],
            activate_final=False,
            layer_norm=self.layer_norm_flag,
            input_dim=in_dim,
        ).to(obs.device)
        self._built = True

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        times: Optional[torch.Tensor] = None,
        is_encoded: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            observations: [B, obs_dim] (encoded or raw)
            actions: [B, action_dim]
            times: [B, 1] or [B, D_t] (optional)
            is_encoded: whether observations are already encoded
        Returns:
            v: [B, action_dim]
        """
        if not is_encoded and self.encoder is not None:
            obs = self.encoder(observations)
        else:
            obs = observations

        if times is not None:
            if self.use_fourier_features:
                t_feat = self.ff(times)
            else:
                t_feat = times
            x = torch.cat([obs, actions, t_feat], dim=-1)
        else:
            x = torch.cat([obs, actions], dim=-1)

        self._build(obs, actions, times)
        v = self.mlp(x)
        return v
