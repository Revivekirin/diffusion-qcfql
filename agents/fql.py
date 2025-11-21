import copy
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from utils.encoders import encoder_modules
from utils.networks import ActorVectorField, Value


class ACFQLNetwork(nn.Module):
    """
    - critic, target_critic
    - actor_bc_flow
    - actor_onestep_flow
    - (optional) encoders
    """
    def __init__(
        self,
        config: Dict[str, Any],
        ob_shape: Tuple[int, ...],
        action_dim: int,
    ):
        super().__init__()
        self.config = config
        self.ob_shape = ob_shape
        self.action_dim = action_dim

        # encoder 정의 (있으면)
        if config.get("encoder", None) is not None:
            enc_cls = encoder_modules[config["encoder"]]
            self.critic_encoder = enc_cls()
            self.actor_bc_flow_encoder = enc_cls()
            self.actor_onestep_flow_encoder = enc_cls()
        else:
            self.critic_encoder = None
            self.actor_bc_flow_encoder = None
            self.actor_onestep_flow_encoder = None

        # action chunking이 켜져 있으면 horizon_length 만큼 concat
        if config["action_chunking"]:
            full_action_dim = action_dim * config["horizon_length"]
        else:
            full_action_dim = action_dim
        self.full_action_dim = full_action_dim

        # critic & target_critic
        self.critic = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=self.critic_encoder,
        )

        self.target_critic = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["num_qs"],
            encoder=self.critic_encoder,  # 보통 같은 encoder 구조 사용
        )
        # 초기에는 critic 파라미터를 복사
        self.target_critic.load_state_dict(self.critic.state_dict())

        # BC flow (multi-step / 벡터장)
        self.actor_bc_flow = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=self.actor_bc_flow_encoder,
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        # one-step flow (distillation target)
        self.actor_onestep_flow = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=self.actor_onestep_flow_encoder,
        )

    # 편의를 위해 간단한 호출 래퍼(Flax의 network.select 대체용)
    def q_value(self, obs, actions, target: bool = False):
        """
        obs: [B, ...]
        actions: [B, full_action_dim]
        return: [num_qs, B] 또는 [B, num_qs] (Value 구현에 따름)
        """
        if target:
            return self.target_critic(obs, actions)
        else:
            return self.critic(obs, actions)

    def bc_flow(self, obs, x_t, t, is_encoded: bool = False):
        return self.actor_bc_flow(obs, x_t, t, is_encoded=is_encoded)

    def onestep_flow(self, obs, noises):
        return self.actor_onestep_flow(obs, noises)

    def encode_actor_bc_flow(self, obs):
        if self.actor_bc_flow_encoder is None:
            return obs
        return self.actor_bc_flow_encoder(obs)


class ACFQLAgent:
    """
    PyTorch implementation of the JAX/Flax ACFQLAgent.
    """
    def __init__(
        self,
        network: ACFQLNetwork,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.config = config
        self.network = network.to(self.device)

        if config["weight_decay"] > 0.0:
            self.optimizer = optim.AdamW(
                self.network.parameters(),
                lr=config["lr"],
                weight_decay=config["weight_decay"],
            )
        else:
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=config["lr"],
            )

    # -----------------------------
    # Utils
    # -----------------------------
    def _soft_update_target(self):
        tau = self.config["tau"]
        with torch.no_grad():
            for p, tp in zip(self.network.critic.parameters(),
                             self.network.target_critic.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    # -----------------------------
    # Losses
    # -----------------------------
    def critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        batch:
            observations: [B, ...] or [B, H, ...]
            actions:      [B, H, action_dim]
            next_observations: [B, H, ...]
            rewards:      [B, H]
            masks:        [B, H]
            valid:        [B, H]  (chunk validity)
        """

        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        next_obs = batch["next_observations"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        masks = batch["masks"].to(self.device)
        valid = batch["valid"].to(self.device)

        # if obs.dim() > 2:
        #     obs = obs.view(obs.shape[0], -1)

        if next_obs.dim() > 3:
            next_obs_last = next_obs[:, -1, :].contiguous()
        elif next_obs.dim() == 3:
            next_obs_last = next_obs[:, -1, :]
        else:
            next_obs_last = next_obs

        B = actions.shape[0]

        # action_chunking 여부에 따라 action flatten
        if self.config["action_chunking"]:
            # actions: [B, H, A] -> [B, H*A]
            batch_actions = actions.reshape(B, -1)
        else:
            # 첫 스텝만 사용: [B, H, A] -> [B, A]
            batch_actions = actions[:, 0, :]

        # next state는 마지막 스텝 기준
        # next_obs: [B, H, obs_dim] 라고 가정
        next_obs_last = next_obs[:, -1, :]  # [..., -1, :]

        # next action 샘플
        with torch.no_grad():
            next_actions = self.sample_actions(next_obs_last)

            next_qs = self.network.q_value(next_obs_last, next_actions, target=True)
            # Value 구현에 따라 shape 다를 수 있음. 여기서는 [num_qs, B] 가정.
            if self.config["q_agg"] == "min":
                next_q, _ = torch.min(next_qs, dim=0)
            else:
                next_q = torch.mean(next_qs, dim=0)

            r_last = rewards[:, -1].view(-1)   # [B]
            m_last = masks[:, -1].view(-1)     # [B]
            v_last = valid[:, -1].view(-1)     # [B]

        # critic forward
        # q = self.network.q_value(obs, batch_actions, target=False)
        # q_mean = q.mean(dim=0)

        target_q = r_last + (self.config["discount"] ** self.config["horizon_length"]) * m_last * next_q
        # td_error = (q_mean - target_q) ** 2
        # critic_loss = (td_error * v_last).mean()

        # info = {
        #     "critic_loss": critic_loss.detach(),
        #     "q_mean": q_mean.mean().detach(),
        #     "q_max": q_mean.max().detach(),
        #     "q_min": q_mean.min().detach(),
        # }
        q = self.network.q_value(obs, batch_actions, target=False)  # [num_qs, B]
        # target_q: [B], v_last: [B]

        target_q_exp = target_q.unsqueeze(0)      # [1, B] → broadcast
        v_last_exp = v_last.unsqueeze(0)          # [1, B]

        td_error = (q - target_q_exp) ** 2        # [num_qs, B]
        critic_loss = (td_error * v_last_exp).mean()

        # 로깅용 통계만 q_mean 써도 됨
        q_mean_for_log = q.mean(dim=0)
        info = {
            "critic_loss": critic_loss.detach(),
            "q_mean": q_mean_for_log.mean().detach(),
            "q_max": q_mean_for_log.max().detach(),
            "q_min": q_mean_for_log.min().detach(),
        }

        return critic_loss, info

    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        valid = batch["valid"].to(self.device)

        B = actions.shape[0]
        H = actions.shape[1]
        A = actions.shape[2]

        # action_chunking flatten
        if self.config["action_chunking"]:
            batch_actions = actions.reshape(B, -1)  # [B, H*A]
            full_action_dim = self.config["action_dim"] * self.config["horizon_length"]
        else:
            batch_actions = actions[:, 0, :]        # [B, A]
            full_action_dim = self.config["action_dim"]

        # BC flow loss: x_0 → x_1 직선보간
        x_0 = torch.randn(B, full_action_dim, device=self.device)
        x_1 = batch_actions
        t = torch.rand(B, 1, device=self.device)  # [B,1]
        x_t = (1.0 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.bc_flow(obs, x_t, t, is_encoded=False)

        if self.config["action_chunking"]:
            # pred, vel : [B, H*A]
            pred_reshaped = pred.reshape(B, self.config["horizon_length"], self.config["action_dim"])
            vel_reshaped = vel.reshape(B, self.config["horizon_length"], self.config["action_dim"])
            bc_sq = (pred_reshaped - vel_reshaped) ** 2
            # valid: [B, H] → [B, H, 1] broadcasting
            bc_flow_loss = (bc_sq * valid[..., None]).mean()
        else:
            bc_flow_loss = ((pred - vel) ** 2).mean()

        # Distillation + Q term (if actor_type == "distill-ddpg")
        if self.config["actor_type"] == "distill-ddpg":
            noises = torch.randn(B, full_action_dim, device=self.device)
            with torch.no_grad():
                target_flow_actions = self.compute_flow_actions(obs, noises)

            actor_actions = self.network.onestep_flow(obs, noises)
            distill_loss = ((actor_actions - target_flow_actions) ** 2).mean()

            # Q term (maximize Q → minimize -Q)
            actor_actions_clipped = actor_actions.clamp(-1.0, 1.0)
            qs = self.network.q_value(obs, actor_actions_clipped, target=False)
            if qs.dim() == 2:  # [num_qs, B]
                q_mean = qs.mean(dim=0)
            else:
                q_mean = qs
            q_loss = -q_mean.mean()
        else:
            distill_loss = torch.zeros((), device=self.device)
            q_loss = torch.zeros((), device=self.device)

        actor_loss = bc_flow_loss + self.config["alpha"] * distill_loss + q_loss

        info = {
            "actor_loss": actor_loss.detach(),
            "bc_flow_loss": bc_flow_loss.detach(),
            "distill_loss": distill_loss.detach(),
        }
        return actor_loss, info

    def total_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        critic_loss, critic_info = self.critic_loss(batch)
        actor_loss, actor_info = self.actor_loss(batch)

        loss = critic_loss + actor_loss
        info = {}
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v
        info["loss"] = loss.detach()
        return loss, info

    # -----------------------------
    # Update
    # -----------------------------
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Single gradient step on the given batch.
        """
        self.optimizer.zero_grad()
        loss, info = self.total_loss(batch)
        loss.backward()
        self.optimizer.step()
        self._soft_update_target()
        return info

    # -----------------------------
    # Action sampling
    # -----------------------------
    def sample_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: [B, obs_dim]
        return:
            actions: [B, full_action_dim] (flattened chunk if action_chunking=True)
        """

        obs = observations.to(self.device)
        ob_dims = self.config["ob_dims"]
        # batch shape = all dims except last len(ob_dims)
        batch_shape = obs.shape[: -len(ob_dims)]
        if self.config["action_chunking"]:
            full_action_dim = self.config["action_dim"] * self.config["horizon_length"]
        else:
            full_action_dim = self.config["action_dim"]

        if self.config["actor_type"] == "distill-ddpg":
            noises = torch.randn(*batch_shape, full_action_dim, device=self.device)
            actions = self.network.onestep_flow(obs, noises)
            actions = actions.clamp(-1.0 + 1e-5, 1.0 - 1e-5)

        elif self.config["actor_type"] == "best-of-n":
            K = self.config["actor_num_samples"]
            noises = torch.randn(*batch_shape, K, full_action_dim, device=self.device)
            # obs: [B, obs_dim] → [B, K, obs_dim]
            expand_dims = (1,) * (noises.dim() - obs.dim())
            obs_rep = obs.view(*obs.shape, *expand_dims).expand(*batch_shape, K, obs.shape[-1])

            # flatten samples into batch dimension for flow 계산
            flat_obs = obs_rep.reshape(-1, obs.shape[-1])
            flat_noises = noises.reshape(-1, full_action_dim)
            flat_actions = self.compute_flow_actions(flat_obs, flat_noises)
            actions_all = flat_actions.reshape(*batch_shape, K, full_action_dim)
            actions_all = actions_all.clamp(-1.0, 1.0)

            # Q 평가
            with torch.no_grad():
                flat_obs_for_q = obs_rep.reshape(-1, obs.shape[-1])
                flat_actions_for_q = actions_all.reshape(-1, full_action_dim)
                qs = self.network.q_value(flat_obs_for_q, flat_actions_for_q, target=False)
                if qs.dim() == 2:  # [num_qs, BK]
                    q_per = qs.mean(dim=0)
                else:
                    q_per = qs
                q_per = q_per.view(*batch_shape, K)

                if self.config["q_agg"] == "mean":
                    # 이미 ensemble mean
                    pass
                else:
                    # ensemble min이면 위에서 min 후 mean 해야 하지만,
                    # 여기서는 간략히 mean 사용 (필요시 수정)
                    pass

                # best index
                best_idx = torch.argmax(q_per, dim=-1)  # [B]
                # gather best action
                # actions_all: [B, K, full_action_dim]
                B_ = actions_all.shape[0]
                best_actions = actions_all[torch.arange(B_, device=self.device), best_idx]
            actions = best_actions
        else:
            raise NotImplementedError(f"Unknown actor_type: {self.config['actor_type']}")

        return actions

    def compute_flow_actions(self, observations: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        """
        Euler integration of BC flow:
            x_{t+1} = x_t + v(s, x_t, t) / flow_steps
        observations: [B, obs_dim] (or [..., obs_dim])
        noises: [B, full_action_dim]
        """
        obs = observations.to(self.device)
        actions = noises.to(self.device)
        flow_steps = self.config["flow_steps"]

        # encoder가 있는 경우 미리 encode
        if self.config.get("encoder", None) is not None:
            obs_encoded = self.network.encode_actor_bc_flow(obs)
        else:
            obs_encoded = obs

        for i in range(flow_steps):
            t_val = float(i) / float(flow_steps)
            t = torch.full((*obs_encoded.shape[:-1], 1), t_val, device=self.device)
            vels = self.network.bc_flow(obs_encoded, actions, t, is_encoded=True)
            actions = actions + vels / flow_steps

        actions = actions.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
        return actions

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations,  # numpy array (or something np.array로 캐스팅 가능한 것)
        ex_actions,       # numpy array
        config: Dict[str, Any],
        device: str = "cuda",
    ) -> "ACFQLAgent":
        torch.manual_seed(seed)
        device_t = torch.device(device)

        # 1) numpy로 강제 캐스팅 후 batch 차원 보장
        ex_obs_np = np.asarray(ex_observations)
        ex_act_np = np.asarray(ex_actions)

        if ex_obs_np.ndim == 1:
            ex_obs_np = ex_obs_np[None, :]      # [obs_dim] -> [1, obs_dim]
        if ex_act_np.ndim == 1:
            ex_act_np = ex_act_np[None, :]      # [act_dim] -> [1, act_dim]

        # 2) shape 정보로 config 채우기
        ob_shape = tuple(ex_obs_np.shape[1:])   # [B, ...] -> ...
        action_dim = ex_act_np.shape[-1]        # primitive action dim (예: 14)

        config = dict(config)  # 복사
        config["ob_dims"] = ob_shape
        config["action_dim"] = action_dim

        # 3) 네트워크 생성
        network = ACFQLNetwork(config, ob_shape, action_dim).to(device_t)

        # 4) lazy MLP building for torchrl
        with torch.no_grad():
            # torch tensor로 변환
            torch_ex_obs = torch.from_numpy(ex_obs_np).float().to(device_t)  # [B, obs_dim]
            torch_ex_act = torch.from_numpy(ex_act_np).float().to(device_t)  # [B, action_dim]

            # obs는 [B, obs_dim] 형태로 맞춰서 사용 (필요시 flatten)
            dummy_obs = torch_ex_obs
            if dummy_obs.ndim > 2:
                dummy_obs = dummy_obs.view(dummy_obs.shape[0], -1)

            # action_chunking 여부에 따라 full_action_dim 맞게 dummy action 생성
            if config["action_chunking"]:
                # ex_actions: [B, A]  ->  [B, H*A] 로 horizon_length 번 concat
                H = config["horizon_length"]
                dummy_act_flat = torch.cat([torch_ex_act] * H, dim=-1)  # [B, H*A]
            else:
                dummy_act_flat = torch_ex_act  # [B, A]

            # critic / target_critic warmup
            _ = network.q_value(dummy_obs, dummy_act_flat, target=False)
            _ = network.q_value(dummy_obs, dummy_act_flat, target=True)

            # actor_bc_flow / actor_onestep_flow warmup
            B = dummy_obs.shape[0]
            dummy_t = torch.zeros(B, 1, device=device_t)
            _ = network.bc_flow(dummy_obs, dummy_act_flat, dummy_t, is_encoded=False)
            _ = network.onestep_flow(dummy_obs, dummy_act_flat)

        # 5) 디버그: 파라미터 개수 확인
        total_params = sum(p.numel() for p in network.parameters())
        print("[DEBUG] total network params after warmup:", total_params)

        # 6) 이제 optimizer를 만드는 agent 생성
        agent = cls(network=network, config=config, device=device)
        return agent



def get_config() -> Dict[str, Any]:
    config = dict(
        agent_name="acfql",
        ob_dims=None,                
        action_dim=None,            
        lr=3e-4,
        batch_size=256,
        actor_hidden_dims=(512, 512, 512, 512),
        value_hidden_dims=(512, 512, 512, 512),
        layer_norm=True,
        actor_layer_norm=False,
        discount=0.99,
        tau=0.005,
        q_agg="mean",
        alpha=100.0,
        num_qs=2,
        flow_steps=10,
        normalize_q_loss=False,
        encoder=None,                
        horizon_length=8,            
        action_chunking=True,
        actor_type="distill-ddpg",  
        actor_num_samples=32,
        use_fourier_features=False,
        fourier_feature_dim=64,
        weight_decay=0.0,
    )
    return config
