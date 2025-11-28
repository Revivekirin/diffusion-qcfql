import os
import json
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from absl import app, flags

import wandb
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.algo.diffusion_policy import DiffusionPolicyUNet


# ======================================================================
# Flags
# ======================================================================

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'SDAC', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'transport-mh-vim', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp_sdac/', 'Save directory.')

flags.DEFINE_integer('total_env_steps', 500000, 'Total number of environment steps.')
flags.DEFINE_integer('buffer_size', 200000, 'Replay buffer size.')
flags.DEFINE_integer('batch_size', 256, 'SDAC batch size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval (env steps).')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval (env steps).')
flags.DEFINE_integer('start_training', 5000, 'Step to start SDAC updates.')
flags.DEFINE_integer('utd_ratio', 1, 'Update-to-data ratio (updates per env step).')

flags.DEFINE_float('discount', 0.99, 'Discount factor.')
flags.DEFINE_float('reward_scale', 0.2, 'Reward scale for SDAC.')
flags.DEFINE_float('tau', 0.005, 'Polyak averaging for target Q-networks.')

flags.DEFINE_string('entity', 'sophia435256-robros', 'wandb entity.')
flags.DEFINE_string('mode', 'online', 'wandb mode (online/offline/disabled).')

flags.DEFINE_string(
    'dp_ckpt',
    '/home/robros/git/robomimic/trained_model/bc_diffusion_policy_ph_low_dim/20251125135939/models/model_epoch_600_low_dim_v15_success_0.7.pth',
    'Pretrained DiffusionPolicy checkpoint path.',
)


# ======================================================================
# Logging helper
# ======================================================================

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        if self.wandb_logger is not None:
            self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)


# ======================================================================
# Replay Buffer : obs_seq (time-stacked) 버전
#   obs_dict, next_obs_dict : 각 key 텐서 shape [1, To, ...]
#   actions : [1, Da], normalized [-1,1]
# ======================================================================

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.storage = []
        self.ptr = 0

    def add(self, obs_dict, action: torch.Tensor, reward: float, next_obs_dict, done: bool):
        data = {
            "obs": {k: v.detach().cpu() for k, v in obs_dict.items()},
            "action": action.detach().cpu(),
            "reward": float(reward),
            "next_obs": {k: v.detach().cpu() for k, v in next_obs_dict.items()},
            "done": bool(done),
        }
        if len(self.storage) < self.capacity:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.capacity

    def __len__(self):
        return len(self.storage)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in idxs]

        obs_batch = {}
        next_obs_batch = {}

        for k in batch[0]["obs"].keys():
            obs_batch[k] = torch.cat([b["obs"][k] for b in batch], dim=0)      # [B, To, ...]
            next_obs_batch[k] = torch.cat([b["next_obs"][k] for b in batch], dim=0)

        actions = torch.cat([b["action"] for b in batch], dim=0)               # [B, Da]
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32)
        dones = torch.tensor([b["done"] for b in batch], dtype=torch.float32)

        obs_batch = TensorUtils.to_device(obs_batch, self.device)
        next_obs_batch = TensorUtils.to_device(next_obs_batch, self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        return {
            "obs": obs_batch,
            "next_obs": next_obs_batch,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }


# ======================================================================
# Q-network
# ======================================================================

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)  # [B, 1]


# ======================================================================
# env obs (벡터 or dict) -> diffusion policy가 학습 때 썼던 obs dict 형태로 변환
#   - dp_algo.obs_shapes (OrderedDict) 기준으로 벡터를 순서대로 split
# ======================================================================

def env_obs_to_dp_obs_dict(obs_raw, dp_algo: DiffusionPolicyUNet, device: torch.device):
    """
    obs_raw:
      - dict 인 경우: 각 value를 [1, ...] 텐서로 변환해서 바로 사용
      - 1D 벡터 (np.ndarray, list, torch.Tensor) 인 경우:
        dp_algo.obs_shapes 순서를 기준으로 필요한 길이만큼 잘라서 각 키 텐서로 mapping
    """
    # 이미 dict인 경우 (robomimic env가 그대로 dict를 줄 때)
    if isinstance(obs_raw, dict):
        obs_dict = {}
        for k, v in obs_raw.items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            obs_dict[k] = torch.from_numpy(arr).to(device)
        return obs_dict

    # 그 외에는 flat vector라고 가정
    vec = obs_raw
    if isinstance(vec, torch.Tensor):
        vec = vec.detach().cpu().numpy()
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec[None, :]  # [1, obs_dim]
    elif vec.ndim > 2:
        # env가 이미 [batch, dim] 주는 케이스는 여기선 고려하지 않음
        raise ValueError(f"Unexpected obs_raw shape {vec.shape}")

    obs_dim = vec.shape[1]
    obs_dict = {}
    idx = 0
    for k, shape in dp_algo.obs_shapes.items():
        size = int(np.prod(shape))
        if idx + size > obs_dim:
            raise ValueError(
                f"env obs dim {obs_dim} too small for obs_shapes split; "
                f"stopped at key {k} (need {idx+size})"
            )
        slice_ = vec[:, idx:idx + size]  # [1, size]
        new_shape = (1,) + tuple(shape)
        obs_dict[k] = torch.from_numpy(slice_.reshape(new_shape)).to(device)
        idx += size

    if idx != obs_dim:
        # 남는 feature가 있으면 warning 정도만
        print(f"[Warning] env obs has extra {obs_dim - idx} dims not used by dp_obs_shapes")

    return obs_dict


# ======================================================================
# DPActor : diffusion policy를 "학습 시 horizon" 그대로 사용하는 actor
#   - env obs -> flat -> dict (env_obs_to_dp_obs_dict)
#   - To-step obs queue 유지해서 obs_seq 만들고
#   - diffusion UNet + scheduler 그대로 돌려서 action 뽑기
# ======================================================================

class DPActor:
    def __init__(self, dp_algo: DiffusionPolicyUNet):
        self.dp = dp_algo
        self.device = dp_algo.device

        self.To = dp_algo.algo_config.horizon.observation_horizon
        self.Ta = dp_algo.algo_config.horizon.action_horizon
        self.Tp = dp_algo.algo_config.horizon.prediction_horizon
        self.action_dim = dp_algo.ac_dim

        self.obs_queue = None  # deque of prepared obs dicts

    def reset(self):
        self.obs_queue = deque(maxlen=self.To)

    def _prepare_obs(self, obs_raw):
        """
        env raw obs -> diffusion policy obs dict (single step, [1, dim])
        normalization은 학습 config에서 hdf5_normalize_obs=False였으므로 생략.
        """
        return env_obs_to_dp_obs_dict(obs_raw, self.dp, self.device)

    def _build_stacked_obs(self, prepared_obs):
        """
        prepared_obs : single-step obs dict (각 value [1, dim])
        obs_queue에 쌓아서 [1, To, dim] 시퀀스로 변환
        """
        if self.obs_queue is None:
            self.reset()

        self.obs_queue.append(prepared_obs)

        # queue 길이가 To보다 짧으면 첫 obs를 앞쪽으로 pad
        if len(self.obs_queue) < self.To:
            first = self.obs_queue[0]
            while len(self.obs_queue) < self.To:
                self.obs_queue.appendleft(first)

        recent = list(self.obs_queue)[-self.To:]

        stacked = {}
        keys = prepared_obs.keys()
        for k in keys:
            seq = [step_obs[k] for step_obs in recent]  # list of [1, dim]
            # [To, 1, dim] → [1, To, dim]
            stacked[k] = torch.stack(seq, dim=1)
        return stacked  # dict of [1, To, dim]

    @torch.no_grad()
    def act(self, obs_raw):
        """
        env raw obs → prepared obs → To-step seq → diffusion policy로 action 하나 뽑기
        """
        prepared = self._prepare_obs(obs_raw)
        obs_seq = self._build_stacked_obs(prepared)  # dict [1, To, dim]

        # nets / ema 선택
        nets = self.dp.nets
        if self.dp.ema is not None:
            nets = self.dp.ema.averaged_model

        To = self.To
        Ta = self.Ta
        Tp = self.Tp
        action_dim = self.action_dim

        # scheduler step 수
        if self.dp.algo_config.ddpm.enabled:
            num_inference_timesteps = self.dp.algo_config.ddpm.num_inference_timesteps
        elif self.dp.algo_config.ddim.enabled:
            num_inference_timesteps = self.dp.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError("Either ddpm or ddim must be enabled")

        # obs encoder
        inputs = {"obs": obs_seq, "goal": None}
        for k in self.dp.obs_shapes:
            x = inputs["obs"][k]
            # [B, To, ...] 이어야 함
            assert x.ndim - 2 == len(self.dp.obs_shapes[k]), \
                f"obs_seq[{k}] shape {x.shape} vs obs_shape {self.dp.obs_shapes[k]}"

        obs_features = TensorUtils.time_distributed(
            inputs,
            nets["policy"]["obs_encoder"],
            inputs_as_kwargs=True,
        )  # [1, To, D]
        assert obs_features.ndim == 3
        B = obs_features.shape[0]

        # flatten → global_cond
        obs_cond = obs_features.flatten(start_dim=1)  # [1, To*D]

        # 학습 시 global_cond_dim과 동일해야 함
        # cond_in_layer = nets["policy"]["noise_pred_net"].cond_encoder[0]
        # cond_dim_expected = cond_in_layer.in_features
        # assert obs_cond.shape[1] == cond_dim_expected, \
        #     f"global_cond dim {obs_cond.shape[1]} != expected {cond_dim_expected}"

        # 초기 noise action
        noisy_action = torch.randn((B, Tp, action_dim), device=self.device)
        naction = noisy_action

        # scheduler 설정
        self.dp.noise_scheduler.set_timesteps(num_inference_timesteps)

        for t in self.dp.noise_scheduler.timesteps:
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction,
                timestep=t,
                global_cond=obs_cond,
            )
            naction = self.dp.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=naction,
            ).prev_sample

        # 학습 코드와 동일하게 Ta 개만 잘라서 사용
        start = To - 1
        end = start + Ta
        action_seq = naction[:, start:end]  # [B, Ta, Da]
        action = action_seq[:, 0, :]        # 첫 액션만 사용 [B, Da]
        return action, obs_seq  # obs_seq는 replay에 저장용


# ======================================================================
# SDAC
#   obs_shapes는 "single step" shape이고,
#   실제 obs는 항상 [B, To, ...] 형태로 들어온다고 가정
# ======================================================================

class DiffusionSDAC:
    def __init__(
        self,
        dp_algo: DiffusionPolicyUNet,
        obs_shapes,
        horizon_config,
        device: torch.device,
        gamma: float = 0.99,
        reward_scale: float = 0.2,
        tau: float = 0.005,
        lr_q: float = 3e-4,
        lr_policy: float = 1e-4,
        lr_alpha: float = 3e-4,
        reverse_mc_num: int = 64,
        target_entropy: float = -10.0,
    ):
        self.dp = dp_algo
        self.device = device
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.tau = tau
        self.reverse_mc_num = reverse_mc_num

        self.obs_shapes = obs_shapes
        self.To = horizon_config.observation_horizon
        self.Ta = horizon_config.action_horizon
        self.Tp = horizon_config.prediction_horizon
        self.action_dim = self.dp.ac_dim

        # ----- obs encoder 출력 차원 계산 (학습 구조 그대로) -----
        self.dp.nets.eval()
        dummy = {}
        for k, shape in obs_shapes.items():
            dummy[k] = torch.zeros((1, self.To) + tuple(shape), device=self.device)
        with torch.no_grad():
            obs_features = TensorUtils.time_distributed(
                {"obs": dummy, "goal": None},
                self.dp.nets["policy"]["obs_encoder"],
                inputs_as_kwargs=True,
            )  # [1, To, D]
        assert obs_features.ndim == 3
        D = obs_features.shape[-1]
        self.obs_cond_dim = D * self.To
        self.action_flat_dim = self.Tp * self.action_dim

        # ----- Q networks -----
        q_input_dim = self.obs_cond_dim + self.action_flat_dim
        self.q1 = QNetwork(q_input_dim).to(self.device)
        self.q2 = QNetwork(q_input_dim).to(self.device)
        self.target_q1 = QNetwork(q_input_dim).to(self.device)
        self.target_q2 = QNetwork(q_input_dim).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # ----- optimizers -----
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_q
        )
        self.policy_optimizer = torch.optim.Adam(
            self.dp.nets["policy"]["noise_pred_net"].parameters(), lr=lr_policy
        )
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = target_entropy

        self.running_q_mean = 0.0
        self.running_q_std = 1.0

    # ----------------------------------------------------------
    # obs_seq (dict of [B, To, ...]) → obs_cond
    # ----------------------------------------------------------

    def encode_obs(self, obs_dict):
        inputs = {"obs": {}, "goal": None}
        for k in self.obs_shapes:
            x = obs_dict[k]
            # 이미 [B, To, dim] 형태만 허용
            assert x.ndim - 2 == len(self.obs_shapes[k]), \
                f"encode_obs expects time-stacked obs, got {x.shape} for key {k}"
            inputs["obs"][k] = x

        with torch.no_grad():
            obs_features = TensorUtils.time_distributed(
                inputs,
                self.dp.nets["policy"]["obs_encoder"],
                inputs_as_kwargs=True,
            )  # [B, To, D]

        B, T, D = obs_features.shape
        assert T == self.To, f"T={T} != To={self.To}"
        obs_cond = obs_features.flatten(start_dim=1)  # [B, To*D]
        assert obs_cond.shape[1] == self.obs_cond_dim
        return obs_cond

    # ----------------------------------------------------------
    # diffusion policy로 action sequence 샘플 (Q / policy 업데이트용)
    #   obs_dict는 항상 time-stacked seq ([B, To, ...])
    # ----------------------------------------------------------

    def sample_action_sequence(self, obs_dict):
        self.dp.nets.eval()
        Tp = self.Tp
        action_dim = self.action_dim

        if self.dp.algo_config.ddpm.enabled:
            num_inference_timesteps = self.dp.algo_config.ddpm.num_inference_timesteps
        elif self.dp.algo_config.ddim.enabled:
            num_inference_timesteps = self.dp.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError("Either ddpm or ddim must be enabled in algo_config.")

        obs_cond = self.encode_obs(obs_dict)  # [B, obs_cond_dim]
        B = obs_cond.shape[0]

        noisy_action = torch.randn((B, Tp, action_dim), device=self.device)
        naction = noisy_action

        self.dp.noise_scheduler.set_timesteps(num_inference_timesteps)

        nets = self.dp.nets
        if self.dp.ema is not None:
            nets = self.dp.ema.averaged_model

        for t in self.dp.noise_scheduler.timesteps:
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction,
                timestep=t,
                global_cond=obs_cond,
            )
            naction = self.dp.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=naction,
            ).prev_sample

        return naction.clamp(-1, 1)  # [B, Tp, Da]

    # ----------------------------------------------------------
    # Q 업데이트
    # ----------------------------------------------------------

    def update_q(self, batch):
        obs = batch["obs"]        # dict of [B, To, ...]
        next_obs = batch["next_obs"]
        actions = batch["actions"]       # [B, Da]
        rewards = batch["rewards"]       # [B]
        dones = batch["dones"]           # [B]

        obs_cond = self.encode_obs(obs)
        next_obs_cond = self.encode_obs(next_obs)

        B = actions.shape[0]
        Tp = self.Tp
        Da = self.action_dim

        # 단일 스텝 action을 Tp 길이로 복제해서 Q 입력
        actions_seq = actions.unsqueeze(1).repeat(1, Tp, 1)         # [B, Tp, Da]
        action_flat = actions_seq.view(B, Tp * Da)

        with torch.no_grad():
            next_action_seq = self.sample_action_sequence(next_obs) # [B, Tp, Da]
            next_action_flat = next_action_seq.view(B, Tp * Da)

            inp_next = torch.cat([next_obs_cond, next_action_flat], dim=-1)
            q1_target = self.target_q1(inp_next).squeeze(-1)
            q2_target = self.target_q2(inp_next).squeeze(-1)
            q_target_min = torch.min(q1_target, q2_target)

            q_backup = self.reward_scale * rewards + (1.0 - dones) * self.gamma * q_target_min

        inp = torch.cat([obs_cond, action_flat], dim=-1)
        q1 = self.q1(inp).squeeze(-1)
        q2 = self.q2(inp).squeeze(-1)

        q1_loss = F.mse_loss(q1, q_backup)
        q2_loss = F.mse_loss(q2, q_backup)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)
            for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

        q_mean = q_target_min.mean().item()
        q_std = q_target_min.std().item()
        self.running_q_mean = 0.999 * self.running_q_mean + 0.001 * q_mean
        self.running_q_std = 0.999 * self.running_q_std + 0.001 * q_std

        return {
            "q_loss": q_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q_mean": q_mean,
            "q_std": q_std,
        }

    # ----------------------------------------------------------
    # Policy 업데이트 (RSM-style)
    # ----------------------------------------------------------

    def update_policy(self, batch):
        obs = batch["obs"]  # time-stacked
        B = list(obs.values())[0].shape[0]

        obs_cond = self.encode_obs(obs)  # [B, obs_cond_dim]

        with torch.no_grad():
            new_action_seq = self.sample_action_sequence(obs)       # [B, Tp, Da]

        timesteps = torch.randint(
            0, self.dp.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        noise1 = torch.randn_like(new_action_seq)
        noisy_actions = self.dp.noise_scheduler.add_noise(
            new_action_seq, noise1, timesteps
        )  # [B, Tp, Da]

        mc = self.reverse_mc_num
        noisy_actions_mc = noisy_actions.repeat_interleave(mc, dim=0)
        obs_cond_mc = obs_cond.repeat_interleave(mc, dim=0)
        timesteps_mc = timesteps.repeat_interleave(mc, dim=0)

        with torch.no_grad():
            noise_pred = self.dp.nets["policy"]["noise_pred_net"](
                noisy_actions_mc, timesteps_mc, global_cond=obs_cond_mc
            )
            recon = self.dp.noise_scheduler.step(
                model_output=noise_pred,
                timestep=timesteps_mc,
                sample=noisy_actions_mc,
            ).prev_sample
            recon = recon.clamp(-1, 1)

            Tp = self.Tp
            Da = self.action_dim
            recon_flat = recon.view(B * mc, Tp * Da)
            inp_q = torch.cat([obs_cond_mc, recon_flat], dim=-1)
            q1_mc = self.q1(inp_q)
            q2_mc = self.q2(inp_q)
            q_min_mc = torch.min(q1_mc, q2_mc).squeeze(-1)     # [B*mc]

            q_min_2d = q_min_mc.view(B, mc)
            Z = torch.logsumexp(q_min_2d, dim=1, keepdim=True)
            q_weights = torch.exp(q_min_2d - Z)                # [B, mc]
            q_weights = (q_weights / (q_weights.sum(dim=1, keepdim=True) + 1e-8)).view(-1)

        self.policy_optimizer.zero_grad()

        noise = torch.randn_like(new_action_seq)
        timesteps2 = torch.randint(
            0, self.dp.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()
        noisy_actions2 = self.dp.noise_scheduler.add_noise(
            new_action_seq.detach(), noise, timesteps2
        )

        noise_pred2 = self.dp.nets["policy"]["noise_pred_net"](
            noisy_actions2, timesteps2, global_cond=obs_cond
        )

        base_loss = F.mse_loss(noise_pred2, noise, reduction="none")  # [B, Tp, Da]
        base_loss = base_loss.mean(dim=(1, 2))                        # [B]

        with torch.no_grad():
            w_batch = q_weights.view(B, mc).mean(dim=1)
            w_batch = w_batch / (w_batch.mean() + 1e-8)

        policy_loss = (w_batch * base_loss).mean()
        policy_loss.backward()
        self.policy_optimizer.step()

        approx_entropy = 0.5 * self.action_dim * torch.log(
            2 * np.pi * np.e * (0.1 * torch.exp(self.log_alpha)) ** 2
        )
        alpha_loss = - self.log_alpha * (-approx_entropy.detach() + self.target_entropy)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "alpha": float(torch.exp(self.log_alpha).item()),
        }

    # ----------------------------------------------------------
    # 한 번의 SDAC update (Q + Policy)
    # ----------------------------------------------------------

    def update(self, batch):
        q_info = self.update_q(batch)
        p_info = self.update_policy(batch)
        info = {}
        info.update(q_info)
        info.update(p_info)
        return info


# ======================================================================
# Main
# ======================================================================

def main(_):
    # Wandb & exp name
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(
        project='sdac_dp',
        group=FLAGS.run_group,
        name=exp_name,
        entity=FLAGS.entity,
        mode=FLAGS.mode,
    )

    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir,
        wandb.run.project,
        FLAGS.run_group,
        FLAGS.env_name,
        exp_name,
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Env
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # Seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    prefixes = ["env", "sdac"]
    logger = LoggingHelper(
        csv_loggers={p: CsvLogger(os.path.join(FLAGS.save_dir, f"{p}.csv")) for p in prefixes},
        wandb_logger=wandb,
    )

    # -----------------------------
    # DiffusionPolicy 로드
    # -----------------------------
    rm_device = TorchUtils.get_torch_device(try_to_use_cuda=(device.type == "cuda"))
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=FLAGS.dp_ckpt,
        device=rm_device,
        verbose=True,
    )
    dp_algo: DiffusionPolicyUNet = rollout_policy.policy
    dp_algo.set_eval()

    horizon_config = dp_algo.algo_config.horizon
    obs_shapes = dp_algo.obs_shapes
    ac_dim = dp_algo.ac_dim

    # Actor (To-step stacking)
    dp_actor = DPActor(dp_algo=dp_algo)

    # SDAC
    sdac = DiffusionSDAC(
        dp_algo=dp_algo,
        obs_shapes=obs_shapes,
        horizon_config=horizon_config,
        device=device,
        gamma=FLAGS.discount,
        reward_scale=FLAGS.reward_scale,
        tau=FLAGS.tau,
    )

    # Replay Buffer
    replay = ReplayBuffer(capacity=FLAGS.buffer_size, device=device)

    # -----------------------------
    # Online RL with SDAC + DP
    # -----------------------------
    total_steps = FLAGS.total_env_steps
    batch_size = FLAGS.batch_size

    step = 0
    while step < total_steps:
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            obs_raw, _ = reset_out
        else:
            obs_raw = reset_out

        dp_actor.reset()
        done = False
        episode_return = 0.0

        # 첫 obs로부터 seq 만들고 action도 같이 얻음
        with torch.no_grad():
            action_tensor, obs_seq = dp_actor.act(obs_raw)  # action_tensor: [1, Da], obs_seq: dict [1, To, ...]

        while not done and step < total_steps:
            action_np = action_tensor.cpu().numpy()[0]

            step_out = env.step(
                np.clip(action_np, -1.0 + 1e-5, 1.0 - 1e-5)
            )
            if len(step_out) == 5:
                next_obs_raw, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_obs_raw, reward, done, info = step_out

            episode_return += float(reward)

            if is_robomimic_env(FLAGS.env_name):
                reward = reward - 1.0

            # 다음 obs 기준 seq & 다음 action 미리 계산
            with torch.no_grad():
                next_action_tensor, next_obs_seq = dp_actor.act(next_obs_raw)

            # replay에는 항상 time-stacked obs_seq 저장
            replay.add(obs_seq, action_tensor, reward, next_obs_seq, done)

            # SDAC 업데이트
            if step >= FLAGS.start_training and len(replay) >= batch_size:
                for _ in range(FLAGS.utd_ratio):
                    batch = replay.sample(batch_size)
                    info_s = sdac.update(batch)
                    logger.log(info_s, "sdac", step=step)

            env_info = {"episode_return": episode_return}
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    env_info[k] = v
            logger.log(env_info, "env", step=step)

            obs_raw = next_obs_raw
            obs_seq = next_obs_seq
            action_tensor = next_action_tensor
            step += 1

            if step >= total_steps:
                break

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)


if __name__ == '__main__':
    app.run(main)
