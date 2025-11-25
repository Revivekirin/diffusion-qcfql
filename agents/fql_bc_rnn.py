import copy
import glob
import os
import pickle
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from utils.encoders import ImpalaEncoder
from utils.networks import Value, ActorVectorField

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy


class ACFQLAgent:
    """Flow Q-learning (FQL) agent with action chunking."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        teacher_ckpt: Optional[str] = None,
    ):
        """Initialize the agent.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            config: Configuration dictionary
            device: Device to run on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Setup encoders
        encoders = {}
        if config.get('encoder') is not None:
            encoder_config = self._get_encoder_config(config['encoder'])
            encoders['critic'] = encoder_config()
            encoders['actor_bc_flow'] = encoder_config()
            encoders['actor_onestep_flow'] = encoder_config()
            
        # Determine full action dimension
        if config['action_chunking']:
            full_action_dim = action_dim * config['horizon_length']
        else:
            full_action_dim = action_dim
            
        # Build networks
        encoded_obs_dim = obs_dim  # Will be set by encoder if used
        
        self.critic = Value(
            input_dim=encoded_obs_dim + full_action_dim,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        ).to(device)
        
        self.target_critic = copy.deepcopy(self.critic).to(device)
        
        self.actor_bc_flow = ActorVectorField(
            obs_dim=encoded_obs_dim,
            action_dim=full_action_dim,
            hidden_dims=config["actor_hidden_dims"],
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get('actor_bc_flow'),
            use_time=True,                            
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        ).to(device)

        # one-step flow: 시간 사용 안 함 
        self.actor_onestep_flow = ActorVectorField(
            obs_dim=encoded_obs_dim,
            action_dim=full_action_dim,
            hidden_dims=config["actor_hidden_dims"],
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get('actor_onestep_flow'),
            use_time=False,                           
            use_fourier_features=False,               
        ).to(device)
        
        print(next(self.actor_onestep_flow.mlp.layers[0].parameters()).device)
        print(next(self.critic.value_nets[0].layers[0].parameters()).device)

        # Optimizers
        all_params = (
            list(self.critic.parameters()) +
            list(self.actor_bc_flow.parameters()) +
            list(self.actor_onestep_flow.parameters())
        )
        
        if config.get('weight_decay', 0) > 0:
            self.optimizer = torch.optim.AdamW(
                all_params, 
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )
        else:
            self.optimizer = torch.optim.Adam(all_params, lr=config['lr'])
        
        # teacher policy config
        self.teacher_policy = None
        self.teacher_obs_splits = None


        if teacher_ckpt is not None:
            rm_device = TorchUtils.get_torch_device(try_to_use_cuda=device.startswith("cuda"))
            policy, ckpt_dict = FileUtils.policy_from_checkpoint(
                ckpt_path=teacher_ckpt,
                device=rm_device,
                verbose=True,
            )
            self.teacher_policy = policy

            # bc_test.json low_dim 순서 기준:
            # ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            #  "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "object"]
            # base_dims = [
            #     ("robot0_eef_pos", 3),
            #     ("robot0_eef_quat", 4),
            #     ("robot0_gripper_qpos", 2),
            #     ("robot1_eef_pos", 3),
            #     ("robot1_eef_quat", 4),
            #     ("robot1_gripper_qpos", 2),

            # ]
            self.teacher_obs_splits = [
                ("robot0_eef_pos", 3),
                ("robot0_eef_quat", 4),
                ("robot0_gripper_qpos", 2),
                ("robot1_eef_pos", 3),
                ("robot1_eef_quat", 4),
                ("robot1_gripper_qpos", 2),
                # ("object", obs_dim - 18),   # 남은 건 전부 object로
                ("object", 41)
            ]

        self.step = 0

    
    # @torch.no_grad()
    # def _flat_obs_to_teacher_dict_from_np(self, obs_flat_np: np.ndarray) -> Dict[str, np.ndarray]:
    #     """
    #     obs_flat_np: [B, obs_dim] np.ndarray
    #     return: {key: np.ndarray[B, dim_k]}
    #     """
    #     assert self.teacher_obs_splits is not None

    #     B, D = obs_flat_np.shape
    #     total_dim = sum(dim for _, dim in self.teacher_obs_splits)
    #     if D != total_dim:
    #         raise ValueError(
    #             f"[teacher obs split mismatch] obs_flat dim={D}, "
    #             f"but teacher_obs_splits sum={total_dim}. "
    #             f"teacher_obs_splits={self.teacher_obs_splits}"
    #         )

    #     obs_dict = {}
    #     start = 0
    #     for key, dim in self.teacher_obs_splits:
    #         end = start + dim
    #         obs_dict[key] = obs_flat_np[:, start:end].reshape(B, dim)
    #         start = end
    #     return obs_dict
    
    @torch.no_grad()
    def _flat_obs_to_teacher_dict_from_np(self, obs_flat_np: np.ndarray) -> Dict[str, np.ndarray]:
        """
        obs_flat_np: [B, obs_dim] np.ndarray (FQL에서 오는 obs)
        return: {key: np.ndarray[B, dim_k]}  # teacher가 기대하는 59차원 분해
        """
        assert self.teacher_obs_splits is not None

        B, D = obs_flat_np.shape
        total_dim = sum(dim for _, dim in self.teacher_obs_splits)  # teacher 기준: 보통 59

        if D < total_dim:
            # 뒤에 0 채워서 teacher_dim에 맞추기
            pad = total_dim - D
            print(f"[ACFQLAgent] WARNING: obs_dim={D} < teacher_dim={total_dim}, "
                f"padding {pad} zeros at the end.")
            obs_flat_np = np.concatenate(
                [obs_flat_np, np.zeros((B, pad), dtype=obs_flat_np.dtype)],
                axis=1,
            )
        elif D > total_dim:
            # 너무 길면 앞 total_dim만 사용
            print(f"[ACFQLAgent] WARNING: obs_dim={D} > teacher_dim={total_dim}, "
                f"truncating to first {total_dim} dims.")
            obs_flat_np = obs_flat_np[:, :total_dim]

        # 여기서부터 obs_flat_np.shape[1] == total_dim 보장
        obs_dict = {}
        start = 0
        for key, dim in self.teacher_obs_splits:
            end = start + dim
            obs_dict[key] = obs_flat_np[:, start:end].reshape(B, dim)
            start = end

        return obs_dict


    @torch.no_grad()
    def teacher_action(self, obs_flat: Tensor) -> Tensor:
        """
        obs_flat: [B, obs_dim]
        return: [B, action_dim]
        """
        if self.teacher_policy is None:
            raise RuntimeError("teacher_policy is None, but teacher_action was called.")

        obs_np = obs_flat.detach().cpu().numpy()  # [B, obs_dim]
        # 여기서 한 번에 dict로 쪼갬
        obs_dict_np = self._flat_obs_to_teacher_dict_from_np(obs_np)  # {k: [B, dim_k]}

        # RolloutPolicy가 batch 입력을 지원한다면:
        acts_np = self.teacher_policy(ob=obs_dict_np, batched_ob=True) 
        acts_np = np.asarray(acts_np)
        acts = torch.from_numpy(acts_np).to(self.device).float()
        return acts


    # @torch.no_grad()
    # def teacher_action(self, obs_flat: Tensor) -> Tensor:
    #     """
    #     obs_flat: [B, obs_dim] torch.Tensor
    #     return: [B, action_dim] torch.Tensor
    #     """
    #     if self.teacher_policy is None:
    #         raise RuntimeError("teacher_policy is None, but teacher_action was called.")

    #     obs_np = obs_flat.detach().cpu().numpy()   # [B, obs_dim]
    #     B = obs_np.shape[0]
    #     actions = []

    #     for i in range(B):
    #         # (1, obs_dim) -> dict {k: (1, dim_k)}
    #         obs_dict_batched = self._flat_obs_to_teacher_dict_from_np(obs_np[i:i+1])
    #         # env 스타일로: 각 value를 (dim,)로 만들어 줌
    #         obs_dict_single = {k: v[0] for k, v in obs_dict_batched.items()}  # shape (dim,)

    #         # RolloutPolicy는 run_trained_agent와 동일하게 np dict를 기대
    #         act_np = self.teacher_policy(ob=obs_dict_single)  # shape (action_dim,) 또는 (1, action_dim)
    #         act_np = np.asarray(act_np)
    #         if act_np.ndim == 2 and act_np.shape[0] == 1:
    #             act_np = act_np[0]
    #         actions.append(act_np)

    #     acts_np = np.stack(actions, axis=0)   # [B, action_dim]
    #     acts = torch.from_numpy(acts_np).to(self.device).float()
    #     return acts



        
    def _get_encoder_config(self, encoder_name: str):
        """Get encoder configuration."""
        encoder_configs = {
            'impala': lambda: ImpalaEncoder(),
            'impala_small': lambda: ImpalaEncoder(num_blocks=1),
            'impala_large': lambda: ImpalaEncoder(
                stack_sizes=(64, 128, 128), 
                mlp_hidden_dims=(1024,)
            ),
        }
        return encoder_configs.get(encoder_name, lambda: None)
        
    def critic_loss(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the FQL critic loss."""
        if self.config['action_chunking']:
            batch_actions = batch['actions'].reshape(batch['actions'].shape[0], -1)
        else:
            batch_actions = batch['actions'][..., 0, :]
            
        # TD loss
        with torch.no_grad():
            next_actions = self.sample_actions(batch['next_observations'][..., -1, :])
            next_qs = self.target_critic(
                batch['next_observations'][..., -1, :], 
                actions=next_actions
            )
            
            if self.config['q_agg'] == 'min':
                next_q = next_qs.min(dim=0)[0]
            else:
                next_q = next_qs.mean(dim=0)
                
            target_q = (
                batch['rewards'][..., -1] +
                (self.config['discount'] ** self.config['horizon_length']) *
                batch['masks'][..., -1] * next_q
            )
        
        q = self.critic(batch['observations'], actions=batch_actions)

        critic_loss = ((q - target_q) ** 2 * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss.item(),
            'q_mean': q.mean().item(),
            'q_max': q.max().item(),
            'q_min': q.min().item(),
        }
        
    def actor_loss(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the FQL actor loss."""

        if self.config['action_chunking']:
            batch_actions = batch['actions'].reshape(batch['actions'].shape[0], -1)
        else:
            batch_actions = batch['actions'][..., 0, :]

        batch_size, action_dim = batch_actions.shape

        # -------------------
        # 1) teacher actions
        # -------------------
        if self.teacher_policy is not None and self.config.get("use_teacher_for_flow", True):
            # obs 시퀀스가 [B, T, obs_dim]이면 마지막 타임스텝 사용 (필요시 수정)
            obs_for_teacher = batch['observations'][..., -1, :] \
                if batch['observations'].ndim == 3 else batch['observations']
            with torch.no_grad():
                teacher_acts = self.teacher_action(obs_for_teacher)   # [B, action_dim]
        else:
            teacher_acts = batch_actions 

        # BC flow loss
        x_0 = torch.randn(batch_size, action_dim, device=self.device)
        beta = self.config.get("teacher_mix_beta", 1.0)  # 1이면 pure teacher
        mix_actions = beta * teacher_acts + (1.0 - beta) * batch_actions
        x_1 = mix_actions
        t = torch.rand(batch_size, 1, device=self.device)
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        # BC flow loss
        # x_0 = torch.randn(batch_size, action_dim, device=self.device)
        # x_1 = batch_actions
        # t = torch.rand(batch_size, 1, device=self.device)
        # x_t = (1 - t) * x_0 + t * x_1
        # vel = x_1 - x_0
        
        pred = self.actor_bc_flow(batch['observations'], x_t, t)
        
        if self.config['action_chunking']:
            pred_reshaped = pred.reshape(
                batch_size, 
                self.config['horizon_length'], 
                self.config['action_dim']
            )
            vel_reshaped = vel.reshape(
                batch_size,
                self.config['horizon_length'],
                self.config['action_dim']
            )
            bc_flow_loss = (
                ((pred_reshaped - vel_reshaped) ** 2) * 
                batch['valid'][..., None]
            ).mean()
        else:
            bc_flow_loss = ((pred - vel) ** 2).mean()
            
        # Distillation and Q loss
        if self.config['actor_type'] == 'distill-ddpg':
            noises = torch.randn(batch_size, action_dim, device=self.device)
            
            with torch.no_grad():
                target_flow_actions = self.compute_flow_actions(
                    batch['observations'], 
                    noises=noises
                )
                
            actor_actions = self.actor_onestep_flow(
                batch['observations'], 
                noises,
                # times=torch.zeros(batch_size, 1, device=self.device)
            )
            distill_loss = ((actor_actions - target_flow_actions) ** 2).mean()
            
            actor_actions = torch.clamp(actor_actions, -1, 1)

            # lambda_q = self.config.get("lambda_q", 0.01)  # 새 하이퍼
            
            qs = self.critic(batch['observations'], actions=actor_actions).detach()
            q = qs.mean(dim=0)
            # q_loss = -lambda_q * q.mean()
            q_loss = -q.mean()
        else:
            distill_loss = torch.zeros(1, device=self.device)
            q_loss = torch.zeros(1, device=self.device)

        actor_loss = (
            bc_flow_loss +
            self.config['alpha'] * distill_loss +
            q_loss
        )

        
        return actor_loss, {
            'actor_loss': actor_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            'distill_loss': distill_loss.item(),
        }
        
    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Update the agent."""
        # Convert batch to tensors on device
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        
        self.optimizer.zero_grad()

        critic_loss, critic_info = self.critic_loss(batch)
        actor_loss, actor_info = self.actor_loss(batch)

        total_loss = critic_loss + actor_loss
        total_loss.backward()

        # grad 통계는 여기서 (clip 전/후 둘 중 원하는 걸로)
        grad_norms = []
        for p in self.critic.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.data.norm().item())
        for p in self.actor_bc_flow.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.data.norm().item())
        for p in self.actor_onestep_flow.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.data.norm().item())

        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) +
            list(self.actor_bc_flow.parameters()) +
            list(self.actor_onestep_flow.parameters()),
            max_norm=10.0,
        )

        self.optimizer.step()

        # Update target network
        self._target_update()
        
        self.step += 1
        
        info = {
            f'critic/{k}': v for k, v in critic_info.items()
        }
        info.update({
            f'actor/{k}': v for k, v in actor_info.items()
        })
        info['grad/norm'] = np.mean(grad_norms) if grad_norms else 0.0
        
        return info
        
    def _target_update(self):
        """Soft update of target network."""
        tau = self.config['tau']
        for target_param, param in zip(
            self.target_critic.parameters(), 
            self.critic.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
            
    @torch.no_grad()
    def sample_actions(self, observations: Tensor) -> Tensor:
        """Sample actions from the policy."""
        if self.config['actor_type'] == 'distill-ddpg':
            batch_size = observations.shape[0]
            action_dim = (
                self.action_dim * self.config['horizon_length']
                if self.config['action_chunking']
                else self.action_dim
            )
            noises = torch.randn(batch_size, action_dim, device=self.device)
            actions = self.actor_onestep_flow(
                observations,
                noises,
                # times=torch.zeros(batch_size, 1, device=self.device)
            )
            actions = torch.clamp(actions, -1, 1)
            
        elif self.config['actor_type'] == 'best-of-n':
            batch_size = observations.shape[0]
            action_dim = (
                self.action_dim * self.config['horizon_length']
                if self.config['action_chunking']
                else self.action_dim
            )
            num_samples = self.config['actor_num_samples']
            
            noises = torch.randn(
                batch_size, num_samples, action_dim, 
                device=self.device
            )
            obs_expanded = observations.unsqueeze(1).expand(
                -1, num_samples, -1
            ).reshape(batch_size * num_samples, -1)
            noises_flat = noises.reshape(batch_size * num_samples, action_dim)
            
            actions = self.compute_flow_actions(obs_expanded, noises_flat)
            actions = torch.clamp(actions, -1, 1)
            actions = actions.reshape(batch_size, num_samples, action_dim)
            
            # Compute Q-values and select best
            obs_for_q = observations.unsqueeze(1).expand(-1, num_samples, -1)
            q_vals = self.critic(
                obs_for_q.reshape(batch_size * num_samples, -1),
                actions.reshape(batch_size * num_samples, action_dim)
            )
            q_vals = q_vals.reshape(
                self.config['num_qs'], batch_size, num_samples
            )
            
            if self.config['q_agg'] == 'mean':
                q = q_vals.mean(dim=0)
            else:
                q = q_vals.min(dim=0)[0]
                
            best_indices = q.argmax(dim=1)
            actions = actions[torch.arange(batch_size), best_indices]
            
        return actions
        
    @torch.no_grad()
    def compute_flow_actions(
        self, 
        observations: Tensor, 
        noises: Tensor
    ) -> Tensor:
        """Compute actions using Euler method for flow."""
        if self.actor_bc_flow.encoder is not None:
            observations = self.actor_bc_flow.encoder(observations)
            
        actions = noises
        for i in range(self.config['flow_steps']):
            t = torch.full(
                (observations.shape[0], 1),
                i / self.config['flow_steps'],
                device=self.device
            )
            vels = self.actor_bc_flow(
                observations, actions, t, is_encoded=True
            )
            actions = actions + vels / self.config['flow_steps']
            
        actions = torch.clamp(actions, -1, 1)
        return actions
        
    def save(self, save_path: str):
        """Save agent."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_bc_flow': self.actor_bc_flow.state_dict(),
            'actor_onestep_flow': self.actor_onestep_flow.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'config': self.config,
        }, save_path)
        print(f'Saved to {save_path}')
        
    def load(self, load_path: str):
        """Load agent."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_bc_flow.load_state_dict(checkpoint['actor_bc_flow'])
        self.actor_onestep_flow.load_state_dict(checkpoint['actor_onestep_flow'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        print(f'Loaded from {load_path}')
    
    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: np.ndarray,
        ex_actions: np.ndarray,
        config: Dict[str, Any],
        teacher_ckpt: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """Create a new agent (JAX-style interface for compatibility).
        
        Args:
            seed: Random seed
            ex_observations: Example observations to infer dimensions
            ex_actions: Example actions to infer dimensions
            config: Configuration dictionary
            device: Device to run on
            
        Returns:
            ACFQLAgent instance
        """
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Infer dimensions from example data
        obs_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        
        # Update config with inferred dimensions
        config = config.copy()
        config['action_dim'] = action_dim
        config['ob_dims'] = ex_observations.shape
        
        # Create and return agent
        return cls(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config,
            teacher_ckpt=teacher_ckpt,
            device=device,
        )


def get_config():
    """Get default configuration."""
    config = {
        'agent_name': 'acfql',
        'lr': 3e-4,
        'batch_size': 256,
        'actor_hidden_dims': (512, 512, 512, 512),
        'value_hidden_dims': (512, 512, 512, 512),
        'layer_norm': True,
        'actor_layer_norm': False,
        'discount': 0.99,
        'tau': 0.005,
        'q_agg': 'mean',
        'alpha': 100.0,
        'num_qs': 2,
        'flow_steps': 10,
        'encoder': None,
        'horizon_length': 1,
        'action_chunking': True,
        'actor_type': 'distill-ddpg',
        'actor_num_samples': 32,
        'use_fourier_features': False,
        'fourier_feature_dim': 64,
        'weight_decay': 0.0,

        'teacher_mix_beta' : 1.0
    }
    return config