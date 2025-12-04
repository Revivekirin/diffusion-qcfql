import copy
from typing import Any, Dict, Optional, Sequence, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from utils.networks import Actor, Value, LogAlpha
import ml_collections


@dataclass
class SACConfig:
    """SAC configuration."""
    agent_name: str = 'sac'
    lr: float = 3e-4
    batch_size: int = 256
    actor_hidden_dims: tuple = (512, 512, 512, 512)
    value_hidden_dims: tuple = (512, 512, 512, 512)
    layer_norm: bool = True
    actor_layer_norm: bool = False
    discount: float = 0.99
    tau: float = 0.005
    target_entropy: Optional[float] = None
    target_entropy_multiplier: float = 0.5
    tanh_squash: bool = True
    state_dependent_std: bool = True
    actor_fc_scale: float = 0.01
    q_agg: str = 'min'  # 'min' or 'mean'
    backup_entropy: bool = False
    alpha: float = 100.0


class SACAgent:
    """Soft Actor-Critic (SAC) agent in PyTorch."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: SACConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        
        # Set target entropy
        if config.target_entropy is None:
            self.target_entropy = -config.target_entropy_multiplier * action_dim
        else:
            self.target_entropy = config.target_entropy
        
        # Create networks
        self.actor = Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.actor_hidden_dims,
            layer_norm=config.actor_layer_norm,
            tanh_squash=config.tanh_squash,
            state_dependent_std=config.state_dependent_std,
            const_std=False,
            final_fc_init_scale=config.actor_fc_scale,
        ).to(device)
        
        critic_input_dim = obs_dim + action_dim
        self.critic = Value(
            input_dim=critic_input_dim,
            hidden_dims=config.value_hidden_dims,
            layer_norm=config.layer_norm,
            num_ensembles=2,
        ).to(device)
        
        self.target_critic = Value(
            input_dim=critic_input_dim,
            hidden_dims=config.value_hidden_dims,
            layer_norm=config.layer_norm,
            num_ensembles=2,
        ).to(device)
        
        # Copy critic parameters to target critic
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Alpha (temperature parameter)
        self.log_alpha = LogAlpha().to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=config.lr)
        print("[DEBUG] SACAgent is loaded!!")

    def critic_loss(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Dict[str, float]]:
        """Compute critic loss."""
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        masks = batch['masks']  # (1 - done)
        
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs = self.actor.get_action_and_log_prob(next_observations)
            
            # Compute target Q values
            next_qs = self.target_critic(next_observations, next_actions)
            
            if self.config.q_agg == 'min':
                next_q = next_qs.min(dim=0)[0]
            else:
                next_q = next_qs.mean(dim=0)
            
            target_q = rewards + self.config.discount * masks * next_q
            
            if self.config.backup_entropy:
                alpha = self.log_alpha()
                target_q = target_q - self.config.discount * masks * next_log_probs * alpha
        
        # Compute current Q values
        current_qs = self.critic(observations, actions)
        critic_loss = F.mse_loss(current_qs, target_q.unsqueeze(0).expand_as(current_qs))
        
        info = {
            'critic_loss': critic_loss.item(),
            'q_mean': current_qs.mean().item(),
            'q_max': current_qs.max().item(),
            'q_min': current_qs.min().item(),
        }
        
        return critic_loss, info
    
    def actor_loss(self, batch: Dict[str, Tensor]) -> tuple[Tensor, Dict[str, float]]:
        """Compute actor and alpha loss."""
        observations = batch['observations']
        
        # Sample actions from current policy
        actions, log_probs = self.actor.get_action_and_log_prob(observations)
        
        # Compute Q values
        qs = self.critic(observations, actions)
        q = qs.mean(dim=0)
        
        # Actor loss
        alpha = self.log_alpha()
        actor_loss = (alpha.detach() * log_probs - q).mean()
        
        # Alpha loss
        entropy = -log_probs.detach().mean()
        alpha_loss = (alpha * (entropy - self.target_entropy)).mean()
        
        total_loss = actor_loss + alpha_loss
        
        info = {
            'total_loss': total_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item(),
            'entropy': -log_probs.mean().item(),
            'q': q.mean().item(),
        }
        
        return total_loss, info
    
    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Update all networks."""
        # Convert batch to tensors on device
        batch = {k: torch.FloatTensor(v).to(self.device) if isinstance(v, np.ndarray) 
                else v.to(self.device) for k, v in batch.items()}
        
        info = {}
        
        # Update critic
        critic_loss, critic_info = self.critic_loss(batch)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v
        
        # Update actor and alpha
        actor_loss, actor_info = self.actor_loss(batch)
        self.actor_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.alpha_optimizer.step()
        
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        
        # Update target critic
        self.soft_update_target()
        
        return info
    
    def soft_update_target(self):
        """Soft update of target network."""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    @torch.no_grad()
    def sample_actions(
        self,
        observations: Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Sample actions from the policy."""
        if isinstance(observations, np.ndarray):
            observations = torch.FloatTensor(observations).to(self.device)
        
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)
        
        if deterministic:
            dist = self.actor(observations, temperature)
            if self.config.tanh_squash:
                actions = torch.tanh(dist.base_dist.mean)
            else:
                actions = dist.mean
        else:
            actions, _ = self.actor.get_action_and_log_prob(observations, temperature)
        
        actions = torch.clamp(actions, -1, 1)
        return actions.cpu().numpy()
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'log_alpha': self.log_alpha.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.log_alpha.load_state_dict(checkpoint['log_alpha'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='sac',
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            target_entropy=None,
            target_entropy_multiplier=0.5,
            tanh_squash=True,
            state_dependent_std=True,
            actor_fc_scale=0.01,
            q_agg='min',
            backup_entropy=False,
            alpha=100.0,
        )
    )