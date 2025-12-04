import os
import json
import random
import time
from typing import Dict, Any

import numpy as np
import torch
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents.sac import SACAgent, SACConfig
from envs.env_utils import make_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.datasets import Dataset
from evaluation import evaluate, flatten
from log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from tensordict import TensorDict

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('agent', 'agents/sac.py', lock_config=False)

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'square-mg-low', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_float('dataset_proportion', 1.0, 'Proportion of dataset to use.')

flags.DEFINE_string('entity', 'sophia435256-robros', 'wandb entity')
flags.DEFINE_string('mode', 'online', 'wandb mode')

flags.DEFINE_bool('sparse', False, "make the task sparse reward")

# SAC specific flags
flags.DEFINE_float('lr', 3e-4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_string('actor_hidden_dims', '512,512,512,512', 'Actor hidden dimensions (comma-separated).')
flags.DEFINE_string('value_hidden_dims', '512,512,512,512', 'Value hidden dimensions (comma-separated).')
flags.DEFINE_boolean('layer_norm', True, 'Whether to use layer normalization.')
flags.DEFINE_boolean('actor_layer_norm', False, 'Whether to use layer normalization for actor.')
flags.DEFINE_float('discount', 0.99, 'Discount factor.')
flags.DEFINE_float('tau', 0.005, 'Target network update rate.')
flags.DEFINE_float('target_entropy', None, 'Target entropy.')
flags.DEFINE_float('target_entropy_multiplier', 0.5, 'Target entropy multiplier.')
flags.DEFINE_boolean('tanh_squash', True, 'Whether to squash actions with tanh.')
flags.DEFINE_boolean('state_dependent_std', True, 'Whether to use state-dependent std.')
flags.DEFINE_float('actor_fc_scale', 0.01, 'Actor final FC init scale.')
flags.DEFINE_string('q_agg', 'min', 'Q aggregation method (min or mean).')
flags.DEFINE_boolean('backup_entropy', False, 'Whether to backup entropy.')


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


def parse_hidden_dims(hidden_dims_str: str) -> tuple:
    """Parse comma-separated string to tuple of ints."""
    return tuple(int(x) for x in hidden_dims_str.split(','))


def get_sac_config() -> SACConfig:
    """Create SAC config from FLAGS."""
    return SACConfig(
        agent_name='sac',
        lr=FLAGS.lr,
        batch_size=FLAGS.batch_size,
        actor_hidden_dims=parse_hidden_dims(FLAGS.actor_hidden_dims),
        value_hidden_dims=parse_hidden_dims(FLAGS.value_hidden_dims),
        layer_norm=FLAGS.layer_norm,
        actor_layer_norm=FLAGS.actor_layer_norm,
        discount=FLAGS.discount,
        tau=FLAGS.tau,
        target_entropy=FLAGS.target_entropy,
        target_entropy_multiplier=FLAGS.target_entropy_multiplier,
        tanh_squash=FLAGS.tanh_squash,
        state_dependent_std=FLAGS.state_dependent_std,
        actor_fc_scale=FLAGS.actor_fc_scale,
        q_agg=FLAGS.q_agg,
        backup_entropy=FLAGS.backup_entropy,
    )


def save_agent(agent: SACAgent, save_dir: str, epoch: int):
    """Save agent checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f'checkpoint_{epoch}.pt')
    agent.save(filepath)
    print(f'Saved checkpoint to {filepath}')


def restore_agent(agent: SACAgent, restore_path: str, restore_epoch: int = None):
    """Restore agent from checkpoint."""
    if restore_epoch is not None:
        filepath = os.path.join(restore_path, f'checkpoint_{restore_epoch}.pt')
    else:
        filepath = restore_path
    
    agent.load(filepath)
    print(f'Restored checkpoint from {filepath}')
    return agent


def evaluate_agent(
    agent: SACAgent,
    env,
    num_eval_episodes: int = 50,
    num_video_episodes: int = 0,
    video_frame_skip: int = 3,
) -> tuple[Dict[str, float], list, list]:
    """Evaluate agent."""
    returns = []
    successes = []
    episode_lengths = []
    renders = []
    
    for ep_idx in range(num_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        episode_renders = []
        
        while not done:
            # Sample action deterministically for evaluation
            action = agent.sample_actions(obs, deterministic=True)
            action = action[0]  # Remove batch dimension
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            
            # Render for video
            if ep_idx < num_video_episodes and episode_length % video_frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    episode_renders.append(frame)
        
        returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        # Check for success
        if 'success' in info:
            successes.append(float(info['success']))
        elif 'is_success' in info:
            successes.append(float(info['is_success']))
        
        if ep_idx < num_video_episodes:
            renders.append(episode_renders)
    
    eval_info = {
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'length_mean': np.mean(episode_lengths),
    }
    
    if len(successes) > 0:
        eval_info['success_rate'] = np.mean(successes)
    
    return eval_info, None, renders


def numpy_to_tensordict(batch: Dict[str, np.ndarray], device: str = 'cpu') -> TensorDict:
    """Convert numpy batch to TensorDict."""
    tensor_batch = {
        k: torch.FloatTensor(v).to(device) 
        for k, v in batch.items()
    }
    return TensorDict(tensor_batch, batch_size=len(batch['observations']))


def tensordict_to_numpy(td: TensorDict) -> Dict[str, np.ndarray]:
    """Convert TensorDict to numpy batch."""
    return {k: v.cpu().numpy() for k, v in td.items()}


def create_replay_buffer_from_dataset(
    dataset: Dataset,
    buffer_size: int,
    batch_size: int,
    device: str = 'cpu'
) -> TensorDictReplayBuffer:
    """Create TorchRL replay buffer and populate with initial dataset."""
    storage = LazyTensorStorage(buffer_size)
    sampler = RandomSampler()
    
    replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=sampler,
        batch_size=batch_size,
    )
    
    # Add initial dataset to replay buffer
    dataset_dict = dict(dataset)
    num_samples = len(dataset_dict['observations'])
    
    print(f"Adding {num_samples} transitions to replay buffer...")
    for i in tqdm.tqdm(range(num_samples), desc="Loading dataset"):
        transition = {k: v[i:i+1] for k, v in dataset_dict.items()}
        td = numpy_to_tensordict(transition, device=device)
        replay_buffer.extend(td)
    
    print(f"Replay buffer size: {len(replay_buffer)}")
    return replay_buffer


def add_transition_to_buffer(
    replay_buffer: TensorDictReplayBuffer,
    transition: Dict[str, Any],
    device: str = 'cpu'
):
    """Add single transition to replay buffer."""
    # Ensure all values are numpy arrays with batch dimension
    transition_batch = {}
    for k, v in transition.items():
        if isinstance(v, (int, float)):
            v = np.array([v], dtype=np.float32)
        elif isinstance(v, np.ndarray):
            if v.ndim == 0:
                v = v.reshape(1)
            elif v.ndim == 1:
                v = v.reshape(1, -1)
        transition_batch[k] = v
    
    td = numpy_to_tensordict(transition_batch, device=device)
    replay_buffer.extend(td)


def main(_):
    # Set random seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
        torch.cuda.manual_seed_all(FLAGS.seed)
        torch.backends.cudnn.deterministic = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Set up logger
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name,
                      entity=FLAGS.entity, mode=FLAGS.mode)
    
    FLAGS.save_dir = os.path.join(
        FLAGS.save_dir, 
        wandb.run.project, 
        FLAGS.run_group, 
        FLAGS.env_name,
        exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)
    
    # Make environment and datasets
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, 
        frame_stack=FLAGS.frame_stack
    )
    
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning not supported for visual environments.'
    
    # Get config
    config = get_sac_config()

    def process_train_dataset(ds):
        """
        - dataset_proportion
        - sparse reward
        - robomimic reward penalty
        """
        ds = Dataset.create(**ds)

        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )

        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)

        if FLAGS.sparse:
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    # Set up datasets
    train_dataset = process_train_dataset(train_dataset)
    
    # Create TorchRL replay buffer
    replay_buffer = create_replay_buffer_from_dataset(
        dataset=train_dataset,
        buffer_size=max(FLAGS.buffer_size, train_dataset.size + 1),
        batch_size=config.batch_size,
        device='cpu'  # Store on CPU, move to GPU during sampling
    )
    
    # Create separate dataset buffer for balanced sampling if needed
    if FLAGS.balanced_sampling:
        dataset_buffer = create_replay_buffer_from_dataset(
            dataset=train_dataset,
            buffer_size=train_dataset.size,
            batch_size=config.batch_size // 2,
            device='cpu'
        )
    
    # Get dimensions from example batch
    example_td = replay_buffer.sample(1)
    obs_dim = example_td['observations'].shape[-1]
    action_dim = example_td['actions'].shape[-1]
    
    print(f'Observation dimension: {obs_dim}')
    print(f'Action dimension: {action_dim}')
    
    # Create agent
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        device=device,
    )

    # --------- Logging setup ---------
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv"))
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    # Restore agent if needed
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    
    # Set up loggers
    # train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    # eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    
    first_time = time.time()
    last_time = time.time()
    
    # Training loop
    step = 0
    done = True
    expl_metrics = dict()
    
    for i in tqdm.tqdm(
        range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), 
        smoothing=0.1, 
        dynamic_ncols=True
    ):
        if i <= FLAGS.offline_steps:
            # Offline RL
            batch_td = replay_buffer.sample()
            batch = tensordict_to_numpy(batch_td)
            update_info = agent.update(batch)
        else:
            # Online fine-tuning
            if done:
                step = 0
                ob, _ = env.reset()
            
            # Sample action
            action = agent.sample_actions(ob, temperature=1.0, deterministic=False)
            action = action[0]  # Remove batch dimension
            
            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated
            
            # Adjust reward for D4RL antmaze
            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                reward = reward - 1.0
            
            # Adjust reward for robomimic
            if is_robomimic_env(FLAGS.env_name):
                reward = reward - 1.0
            
            # Add transition to replay buffer
            transition = dict(
                observations=ob,
                actions=action,
                rewards=reward,
                terminals=float(done),
                masks=1.0 - float(terminated),
                next_observations=next_ob,
            )
            add_transition_to_buffer(replay_buffer, transition, device='cpu')
            ob = next_ob
            
            if done:
                expl_metrics = {k: np.mean(v) for k, v in flatten(info).items()}
            
            step += 1
            
            # Update agent
            if FLAGS.balanced_sampling:
                # Half-and-half sampling
                dataset_batch_td = dataset_buffer.sample()
                replay_batch_td = replay_buffer.sample(config.batch_size // 2)
                
                # Combine batches
                batch_dict = {}
                for k in dataset_batch_td.keys():
                    batch_dict[k] = torch.cat([
                        dataset_batch_td[k],
                        replay_batch_td[k]
                    ], dim=0)
                
                batch = {k: v.cpu().numpy() for k, v in batch_dict.items()}
            else:
                batch_td = replay_buffer.sample()
                batch = tensordict_to_numpy(batch_td)
            
            update_info = agent.update(batch)
        
        # Log metrics
        if i % FLAGS.log_interval == 0:
            train_metrics = {k: v for k, v in update_info.items()}
            
            # Validation metrics
            if val_dataset is not None:
                val_batch = val_dataset.sample(config.batch_size)
                val_batch = {k: torch.FloatTensor(v).to(device) for k, v in val_batch.items()}
                
                with torch.no_grad():
                    critic_loss, critic_info = agent.critic_loss(val_batch)
                    actor_loss, actor_info = agent.actor_loss(val_batch)
                    val_info = {**critic_info, **actor_info}
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            
            train_metrics['epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            
            prefix = "offline_agent" if i <= FLAGS.offline_steps else "online_agent"
            logger.log(train_metrics, prefix, step=i)
        
        # Evaluate agent
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            agent.actor.eval()
            agent.critic.eval()
            
            renders = []
            eval_metrics = {}
            
            eval_info, trajs, cur_renders = evaluate_agent(
                agent=agent,
                env=eval_env,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)
            
            eval_metrics.update(eval_info)
            
            if FLAGS.video_episodes > 0 and len(renders) > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video
            
            logger.log(eval_metrics, "eval", step=i)
            
            agent.actor.train()
            agent.critic.train()
        
        # Save agent
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)
    
    # Final save
    if FLAGS.save_interval > 0:
        save_agent(agent, FLAGS.save_dir, FLAGS.offline_steps + FLAGS.online_steps)
    
    # Close loggers
    for csv_logger in logger.csv_loggers.values():
        csv_logger.close()
    
    print(f'Training completed! Total time: {time.time() - first_time:.2f}s')

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)


if __name__ == '__main__':
    app.run(main)