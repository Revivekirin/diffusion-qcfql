import os
import glob
import json
import random
import time
import tqdm
from collections import defaultdict, deque
from typing import Dict

import numpy as np
import torch

from absl import app, flags
from ml_collections import config_flags

import wandb
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.datasets import Dataset 
from evaluation import evaluate

from agents.fql_ptr import ACFQLAgent_PTR, get_config as get_acfql_config

from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from tensordict import TensorDict

from utils.ptr_buffer import PrioritizedChunkReplayBuffer, PrioritySampler, build_traj_index, init_active_trajs, sample_traj_batch, compute_episode_scores


# ======================================================================
# Flags
# ======================================================================

FLAGS = flags.FLAGS
EPISODE_TO_INDICES = defaultdict(list)
TR_INTERVAL = 1000

DEBUG_TR = True
DEBUG_TR_EP = 0  
TR_DEBUG_LOGS = []

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'transport-mh-vim', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")
flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

flags.DEFINE_string('entity', 'sophia435256-robros', 'wandb entity')
flags.DEFINE_string('mode', 'online', 'wandb mode')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')

# PTR configuration
flags.DEFINE_float('ptr_alpha', 1.0, 'PTR priority scaling factor')
flags.DEFINE_float('ptr_eps_uniform', 0.2, 'Uniform sampling probability for PTR')
flags.DEFINE_bool('use_shaped_for_priority', True, 'Use shaped reward for priority calculation')


flags.DEFINE_enum('priority_mode', 'chunk', ['chunk', 'episode'],
                  'Priority calculation mode')

# Log config
PTR_CHUNK_LOGS = []
PTR_SNAPSHOT_LOGS = []

def log_priority_snapshot(step, priorities, N, CURRENT_STAGE:str = "offline_prefill"):
    pri = priorities[:N].detach().cpu().numpy().astype(np.float64)
    pri_clip = pri.copy()
    pri_clip[pri_clip <= 0] = 1.0
    p = pri_clip / (pri_clip.sum() + 1e-8)

    entropy = -np.sum(p * np.log(p + 1e-12))

    k_ratio = 0.05
    k = max(1, int(len(p) * k_ratio))
    idx_sorted = np.argsort(-p)
    topk_mass = p[idx_sorted[:k]].sum()

    PTR_SNAPSHOT_LOGS.append({
        "stage": CURRENT_STAGE,
        "step": int(step),
        "H": int(FLAGS.horizon_length),
        "pri_min": float(pri.min()),
        "pri_max": float(pri.max()),
        "pri_mean": float(pri.mean()),
        "pri_std": float(pri.std()),
        "entropy": float(entropy),
        "top5pct_mass": float(topk_mass),
    })

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
# Helper: batch numpy -> torch
# ======================================================================

def numpy_batch_to_torch(batch, device):
    torch_batch = {}
    for k, v in batch.items():
        arr = np.asarray(v)
        if arr.dtype == np.bool_:
            t = torch.from_numpy(arr.astype(np.float32))
        elif np.issubdtype(arr.dtype, np.integer):
            t = torch.from_numpy(arr.astype(np.int64))
        else:
            t = torch.from_numpy(arr.astype(np.float32))
        torch_batch[k] = t.to(device)
    return torch_batch

# ======================================================================
# PTR: compute chunk-level quality (FIXED)
# ======================================================================

# def compute_quality(td, CURRENT_STAGE="offline_prefill"):
#     """
#     개선된 PTR priority 계산 (음수 reward 대응):
#     - 원본 reward(rewards_ptr) 기반으로 quality 계산
#     - 음수 값을 양수로 변환하여 priority 할당
#     - 상대적 순위 기반 접근
#     """
#     # 1) 항상 원본 reward 사용 (shaped reward는 음수가 많아 priority 계산에 부적합)
#     rewards_ptr = td["rewards_ptr"].squeeze(-1).cpu().numpy()  # [H]
#     valid = td["valid"].cpu().numpy().astype(bool)
#     r_valid = rewards_ptr[valid]
    
#     if r_valid.size == 0:
#         return 1.0  # 기본값

#     # 2) 통계량 계산
#     traj_return = float(r_valid.sum())
#     avg_r = float(r_valid.mean())
    
#     # Upper quartile mean (UQM)
#     k = max(1, int(0.25 * r_valid.size))
#     topk = np.partition(r_valid, -k)[-k:]
#     uqm = float(topk.mean())
    
#     min_r = float(r_valid.min())
#     max_r = float(r_valid.max())

#     # 3) Priority 계산 전략 (Robomimic 최적화)
#     if is_robomimic_env(FLAGS.env_name):
#         # Strategy 1: 성공 여부 기반 (이진 분류)
#         if traj_return > 0.5:  # 성공
#             q = 100.0 + traj_return * 10.0  # 높은 base + 추가 보상
#         elif traj_return > 0.0:  # 부분 성공
#             q = 50.0 + traj_return * 10.0
#         elif avg_r > -0.5:  # 실패했지만 괜찮은 trajectory
#             q = 10.0 + (avg_r + 1.0) * 10.0  # -1 penalty 보정
#         else:  # 매우 나쁜 trajectory
#             q = 1.0 + max(0, avg_r + 1.0) * 5.0
#     else:
#         # 일반 환경: 논문의 공식 사용
#         q = traj_return + 0.5 * avg_r + 0.5 * uqm
#         # 음수를 양수로 변환 (offset 추가)
#         if q < 0:
#             q = np.exp(q / 10.0)  # 지수 변환으로 양수화
#         q = max(q, 0.1)  # 최소값 보장
    
#     # 4) 스케일링
#     q = q * FLAGS.ptr_alpha
#     q = max(q, 0.01)  # 절대 최소값

#     PTR_CHUNK_LOGS.append({
#         "stage": CURRENT_STAGE,
#         "traj_return": traj_return,
#         "avg_r": avg_r,
#         "uqm": uqm,
#         "min_r": min_r,
#         "max_r": max_r,
#         "quality": q,
#     })

#     return q

def compute_quality(td, CURRENT_STAGE="offline_prefill"):
    """
    Chunk-level priority 계산:
    - 해당 chunk의 H-step만 보고 계산
    - 빠르고 세밀한 우선순위
    """
    rewards_ptr = td["rewards_ptr"].squeeze(-1).cpu().numpy()  # [H]
    valid = td["valid"].cpu().numpy().astype(bool)
    r_valid = rewards_ptr[valid]
    
    if r_valid.size == 0:
        return 1.0
    
    # 통계량 계산
    traj_return = float(r_valid.sum())
    avg_r = float(r_valid.mean())
    
    k = max(1, int(0.25 * r_valid.size))
    topk = np.partition(r_valid, -k)[-k:]
    uqm = float(topk.mean())
    
    min_r = float(r_valid.min())
    max_r = float(r_valid.max())
    
    # Priority 계산
    if is_robomimic_env(FLAGS.env_name):
        base_score = traj_return + 0.5 * avg_r + 0.5 * uqm
        
        if traj_return > 0.5:
            q = 100.0 + np.clip(base_score * 10, 0, 100)
        elif traj_return > -0.5 * r_valid.size:
            normalized = (traj_return + 0.5 * r_valid.size) / (0.5 * r_valid.size)
            q = 10.0 + normalized * 90.0
        else:
            q = 1.0 + max(0, (traj_return + r_valid.size) / r_valid.size) * 9.0
        
        if uqm > avg_r + 0.1:
            q *= 1.2
    else:
        q = traj_return + 0.5 * avg_r + 0.5 * uqm
        if q < 0:
            q = np.exp(q / 10.0)
        q = max(q, 0.1)
    
    q = q * FLAGS.ptr_alpha
    q = max(q, 0.1)
    
    PTR_CHUNK_LOGS.append({
        "stage": CURRENT_STAGE,
        "mode": "chunk",
        "traj_return": traj_return,
        "avg_r": avg_r,
        "uqm": uqm,
        "min_r": min_r,
        "max_r": max_r,
        "quality": q,
    })
    
    return q


def compute_episode_quality(ep_id, episode_to_indices, storage):
    """
    Episode-level priority 계산:
    - 전체 episode의 모든 chunk를 합쳐서 계산
    - 논문의 정확한 구현, 더 안정적
    """
    idx_list = episode_to_indices.get(ep_id, [])
    if len(idx_list) == 0:
        return 1.0
    
    # 전체 episode의 모든 reward 수집
    all_rewards = []
    for idx in idx_list:
        td = storage[idx]
        rewards = td["rewards_ptr"].squeeze(-1).cpu().numpy()
        valid = td["valid"].cpu().numpy().astype(bool)
        all_rewards.extend(rewards[valid].tolist())
    
    if len(all_rewards) == 0:
        return 1.0
    
    r = np.array(all_rewards, dtype=np.float32)
    
    # 통계량 계산
    traj_return = float(r.sum())
    avg_r = float(r.mean())
    
    k = max(1, int(0.25 * len(r)))
    uqm = float(np.partition(r, -k)[-k:].mean())
    
    min_r = float(r.min())
    max_r = float(r.max())
    
    # Priority 계산
    if is_robomimic_env(FLAGS.env_name):
        base_score = traj_return + 0.5 * avg_r + 0.5 * uqm
        
        if traj_return > 0.5:
            q = 100.0 + np.clip(base_score * 10, 0, 100)
        elif traj_return > -0.5 * len(r):
            normalized = (traj_return + 0.5 * len(r)) / (0.5 * len(r))
            q = 10.0 + normalized * 90.0
        else:
            q = 1.0 + max(0, (traj_return + len(r)) / len(r)) * 9.0
        
        if uqm > avg_r + 0.1:
            q *= 1.2
    else:
        q = traj_return + 0.5 * avg_r + 0.5 * uqm
        if q < 0:
            q = np.exp(q / 10.0)
        q = max(q, 0.1)
    
    q = q * FLAGS.ptr_alpha
    q = max(q, 0.1)
    
    PTR_CHUNK_LOGS.append({
        "stage": "episode_init",
        "mode": "episode",
        "ep_id": int(ep_id),
        "num_chunks": len(idx_list),
        "traj_return": traj_return,
        "avg_r": avg_r,
        "uqm": uqm,
        "min_r": min_r,
        "max_r": max_r,
        "quality": q,
    })
    
    return q


def init_priorities(episode_to_indices, storage, priorities, mode='chunk'):
    """
    Priority 초기화 - chunk 또는 episode mode 선택
    
    Args:
        mode: 'chunk' 또는 'episode'
    """
    if mode == 'chunk':
        # Chunk-level: 각 chunk 개별 계산
        print(f"[PTR] Initializing chunk-level priorities...")
        for idx in range(len(storage)):
            td = storage[idx]
            q = compute_quality(td, CURRENT_STAGE="offline_prefill")
            priorities[idx] = q
            
    elif mode == 'episode':
        # Episode-level: episode별 계산 후 모든 chunk에 동일하게 할당
        print(f"[PTR] Initializing episode-level priorities...")
        for ep_id, idx_list in episode_to_indices.items():
            if len(idx_list) == 0:
                continue
            
            ep_quality = compute_episode_quality(ep_id, episode_to_indices, storage)
            
            # 해당 episode의 모든 chunk에 동일한 priority 할당
            for idx in idx_list:
                priorities[idx] = ep_quality
    
    else:
        raise ValueError(f"Unknown priority mode: {mode}")
    
    # 통계 출력
    nonzero = priorities[priorities > 0]
    print(f"[PTR] Priority initialization ({mode} mode) complete:")
    print(f"  Total chunks: {len(storage)}")
    print(f"  Non-zero priorities: {len(nonzero)}")
    print(f"  min={nonzero.min():.2f}, max={nonzero.max():.2f}, "
          f"mean={nonzero.mean():.2f}, median={np.median(nonzero.cpu().numpy()):.2f}")
    
    # 분포 시각화
    pri_vals = priorities[:len(storage)].cpu().numpy()
    pri_nonzero = pri_vals[pri_vals > 0]
    print(f"\n[PTR] Priority distribution:")
    print(f"  0-10:   {np.sum((pri_nonzero > 0) & (pri_nonzero < 10))} chunks "
          f"({100*np.mean((pri_nonzero > 0) & (pri_nonzero < 10)):.1f}%)")
    print(f"  10-50:  {np.sum((pri_nonzero >= 10) & (pri_nonzero < 50))} chunks "
          f"({100*np.mean((pri_nonzero >= 10) & (pri_nonzero < 50)):.1f}%)")
    print(f"  50-100: {np.sum((pri_nonzero >= 50) & (pri_nonzero < 100))} chunks "
          f"({100*np.mean((pri_nonzero >= 50) & (pri_nonzero < 100)):.1f}%)")
    print(f"  100+:   {np.sum(pri_nonzero >= 100)} chunks "
          f"({100*np.mean(pri_nonzero >= 100):.1f}%)")
    
    return priorities

# ======================================================================
# PTR: Weighted Backward Update based on TR (FIXED)
# ======================================================================

def get_chunks_by_episode(ep_id: int, storage, episode_to_indices):
    """ep_id에 해당하는 모든 chunk들을 (idx, TensorDict) 리스트로 반환."""
    idx_list = episode_to_indices.get(ep_id, [])
    pairs = []
    for idx in idx_list:
        if idx < len(storage):  
            td = storage[idx]
            pairs.append((idx, td))
    return pairs


def convert_td_to_batch(td: TensorDict) -> Dict[str, torch.Tensor]:
    """TensorDict를 ACFQLAgent.update_critic용 batch dict로 변환."""
    rewards = td["rewards"]
    if rewards.ndim == 2 and rewards.shape[-1] == 1:
        rewards = rewards.squeeze(-1)

    masks = td["masks"]
    if masks.ndim == 2 and masks.shape[-1] == 1:
        masks = masks.squeeze(-1)

    terminals = td["terminals"]
    if terminals.ndim == 2 and terminals.shape[-1] == 1:
        terminals = terminals.squeeze(-1)

    batch = {
        "observations": td["observations"].unsqueeze(0),
        "actions": td["actions"].unsqueeze(0),
        "rewards": rewards.unsqueeze(0),
        "terminals": terminals.unsqueeze(0),
        "masks": masks.unsqueeze(0),
        "next_observations": td["next_observations"].unsqueeze(0),
        "valid": td["valid"].unsqueeze(0),
    }

    return batch

def update_with_tr_lite(agent, 
                        replay_buffer, 
                        sampler, 
                        storage, 
                        gamma, 
                        H, 
                        logger, 
                        log_step):
    
    td_batch = replay_buffer.sample(1)
    ep_id = int(td_batch["episode_id"][0].item())

    idx_td_list = get_chunks_by_episode(ep_id, storage, EPISODE_TO_INDICES)
    if len(idx_td_list) == 0:
        return

    idx_td_list = sorted(
        idx_td_list,
        key=lambda pair: int(pair[1]["chunk_index"].item())
    )

    # Backward return 계산
    G = 0.0
    returns = []
    for idx, c in reversed(idx_td_list):
        r = c["rewards"]
        if r.ndim == 2 and r.shape[-1] == 1:
            r = r.squeeze(-1)
        if "valid" in c.keys():
            v = c["valid"].to(r.dtype)
        else:
            v = torch.ones_like(r)

        R_k = float((r * v).sum().item())
        G = R_k + (gamma ** H) * G
        returns.append(G)

    returns.reverse()

    # Critic 업데이트 및 priority 재계산
    if FLAGS.priority_mode == 'episode':
        # Episode mode: 한 번만 계산하여 모든 chunk에 동일 할당
        ep_quality = compute_episode_quality(ep_id, EPISODE_TO_INDICES, storage)
        all_indices = [idx for idx, _ in idx_td_list]
        all_priorities = [ep_quality] * len(all_indices)
        
        for (idx, chunk_td), G_k in zip(idx_td_list, returns):
            batch = convert_td_to_batch(chunk_td)
            ptr_critic_info = agent.update_critic(batch, G_k)
            logger.log(ptr_critic_info, "offline_agent", step=log_step)
    
    else:  # chunk mode
        # Chunk mode: 각 chunk 개별 계산
        all_indices = []
        all_priorities = []
        
        for (idx, chunk_td), G_k in zip(idx_td_list, returns):
            batch = convert_td_to_batch(chunk_td)
            ptr_critic_info = agent.update_critic(batch, G_k)
            logger.log(ptr_critic_info, "offline_agent", step=log_step)
            
            all_indices.append(idx)
            q_new = compute_quality(chunk_td, CURRENT_STAGE="tr_update")
            all_priorities.append(q_new)

    # Priority 업데이트
    if len(all_indices) > 0:
        indices_tensor = torch.tensor(all_indices, dtype=torch.long)
        pri_tensor = torch.tensor(all_priorities, dtype=torch.float32)
        sampler.update_priority(indices_tensor, pri_tensor)

# ======================================================================
# Main
# ======================================================================

def main(_):
    # Wandb & exp name
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name,
                      entity=FLAGS.entity, mode=FLAGS.mode)

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

    # Config
    config = dict(get_acfql_config())
    config["horizon_length"] = FLAGS.horizon_length

    # Env & dataset
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # Seed
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_step = 0
    discount = FLAGS.discount

    # Dataset 처리
    def process_train_dataset(ds):
        ds = Dataset.create(**ds)

        # dataset_proportion
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )

        # PTR용 원본 보상 저장
        orig_rewards = ds["rewards"].copy()

        # RL 학습용 shaped reward
        shaped_rewards = ds["rewards"].copy()

        if is_robomimic_env(FLAGS.env_name):
            shaped_rewards = shaped_rewards - 1.0

        if FLAGS.sparse:
            shaped_rewards = (shaped_rewards != 0.0) * -1.0

        ds = ds.copy(add_or_replace={
            "rewards": shaped_rewards,
            "rewards_ptr": orig_rewards
        })

        return ds

    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())

    # Agent 생성
    agent = ACFQLAgent_PTR.create(
        seed=FLAGS.seed,
        ex_observations=np.asarray(example_batch['observations']),
        ex_actions=np.asarray(example_batch['actions']),
        config=config,
        device=str(device),
    )
    
    # Logging setup
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

    # ==================================================================
    # Offline RL ReplayBuffer for PTR
    # ==================================================================

    H = FLAGS.horizon_length
    buffer_size = FLAGS.buffer_size
    capacity = FLAGS.buffer_size
    priorities = torch.ones(capacity)

    sampler = PrioritySampler(priorities, eps_uniform=FLAGS.ptr_eps_uniform)
    storage = LazyTensorStorage(capacity)
    batch_size_offline = config["batch_size"]

    storage_episode_ids = torch.full((capacity,), -1, dtype=torch.long)
    storage_chunk_indices = torch.full((capacity,), -1, dtype=torch.long)

    def deregister_slot(slot_idx: int):
        old_ep = int(storage_episode_ids[slot_idx].item())
        if old_ep >= 0:
            ep_list = EPISODE_TO_INDICES.get(old_ep, [])
            if slot_idx in ep_list:
                ep_list.remove(slot_idx)
                if len(ep_list) == 0:
                    EPISODE_TO_INDICES.pop(old_ep, None)
        storage_episode_ids[slot_idx] = -1
        storage_chunk_indices[slot_idx] = -1

    def register_slot(slot_idx: int, ep_id: int, chunk_idx: int):
        EPISODE_TO_INDICES[ep_id].append(slot_idx)
        storage_episode_ids[slot_idx] = ep_id
        storage_chunk_indices[slot_idx] = chunk_idx

    replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=sampler,
        batch_size=batch_size_offline,
    )

    # Offline dataset으로 replay 채우기
    CURRENT_STAGE = "offline_prefill"
    approx_num_chunks = train_dataset.size // H
    max_init_offline = min(buffer_size, approx_num_chunks)

    cursor = 0
    num_added = 0
    print(f"[PTR] Filling offline buffer: target={max_init_offline}")

    dataset_indices = np.arange(0, train_dataset.size - H, H)  
    np.random.shuffle(dataset_indices)  
    
    for start_idx in dataset_indices[:max_init_offline]:
        end_idx = min(start_idx + H, train_dataset.size)
        actual_H = end_idx - start_idx
        
        if actual_H < H:
            continue 

        # Dataset에서 직접 추출
        obs_seq = train_dataset["observations"][start_idx:end_idx]
        actions_seq = train_dataset["actions"][start_idx:end_idx]
        rewards_seq = train_dataset["rewards"][start_idx:end_idx]
        rewards_seq_ptr = train_dataset["rewards_ptr"][start_idx:end_idx]
        terminals_seq = train_dataset["terminals"][start_idx:end_idx]
        masks_seq = train_dataset["masks"][start_idx:end_idx]
        next_obs_seq = train_dataset["next_observations"][start_idx:end_idx]

        # Episode 정보
        ep_idx = np.searchsorted(train_dataset.initial_locs, start_idx, side="right") - 1
        ep_idx = np.clip(ep_idx, 0, len(train_dataset.initial_locs) - 1)
        current_episode_id = int(ep_idx)
        start_step_in_ep = int(start_idx - train_dataset.initial_locs[ep_idx])
        current_chunk_index = start_step_in_ep // H

        # 첫 obs
        obs0 = np.asarray(obs_seq[0], dtype=np.float32)
        if obs0.ndim > 1:
            obs0 = obs0.reshape(-1)

        # Valid mask
        valid_seq = np.ones(H, dtype=np.float32)
        for t_idx in range(1, H):
            if terminals_seq[t_idx - 1] > 0.5:
                valid_seq[t_idx:] = 0.0
                break

        # TensorDict 생성
        td = TensorDict(
            {
                "observations": torch.from_numpy(obs0).float(),
                "actions": torch.from_numpy(np.asarray(actions_seq)).float(),
                "rewards": torch.from_numpy(np.asarray(rewards_seq)).unsqueeze(-1).float(),
                "rewards_ptr": torch.from_numpy(np.asarray(rewards_seq_ptr)).unsqueeze(-1).float(),
                "terminals": torch.from_numpy(np.asarray(terminals_seq)).unsqueeze(-1).float(),
                "masks": torch.from_numpy(np.asarray(masks_seq)).unsqueeze(-1).float(),
                "next_observations": torch.from_numpy(np.asarray(next_obs_seq)).float(),
                "valid": torch.from_numpy(valid_seq).float(),
                "episode_id": torch.tensor(current_episode_id, dtype=torch.int32),
                "chunk_start_step": torch.tensor(start_step_in_ep, dtype=torch.int32),
                "chunk_index": torch.tensor(current_chunk_index, dtype=torch.int32),
            },
            batch_size=[],
        )

        replay_buffer.add(td)
        register_slot(cursor, current_episode_id, current_chunk_index)

        cursor = (cursor + 1) % capacity
        num_added += 1
        
        if num_added >= max_init_offline:
            break

    # keep track of the next index whose priority should be updated when new
    # chunks are appended (online stage continues from the offline cursor)
    priority_cursor = cursor

    print(f"\n[PTR] Offline buffer filled: {num_added} chunks")
    print(f"[PTR] Unique episodes: {len(EPISODE_TO_INDICES)}")
    print(f"[PTR] Avg chunks per episode: {np.mean([len(v) for v in EPISODE_TO_INDICES.values()]):.2f}")

    for ep_id in EPISODE_TO_INDICES:
        EPISODE_TO_INDICES[ep_id] = sorted(
            EPISODE_TO_INDICES[ep_id],
            key=lambda i: int(storage[i]["chunk_index"].item())
        )
    
    priorities = init_priorities(
        episode_to_indices=EPISODE_TO_INDICES,
        storage=storage,
        priorities=priorities,
        mode=FLAGS.priority_mode  # chunk or episode
    )
    
    log_priority_snapshot(0, priorities, len(replay_buffer), CURRENT_STAGE="after_init")

    existing_eps = list(EPISODE_TO_INDICES.keys())
    max_existing_ep = max(existing_eps) if len(existing_eps) > 0 else -1
    next_episode_id = max_existing_ep + 1

    # ==================================================================
    # TR trajectory index 정렬 & active set 초기화
    # ==================================================================
    
    build_traj_index(storage, EPISODE_TO_INDICES)
    
    active_trajs, available_eps, ep_ids, ep_scores = init_active_trajs(
            EPISODE_TO_INDICES, priorities, num_active=batch_size_offline,
        )

    # ==================================================================
    # Offline RL Training
    # ==================================================================
    
    offline_init_time = time.time()
    print(f"\n[Offline RL] Starting {FLAGS.offline_steps} steps")
    print(f"[PTR] Using {FLAGS.priority_mode} mode for priority calculation")
    
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1

        # TR-style backward sampling
        batch_td, active_trajs, available_eps = sample_traj_batch(
            storage=storage,
            episode_to_indices=EPISODE_TO_INDICES,
            ep_ids=ep_ids,
            ep_scores=ep_scores,
            active_trajs=active_trajs,
            available_eps=available_eps,
            batch_size=config["batch_size"],
        )

        rewards = batch_td["rewards"]
        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)

        masks = batch_td["masks"]
        if masks.ndim == 3 and masks.shape[-1] == 1:
            masks = masks.squeeze(-1)

        terminals = batch_td["terminals"]
        if terminals.ndim == 3 and terminals.shape[-1] == 1:
            terminals = terminals.squeeze(-1)

        batch = {
            "observations": batch_td["observations"],
            "actions": batch_td["actions"],
            "rewards": rewards,
            "terminals": terminals,
            "masks": masks,
            "next_observations": batch_td["next_observations"],
            "valid": batch_td["valid"],
        }

        offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

        if i % TR_INTERVAL == 0:
            update_with_tr_lite(
                agent=agent,
                replay_buffer=replay_buffer,
                sampler=sampler,
                storage=storage,
                gamma=discount,
                H=H,
                logger=logger,
                log_step=log_step,
            )
        # Evaluation
        if (
            i == FLAGS.offline_steps - 1
            or (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0)
        ):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            print(f"\n[Eval @ step {i}] {eval_info}")
            logger.log(eval_info, "eval", step=log_step)
        
        if i % 5000 == 0:
            N = len(replay_buffer)
            log_priority_snapshot(i, priorities, N, CURRENT_STAGE=f"offline_step_{i}")
    
    # ==================================================================
    # Online RL
    # ==================================================================
    data = defaultdict(list)
    online_init_time = time.time()

    H = FLAGS.horizon_length
    action_dim = example_batch["actions"].shape[-1]

    ob, _ = env.reset()         
    action_queue = []           
    trans_window = deque(maxlen=H)  

    update_info = {}

    priority_cursor = priority_cursor % capacity
    current_online_episode_id = next_episode_id
    next_episode_id += 1
    online_chunk_index = 0
    online_chunk_start_step = 0
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        CURRENT_STAGE="online_add"

        # -----------------------------------------------------------
        # 액션 샘플링: queue 비면 새 chunk 뽑기
        # -----------------------------------------------------------
        if len(action_queue) == 0:
            obs_t = torch.from_numpy(np.asarray(ob, dtype=np.float32)).float().to(device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)  # [1, obs_dim]

            with torch.no_grad():
                action_chunk_t = agent.sample_actions(obs_t)  # [1, H*A] 또는 [1, A]
            action_chunk = action_chunk_t.cpu().numpy().reshape(-1, action_dim)  # [H, A] or [1, A]

            for a in action_chunk:
                action_queue.append(a)

        action = action_queue.pop(0)

        # -----------------------------------------------------------
        # env step
        # -----------------------------------------------------------
        next_ob, int_reward_raw, terminated, truncated, info = env.step(
            np.clip(action, -1.0 + 1e-5, 1.0 - 1e-5)
        )
        done = terminated or truncated

        # save_all_online_states
        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))

        # env metrics logging
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        if len(env_info) > 0:
            logger.log(env_info, "env", step=log_step)

        int_reward_rl = int_reward_raw
        # reward shaping
        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            int_reward_rl = int_reward_rl - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            int_reward_rl = int_reward_rl - 1.0

        if FLAGS.sparse:
            assert int_reward_rl <= 0.0
            int_reward_rl = (int_reward_rl != 0.0) * -1.0

        # -----------------------------------------------------------
        # 1-step transition을 window에 쌓기
        # -----------------------------------------------------------
        trans_window.append(
            dict(
                observations=np.asarray(ob, dtype=np.float32),
                actions=np.asarray(action, dtype=np.float32),
                rewards=float(int_reward_rl),
                rewards_ptr = float(int_reward_raw),
                terminals=float(done),
                masks=float(1.0 - float(terminated)),
                next_observations=np.asarray(next_ob, dtype=np.float32),
            )
        )

        # -----------------------------------------------------------
        # window 길이가 H가 되면 H-step chunk로 만들어 ReplayBuffer에 추가
        # -----------------------------------------------------------
        if len(trans_window) == H:
            obs0 = np.asarray(trans_window[0]["observations"], dtype=np.float32)  # [obs_dim]
            if obs0.shape == ():  # obs_dim=1일 때 안전 보정
                obs0 = obs0.reshape(1,)

            actions_seq = np.stack([t["actions"] for t in trans_window], axis=0)              # [H, act_dim]
            rewards_seq_rl = np.stack([t["rewards"] for t in trans_window], axis=0)              # [H]
            rewards_seq_ptr = np.stack([t["rewards_ptr"] for t in trans_window], axis=0)
            terminals_seq = np.stack([t["terminals"] for t in trans_window], axis=0)          # [H]
            masks_seq = np.stack([t["masks"] for t in trans_window], axis=0)                  # [H]
            next_obs_seq = np.stack([t["next_observations"] for t in trans_window], axis=0)   # [H, obs_dim]

            # valid 마스크: 첫 terminal 이후는 0
            valid_seq = np.ones(H, dtype=np.float32)
            for t_idx in range(1, H):
                if terminals_seq[t_idx - 1] > 0.5:
                    valid_seq[t_idx:] = 0.0
                    break

            chunk_start_step = online_chunk_start_step
            chunk_index_value = online_chunk_index

            td = TensorDict(
                {
                    "observations": torch.from_numpy(obs0).float(),
                    "actions": torch.from_numpy(actions_seq).float(),
                    "rewards": torch.from_numpy(rewards_seq_rl).unsqueeze(-1).float(),
                    "rewards_ptr": torch.from_numpy(rewards_seq_ptr).unsqueeze(-1).float(),
                    "terminals": torch.from_numpy(terminals_seq).unsqueeze(-1).float(),
                    "masks": torch.from_numpy(masks_seq).unsqueeze(-1).float(),
                    "next_observations": torch.from_numpy(next_obs_seq).float(),
                    "valid": torch.from_numpy(valid_seq).float(),
                    "episode_id": torch.tensor(current_online_episode_id, dtype=torch.int32),
                    "chunk_start_step": torch.tensor(int(chunk_start_step), dtype=torch.int32),
                    "chunk_index": torch.tensor(int(chunk_index_value), dtype=torch.int32),
                },
                batch_size=[],
            )

            replay_buffer.add(td)
            deregister_slot(priority_cursor)
            register_slot(priority_cursor, current_online_episode_id, chunk_index_value)
            # debug quality
            q = compute_quality(td, CURRENT_STAGE=CURRENT_STAGE)
            priorities[priority_cursor] = q

            if priority_cursor % 1000 == 0:
                N = len(replay_buffer)
                log_priority_snapshot(i, priorities, N, CURRENT_STAGE=CURRENT_STAGE)
            priority_cursor = (priority_cursor + 1) % capacity
            online_chunk_index += 1
            online_chunk_start_step += 1

        if done:
            ob, _ = env.reset()
            action_queue = []
            trans_window.clear()
            current_online_episode_id = next_episode_id
            next_episode_id += 1
            online_chunk_index = 0
            online_chunk_start_step = 0
        else:
            ob = next_ob

        # ---------------------------------------------------------------
        # Online 학습
        # ---------------------------------------------------------------
        if i >= FLAGS.start_training and len(replay_buffer) >= config["batch_size"]:
            batch_td = replay_buffer.sample(batch_size=config["batch_size"])
            # batch_td field shapes:
            #  observations      : [B, obs_dim]
            #  actions           : [B, H, act_dim]
            #  rewards           : [B, H, 1]
            #  terminals         : [B, H, 1]
            #  masks             : [B, H, 1]
            #  next_observations : [B, H, obs_dim]
            #  valid             : [B, H]

            rewards = batch_td["rewards"]
            if rewards.ndim == 3 and rewards.shape[-1] == 1:
                rewards = rewards.squeeze(-1)

            masks = batch_td["masks"]
            if masks.ndim == 3 and masks.shape[-1] == 1:
                masks = masks.squeeze(-1)

            terminals = batch_td["terminals"]
            if terminals.ndim == 3 and terminals.shape[-1] == 1:
                terminals = terminals.squeeze(-1)

            batch = {
                "observations": batch_td["observations"],             # [B, obs_dim]
                "actions": batch_td["actions"],                       # [B, H, act_dim]
                "rewards": rewards,                                   # [B, H]
                "terminals": terminals,                               # [B, H]
                "masks": masks,                                       # [B, H]
                "next_observations": batch_td["next_observations"],   # [B, H, obs_dim]
                "valid": batch_td["valid"],                           # [B, H]
            }

            online_info = agent.update(batch)
            update_info["online_agent"] = online_info


        if i % FLAGS.log_interval == 0:
            for key, info_ in update_info.items():
                logger.log(info_, key, step=log_step)
            update_info = {}

        # eval
        if (
            i == FLAGS.online_steps - 1
            or (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0)
        ):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            # TODO: PyTorch 버전 save_agent 구현
            # save_agent(agent, FLAGS.save_dir, log_step)
            pass
        
        # debug log priority
        if i % 1000 == 0:
            N = len(replay_buffer)
            log_priority_snapshot(i, priorities, N, CURRENT_STAGE=CURRENT_STAGE)

    end_time = time.time()

    # close csv
    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    # save_all_online_states
    if FLAGS.save_all_online_states:
        c_data = {
            "steps": np.array(data["steps"]),
            "qpos": np.stack(data["qpos"], axis=0),
            "qvel": np.stack(data["qvel"], axis=0),
            "obs": np.stack(data["obs"], axis=0),
            "offline_time": online_init_time - offline_init_time,
            "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

    
    # 로그 저장
    import pandas as pd

    if len(PTR_CHUNK_LOGS) > 0:
        df_chunks = pd.DataFrame(PTR_CHUNK_LOGS)
        csv_path_chunks = os.path.join(FLAGS.save_dir, "ptr_chunks.csv")
        df_chunks.to_csv(csv_path_chunks, index=False)
        print(f"[PTR] Chunk logs saved to {csv_path_chunks}")

    if len(PTR_SNAPSHOT_LOGS) > 0:
        df_snap = pd.DataFrame(PTR_SNAPSHOT_LOGS)
        csv_path_snap = os.path.join(FLAGS.save_dir, "ptr_snapshots.csv")
        df_snap.to_csv(csv_path_snap, index=False)
        print(f"[PTR] Snapshot logs saved to {csv_path_snap}")

    # === TR Debug log ===
    if len(TR_DEBUG_LOGS) > 0:
        df_tr = pd.DataFrame(TR_DEBUG_LOGS)
        csv_path_tr = os.path.join(FLAGS.save_dir, "tr_debug.csv")
        df_tr.to_csv(csv_path_tr, index=False)
        print(f"[TR] Debug logs saved to {csv_path_tr}")



if __name__ == '__main__':
    app.run(main)
