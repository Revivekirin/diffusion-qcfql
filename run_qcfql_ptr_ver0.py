import os
import glob
import json
import random
import time
import tqdm
from collections import defaultdict, deque

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

from agents.fql import ACFQLAgent, get_config as get_acfql_config

from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from tensordict import TensorDict

from utils.ptr_buffer import PrioritizedChunkReplayBuffer, PrioritySampler


# ======================================================================
# Flags
# ======================================================================

FLAGS = flags.FLAGS

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

# Log config
PTR_CHUNK_LOGS = []
PTR_SNAPSHOT_LOGS = []

def log_priority_snapshot(step, priorities, N, CURRNET_STAGE:str = "offline_prefill"):
    # N = len(storage)와 같게 맞추거나 min(N, len(priorities))
    pri = priorities[:N].detach().cpu().numpy().astype(np.float64)

    # 음수/0 없게 한 번 정리 (PrioritySampler에서도 했던 것과 동일하게)
    pri_clip = pri.copy()
    pri_clip[pri_clip <= 0] = 1.0
    p = pri_clip / (pri_clip.sum() + 1e-8)

    # entropy
    entropy = -np.sum(p * np.log(p + 1e-12))

    k_ratio = 0.05
    k = max(1, int(len(p) * k_ratio))
    idx_sorted = np.argsort(-p)
    topk_mass = p[idx_sorted[:k]].sum()

    PTR_SNAPSHOT_LOGS.append({
        "stage": CURRNET_STAGE,
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
            # rewards, masks, terminals, valid, obs, actions 모두 float32로 통일
            t = torch.from_numpy(arr.astype(np.float32))

        torch_batch[k] = t.to(device)

    return torch_batch

# ======================================================================
# PTR: compute chunk-level quality
# ======================================================================
# Discrete compute
# def compute_quality(td, CURRENT_STAGE: str = "offline_prefill") -> float:
#     """
#     이산형 priority (성공: 1.0, 실패: 0.1) + 디버그 + 로그 기록

#     td["rewards"]: [H, 1] or [H]
#     td["valid"]  : [H]
#     """
#     rewards = td["rewards"].squeeze(-1)        # [H]
#     valid = td["valid"].to(rewards.dtype)      # [H]

#     traj_return = (rewards * valid).sum().item()
#     H = rewards.shape[0]

#     # ----- Tuning 구간 -----
#     # 예: traj_return > 0.0 을 성공으로 간주
#     success_flag = traj_return > 0.0

#     # priority 값을 이산으로 부여
#     if success_flag:
#         q = 1.0
#     else:
#         q = 0.1  # 실패도 완전 무시하진 않음

#     # 디버그 프린트 (원하면 간헐적으로만 찍도록 조건 추가 가능)
#     print(f"[DEBUG][compute_quality_discrete] stage={CURRENT_STAGE}")
#     print(f"  len(rewards): {H}")
#     print(f"  traj_return: {traj_return:.6f}")
#     print(f"  success_flag: {success_flag}")
#     print(f"  quality (q): {q}")

#     # --- 로그 저장 ---
#     PTR_CHUNK_LOGS.append({
#         "stage": CURRENT_STAGE,
#         "mode": "discrete",
#         "H": int(H),
#         "traj_return": float(traj_return),
#         "quality": float(q),
#         "success_flag": bool(success_flag),
#     })

#     return float(q)

def compute_quality(td, CURRENT_STAGE="offline_prefill"):
    #rewards = td["rewards"].squeeze(-1).cpu().numpy()
    rewards = td["rewards_ptr"].squeeze(-1).cpu().numpy()
    valid = td["valid"].cpu().numpy().astype(bool)

    r_valid = rewards[valid]
    if r_valid.size == 0:
        return 1e-3

    # 1) Return
    ret = float(r_valid.sum())

    # 2) Avg reward
    avg_r = float(r_valid.mean())

    # 3) UQM (상위 25% 평균)
    k = max(1, int(0.25 * r_valid.size))
    topk = np.partition(r_valid, -k)[-k:]
    uqm = float(topk.mean())

    # 예: sparse 환경에서는 avg_r, uqm, min_r를 조합해서 priority
    min_r = float(r_valid.min())

    # 간단히: ret + λ1*avg_r + λ2*uqm + λ3*min_r 같은 형태도 가능
    q = ret + 0.5 * avg_r + 0.5 * uqm + 0.1 * min_r

    PTR_CHUNK_LOGS.append({
        "stage": CURRENT_STAGE,
        "mode": "quality_combo",
        "traj_return": ret,
        "avg_r": avg_r,
        "uqm": uqm,
        "min_r": min_r,
        "quality": q,
    })
    return float(max(q, 1e-6))


    
# def compute_quality(td, alpha: float = 1.0, CURRENT_STAGE: str = "offline_prefill") -> float:
#     """
#     chunk 하나(td)에 대해 quality 스칼라를 계산하고,
#     PTR_CHUNK_LOGS에 로깅까지 담당하는 함수.

#     td["rewards"]: [H, 1] or [H]
#     td["valid"]  : [H]
#     """
#     rewards = td["rewards"].squeeze(-1)      # [H]
#     valid = td["valid"].to(rewards.dtype)    # [H]

#     traj_return = (rewards * valid).sum().item()

#     # (선택) binary success flag - purely for 분석용
#     success_flag = traj_return > 0.5

#     import math
#     q = math.log1p(math.exp(alpha * traj_return))

#     H = rewards.shape[0]
#     PTR_CHUNK_LOGS.append({
#         "stage": CURRENT_STAGE,
#         "H": int(H),
#         "traj_return": float(traj_return),
#         "quality": float(q),
#         "success_flag": bool(success_flag),
#     })

#     return float(q)



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

    # --------- Config (agent) ---------
    # 기존처럼 FLAGS.agent를 쓸 수도 있고,
    # 그냥 PyTorch용 get_acfql_config()로 dict를 가져와도 된다.
    # 여기서는 config_flags로 읽어온 뒤, 거기에 horizon_length만 덮어쓴다고 가정.
    # (원하면 get_acfql_config()를 직접 호출해도 무방)
    config = dict(get_acfql_config())
    config["horizon_length"] = FLAGS.horizon_length

    # --------- Env & dataset (no OGBench) ---------
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # --------- Seed ---------
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_step = 0
    discount = FLAGS.discount

    # --------- Dataset 처리 함수 ---------
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

        ds_dict = {k: v for k, v in ds.items()}
        ds_dict["rewards_ptr"] = ds_dict["rewards"].copy() 

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

    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())

    # --------- Agent 생성 (PyTorch) ---------
    agent = ACFQLAgent.create(
        seed=FLAGS.seed,
        ex_observations=np.asarray(example_batch['observations']),
        ex_actions=np.asarray(example_batch['actions']),
        config=config,
        device=str(device),
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
    # ==================================================================
    # Offline RL ReplayBuffer for PTR
    # ==================================================================

    H = FLAGS.horizon_length
    buffer_size = FLAGS.buffer_size   
    capacity = FLAGS.buffer_size
    priorities = torch.ones(capacity)

    sampler = PrioritySampler(priorities, eps_uniform=0.2)
    storage = LazyTensorStorage(capacity)
    batch_size_offline = config["batch_size"]


    replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=sampler,
        batch_size=batch_size_offline
    )

    # offline dataset으로 offline_replay pre-fill
    CURRENT_STAGE = "offline_prefill"
    max_init_offline = min(buffer_size, train_dataset.size)  

    cursor = 0
    for _ in range(max_init_offline):
        seq = train_dataset.sample_sequence(
            batch_size=1,
            sequence_length=H,
            discount=discount,
        )

        obs_all = np.asarray(seq["observations"])  # [1, H, obs_dim] or [1, H+1, obs_dim]
        actions_seq = np.asarray(seq["actions"])[0]         # [H, act_dim]
        rewards_seq = np.asarray(seq["rewards"])[0]         # [H]
        rewards_seq_ptr= np.asarray(seq["rewards_ptr"])[0] # Added
        terminals_seq = np.asarray(seq["terminals"])[0]     # [H]
        masks_seq = np.asarray(seq["masks"])[0]             # [H]
        next_obs_seq = np.asarray(seq["next_observations"])[0]  # [H, obs_dim]

        # obs_seq: [H, obs_dim]로 정규화
        if obs_all.ndim == 3:
            obs_seq = obs_all[0]
        elif obs_all.ndim == 2:
            obs_seq = obs_all
        else:
            raise ValueError(f"Unexpected observations shape: {obs_all.shape}")

        obs0 = np.asarray(obs_seq[0], dtype=np.float32)  # [obs_dim] 기대

        if obs0.ndim > 1:
            obs0 = obs0.reshape(-1)

        assert obs0.ndim == 1, f"unexpected obs0.ndim: {obs0.ndim}"

        # valid mask: 첫 done 이후 0
        valid_seq = np.ones(H, dtype=np.float32)
        for t_idx in range(1, H):
            if terminals_seq[t_idx - 1] > 0.5:
                valid_seq[t_idx:] = 0.0
                break

        td = TensorDict(
            {
                "observations": torch.from_numpy(obs0).float(),
                "actions": torch.from_numpy(actions_seq).float(),
                "rewards": torch.from_numpy(rewards_seq).unsqueeze(-1).float(),
                "rewards_ptr": torch.from_numpy(rewards_seq_ptr).unsqueeze(-1).float(), #Added
                "terminals": torch.from_numpy(terminals_seq).unsqueeze(-1).float(),
                "masks": torch.from_numpy(masks_seq).unsqueeze(-1).float(),
                "next_observations": torch.from_numpy(next_obs_seq).float(),
                "valid": torch.from_numpy(valid_seq).float(),
            },
            batch_size=[],
        )

        replay_buffer.add(td)
        q = compute_quality(td, CURRENT_STAGE=CURRENT_STAGE)
        priorities[cursor] = q
        
        # debug log_priority
        if cursor % 1000 == 0:
            nonzero = priorities[priorities > 0]
            print(f"[DEBUG][offline prefill] cursor={cursor}")
            print("  pri stats: min={:.4f}, max={:.4f}, mean={:.4f}, std={:.4f}".format(
                nonzero.min().item(), nonzero.max().item(),
                nonzero.mean().item(), nonzero.std().item()
            ))

            N = len(replay_buffer)
            log_priority_snapshot(cursor, priorities, N, CURRNET_STAGE=CURRENT_STAGE)

        cursor = (cursor + 1) % capacity

    # ==================================================================
    # Offline RL
    # ==================================================================
    offline_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step+=1
        # PTR 기반 offline batch 샘플
        batch_td = replay_buffer.sample(batch_size=config["batch_size"])

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

        offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            # TODO: PyTorch 버전 save_agent 구현 필요 (state_dict 저장 등)
            # save_agent(agent, FLAGS.save_dir, log_step)
            pass

        # eval
        if (
            i == FLAGS.offline_steps - 1
            or (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0)
        ):
            # evaluate 함수도 PyTorch agent를 받도록 수정해야 함 (TODO).
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            print("[DEBUG] eval info :", eval_info)
            logger.log(eval_info, "eval", step=log_step)
        
        if i % 1000 == 0:
            N = len(replay_buffer)
            log_priority_snapshot(i, priorities, N, CURRNET_STAGE=CURRENT_STAGE)

    # # ==================================================================
    # # Replay Buffer (TorchRL) 초기화
    # # ==================================================================
    # buffer_size = FLAGS.buffer_size
    # batch_size_rb = config["batch_size"] * FLAGS.utd_ratio

    # capacity = FLAGS.buffer_size
    # priorities = torch.ones(capacity)

    # sampler = PrioritySampler(priorities, eps_uniform=0.2)
    # storage = LazyTensorStorage(capacity)

    # replay_buffer = TensorDictReplayBuffer(
    #     storage=storage,
    #     sampler=sampler,
    #     batch_size=batch_size_rb
    # )


    # # offline dataset으로 replay buffer pre-fill
    # H = FLAGS.horizon_length
    # max_init = min(buffer_size, train_dataset.size)  
    # cursor = 0
    # for _ in range(max_init):
    #     seq = train_dataset.sample_sequence(
    #         batch_size=1,
    #         sequence_length=H,
    #         discount=discount,
    #     )

    #     obs_all = np.asarray(seq["observations"])  # [1, H, obs_dim] 또는 [1, H+1, obs_dim]
    #     actions_seq = np.asarray(seq["actions"])[0]         # [H, act_dim]
    #     rewards_seq = np.asarray(seq["rewards"])[0]         # [H]
    #     terminals_seq = np.asarray(seq["terminals"])[0]     # [H]
    #     masks_seq = np.asarray(seq["masks"])[0]             # [H]
    #     next_obs_seq = np.asarray(seq["next_observations"])[0]  # [H, obs_dim]

    #     # obs_seq: [H, obs_dim]로 정규화
    #     if obs_all.ndim == 3:
    #         obs_seq = obs_all[0]
    #     elif obs_all.ndim == 2:
    #         obs_seq = obs_all
    #     else:
    #         raise ValueError(f"Unexpected observations shape: {obs_all.shape}")

    #     # 첫 시점 관측
    #     obs0 = np.asarray(obs_seq[0], dtype=np.float32)  # [obs_dim] 기대

    #     # 혹시 [1, obs_dim] 같은 경우 flatten
    #     if obs0.ndim > 1:
    #         obs0 = obs0.reshape(-1)

    #     assert obs0.ndim == 1, f"unexpected obs0.ndim: {obs0.ndim}"

    #     # valid 마스크: 첫 terminal 이후는 0
    #     valid_seq = np.ones(H, dtype=np.float32)
    #     for t_idx in range(1, H):
    #         if terminals_seq[t_idx - 1] > 0.5:
    #             valid_seq[t_idx:] = 0.0
    #             break

    #     td = TensorDict(
    #         {
    #             "observations": torch.from_numpy(obs0).float(),
    #             "actions": torch.from_numpy(actions_seq).float(),
    #             "rewards": torch.from_numpy(rewards_seq).unsqueeze(-1).float(),
    #             "terminals": torch.from_numpy(terminals_seq).unsqueeze(-1).float(),
    #             "masks": torch.from_numpy(masks_seq).unsqueeze(-1).float(),
    #             "next_observations": torch.from_numpy(next_obs_seq).float(),
    #             "valid": torch.from_numpy(valid_seq).float(),
    #         },
    #         batch_size=[],
    #     )

    #     replay_buffer.add(td)

    #     # chunk quality 계산 (예: valid 구간 평균 reward)
    #     quality = compute_quality(td)  # TensorDict -> float

    #     priorities[cursor] = quality
    #     cursor = (cursor + 1) % capacity



    # ==================================================================
    # Online RL
    # ==================================================================
    data = defaultdict(list)
    online_init_time = time.time()

    # ---- 온라인 시작 전 초기화 ----
    H = FLAGS.horizon_length
    action_dim = example_batch["actions"].shape[-1]

    ob, _ = env.reset()         
    action_queue = []           
    trans_window = deque(maxlen=H)  

    update_info = {}

    cursor=0
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
                },
                batch_size=[],
            )

            replay_buffer.add(td)
            # debug quality
            q = compute_quality(td, CURRENT_STAGE=CURRENT_STAGE)
            priorities[cursor] = q

            if cursor % 1000 == 0:
                N = len(replay_buffer)
                log_priority_snapshot(i, priorities, N, CURRNET_STAGE=CURRENT_STAGE)
            cursor = (cursor + 1) % capacity

        # -----------------------------------------------------------
        # 에피소드 종료 처리
        # -----------------------------------------------------------
        if done:
            ob, _ = env.reset()
            action_queue = []
            trans_window.clear()
        else:
            ob = next_ob

        # ---------------------------------------------------------------
        # Online 학습
        # ---------------------------------------------------------------
        if i >= FLAGS.start_training and len(replay_buffer) >= config["batch_size"]:
            batch_td = replay_buffer.sample(batch_size = config["batch_size"])
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
            log_priority_snapshot(i, priorities, N, CURRNET_STAGE=CURRENT_STAGE)

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
    
    import pandas as pd

    # chunk-level 로그
    if len(PTR_CHUNK_LOGS) > 0:
        df_chunks = pd.DataFrame(PTR_CHUNK_LOGS)
        csv_path_chunks = os.path.join(FLAGS.save_dir, "ptr_chunks.csv")
        df_chunks.to_csv(csv_path_chunks, index=False)
        print(f"[PTR] chunk-level logs saved to {csv_path_chunks}")

    # priority 분포 스냅샷
    if len(PTR_SNAPSHOT_LOGS) > 0:
        df_snap = pd.DataFrame(PTR_SNAPSHOT_LOGS)
        csv_path_snap = os.path.join(FLAGS.save_dir, "ptr_snapshots.csv")
        df_snap.to_csv(csv_path_snap, index=False)
        print(f"[PTR] snapshot logs saved to {csv_path_snap}")




if __name__ == '__main__':
    app.run(main)