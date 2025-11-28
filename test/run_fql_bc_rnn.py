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

from agents.fql_bc_rnn import ACFQLAgent, get_config as get_acfql_config

from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from tensordict import TensorDict


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

flags.DEFINE_string('teacher_ckpt', "/home/robros/git/robomimic/trained_model/transport_ph_low_dim_epoch_1000_succ_78.pth", "pretrained teacher file path")

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
    print("[DEBUG] example batch :", example_batch)

    # --------- Agent 생성 (PyTorch) ---------
    agent = ACFQLAgent.create(
        seed=FLAGS.seed,
        ex_observations=np.asarray(example_batch['observations']),
        ex_actions=np.asarray(example_batch['actions']),
        config=config,
        device=str(device),
        teacher_ckpt=FLAGS.teacher_ckpt
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
    # Offline RL
    # ==================================================================
    offline_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1

        # (B, H, ...) 시퀀스 샘플
        batch_np = train_dataset.sample_sequence(
            config['batch_size'],
            sequence_length=FLAGS.horizon_length,
            discount=discount,
        )
        batch_torch = numpy_batch_to_torch(batch_np, device=device)

        offline_info = agent.update(batch_torch)  

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

    # ==================================================================
    # Replay Buffer (TorchRL) 초기화
    # ==================================================================
    buffer_size = FLAGS.buffer_size
    batch_size_rb = config["batch_size"] * FLAGS.utd_ratio

    storage = LazyTensorStorage(buffer_size)
    sampler = RandomSampler()

    replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=sampler,
        batch_size=batch_size_rb,
    )

    # offline dataset으로 replay buffer pre-fill
    H = FLAGS.horizon_length
    max_init = min(buffer_size, train_dataset.size)  # 너무 오래 안 돌게 제한

    for _ in range(max_init):
        # (1, H, ...) 시퀀스 하나 뽑기
        seq = train_dataset.sample_sequence(
            batch_size=1,
            sequence_length=H,
            discount=discount,
        )
        # 아래 shape는 Dataset 구현에 따라 약간 다를 수 있음
        # 보통:
        #   seq["observations"]      : [1, H, obs_dim]
        #   seq["actions"]           : [1, H, act_dim]
        #   seq["rewards"]           : [1, H]
        #   seq["terminals"]         : [1, H]
        #   seq["masks"]             : [1, H]
        #   seq["next_observations"] : [1, H, obs_dim]

        obs_seq = np.asarray(seq["observations"])[0]        # [H, obs_dim]
        actions_seq = np.asarray(seq["actions"])[0]         # [H, act_dim]
        rewards_seq = np.asarray(seq["rewards"])[0]         # [H]
        terminals_seq = np.asarray(seq["terminals"])[0]     # [H]
        masks_seq = np.asarray(seq["masks"])[0]             # [H]
        next_obs_seq = np.asarray(seq["next_observations"])[0]  # [H, obs_dim]

        obs0 = np.asarray(seq["observations"][0], dtype=np.float32)  # [obs_dim]

        # obs0는 이미 (19,) 이므로 여기서 shape 건드리지 않는 게 맞음
        assert obs0.ndim == 1, f"unexpected obs0.ndim: {obs0.ndim}"

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
                "rewards": torch.from_numpy(rewards_seq).unsqueeze(-1).float(),
                "terminals": torch.from_numpy(terminals_seq).unsqueeze(-1).float(),
                "masks": torch.from_numpy(masks_seq).unsqueeze(-1).float(),
                "next_observations": torch.from_numpy(next_obs_seq).float(),
                "valid": torch.from_numpy(valid_seq).float(),
            },
            batch_size=[],
        )
        replay_buffer.add(td)


    # ==================================================================
    # Online RL
    # ==================================================================
    data = defaultdict(list)
    online_init_time = time.time()

    # ---- 온라인 시작 전 초기화 ----
    H = FLAGS.horizon_length
    action_dim = example_batch["actions"].shape[-1]

    ob, _ = env.reset()          # 첫 관측
    action_queue = []            # chunk에서 꺼내 쓸 action 큐
    trans_window = deque(maxlen=H)  # H-step window

    update_info = {}

    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1

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

        # 큐에서 한 스텝 액션 꺼내기
        action = action_queue.pop(0)

        # -----------------------------------------------------------
        # env step
        # -----------------------------------------------------------
        next_ob, int_reward, terminated, truncated, info = env.step(
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

        # reward shaping
        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        # -----------------------------------------------------------
        # 1-step transition을 window에 쌓기
        # -----------------------------------------------------------
        trans_window.append(
            dict(
                observations=np.asarray(ob, dtype=np.float32),
                actions=np.asarray(action, dtype=np.float32),
                rewards=float(int_reward),
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
            rewards_seq = np.stack([t["rewards"] for t in trans_window], axis=0)              # [H]
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
                    "rewards": torch.from_numpy(rewards_seq).unsqueeze(-1).float(),
                    "terminals": torch.from_numpy(terminals_seq).unsqueeze(-1).float(),
                    "masks": torch.from_numpy(masks_seq).unsqueeze(-1).float(),
                    "next_observations": torch.from_numpy(next_obs_seq).float(),
                    "valid": torch.from_numpy(valid_seq).float(),
                },
                batch_size=[],
            )
            replay_buffer.add(td)

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
        if i >= FLAGS.start_training and len(replay_buffer) >= batch_size_rb:
            batch_td = replay_buffer.sample().to(device)
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


if __name__ == '__main__':
    app.run(main)