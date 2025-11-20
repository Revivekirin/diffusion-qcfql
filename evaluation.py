from collections import defaultdict
from functools import partial

import numpy as np
from tqdm import trange
import torch


def flatten(d, parent_key='', sep='.'):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0.0,  
    eval_gaussian=None,
    action_shape=None,      
    observation_shape=None,
    action_dim=None,
    device=None,
):
    """
    PyTorch 버전 evaluate.

    Args:
        agent: PyTorch ACFQLAgent (sample_actions: obs_tensor -> action_tensor).
        env: Gym-like environment.
        num_eval_episodes: 통계 계산용 에피소드 수.
        num_video_episodes: 렌더용 에피소드 수 (통계에서는 제외).
        video_frame_skip: 몇 step마다 한 프레임씩 저장할지.
        eval_temperature: (옵션) 온도 파라미터. 현재 구현에서는 사용 X.
        eval_gaussian: 액션에 더할 Gaussian noise의 std (None이면 noise X).
        action_shape: 사용하지 않음 (옛 인터페이스 유지용).
        observation_shape: 사용하지 않음.
        action_dim: primitive action dimension (chunk reshape에 필요).
        device: PyTorch device (str or torch.device). None이면 agent.device 추정.

    Returns:
        stats: dict(key -> 평균값)
        trajs: list of dict(trajectory)
        renders: list of np.ndarray (각각 video 프레임 배열)
    """
    assert action_dim is not None, "evaluate() 호출 시 action_dim 을 반드시 지정해야 합니다."

    if device is None:
        # agent 내부에 device 속성이 있다고 가정
        device = getattr(agent, "device", torch.device("cpu"))
    else:
        device = torch.device(device)

    # agent network eval 모드로 전환 (dropout, BN 대응)
    net = getattr(agent, "network", None)
    prev_training_mode = None
    if isinstance(net, torch.nn.Module):
        prev_training_mode = net.training
        net.eval()

    trajs = []
    stats = defaultdict(list)
    renders = []

    # actor 호출 함수: numpy obs -> numpy action_chunk
    def actor_fn(obs_np: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(np.asarray(obs_np)).float().to(device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)  # [1, obs_dim]
        with torch.no_grad():
            action_t = agent.sample_actions(obs_t)  # [B, H*A] 또는 [B, A]
        return action_t.cpu().numpy()  # [B, ...], 보통 B=1

    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()

        done = False
        step = 0
        render = []

        action_queue = []

        gripper_contact_lengths = []
        gripper_contact_length = 0

        while not done:
            # 액션 chunk 뽑기
            if len(action_queue) == 0:
                # 새 chunk 샘플
                action_chunk = actor_fn(observation)  # [1, H*A] 또는 [1, A]
                action_chunk = np.array(action_chunk).reshape(-1, action_dim)  # [H, A] 또는 [1, A]
                for a in action_chunk:
                    action_queue.append(a)

            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)

            # gripper_contact 통계
            observation = next_observation
            if "proprio" in info and "gripper_contact" in info["proprio"]:
                gripper_contact = info["proprio"]["gripper_contact"]
            elif "gripper_contact" in info:
                gripper_contact = info["gripper_contact"]
            else:
                gripper_contact = None

            if gripper_contact is not None:
                if gripper_contact > 0.1:
                    gripper_contact_length += 1
                else:
                    if gripper_contact_length > 0:
                        gripper_contact_lengths.append(gripper_contact_length)
                    gripper_contact_length = 0

        # 에피소드 종료 후 마지막 contact 길이 반영
        if gripper_contact_length > 0:
            gripper_contact_lengths.append(gripper_contact_length)

        num_gripper_contacts = len(gripper_contact_lengths)
        if num_gripper_contacts > 0:
            avg_gripper_contact_length = np.mean(np.array(gripper_contact_lengths))
        else:
            avg_gripper_contact_length = 0.0

        add_to(
            stats,
            {
                "avg_gripper_contact_length": avg_gripper_contact_length,
                "num_gripper_contacts": num_gripper_contacts,
            },
        )

        if i < num_eval_episodes:
            # 마지막 step의 info 를 flatten 해서 통계에 추가
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    # 평균내기
    for k, v in stats.items():
        stats[k] = float(np.mean(v))

    # network mode 복원
    if isinstance(net, torch.nn.Module) and prev_training_mode is not None:
        net.train(prev_training_mode)

    return stats, trajs, renders
