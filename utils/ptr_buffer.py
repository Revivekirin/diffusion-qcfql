import numpy as np
import torch
from tensordict import TensorDict

from torchrl.data.replay_buffers.samplers import Sampler

TR_SAMPLING_LOGS = []

class PrioritySampler(Sampler):
    def __init__(self, priorities: torch.Tensor, eps_uniform: float = 0.1):
        super().__init__()
        self.priorities = priorities  
        self.eps_uniform = eps_uniform
        self.debug = True
        self.debug_interval = 100
        self._sample_calls = 0

    def sample(self, storage, batch_size: int):
        N = len(storage)
        if N == 0:
            raise RuntimeError("Storage is empty in PrioritySampler.")

        p = self.priorities[:N].clone().detach().to(dtype=torch.float32)
        p = torch.clamp(p, min=1e-6)
        p = p / (p.sum() + 1e-8)

        if torch.rand(()) < self.eps_uniform:
            mode = "uniform"
            indices = torch.randint(low=0, high=N, size=(batch_size,), dtype=torch.int64)
        else:
            mode = "prioritized"
            indices = torch.multinomial(p, num_samples=batch_size, replacement=False)

        p_batch = p[indices]                   # [B]
        p_all = p                              # [N]

        info = {
            "mode": mode,
            "p_batch_min": float(p_batch.min().item()),
            "p_batch_max": float(p_batch.max().item()),
            "p_batch_mean": float(p_batch.mean().item()),
            "p_batch_std": float(p_batch.std(unbiased=False).item()),
        }

        # with torch.no_grad():
        #     entropy = -(p_all * (p_all + 1e-12).log()).sum()
        # info.update({
        #     "p_all_min": float(p_all.min().item()),
        #     "p_all_max": float(p_all.max().item()),
        #     "p_all_mean": float(p_all.mean().item()),
        #     "p_all_std": float(p_all.std(unbiased=False).item()),
        #     "p_all_entropy": float(entropy.item()),
        # })

        k = max(1, int(0.05 * N))
        topk_vals, _ = torch.topk(p_all, k)
        info["p_all_top5pct_mass"] = float(topk_vals.sum().item())

        self._sample_calls += 1
        return indices, info


    def update_priority(self, index, priority, *, storage=None):
        """
        Sampler.update_priority 인터페이스 오버라이드 (선택).
        replay_buffer 쪽에서 호출하면 priority tensor 업데이트.
        """
        with torch.no_grad():
            if isinstance(index, torch.Tensor):
                self.priorities[index] = priority
            else:
                self.priorities[index] = float(priority)
        return {}


    def state_dict(self) -> dict:
        return {
            "priorities": self.priorities,
            "eps_uniform": self.eps_uniform,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.priorities = state_dict["priorities"]
        self.eps_uniform = state_dict.get("eps_uniform", self.eps_uniform)

    def _empty(self):
        """버퍼가 비워질 때 priority 초기화 용도로 호출."""
        with torch.no_grad():
            self.priorities.zero_()

    def dumps(self, path):
        """우리가 쓴 Sampler 상태를 파일로 저장 (선택적으로 사용)."""
        torch.save(self.state_dict(), path)

    def loads(self, path):
        """파일에서 Sampler 상태를 읽어옴."""
        state = torch.load(path)
        self.load_state_dict(state)


# ======================================================================
# TR-style trajectory sampling helpers
# ======================================================================

class TrajCursor:
    """하나의 active trajectory 상태를 저장하는 커서."""
    def __init__(self, ep_id: int, pos: int):
        # ep_id: episode id
        # pos  : EPISODE_TO_INDICES[ep_id] 리스트 안에서의 인덱스 (뒤에서 앞으로 이동)
        self.ep_id = ep_id
        self.pos = pos


def build_traj_index(storage, episode_to_indices):
    """
    EPISODE_TO_INDICES를 chunk_index 기준으로 정렬해두기.
    - ep_id -> [chunk_idx0, chunk_idx1, ...] (시간 순서)
    """
    for ep_id, idx_list in episode_to_indices.items():
        sorted_list = sorted(
            idx_list,
            key=lambda i: int(storage[i]["chunk_index"].item())
        )
        episode_to_indices[ep_id] = sorted_list

    return episode_to_indices


def compute_episode_scores(episode_to_indices, priorities: torch.Tensor):
    ep_ids = list(episode_to_indices.keys())
    scores = torch.zeros(len(ep_ids), device=priorities.device)
    for i, ep_id in enumerate(ep_ids):
        idxs = episode_to_indices[ep_id]
        if len(idxs) == 0:
            scores[i] = 0.0
        else:
            idx_tensor = torch.tensor(idxs, dtype=torch.long, device=priorities.device)
            scores[i] = priorities[idx_tensor].mean()
    return ep_ids, scores  # ep_ids: list[int], scores: torch.Tensor [E]


def init_active_trajs(
    episode_to_indices,
    priorities: torch.Tensor,
    num_active: int,
    ep_ids=None,
    ep_scores=None,
):
    if ep_ids is None or ep_scores is None:
        ep_ids, ep_scores = compute_episode_scores(episode_to_indices, priorities)

    valid_mask = ep_scores > 0
    ep_ids = [ep_ids[i] for i in range(len(ep_ids)) if valid_mask[i]]
    scores = ep_scores[valid_mask]
    if len(ep_ids) == 0:
        return [], set(), [], ep_ids, ep_scores 

    scores = torch.clamp(scores, min=1e-6)
    probs = (scores / scores.sum()).cpu().numpy()

    num_active = min(num_active, len(ep_ids))
    chosen_idx = np.random.choice(len(ep_ids), size=num_active, replace=False, p=probs)
    chosen_eps = [ep_ids[i] for i in chosen_idx]

    active_trajs = []
    active_set = set()
    for ep_id in chosen_eps:
        idx_list = episode_to_indices[ep_id]
        cursor = TrajCursor(ep_id=ep_id, pos=len(idx_list) - 1)
        active_trajs.append(cursor)
        active_set.add(ep_id)

    available_eps = [ep_id for ep_id in ep_ids if ep_id not in active_set]
    return active_trajs, set(available_eps), ep_ids, ep_scores

def pick_new_episode(available_eps: set,
                     episode_to_indices,
                     ep_ids,
                     ep_scores: torch.Tensor):
    if len(available_eps) == 0:
        return None

    eps_list = list(available_eps)
    # ep_id -> index 매핑
    ep_id_to_idx = {ep_id: i for i, ep_id in enumerate(ep_ids)}
    idx_list = [ep_id_to_idx[ep] for ep in eps_list]

    scores = ep_scores[idx_list]
    scores = torch.clamp(scores, min=1e-6)
    probs = (scores / scores.sum()).cpu().numpy()
    chosen = np.random.choice(len(eps_list), size=1, replace=False, p=probs)[0]
    return eps_list[chosen]


def sample_traj_batch(
    storage,
    episode_to_indices,
    ep_ids,
    ep_scores,
    active_trajs: list,
    available_eps: set,
    batch_size: int,
):
    """
    TR 논문 스타일의 batch 샘플링:
    - active_trajs에 있는 trajectory들에서 맨 끝 chunk를 pop
    - pos<0가 되면 해당 trajectory 다 쓴 것 → available에서 새 episode를 채워서 fill
    - 각 step마다 active_trajs에서 1개씩 모아 batch (B = len(active_trajs))
    """
    assert len(active_trajs) > 0, "active_trajs is empty"

    td_list = []
    use_trajs = active_trajs[:batch_size]

    # 각 active trajectory에서 하나씩 chunk 꺼내기
    for cursor in use_trajs:
        idx_list = episode_to_indices.get(cursor.ep_id, [])
        if len(idx_list) == 0:
            continue

        chunk_idx = idx_list[cursor.pos]
        td = storage[chunk_idx]
        td_list.append(td)

        cursor.pos -= 1

        # trajectory를 다 소모했으면 새 episode로 교체 시도
        if cursor.pos < 0:
            new_ep = pick_new_episode(available_eps, episode_to_indices, ep_ids, ep_scores)
            if new_ep is not None:
                available_eps.remove(new_ep)
                cursor.ep_id = new_ep
                cursor.pos = len(episode_to_indices[new_ep]) - 1
            else:
                cursor.pos = len(idx_list) - 1

    if len(td_list) == 0:
        raise RuntimeError("TR sampler could not produce any samples.")

    # TensorDict.stack 사용
    batch_td = TensorDict.stack(td_list, dim=0)
    return batch_td, active_trajs, available_eps