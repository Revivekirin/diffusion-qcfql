import numpy as np
import torch
from tensordict import TensorDict

from torchrl.data.replay_buffers.samplers import Sampler

TR_SAMPLING_LOGS = []

class PrioritySampler(Sampler):
    """
    torchrl Sampler 기반 PTR 샘플러.
    - priorities: shape [capacity] 인 torch.Tensor (float)
    - eps_uniform 확률로 uniform sampling, 나머지는 priority-based sampling
    """

    def __init__(self, priorities: torch.Tensor, eps_uniform: float = 0.1):
        super().__init__()
        self.priorities = priorities  
        self.eps_uniform = eps_uniform
        self.debug = True
        self.debug_interval = 100
        self._sample_calls = 0

    def sample(self, storage, batch_size: int):
        """
        storage: LazyTensorStorage (replay buffer 내부 저장소)
        batch_size: 한 번에 뽑을 개수

        반환:
            indices: torch.LongTensor [batch_size]
            info   : dict (추후 IS weight 등 넣고 싶으면 사용)
        """
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
            # multinomial sampling
            mode = "prioritized"
            indices = torch.multinomial(p, num_samples=batch_size, replacement=False)

        info = {} 
        if self.debug and (self._sample_calls % self.debug_interval == 0):
            show_k = min(5, batch_size)
            idx_show = indices[:show_k]
            print(f"[DEBUG][PrioritySampler] call={self._sample_calls}, mode={mode}")
            print("  sampled idx :", idx_show.tolist())
            print("  raw pri     :", [round(self.priorities[j].item(), 4) for j in idx_show])
            print("  prob        :", [float(f"{p[j].item():.8e}") for j in idx_show])

        self._sample_calls +=1
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


class PrioritizedChunkReplayBuffer:
    def __init__(self, capacity, alpha=1.0, eps_uniform=0.1, device="cpu"):
        """
        capacity    : 최대 저장 개수
        alpha       : priority exponent (0 = uniform, 1 = 강한 priority)
        eps_uniform : eps 확률로 uniform sampling 수행
        device      : 샘플 뽑을 때 옮길 디바이스
        """
        self.capacity = capacity
        self.alpha = alpha
        self.eps_uniform = eps_uniform
        self.device = device

        self.storage = []    # TensorDict 리스트
        self.qualities = []  # 각 chunk의 quality (float)
        self.pos = 0         # ring buffer 포인터

    def __len__(self):
        return len(self.storage)

    def _compute_quality(self, td: TensorDict) -> float:
        """
        td 안의 rewards, valid를 이용해 chunk-level quality 계산.
        여기서는 valid 구간 평균 reward를 예시로 사용.
        """
        # [H, 1] 혹은 [H]
        rewards = td["rewards"]
        valid   = td["valid"]

        # shape 정리
        rewards = rewards.squeeze(-1).cpu().numpy()  # [H]
        valid   = valid.cpu().numpy().astype(np.float32)  # [H]

        # valid 구간만 사용
        effective = rewards * valid
        if valid.sum() > 0:
            quality = float(effective.sum() / (valid.sum() + 1e-6))
        else:
            # 전부 invalid인 경우는 거의 없겠지만 방어적으로 0
            quality = 0.0
        return quality

    def add(self, td: TensorDict):
        """
        새로운 chunk(td)를 버퍼에 추가하고 priority용 quality 갱신
        """
        # td는 batch_size=[] (scalar tensordict) 라고 가정 (당신 코드 그대로)
        # 저장은 CPU에 두는 걸 권장 (샘플할 때 to(device))
        td_cpu = td.to("cpu")

        quality = self._compute_quality(td_cpu)

        if len(self.storage) < self.capacity:
            self.storage.append(td_cpu)
            self.qualities.append(quality)
        else:
            # ring buffer 방식으로 덮어쓰기
            self.storage[self.pos] = td_cpu
            self.qualities[self.pos] = quality
            self.pos = (self.pos + 1) % self.capacity

    def _get_sampling_probs(self):
        """
        qualities -> rank-based priority -> sampling prob 계산
        """
        q = np.array(self.qualities, dtype=np.float32)
        N = len(q)
        if N == 0:
            raise ValueError("Replay buffer is empty.")

        # rank: 작은 값부터 0,1,2,... (여기서는 quality가 클수록 좋다고 가정하므로
        # argsort를 한 번 더 감싸서 실제 순위로 변환)
        ranks = q.argsort().argsort()  # [0..N-1]
        # quality가 클수록 rank도 커지게 되어 있음
        # 우리는 보통 "high-quality에 더 큰 weight"를 주고 싶으니,
        # 아래처럼 (rank + 1)으로 나눌지, 그냥 (ranks+1)**something 등을 바꿔가며 실험 가능
        # 여기서는 가장 high rank가 가장 큰 p가 되도록 반대로 한번 뒤집어주자:
        inv_ranks = (N - ranks)  # 가장 큰 quality -> inv_ranks ~ N

        p = inv_ranks.astype(np.float32)
        p = np.maximum(p, 1e-6)  # 0 방지

        probs = p ** self.alpha
        probs /= probs.sum()
        return probs

    def sample(self, batch_size: int) -> TensorDict:
        """
        eps_uniform 비율로 uniform + PTR priority 섞어서 batch 추출.
        """
        N = len(self.storage)
        assert N >= batch_size, f"Not enough samples: {N} < {batch_size}"

        # eps 확률로 완전 uniform
        if np.random.rand() < self.eps_uniform:
            idxs = np.random.choice(N, size=batch_size, replace=False)
        else:
            probs = self._get_sampling_probs()  # [N]
            idxs = np.random.choice(N, size=batch_size, p=probs, replace=False)

        # 선택된 tensordict들을 stack해서 [B, ...] batch 만들기
        td_list = [self.storage[i] for i in idxs]
        batch_td = torch.stack(td_list, dim=0)  # TensorDict stack

        return batch_td.to(self.device)
