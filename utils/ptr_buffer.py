import numpy as np
import torch
from tensordict import TensorDict

from torchrl.data.replay_buffers.samplers import Sampler


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

        # 0 이하 priority는 아주 작은 양수로 클램프만 하기
        p = torch.clamp(p, min=1e-6)

        # 정규화
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
