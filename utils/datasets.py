from functools import partial
import numpy as np


def get_size(data):
    """Return the size of the dataset (max length over all fields)."""
    sizes = [len(arr) for arr in data.values()]
    return max(sizes)


def random_crop(img, crop_from, padding):
    """Randomly crop an image (numpy version).

    Args:
        img: (H, W, C) image.
        crop_from: (y, x, 0) style crop origin (마지막 0은 채널 dummy).
        padding: padding size.
    """
    # pad: ((top,bottom),(left,right),(channel_pad))
    padded_img = np.pad(
        img,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="edge",
    )
    y, x, _ = crop_from
    H, W, C = img.shape
    return padded_img[y : y + H, x : x + W, :]


def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop.

    imgs: (B, H, W, C)
    crop_froms: (B, 3)
    """
    out = []
    for img, cf in zip(imgs, crop_froms):
        out.append(random_crop(img, cf, padding))
    return np.stack(out, axis=0)


class Dataset(dict):
    """Numpy-only Dataset class.

    - 내부는 그냥 dict[k] = np.ndarray
    - size, frame_stack, p_aug, return_next_actions 등 속성 유지
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to set arrays read-only.
            **fields: Keys and values of the dataset.
        """
        data = {}
        assert "observations" in fields

        for k, v in fields.items():
            arr = np.asarray(v)
            if freeze:
                arr.setflags(write=False)
            data[k] = arr

        return cls(data)

    def __init__(self, data):
        super().__init__(data)
        self.size = get_size(self)
        self.frame_stack = None         # Number of frames to stack; set outside the class.
        self.p_aug = None              # Image augmentation probability; set outside the class.
        self.return_next_actions = False

        # Compute terminal and initial locations (numpy only).
        self.terminal_locs = np.nonzero(self["terminals"] > 0)[0]
        self.initial_locs = np.concatenate(
            [[0], self.terminal_locs[:-1] + 1]
        )
    
    def copy(self, add_or_replace=None):
        """
        - self 내용을 dict로 복사한 뒤
        - add_or_replace 에 있는 key들을 덮어쓰기
        - 새 Dataset 인스턴스를 반환
        """
        new_data = {k: v for k, v in self.items()}

        if add_or_replace is not None:
            for k, v in add_or_replace.items():
                arr = np.asarray(v)
                arr.setflags(write=False)
                new_data[k] = arr

        return Dataset.create(freeze=True, **new_data)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices (plain dict)."""
        result = {k: v[idxs] for k, v in self.items()}
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result["next_actions"] = self["actions"][np.minimum(idxs + 1, self.size - 1)]
        return result

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions (single-step)."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)  # plain dict

        # frame stacking
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[
                np.searchsorted(self.initial_locs, idxs, side="right") - 1
            ]
            obs_list = []       # [ob[t-f+1], ..., ob[t]]
            next_obs_list = []  # [ob[t-f+2], ..., ob[t], next_ob[t]]

            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                cur_obs = self["observations"][cur_idxs]
                obs_list.append(cur_obs)
                if i != self.frame_stack - 1:
                    next_obs_list.append(cur_obs)
            next_obs_list.append(self["next_observations"][idxs])

            # concat along last dim
            batch["observations"] = np.concatenate(obs_list, axis=-1)
            batch["next_observations"] = np.concatenate(next_obs_list, axis=-1)

        # augmentation
        if self.p_aug is not None:
            if np.random.rand() < self.p_aug:
                self.augment(batch, ["observations", "next_observations"])

        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        """Sample sequences of length `sequence_length`.

        반환 형식은 기존 JAX 버전과 동일:
        dict(
            observations=data['observations'].copy(),
            full_observations=observations,
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=next_actions,
        )
        """
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)

        data = {k: v[idxs] for k, v in self.items()}

        # Pre-compute all required indices
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]  # (B, L)
        all_idxs = all_idxs.flatten()  # (B*L,)

        # Batch fetch
        obs_all = self["observations"]
        next_obs_all = self["next_observations"]
        act_all = self["actions"]
        rew_all = self["rewards"]
        masks_all = self["masks"]
        terms_all = self["terminals"]

        # reshape to [B, L, ...]
        batch_observations = obs_all[all_idxs].reshape(
            batch_size, sequence_length, *obs_all.shape[1:]
        )
        batch_next_observations = next_obs_all[all_idxs].reshape(
            batch_size, sequence_length, *next_obs_all.shape[1:]
        )
        batch_actions = act_all[all_idxs].reshape(
            batch_size, sequence_length, *act_all.shape[1:]
        )
        batch_rewards = rew_all[all_idxs].reshape(
            batch_size, sequence_length, *rew_all.shape[1:]
        )
        batch_masks = masks_all[all_idxs].reshape(
            batch_size, sequence_length, *masks_all.shape[1:]
        )
        batch_terminals = terms_all[all_idxs].reshape(
            batch_size, sequence_length, *terms_all.shape[1:]
        )

        # next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = act_all[next_action_idxs].reshape(
            batch_size, sequence_length, *act_all.shape[1:]
        )

        # cumulative rewards, masks, terminals, valid
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)

        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()

        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            r_i = batch_rewards[:, i].squeeze()
            m_i = batch_masks[:, i].squeeze()
            t_i = batch_terminals[:, i].squeeze()

            rewards[:, i] = rewards[:, i - 1] + r_i * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i - 1], m_i)
            terminals[:, i] = np.maximum(terminals[:, i - 1], t_i)
            valid[:, i] = 1.0 - terminals[:, i - 1]

        # observations 포맷
        if batch_observations.ndim == 5:  # (B, L, H, W, C) 가정
            # (B, H, W, L, C) 로 transpose (원 코드와 동일)
            full_observations = batch_observations.transpose(0, 2, 3, 1, 4)
            next_observations = batch_next_observations.transpose(0, 2, 3, 1, 4)
        else:
            # state: (B, L, D)
            full_observations = batch_observations
            next_observations = batch_next_observations

        actions = batch_actions
        next_actions = batch_next_actions

        return dict(
            observations=data["observations"].copy(),  # 시작 시점 관측
            full_observations=full_observations,
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=next_actions,
        )

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def augment(self, batch, keys):
        """Apply image augmentation (random crop) to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])

        # crop_froms: (B, 3) = (dy, dx, 0)
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate(
            [crop_froms, np.zeros((batch_size, 1), dtype=np.int64)],
            axis=1,
        )

        for key in keys:
            arr = batch[key]
            # arr: (B, H, W, C) 인 경우만 crop
            if isinstance(arr, np.ndarray) and arr.ndim == 4:
                batch[key] = batched_random_crop(arr, crop_froms, padding)
