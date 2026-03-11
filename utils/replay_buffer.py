import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer:
    """Replay buffer as a plain Python iterable (no torch dependency)."""
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        # negative samples for the contrastive loss
        neg_idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1

        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        neg_obs = episode['observation'][neg_idx]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs, neg_obs)

    def __iter__(self):
        while True:
            yield self._sample()


class CircularReplayBuffer:
    """
    Fast in-memory numpy circular buffer for off-policy RL.

    Accepts batch adds (from vectorized envs) and batch samples.
    No disk I/O — all data lives in pre-allocated numpy arrays.
    """

    def __init__(self, capacity, obs_dim, action_dim):
        self._capacity = int(capacity)
        self._obs = np.zeros((self._capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((self._capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._capacity, dtype=np.float32)
        self._next_obs = np.zeros((self._capacity, obs_dim), dtype=np.float32)
        self._dones = np.zeros(self._capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, obs, actions, rewards, next_obs, dones):
        """Add a batch of N transitions."""
        n = len(obs)
        indices = np.arange(self._ptr, self._ptr + n) % self._capacity
        self._obs[indices] = obs
        self._actions[indices] = actions
        self._rewards[indices] = rewards
        self._next_obs[indices] = next_obs
        self._dones[indices] = dones
        self._ptr = (self._ptr + n) % self._capacity
        self._size = min(self._size + n, self._capacity)

    def sample(self, batch_size):
        """Sample a random batch. Returns a dict of numpy arrays."""
        assert self._size >= batch_size, (
            f"Buffer has only {self._size} transitions, need {batch_size}"
        )
        idx = np.random.randint(0, self._size, batch_size)
        return {
            'obs':      self._obs[idx],
            'actions':  self._actions[idx],
            'rewards':  self._rewards[idx],
            'next_obs': self._next_obs[idx],
            'dones':    self._dones[idx],
        }

    def __len__(self):
        return self._size


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    return BatchedIterator(iterable, batch_size)


class BatchedIterator:
    """Simple batched iterator that replaces torch DataLoader."""
    def __init__(self, iterable, batch_size):
        self._iterable = iterable
        self._batch_size = batch_size
        self._iter = None

    def __iter__(self):
        self._iter = iter(self._iterable)
        return self

    def __next__(self):
        samples = []
        for _ in range(self._batch_size):
            samples.append(next(self._iter))
        # Stack into batched numpy arrays
        batch = tuple(np.stack([s[i] for s in samples]) for i in range(len(samples[0])))
        return batch
