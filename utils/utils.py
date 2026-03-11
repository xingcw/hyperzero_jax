import random
import re
import csv
import time
import os
import git
import json
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf


_STATE_AGENTS = ['td3', 'ppo', 'random', 'lapleig']
_PIXEL_AGENTS = ['drqv2', 'random']


@contextmanager
def eval_mode(*models):
    """No-op context manager. Flax has no train/eval mode for these architectures."""
    yield


def assert_agent(agent_name, pixel_obs):
    agent_name = agent_name.partition('_')[0]
    if pixel_obs:
        assert agent_name in _PIXEL_AGENTS, f"{agent_name} does not support pixel observations"
    else:
        assert agent_name in _STATE_AGENTS, f"{agent_name} does not support state observations"


def set_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)
    return jax.random.PRNGKey(seed)


def soft_update_params(params, target_params, tau):
    return jax.tree.map(
        lambda p, tp: tau * p + (1 - tau) * tp,
        params, target_params
    )


def to_jax(xs):
    return tuple(jnp.asarray(x) for x in xs)


def to_device(xs):
    """No-op: JAX arrays auto-placed on default device."""
    return xs


def select_indices(xs, indices):
    return tuple(x[indices] for x in xs)


def preprocess_obs(obs, rng_key, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == jnp.float32
    if bits < 8:
        obs = jnp.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + jax.random.uniform(rng_key, obs.shape) / bins
    obs = obs - 0.5
    return obs


def save_cfg(cfg, dir):
    with open(os.path.join(dir, 'cfg.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f.name)


def save_args(args, dir):
    with open(os.path.join(dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)


def save_git_sha(dir):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(dir, 'git_sha.txt'), 'w') as f:
        f.write(sha)


def get_last_model(model_dir):
    if not isinstance(model_dir, Path):
        model_dir = Path(model_dir)
    # return the step of the last saved model
    saved_models = [f for f in sorted(model_dir.glob(f'**/')) if not 'best' in str(f)]
    last_saved = saved_models[-1]
    last_step = str(last_saved.stem).partition('_')[-1]
    return int(last_step)


def dump_dict(fname, logs):
    with open(fname, "a") as f:
        writer = csv.DictWriter(f, logs.keys())
        if not os.path.getsize(fname):
            writer.writeheader()
        writer.writerow(logs)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


def truncated_normal(rng_key, loc, scale, shape, low=-1.0, high=1.0, eps=1e-6):
    """Sample from truncated normal distribution."""
    noise = jax.random.normal(rng_key, shape) * scale
    x = loc + noise
    x = jnp.clip(x, low + eps, high - eps)
    return x


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)
