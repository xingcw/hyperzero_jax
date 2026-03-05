"""
Brax-based environments for the training loop. State-only.
Uses env_common types (no dm_env).
"""

import numpy as np

from utils.env_common import (
    BoundedArraySpec,
    ArraySpec,
    ExtendedTimeStepWrapper,
    TimeStep,
    STEP_FIRST,
    STEP_MID,
    STEP_LAST,
)


class ActionRepeatWrapper:
    """Repeat the same action N times and accumulate reward/discount."""

    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break
        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class BraxWrapper:
    """
    Wraps a Brax env (stateless step(state, action)) with a single-instance
    interface. Returns TimeStep with observation, reward, step_type, discount.
    """

    def __init__(self, env, seed):
        self._env = env
        self._rng = __import__('jax').random.PRNGKey(seed)
        self._state = None
        self._obs_spec = None
        self._action_spec = None
        self._init_specs()

    def _init_specs(self):
        import jax
        rng = jax.random.PRNGKey(0)
        state = self._env.reset(rng)
        obs = np.array(state.obs, dtype=np.float32)
        if obs.ndim > 1:
            obs = obs.squeeze(0)
        obs_size = int(obs.shape[0])
        action_size = getattr(self._env, 'action_size', None)
        if action_size is None and hasattr(self._env, 'sys'):
            action_size = self._env.sys.act_size()
        if action_size is None:
            action_size = obs_size
        action_size = int(action_size)
        self._obs_spec = ArraySpec(shape=(obs_size,), dtype=np.float32, name='observation')
        self._action_spec = BoundedArraySpec(
            shape=(action_size,), dtype=np.float32,
            minimum=-1.0, maximum=1.0, name='action'
        )

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def reset(self):
        import jax
        self._rng, rng_use = jax.random.split(self._rng)
        self._state = self._env.reset(rng_use)
        obs = self._obs_from_state(self._state)
        return TimeStep(
            observation=obs,
            reward=0.0,
            discount=1.0,
            step_type=STEP_FIRST,
        )

    def _obs_from_state(self, state):
        obs = np.array(state.obs, dtype=np.float32)
        if obs.ndim > 1:
            obs = obs.squeeze(0)
        return obs

    def step(self, action):
        import jax
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action_batch = action[None, :]
        else:
            action_batch = action
        action_jax = jax.numpy.array(action_batch)
        self._state = self._env.step(self._state, action_jax)
        obs = self._obs_from_state(self._state)
        reward = float(np.array(self._state.reward).ravel().item())
        done = bool(np.array(self._state.done).ravel().item())
        step_type = STEP_LAST if done else STEP_MID
        discount = 1.0 - float(done)
        return TimeStep(
            observation=obs,
            reward=reward,
            discount=discount,
            step_type=step_type,
        )


def make(
    name,
    frame_stack,
    action_repeat,
    reward_kwargs,
    dynamics_kwargs,
    seed,
    pixel_obs,
    episode_length=1000,
):
    """
    Create a Brax env (state-only). reward_kwargs, dynamics_kwargs,
    frame_stack, and pixel_obs are ignored.
    """
    from brax import envs

    name = name.lower().replace('-', '_')
    if name == 'cheetah_run':
        name = 'halfcheetah'
    elif name == 'walker_walk':
        name = 'walker2d'

    env = envs.create(
        env_name=name,
        episode_length=episode_length,
        action_repeat=1,
        auto_reset=False,
        batch_size=1,
    )

    wrapper = BraxWrapper(env, seed=seed)
    if action_repeat > 1:
        wrapper = ActionRepeatWrapper(wrapper, action_repeat)
    wrapper = ExtendedTimeStepWrapper(wrapper)
    return wrapper
