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
        import jax
        self._env = env
        self._rng = jax.random.PRNGKey(seed)
        self._state = None
        self._obs_spec = None
        self._action_spec = None
        self._init_specs()
        # JIT-compile step and reset so eval doesn't retrace every call
        self._jit_step = jax.jit(env.step)
        self._jit_reset = jax.jit(env.reset)

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
        self._state = self._jit_reset(rng_use)
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
        self._state = self._jit_step(self._state, action_jax)
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


class VectorizedBraxWrapper:
    """
    Vectorized Brax env using pmap across all TPU/GPU devices.
    Each device runs n_envs_per_device envs (auto_reset=True).
    Total envs = n_devices * n_envs_per_device.

    Both reset and step are pmap'd + JIT-compiled, so all devices do
    env work in parallel. Returns flat numpy arrays (n_envs, ...).
    """

    def __init__(self, env, n_devices, n_envs_per_device, seed):
        import jax
        self._env = env
        self._n_devices = n_devices
        self._n_envs_per_device = n_envs_per_device
        self._n_envs = n_devices * n_envs_per_device
        self._rng = jax.random.PRNGKey(seed)
        self._state = None

        # Determine obs/action sizes via a cheap single-device reset
        _state0 = env.reset(jax.random.PRNGKey(0))
        self._obs_size = int(np.array(_state0.obs).shape[-1])
        self._action_size = int(env.action_size)

        # pmap'd reset and step — compiled once, runs on all devices every call
        self._pmap_reset = jax.pmap(env.reset)
        self._pmap_step = jax.pmap(env.step)

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def obs_size(self):
        return self._obs_size

    @property
    def action_size(self):
        return self._action_size

    def observation_spec(self):
        return ArraySpec(shape=(self._obs_size,), dtype=np.float32, name='observation')

    def action_spec(self):
        return BoundedArraySpec(
            shape=(self._action_size,), dtype=np.float32,
            minimum=-1.0, maximum=1.0, name='action',
        )

    def reset(self):
        """Reset all envs on all devices. Returns obs of shape (n_envs, obs_size)."""
        import jax
        import jax.numpy as jnp
        self._rng, *device_rngs = jax.random.split(self._rng, self._n_devices + 1)
        # Stack → (n_devices, 2); pmap sends each row to the corresponding device
        device_rngs = jnp.stack(device_rngs)
        self._state = self._pmap_reset(device_rngs)
        return np.array(self._state.obs, dtype=np.float32).reshape(
            self._n_envs, self._obs_size
        )

    def step(self, actions):
        """
        Step all envs on all devices.
        actions: (n_envs, action_size) numpy array
        Returns: obs (n_envs, obs_size), rewards (n_envs,), dones (n_envs,).
        """
        import jax.numpy as jnp
        # Shard actions across devices: (n_devices, n_envs_per_device, action_size)
        actions_sharded = jnp.asarray(actions, dtype=jnp.float32).reshape(
            self._n_devices, self._n_envs_per_device, self._action_size
        )
        self._state = self._pmap_step(self._state, actions_sharded)
        obs = np.array(self._state.obs, dtype=np.float32).reshape(
            self._n_envs, self._obs_size
        )
        rewards = np.array(self._state.reward, dtype=np.float32).reshape(self._n_envs)
        dones = np.array(self._state.done, dtype=np.float32).reshape(self._n_envs)
        return obs, rewards, dones


def make_vectorized(name, n_envs, episode_length, seed):
    """Create a pmap-vectorized Brax env across all available devices."""
    import jax
    from brax import envs

    n_devices = jax.device_count()
    n_envs_per_device = max(1, n_envs // n_devices)

    name = name.lower().replace('-', '_')
    if name == 'cheetah_run':
        name = 'halfcheetah'
    elif name == 'walker_walk':
        name = 'walker2d'

    env = envs.create(
        env_name=name,
        episode_length=episode_length,
        action_repeat=1,
        auto_reset=True,
        batch_size=n_envs_per_device,
    )
    return VectorizedBraxWrapper(env, n_devices, n_envs_per_device, seed=seed)


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


def make_scan_eval(name, num_episodes, episode_length, actor_apply_fn, seed):
    """
    Build a JIT'd eval function using lax.scan + batched episodes.

    All num_episodes run in parallel on-device; lax.scan replaces the
    Python step loop — zero Python overhead after the first (compile) call.

    Returns: (jit_eval_fn, initial_rng)
      jit_eval_fn(actor_params, rng) -> (mean_episode_reward, next_rng)
    """
    import jax
    import jax.numpy as jnp
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
        batch_size=num_episodes,
    )
    jit_reset = jax.jit(env.reset)
    jit_step  = jax.jit(env.step)

    @jax.jit
    def jit_eval(actor_params, rng):
        rng, reset_rng = jax.random.split(rng)
        state = jit_reset(reset_rng)

        def step_fn(carry, _):
            state, has_ended = carry
            # Actor forward pass on batched obs (num_episodes, obs_dim)
            actions = actor_apply_fn({'params': actor_params}, state.obs)
            next_state = jit_step(state, actions)
            # Mask rewards for steps after episode termination
            reward = next_state.reward * (1.0 - has_ended)
            has_ended = jnp.maximum(has_ended, next_state.done)
            return (next_state, has_ended), reward

        has_ended = jnp.zeros(num_episodes)
        (_, _), rewards = jax.lax.scan(
            step_fn,
            (state, has_ended),
            None,
            length=episode_length,
        )
        # rewards: (episode_length, num_episodes) → sum per episode, then mean
        return jnp.mean(jnp.sum(rewards, axis=0)), rng

    return jit_eval, jax.random.PRNGKey(seed + 100)
