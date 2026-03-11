"""
Proximal Policy Optimization (PPO) with GAE.
Continuous actions via diagonal Gaussian policy.

Multi-device training via jax.pmap with lax.pmean gradient averaging.
Compatible interface with TD3Agent: act(), observe(), save(), load().
On-policy training is handled externally by train_ppo.py.
"""

import pickle
import numpy as np
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState


class GaussianActor(nn.Module):
    """Stochastic Gaussian policy with state-independent log-std."""
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(2**0.5))(obs)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(2**0.5))(x)
        x = nn.tanh(x)
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        return mean, log_std


class ValueFunction(nn.Module):
    """State value estimator V(s)."""
    hidden_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(2**0.5))(obs)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(2**0.5))(x)
        x = nn.tanh(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return value[..., 0]


class PPOAgent:
    """
    PPO agent for continuous control with multi-device (pmap) training support.

    act() / observe() / save() / load() match the TD3Agent interface.
    Training is done in train_ppo.py via update(rollout_data, step).
    """

    def __init__(self, obs_shape, action_shape, device, lr, hidden_dim,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
                 n_epochs=10, minibatch_size=64, max_grad_norm=0.5):
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.n_devices = jax.device_count()

        rng_key = jax.random.PRNGKey(0)
        rng_key, actor_key, critic_key = jax.random.split(rng_key, 3)
        self.rng_key = rng_key

        dummy_obs = jnp.ones((1,) + obs_shape)

        self.actor_module = GaussianActor(action_dim=action_shape[0], hidden_dim=hidden_dim)
        actor_params = self.actor_module.init(actor_key, dummy_obs)['params']
        self.actor_state = TrainState.create(
            apply_fn=self.actor_module.apply,
            params=actor_params,
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr, eps=1e-5),
            ),
        )

        self.critic_module = ValueFunction(hidden_dim=hidden_dim)
        critic_params = self.critic_module.init(critic_key, dummy_obs)['params']
        self.critic_state = TrainState.create(
            apply_fn=self.critic_module.apply,
            params=critic_params,
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr, eps=1e-5),
            ),
        )

        # Replicated states for pmap (lazily initialized on first update())
        self._actor_state_rep = None
        self._critic_state_rep = None

        # Build pmap update function (captures hyperparams + apply_fns in closure)
        self._pmap_update = self._build_pmap_update()

    def _build_pmap_update(self):
        """Build a pmap'd update step with gradient averaging via lax.pmean."""
        actor_apply = self.actor_module.apply
        critic_apply = self.critic_module.apply
        clip_eps = self.clip_eps
        value_coef = self.value_coef
        entropy_coef = self.entropy_coef

        @partial(jax.pmap, axis_name='devices')
        def pmap_update(actor_state, critic_state,
                        obs, actions, old_log_probs, advantages, returns):
            def loss_fn(actor_params, critic_params):
                mean, log_std = actor_apply({'params': actor_params}, obs)
                std = jnp.exp(log_std)
                log_probs = (
                    -0.5 * (((actions - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
                ).sum(-1)

                ratio = jnp.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))

                entropy = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * std ** 2))
                entropy_loss = -entropy_coef * entropy

                values = critic_apply({'params': critic_params}, obs)
                value_loss = value_coef * jnp.mean((values - returns) ** 2)

                total_loss = actor_loss + entropy_loss + value_loss
                return total_loss, (actor_loss, value_loss, entropy_loss,
                                    jnp.mean(jnp.abs(ratio - 1)))

            (_, aux), grads = jax.value_and_grad(
                loss_fn, argnums=(0, 1), has_aux=True
            )(actor_state.params, critic_state.params)

            # Average gradients across all devices
            actor_grads = jax.lax.pmean(grads[0], axis_name='devices')
            critic_grads = jax.lax.pmean(grads[1], axis_name='devices')

            actor_state = actor_state.apply_gradients(grads=actor_grads)
            critic_state = critic_state.apply_gradients(grads=critic_grads)

            # Average metrics across devices for consistent logging
            metrics = jax.lax.pmean(jnp.stack(aux), axis_name='devices')
            return actor_state, critic_state, metrics

        return pmap_update

    def _get_replicated_states(self):
        """Lazily replicate train states across all devices."""
        if self._actor_state_rep is None:
            self._actor_state_rep = jax.device_put_replicated(
                self.actor_state, jax.devices()
            )
            self._critic_state_rep = jax.device_put_replicated(
                self.critic_state, jax.devices()
            )
        return self._actor_state_rep, self._critic_state_rep

    def _invalidate_replicated_states(self):
        """Force re-replication on next update (e.g., after load_snapshot)."""
        self._actor_state_rep = None
        self._critic_state_rep = None

    # ------------------------------------------------------------------
    # eval.py interface
    # ------------------------------------------------------------------

    def act(self, obs, step, eval_mode):
        """Select action. Returns deterministic mean in eval_mode."""
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)[None]
        mean, log_std = self.actor_module.apply({'params': self.actor_state.params}, obs_jax)
        if eval_mode:
            action = mean[0]
        else:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            std = jnp.exp(log_std)
            action = mean[0] + std * jax.random.normal(subkey, mean.shape[1:])
        action = jnp.clip(action, -1.0, 1.0)
        return np.asarray(action, dtype=np.float32)

    def observe(self, obs, action):
        """Return V(s) as value estimate for dataset labelling."""
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)[None]
        value = self.critic_module.apply({'params': self.critic_state.params}, obs_jax)
        return {
            'state': np.asarray(obs),
            'value': float(np.asarray(value[0])),
        }

    # ------------------------------------------------------------------
    # train_ppo.py helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def get_value(self, obs):
        """Batched V(s): obs (N, obs_dim) → values (N,)."""
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)
        if obs_jax.ndim == 1:
            obs_jax = obs_jax[None]
        return np.asarray(self.critic_module.apply(
            {'params': self.critic_state.params}, obs_jax
        ))

    def get_action_logprob_value(self, obs, rng_key=None):
        """
        Batched rollout step: obs (N, obs_dim) →
            actions (N, action_dim), log_probs (N,), values (N,).
        """
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)
        if rng_key is None:
            self.rng_key, rng_key = jax.random.split(self.rng_key)
        mean, log_std = self.actor_module.apply({'params': self.actor_state.params}, obs_jax)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng_key, mean.shape)
        action = jnp.clip(mean + std * noise, -1.0, 1.0)
        log_prob = (
            -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        ).sum(-1)
        value = self.critic_module.apply({'params': self.critic_state.params}, obs_jax)
        return np.asarray(action), np.asarray(log_prob), np.asarray(value)

    # ------------------------------------------------------------------
    # PPO update — pmap across devices with pmean gradient averaging
    # ------------------------------------------------------------------

    def update(self, rollout_data, step=None):
        """PPO update over collected on-policy rollout_data dict."""
        obs = jnp.asarray(rollout_data['obs'], dtype=jnp.float32)
        actions = jnp.asarray(rollout_data['actions'], dtype=jnp.float32)
        old_log_probs = jnp.asarray(rollout_data['log_probs'], dtype=jnp.float32)
        advantages = jnp.asarray(rollout_data['advantages'], dtype=jnp.float32)
        returns = jnp.asarray(rollout_data['returns'], dtype=jnp.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        n_devices = self.n_devices
        # Ensure minibatch size is divisible by n_devices
        mb_per_device = self.minibatch_size // n_devices
        mb_size = mb_per_device * n_devices

        actor_state_rep, critic_state_rep = self._get_replicated_states()

        last_metrics = {}
        for _ in range(self.n_epochs):
            self.rng_key, subkey = jax.random.split(self.rng_key)
            perm = jax.random.permutation(subkey, n)

            for start in range(0, n - mb_size + 1, mb_size):
                idx = perm[start:start + mb_size]

                def shard(x):
                    return x[idx].reshape(n_devices, mb_per_device, *x.shape[1:])

                actor_state_rep, critic_state_rep, metrics = self._pmap_update(
                    actor_state_rep, critic_state_rep,
                    shard(obs), shard(actions), shard(old_log_probs),
                    shard(advantages), shard(returns),
                )
                last_metrics = {
                    'actor_loss':   float(metrics[0]),
                    'critic_loss':  float(metrics[1]),
                    'entropy_loss': float(metrics[2]),
                    'approx_kl':    float(metrics[3]),
                }

        # Save updated replicated states
        self._actor_state_rep = actor_state_rep
        self._critic_state_rep = critic_state_rep

        # Unreplicate: extract from device 0 (all devices are in sync)
        self.actor_state = jax.tree_util.tree_map(lambda x: x[0], actor_state_rep)
        self.critic_state = jax.tree_util.tree_map(lambda x: x[0], critic_state_rep)

        return last_metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)
        with open(f'{model_save_dir}/actor.pkl', 'wb') as f:
            pickle.dump({'params': jax.device_get(self.actor_state.params)}, f)
        with open(f'{model_save_dir}/critic.pkl', 'wb') as f:
            pickle.dump({'params': jax.device_get(self.critic_state.params)}, f)

    def load(self, model_dir, step):
        print(f"Loading PPO model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        with open(f'{model_load_dir}/actor.pkl', 'rb') as f:
            actor_data = pickle.load(f)
            self.actor_state = self.actor_state.replace(
                params=jax.device_put(actor_data['params'])
            )
        with open(f'{model_load_dir}/critic.pkl', 'rb') as f:
            critic_data = pickle.load(f)
            self.critic_state = self.critic_state.replace(
                params=jax.device_put(critic_data['params'])
            )
        # Invalidate replicated states so next update() re-replicates fresh params
        self._invalidate_replicated_states()
