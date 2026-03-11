"""
Proximal Policy Optimization (PPO) with GAE.
Continuous actions via diagonal Gaussian policy.

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
    """State value estimator V(s). Used to label the dataset in place of Q(s,a)."""
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
    PPO agent for continuous control.

    act() / observe() / save() / load() match the TD3Agent interface
    so eval.py and the rollout pipeline work without changes.

    Training is done in train_ppo.py via update(rollout_data, step).
    The rollout_data dict is produced by collect_rollouts() there.
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
        """Return V(s) as value estimate for dataset labelling (replaces Q(s,a))."""
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)[None]
        value = self.critic_module.apply({'params': self.critic_state.params}, obs_jax)
        return {
            'state': np.asarray(obs),
            'value': float(np.asarray(value[0])),
        }

    # ------------------------------------------------------------------
    # train_ppo.py helpers
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
    # PPO update (JIT-compiled minibatch step)
    # ------------------------------------------------------------------

    @staticmethod
    @partial(jax.jit, static_argnames=('actor_apply_fn', 'critic_apply_fn',
                                       'clip_eps', 'value_coef', 'entropy_coef'))
    def _ppo_update_step(actor_state, critic_state, actor_apply_fn, critic_apply_fn,
                         obs, actions, old_log_probs, advantages, returns,
                         clip_eps, value_coef, entropy_coef):
        def loss_fn(actor_params, critic_params):
            mean, log_std = actor_apply_fn({'params': actor_params}, obs)
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

            values = critic_apply_fn({'params': critic_params}, obs)
            value_loss = value_coef * jnp.mean((values - returns) ** 2)

            total_loss = actor_loss + entropy_loss + value_loss
            return total_loss, (actor_loss, value_loss, entropy_loss,
                                jnp.mean(jnp.abs(ratio - 1)))

        (_, aux), grads = jax.value_and_grad(
            lambda ap, cp: loss_fn(ap, cp), argnums=(0, 1), has_aux=True
        )(actor_state.params, critic_state.params)
        actor_grads, critic_grads = grads
        actor_state = actor_state.apply_gradients(grads=actor_grads)
        critic_state = critic_state.apply_gradients(grads=critic_grads)
        actor_loss, value_loss, entropy_loss, approx_kl = aux
        return actor_state, critic_state, {
            'actor_loss': actor_loss,
            'critic_loss': value_loss,
            'entropy_loss': entropy_loss,
            'approx_kl': approx_kl,
        }

    def update(self, rollout_data, step=None):
        """PPO update over collected on-policy rollout_data dict."""
        obs = jnp.asarray(rollout_data['obs'], dtype=jnp.float32)
        actions = jnp.asarray(rollout_data['actions'], dtype=jnp.float32)
        old_log_probs = jnp.asarray(rollout_data['log_probs'], dtype=jnp.float32)
        advantages = jnp.asarray(rollout_data['advantages'], dtype=jnp.float32)
        returns = jnp.asarray(rollout_data['returns'], dtype=jnp.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        last_metrics = {}
        for _ in range(self.n_epochs):
            self.rng_key, subkey = jax.random.split(self.rng_key)
            perm = jax.random.permutation(subkey, n)
            for start in range(0, n, self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]
                self.actor_state, self.critic_state, m = self._ppo_update_step(
                    self.actor_state, self.critic_state,
                    self.actor_module.apply, self.critic_module.apply,
                    obs[idx], actions[idx], old_log_probs[idx],
                    advantages[idx], returns[idx],
                    self.clip_eps, self.value_coef, self.entropy_coef,
                )
                last_metrics = {k: float(v) for k, v in m.items()}
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
