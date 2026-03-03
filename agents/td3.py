"""
Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
https://arxiv.org/abs/1802.09477
"""

import pickle
import numpy as np
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from models.core import DeterministicActor, Critic
import utils.utils as utils


class TD3Agent:
    def __init__(self, obs_shape, action_shape, device, lr, hidden_dim,
                 critic_target_tau, num_expl_steps, update_every_steps,
                 stddev_schedule, stddev_clip):

        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Initialize PRNG
        rng_key = jax.random.PRNGKey(0)
        rng_key, actor_key, critic_key = jax.random.split(rng_key, 3)
        self.rng_key = rng_key

        # Dummy inputs for init
        dummy_obs = jnp.ones((1,) + obs_shape)
        dummy_action = jnp.ones((1,) + action_shape)

        # Actor
        self.actor_module = DeterministicActor(action_dim=action_shape[0],
                                               hidden_dim=hidden_dim)
        actor_params = self.actor_module.init(actor_key, dummy_obs)['params']
        self.actor_state = TrainState.create(
            apply_fn=self.actor_module.apply,
            params=actor_params,
            tx=optax.adam(lr),
        )
        self.actor_target_params = jax.tree.map(jnp.copy, actor_params)

        # Critic
        self.critic_module = Critic(hidden_dim=hidden_dim)
        critic_params = self.critic_module.init(critic_key, dummy_obs, dummy_action)['params']
        self.critic_state = TrainState.create(
            apply_fn=self.critic_module.apply,
            params=critic_params,
            tx=optax.adam(lr),
        )
        self.critic_target_params = jax.tree.map(jnp.copy, critic_params)

    def act(self, obs, step, eval_mode):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        action = self.actor_module.apply(
            {'params': self.actor_state.params}, obs[None]
        )
        action = np.asarray(action[0])

        if eval_mode:
            pass  # deterministic
        else:
            stddev = utils.schedule(self.stddev_schedule, step)
            action = action + np.random.normal(0, stddev, size=self.action_dim)
            if step < self.num_expl_steps:
                action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        return action.astype(np.float32)

    def observe(self, obs, action):
        obs = jnp.asarray(obs, dtype=jnp.float32)[None]
        action = jnp.asarray(action, dtype=jnp.float32)[None]

        q, _ = self.critic_module.apply(
            {'params': self.critic_state.params}, obs, action
        )

        return {
            'state': np.asarray(obs[0]),
            'value': np.asarray(q[0])
        }

    @staticmethod
    @partial(jax.jit, static_argnames=('actor_apply_fn', 'critic_apply_fn'))
    def _update_critic(critic_state, critic_target_params,
                       actor_target_params, actor_apply_fn, critic_apply_fn,
                       obs, action, reward, discount, next_obs,
                       stddev, stddev_clip, rng_key):
        # Compute target Q
        next_action = actor_apply_fn({'params': actor_target_params}, next_obs)
        noise = jnp.clip(
            jax.random.normal(rng_key, action.shape) * stddev,
            -stddev_clip, stddev_clip
        )
        next_action = jnp.clip(next_action + noise, -1.0, 1.0)

        target_Q1, target_Q2 = critic_apply_fn(
            {'params': critic_target_params}, next_obs, next_action
        )
        target_Q = jnp.minimum(target_Q1, target_Q2)
        target_Q = reward + discount * target_Q

        def critic_loss_fn(critic_params):
            current_Q1, current_Q2 = critic_apply_fn(
                {'params': critic_params}, obs, action
            )
            loss = jnp.mean((current_Q1 - target_Q)**2) + jnp.mean((current_Q2 - target_Q)**2)
            return loss, (current_Q1, current_Q2, target_Q)

        (critic_loss, (q1, q2, tq)), grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(critic_state.params)

        critic_state = critic_state.apply_gradients(grads=grads)

        metrics = {
            'critic_loss': critic_loss,
            'critic_q1': jnp.mean(q1),
            'critic_q2': jnp.mean(q2),
            'critic_target_q': jnp.mean(tq),
        }

        return critic_state, metrics

    @staticmethod
    @partial(jax.jit, static_argnames=('actor_apply_fn', 'critic_apply_fn'))
    def _update_actor(actor_state, critic_params, actor_apply_fn, critic_apply_fn, obs):
        def actor_loss_fn(actor_params):
            action = actor_apply_fn({'params': actor_params}, obs)
            q1 = critic_apply_fn(
                {'params': critic_params}, obs, action,
                method=Critic.Q1
            )
            return -jnp.mean(q1)

        actor_loss, grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, {'actor_loss': actor_loss}

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        stddev = utils.schedule(self.stddev_schedule, step)
        self.rng_key, subkey = jax.random.split(self.rng_key)

        self.critic_state, metrics = self._update_critic(
            self.critic_state, self.critic_target_params,
            self.actor_target_params,
            self.actor_module.apply, self.critic_module.apply,
            obs, action, reward, discount, next_obs,
            stddev, self.stddev_clip, subkey
        )

        return {k: float(v) for k, v in metrics.items()}

    def update_actor(self, obs, step):
        self.actor_state, metrics = self._update_actor(
            self.actor_state, self.critic_state.params,
            self.actor_module.apply, self.critic_module.apply, obs
        )

        return {k: float(v) for k, v in metrics.items()}

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_jax(batch)

        obs = obs.astype(jnp.float32)
        next_obs = next_obs.astype(jnp.float32)

        metrics['batch_reward'] = float(jnp.mean(reward))

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor (delayed)
        if step % self.update_every_steps == 0:
            metrics.update(self.update_actor(obs, step))

            # update target networks
            self.critic_target_params = utils.soft_update_params(
                self.critic_state.params, self.critic_target_params,
                self.critic_target_tau
            )
            self.actor_target_params = utils.soft_update_params(
                self.actor_state.params, self.actor_target_params,
                self.critic_target_tau
            )

        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        with open(f'{model_save_dir}/actor.pkl', 'wb') as f:
            pickle.dump({
                'params': jax.device_get(self.actor_state.params),
                'target_params': jax.device_get(self.actor_target_params),
            }, f)
        with open(f'{model_save_dir}/critic.pkl', 'wb') as f:
            pickle.dump({
                'params': jax.device_get(self.critic_state.params),
                'target_params': jax.device_get(self.critic_target_params),
            }, f)

    def load(self, model_dir, step):
        print(f"Loading the model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

        with open(f'{model_load_dir}/actor.pkl', 'rb') as f:
            actor_data = pickle.load(f)
            self.actor_state = self.actor_state.replace(
                params=jax.device_put(actor_data['params'])
            )
            self.actor_target_params = jax.device_put(actor_data['target_params'])

        with open(f'{model_load_dir}/critic.pkl', 'rb') as f:
            critic_data = pickle.load(f)
            self.critic_state = self.critic_state.replace(
                params=jax.device_put(critic_data['params'])
            )
            self.critic_target_params = jax.device_put(critic_data['target_params'])
