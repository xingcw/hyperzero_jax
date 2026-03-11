"""
Twin Delayed Deep Deterministic Policy Gradients (TD3)
https://arxiv.org/abs/1802.09477

Optimized for TPU:
  - jax.pmap across all available devices for critic/actor gradient updates
  - lax.pmean averages gradients so all devices remain in sync
  - act_batch() uses a JIT-compiled actor forward pass for vectorized envs
  - update_batch() accepts a dict from CircularReplayBuffer for zero-copy I/O
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
        self.n_devices = jax.device_count()

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

        # Replicated states for pmap (lazily initialized on first update_batch)
        self._actor_state_rep = None
        self._critic_state_rep = None
        self._actor_target_rep = None
        self._critic_target_rep = None

        # JIT-compiled actor for batch inference
        self._jit_actor = jax.jit(self.actor_module.apply)

        # Build pmap'd update functions
        self._pmap_critic_update, self._pmap_full_update = self._build_pmap_updates()
        # Build scan-based multi-update (main training path)
        self._pmap_scan_update = self._build_pmap_scan_update()

    # ------------------------------------------------------------------
    # Replicated state management
    # ------------------------------------------------------------------

    def _init_replicated_states(self):
        devs = jax.devices()
        self._actor_state_rep = jax.device_put_replicated(self.actor_state, devs)
        self._critic_state_rep = jax.device_put_replicated(self.critic_state, devs)
        self._actor_target_rep = jax.device_put_replicated(self.actor_target_params, devs)
        self._critic_target_rep = jax.device_put_replicated(self.critic_target_params, devs)

    def _invalidate_replicated_states(self):
        self._actor_state_rep = None
        self._critic_state_rep = None
        self._actor_target_rep = None
        self._critic_target_rep = None

    def _unreplicate(self):
        """Extract single-device state from replicated state (all devices are in sync)."""
        self.actor_state = jax.tree_util.tree_map(lambda x: x[0], self._actor_state_rep)
        self.critic_state = jax.tree_util.tree_map(lambda x: x[0], self._critic_state_rep)
        self.actor_target_params = jax.tree_util.tree_map(lambda x: x[0], self._actor_target_rep)
        self.critic_target_params = jax.tree_util.tree_map(lambda x: x[0], self._critic_target_rep)

    # ------------------------------------------------------------------
    # pmap update builders
    # ------------------------------------------------------------------

    def _build_pmap_updates(self):
        actor_apply = self.actor_module.apply
        critic_apply = self.critic_module.apply
        tau = self.critic_target_tau

        @partial(jax.pmap, axis_name='devices')
        def pmap_critic_update(critic_state, critic_target_params, actor_target_params,
                               obs, action, reward, discount, next_obs,
                               stddev, stddev_clip, rng_key):
            """Critic-only update (called every step)."""
            next_action = actor_apply({'params': actor_target_params}, next_obs)
            noise = jnp.clip(
                jax.random.normal(rng_key, action.shape) * stddev, -stddev_clip, stddev_clip
            )
            next_action = jnp.clip(next_action + noise, -1.0, 1.0)
            target_Q1, target_Q2 = critic_apply(
                {'params': critic_target_params}, next_obs, next_action
            )
            target_Q = reward + discount * jnp.minimum(target_Q1, target_Q2)

            def critic_loss_fn(critic_params):
                Q1, Q2 = critic_apply({'params': critic_params}, obs, action)
                loss = jnp.mean((Q1 - target_Q) ** 2) + jnp.mean((Q2 - target_Q) ** 2)
                return loss, (jnp.mean(Q1), jnp.mean(Q2), jnp.mean(target_Q))

            (loss, (q1, q2, tq)), grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(critic_state.params)

            # Average gradients across devices
            grads = jax.lax.pmean(grads, axis_name='devices')
            critic_state = critic_state.apply_gradients(grads=grads)

            metrics = jax.lax.pmean(
                jnp.stack([loss, q1, q2, tq]), axis_name='devices'
            )
            return critic_state, metrics

        @partial(jax.pmap, axis_name='devices')
        def pmap_full_update(actor_state, critic_state,
                             actor_target_params, critic_target_params,
                             obs, action, reward, discount, next_obs,
                             stddev, stddev_clip, rng_key):
            """Critic + actor update + soft target update (called every update_every_steps)."""
            # --- Critic update ---
            next_action = actor_apply({'params': actor_target_params}, next_obs)
            noise = jnp.clip(
                jax.random.normal(rng_key, action.shape) * stddev, -stddev_clip, stddev_clip
            )
            next_action = jnp.clip(next_action + noise, -1.0, 1.0)
            target_Q1, target_Q2 = critic_apply(
                {'params': critic_target_params}, next_obs, next_action
            )
            target_Q = reward + discount * jnp.minimum(target_Q1, target_Q2)

            def critic_loss_fn(critic_params):
                Q1, Q2 = critic_apply({'params': critic_params}, obs, action)
                loss = jnp.mean((Q1 - target_Q) ** 2) + jnp.mean((Q2 - target_Q) ** 2)
                return loss, (jnp.mean(Q1), jnp.mean(Q2), jnp.mean(target_Q))

            (critic_loss, (q1, q2, tq)), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(critic_state.params)
            critic_grads = jax.lax.pmean(critic_grads, axis_name='devices')
            critic_state = critic_state.apply_gradients(grads=critic_grads)

            # --- Actor update ---
            def actor_loss_fn(actor_params):
                a = actor_apply({'params': actor_params}, obs)
                q1_val = critic_apply(
                    {'params': critic_state.params}, obs, a, method=Critic.Q1
                )
                return -jnp.mean(q1_val)

            actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
            actor_grads = jax.lax.pmean(actor_grads, axis_name='devices')
            actor_state = actor_state.apply_gradients(grads=actor_grads)

            # --- Soft-update targets ---
            # All devices have identical params after pmean → targets are identical too
            new_actor_target = jax.tree_util.tree_map(
                lambda p, tp: tau * p + (1 - tau) * tp,
                actor_state.params, actor_target_params,
            )
            new_critic_target = jax.tree_util.tree_map(
                lambda p, tp: tau * p + (1 - tau) * tp,
                critic_state.params, critic_target_params,
            )

            critic_loss = jax.lax.pmean(critic_loss, axis_name='devices')
            actor_loss = jax.lax.pmean(actor_loss, axis_name='devices')
            metrics = jax.lax.pmean(
                jnp.stack([critic_loss, q1, q2, tq, actor_loss]), axis_name='devices'
            )
            return (actor_state, critic_state,
                    new_actor_target, new_critic_target, metrics)

        return pmap_critic_update, pmap_full_update

    def _build_pmap_scan_update(self):
        """
        Single pmap call that scans over n_updates gradient steps on-device.

        Inputs per device have shape (n_updates, mb_per_device, ...).
        lax.scan loops over n_updates entirely in XLA — no Python overhead,
        no extra host-device transfers.  donate_argnums reuses state buffers.
        """
        actor_apply = self.actor_module.apply
        critic_apply = self.critic_module.apply
        tau = self.critic_target_tau

        @partial(jax.pmap, axis_name='devices', donate_argnums=(0, 1, 2, 3))
        def pmap_scan_update(actor_state, critic_state,
                             actor_target_params, critic_target_params,
                             obs_seq, action_seq, reward_seq, disc_seq, next_obs_seq,
                             stddev_seq, stddev_clip_seq, rng_seq):
            # obs_seq: (n_updates, mb, obs_dim) — already per-device slice

            def one_update(carry, x):
                actor_state, critic_state, actor_target, critic_target = carry
                obs, action, reward, disc, next_obs, stddev, stddev_clip, rng = x

                # --- Critic update ---
                next_action = actor_apply({'params': actor_target}, next_obs)
                noise = jnp.clip(
                    jax.random.normal(rng, action.shape) * stddev,
                    -stddev_clip, stddev_clip,
                )
                next_action = jnp.clip(next_action + noise, -1.0, 1.0)
                target_Q1, target_Q2 = critic_apply(
                    {'params': critic_target}, next_obs, next_action
                )
                target_Q = reward + disc * jnp.minimum(target_Q1, target_Q2)

                def critic_loss_fn(params):
                    Q1, Q2 = critic_apply({'params': params}, obs, action)
                    loss = (jnp.mean((Q1 - target_Q) ** 2) +
                            jnp.mean((Q2 - target_Q) ** 2))
                    return loss, (jnp.mean(Q1), jnp.mean(Q2), jnp.mean(target_Q))

                (closs, (q1, q2, tq)), cgrads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True
                )(critic_state.params)
                cgrads = jax.lax.pmean(cgrads, axis_name='devices')
                critic_state = critic_state.apply_gradients(grads=cgrads)

                # --- Actor update ---
                def actor_loss_fn(params):
                    a = actor_apply({'params': params}, obs)
                    q = critic_apply(
                        {'params': critic_state.params}, obs, a, method=Critic.Q1
                    )
                    return -jnp.mean(q)

                aloss, agrads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
                agrads = jax.lax.pmean(agrads, axis_name='devices')
                actor_state = actor_state.apply_gradients(grads=agrads)

                # --- Soft target update ---
                new_actor_target = jax.tree_util.tree_map(
                    lambda p, tp: tau * p + (1 - tau) * tp,
                    actor_state.params, actor_target,
                )
                new_critic_target = jax.tree_util.tree_map(
                    lambda p, tp: tau * p + (1 - tau) * tp,
                    critic_state.params, critic_target,
                )

                closs = jax.lax.pmean(closs, axis_name='devices')
                aloss = jax.lax.pmean(aloss, axis_name='devices')
                metrics = jnp.stack([closs, q1, q2, tq, aloss])
                return (actor_state, critic_state,
                        new_actor_target, new_critic_target), metrics

            (actor_state, critic_state,
             actor_target_params, critic_target_params), all_metrics = jax.lax.scan(
                one_update,
                (actor_state, critic_state, actor_target_params, critic_target_params),
                (obs_seq, action_seq, reward_seq, disc_seq, next_obs_seq,
                 stddev_seq, stddev_clip_seq, rng_seq),
            )
            # Average metrics over all scan steps
            mean_metrics = jax.lax.pmean(
                jnp.mean(all_metrics, axis=0), axis_name='devices'
            )
            return (actor_state, critic_state,
                    actor_target_params, critic_target_params, mean_metrics)

        return pmap_scan_update

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(self, obs, step, eval_mode):
        """Single-obs action for eval loop."""
        obs = jnp.asarray(obs, dtype=jnp.float32)
        action = self._jit_actor({'params': self.actor_state.params}, obs[None])
        action = np.asarray(action[0])

        if eval_mode:
            pass  # deterministic
        else:
            stddev = utils.schedule(self.stddev_schedule, step)
            action = np.clip(
                action + np.random.normal(0, stddev, size=self.action_dim), -1.0, 1.0
            )
            if step < self.num_expl_steps:
                action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        return action.astype(np.float32)

    def act_batch(self, obs, step, eval_mode=False):
        """
        Vectorized action selection for N parallel envs.
        obs: (N, obs_dim) numpy array
        Returns: (N, action_dim) numpy array
        """
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)
        actions = self._jit_actor({'params': self.actor_state.params}, obs_jax)
        actions = np.asarray(actions)

        if not eval_mode:
            stddev = utils.schedule(self.stddev_schedule, step)
            actions = np.clip(
                actions + np.random.normal(0, stddev, size=actions.shape), -1.0, 1.0
            )
            if step < self.num_expl_steps:
                actions = np.random.uniform(-1.0, 1.0, size=actions.shape).astype(np.float32)
        return actions.astype(np.float32)

    def observe(self, obs, action):
        obs = jnp.asarray(obs, dtype=jnp.float32)[None]
        action = jnp.asarray(action, dtype=jnp.float32)[None]
        q, _ = self.critic_module.apply({'params': self.critic_state.params}, obs, action)
        return {'state': np.asarray(obs[0]), 'value': np.asarray(q[0])}

    # ------------------------------------------------------------------
    # Update — new vectorized-env interface (CircularReplayBuffer dict)
    # ------------------------------------------------------------------

    def update_batch(self, batch_dict, step, discount=0.99):
        """
        Update from a CircularReplayBuffer sample dict.

        batch_dict keys: 'obs', 'actions', 'rewards', 'next_obs', 'dones'
        """
        n = len(batch_dict['obs'])
        n_devices = self.n_devices
        n_per = (n // n_devices) * n_devices  # trim to multiple of n_devices
        mb = n_per // n_devices

        obs = jnp.asarray(batch_dict['obs'][:n_per], dtype=jnp.float32)
        action = jnp.asarray(batch_dict['actions'][:n_per], dtype=jnp.float32)
        reward = jnp.asarray(batch_dict['rewards'][:n_per], dtype=jnp.float32)[:, None]
        dones = jnp.asarray(batch_dict['dones'][:n_per], dtype=jnp.float32)
        disc = (1.0 - dones)[:, None] * discount
        next_obs = jnp.asarray(batch_dict['next_obs'][:n_per], dtype=jnp.float32)

        def shard(x):
            return x.reshape(n_devices, mb, *x.shape[1:])

        stddev = utils.schedule(self.stddev_schedule, step)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        device_rngs = jax.random.split(subkey, n_devices)

        # Broadcast scalars to (n_devices,) so pmap can shard them
        stddev_arr = jnp.full((n_devices,), stddev)
        stddev_clip_arr = jnp.full((n_devices,), self.stddev_clip)

        # Lazy init replicated states
        if self._critic_state_rep is None:
            self._init_replicated_states()

        do_actor_update = (step % self.update_every_steps == 0)

        if do_actor_update:
            (self._actor_state_rep,
             self._critic_state_rep,
             self._actor_target_rep,
             self._critic_target_rep,
             metrics_raw) = self._pmap_full_update(
                self._actor_state_rep, self._critic_state_rep,
                self._actor_target_rep, self._critic_target_rep,
                shard(obs), shard(action), shard(reward), shard(disc),
                shard(next_obs), stddev_arr, stddev_clip_arr, device_rngs,
            )
            m = metrics_raw[0]  # same on all devices
            metrics = {
                'critic_loss': float(m[0]),
                'critic_q1':   float(m[1]),
                'critic_q2':   float(m[2]),
                'target_q':    float(m[3]),
                'actor_loss':  float(m[4]),
            }
        else:
            self._critic_state_rep, metrics_raw = self._pmap_critic_update(
                self._critic_state_rep,
                self._critic_target_rep,
                self._actor_target_rep,
                shard(obs), shard(action), shard(reward), shard(disc),
                shard(next_obs), stddev_arr, stddev_clip_arr, device_rngs,
            )
            m = metrics_raw[0]
            metrics = {
                'critic_loss': float(m[0]),
                'critic_q1':   float(m[1]),
                'critic_q2':   float(m[2]),
                'target_q':    float(m[3]),
            }

        # Keep non-replicated state in sync for act() / save() / eval
        self._unreplicate()
        return metrics

    def update_many(self, big_batch_dict, n_updates, step, discount=0.99):
        """
        Run n_updates gradient steps in a single pmap call via lax.scan.

        big_batch_dict: CircularReplayBuffer.sample(n_updates * batch_size)
        Compared to calling update_batch n_updates times:
          - 1 host→device transfer instead of n_updates
          - 1 pmap dispatch instead of n_updates
          - XLA fuses + optimizes the full scan loop
        """
        n_devices = self.n_devices
        total = len(big_batch_dict['obs'])
        batch_size = total // n_updates
        mb = batch_size // n_devices
        # Trim to exact multiple
        total = n_devices * n_updates * mb

        obs      = jnp.asarray(big_batch_dict['obs'][:total],     dtype=jnp.float32)
        action   = jnp.asarray(big_batch_dict['actions'][:total], dtype=jnp.float32)
        reward   = jnp.asarray(big_batch_dict['rewards'][:total], dtype=jnp.float32).reshape(total, 1)
        dones    = jnp.asarray(big_batch_dict['dones'][:total],   dtype=jnp.float32)
        disc     = (1.0 - dones).reshape(total, 1) * discount
        next_obs = jnp.asarray(big_batch_dict['next_obs'][:total],dtype=jnp.float32)

        def shard_seq(x):
            # (total, ...) → (n_devices, n_updates, mb, ...)
            return x.reshape(n_devices, n_updates, mb, *x.shape[1:])

        stddev = utils.schedule(self.stddev_schedule, step)
        stddev_seq      = jnp.full((n_devices, n_updates), stddev)
        stddev_clip_seq = jnp.full((n_devices, n_updates), self.stddev_clip)

        self.rng_key, subkey = jax.random.split(self.rng_key)
        rng_seq = jax.random.split(subkey, n_devices * n_updates).reshape(
            n_devices, n_updates, 2
        )

        if self._critic_state_rep is None:
            self._init_replicated_states()

        (self._actor_state_rep,
         self._critic_state_rep,
         self._actor_target_rep,
         self._critic_target_rep,
         metrics_raw) = self._pmap_scan_update(
            self._actor_state_rep, self._critic_state_rep,
            self._actor_target_rep, self._critic_target_rep,
            shard_seq(obs), shard_seq(action), shard_seq(reward), shard_seq(disc),
            shard_seq(next_obs), stddev_seq, stddev_clip_seq, rng_seq,
        )

        self._unreplicate()
        m = metrics_raw[0]  # same on all devices after pmean
        return {
            'critic_loss': float(m[0]),
            'critic_q1':   float(m[1]),
            'critic_q2':   float(m[2]),
            'target_q':    float(m[3]),
            'actor_loss':  float(m[4]),
        }

    # ------------------------------------------------------------------
    # Legacy update interface (single-env disk-based replay — kept for compat)
    # ------------------------------------------------------------------

    @staticmethod
    @partial(jax.jit, static_argnames=('actor_apply_fn', 'critic_apply_fn'))
    def _update_critic(critic_state, critic_target_params,
                       actor_target_params, actor_apply_fn, critic_apply_fn,
                       obs, action, reward, discount, next_obs,
                       stddev, stddev_clip, rng_key):
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

        return critic_state, {
            'critic_loss': critic_loss,
            'critic_q1': jnp.mean(q1),
            'critic_q2': jnp.mean(q2),
            'critic_target_q': jnp.mean(tq),
        }

    @staticmethod
    @partial(jax.jit, static_argnames=('actor_apply_fn', 'critic_apply_fn'))
    def _update_actor(actor_state, critic_params, actor_apply_fn, critic_apply_fn, obs):
        def actor_loss_fn(actor_params):
            action = actor_apply_fn({'params': actor_params}, obs)
            q1 = critic_apply_fn(
                {'params': critic_params}, obs, action, method=Critic.Q1
            )
            return -jnp.mean(q1)

        actor_loss, grads = jax.value_and_grad(actor_loss_fn)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, {'actor_loss': actor_loss}

    def update(self, replay_iter, step):
        """Legacy single-device update for backward compatibility."""
        metrics = {}
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_jax(batch)
        obs = obs.astype(jnp.float32)
        next_obs = next_obs.astype(jnp.float32)

        metrics['batch_reward'] = float(jnp.mean(reward))
        stddev = utils.schedule(self.stddev_schedule, step)
        self.rng_key, subkey = jax.random.split(self.rng_key)

        self.critic_state, m = self._update_critic(
            self.critic_state, self.critic_target_params,
            self.actor_target_params,
            self.actor_module.apply, self.critic_module.apply,
            obs, action, reward, discount, next_obs,
            stddev, self.stddev_clip, subkey,
        )
        metrics.update({k: float(v) for k, v in m.items()})

        if step % self.update_every_steps == 0:
            self.actor_state, m = self._update_actor(
                self.actor_state, self.critic_state.params,
                self.actor_module.apply, self.critic_module.apply, obs,
            )
            metrics.update({k: float(v) for k, v in m.items()})

            self.critic_target_params = utils.soft_update_params(
                self.critic_state.params, self.critic_target_params, self.critic_target_tau
            )
            self.actor_target_params = utils.soft_update_params(
                self.actor_state.params, self.actor_target_params, self.critic_target_tau
            )

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

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

        self._invalidate_replicated_states()
