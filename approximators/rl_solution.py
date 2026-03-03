import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from models.rl_regressor import MLPRLSolution, HyperRLSolution
import utils.utils as utils


class RLApproximator:
    """
    Approximates a family of near-optimal RL solutions.
    Uses either a conditional MLP or a hypernetwork.
    """
    def __init__(self, model, input_dim, state_dim, action_dim,
                 device, lr, embed_dim, hidden_dim, noise_clip,
                 use_clipped_noise, use_td, td_weight, value_weight):
        self.lr = lr
        self.model_type = model
        self.use_td_error = use_td
        self.use_clipped_noise = use_clipped_noise
        self.noise_clip = noise_clip
        self.td_weight = td_weight
        self.value_weight = value_weight

        # model
        if model == 'mlp':
            self.rl_module = MLPRLSolution(
                input_param_dim=input_dim, state_dim=state_dim,
                action_dim=action_dim, embed_dim=embed_dim, hidden_dim=hidden_dim
            )
        elif model == 'hyper':
            self.rl_module = HyperRLSolution(
                input_param_dim=input_dim, state_dim=state_dim,
                action_dim=action_dim, embed_dim=embed_dim, hidden_dim=hidden_dim
            )
        else:
            raise NotImplementedError

        # Initialize
        rng_key = jax.random.PRNGKey(0)
        rng_key, init_key = jax.random.split(rng_key)
        self.rng_key = rng_key

        dummy_param = jnp.ones((1, input_dim))
        dummy_state = jnp.ones((1, state_dim))
        dummy_action = jnp.ones((1, action_dim))
        params = self.rl_module.init(init_key, dummy_param, dummy_state, dummy_action)['params']

        self.rl_state = TrainState.create(
            apply_fn=self.rl_module.apply,
            params=params,
            tx=optax.adam(lr),
        )

    def act(self, input_param, obs):
        input_param = jnp.asarray(input_param, dtype=jnp.float32)[None]
        obs = jnp.asarray(obs, dtype=jnp.float32)[None]
        task_emb = self.rl_module.apply(
            {'params': self.rl_state.params}, input_param,
            method=self.rl_module.embed_task
        )
        action = self.rl_module.apply(
            {'params': self.rl_state.params}, task_emb, obs,
            method=self.rl_module.predict_action
        )
        return np.asarray(action[0]).astype(np.float32)

    def q(self, input_param, obs, action):
        input_param = jnp.asarray(input_param, dtype=jnp.float32)[None]
        obs = jnp.asarray(obs, dtype=jnp.float32)[None]
        action = jnp.asarray(action, dtype=jnp.float32)[None]
        task_emb = self.rl_module.apply(
            {'params': self.rl_state.params}, input_param,
            method=self.rl_module.embed_task
        )
        q_val = self.rl_module.apply(
            {'params': self.rl_state.params}, task_emb, obs, action,
            method=self.rl_module.predict_q_value
        )
        return np.asarray(q_val[0]).astype(np.float32)

    def _get_td_error(self, params, task_emb, next_state, reward, discount, q):
        """Compute TD error with stop_gradient on next_action."""
        next_action = jax.lax.stop_gradient(
            self.rl_module.apply(
                {'params': params}, task_emb, next_state,
                method=self.rl_module.predict_action
            )
        )
        target_q = self.rl_module.apply(
            {'params': params}, task_emb, next_state, next_action,
            method=self.rl_module.predict_q_value
        )
        target_q = reward + discount * target_q
        td_error = jnp.mean((q - target_q)**2)
        return td_error

    def eval(self, data_loader):
        metrics = defaultdict(lambda: 0)
        num_batches = len(data_loader)

        for batch_idx, batch in enumerate(data_loader):
            input_param, state, action, next_state, reward, discount, value = batch

            task_emb, predicted_action, predicted_value = self.rl_module.apply(
                {'params': self.rl_state.params}, input_param, state, action
            )

            loss_action = jnp.mean((predicted_action - action)**2)
            loss_value = jnp.mean((predicted_value - value)**2)
            loss = loss_action + self.value_weight * loss_value

            # evaluate the TD error in any case
            loss_td = self._get_td_error(
                self.rl_state.params, task_emb, next_state, reward, discount, value
            )

            if self.use_td_error:
                loss = loss + self.td_weight * loss_td

            metrics['valid/loss_action_pred'] += float(loss_action)
            metrics['valid/loss_value_pred'] += float(self.value_weight * loss_value)
            metrics['valid/loss_td'] += float(self.td_weight * loss_td)
            metrics['valid/loss_total'] += float(loss)

        for k in metrics.keys():
            metrics[k] /= num_batches
        return metrics

    def update(self, data_loader):
        metrics = defaultdict(lambda: 0)
        num_batches = len(data_loader)

        for batch_idx, batch in enumerate(data_loader):
            input_param, state, action, next_state, reward, discount, value = batch

            if self.use_clipped_noise:
                self.rng_key, noise_key = jax.random.split(self.rng_key)
                input_param_noise = jnp.clip(
                    jax.random.normal(noise_key, input_param.shape),
                    -self.noise_clip, self.noise_clip
                )
                input_param = input_param + input_param_noise

            def loss_fn(params):
                task_emb, predicted_action, predicted_value = self.rl_module.apply(
                    {'params': params}, input_param, state, action
                )

                loss_action = jnp.mean((predicted_action - action)**2)
                loss_value = jnp.mean((predicted_value - value)**2)
                total_loss = loss_action + self.value_weight * loss_value

                aux = {
                    'loss_action': loss_action,
                    'loss_value': loss_value,
                }

                if self.use_td_error:
                    loss_td = self._get_td_error(
                        params, task_emb, next_state, reward, discount, value
                    )
                    total_loss = total_loss + self.td_weight * loss_td
                    aux['loss_td'] = loss_td

                return total_loss, aux

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.rl_state.params)
            self.rl_state = self.rl_state.apply_gradients(grads=grads)

            metrics['train/loss_action_pred'] += float(aux['loss_action'])
            metrics['train/loss_value_pred'] += float(self.value_weight * aux['loss_value'])
            metrics['train/loss_total'] += float(loss)
            if self.use_td_error:
                metrics['train/loss_td'] += float(self.td_weight * aux['loss_td'])

        for k in metrics.keys():
            metrics[k] /= num_batches
        return metrics

    def save(self, model_dir, name):
        model_save_dir = Path(f'{model_dir}/step_{str(name).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        with open(f'{model_save_dir}/rl_net.pkl', 'wb') as f:
            pickle.dump(jax.device_get(self.rl_state.params), f)

    def load(self, model_dir, name):
        print(f"Loading the model from {model_dir}, name: {name}")
        model_load_dir = Path(f'{model_dir}/step_{str(name).zfill(8)}')

        with open(f'{model_load_dir}/rl_net.pkl', 'rb') as f:
            params = pickle.load(f)
            self.rl_state = self.rl_state.replace(
                params=jax.device_put(params)
            )


class MetaRLApproximator(RLApproximator):
    """
    Approximates a family of near-optimal policies. Uses MAML.
    DEFERRED: requires learn2learn dependency.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MetaRLApproximator (MAML) is deferred to a later phase. "
            "This requires the learn2learn dependency which is not yet converted to JAX."
        )
