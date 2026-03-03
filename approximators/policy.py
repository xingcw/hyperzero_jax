import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from models.rl_regressor import MLPActionPredictor, HyperPolicy
from utils.dataloader import FastTensorDataLoader


class PolicyApproximator:
    """
    Approximates a family of near-optimal policies.
    Uses either a conditional MLP or a hypernetwork.
    """
    def __init__(self, model, input_dim, state_dim,
                 action_dim, device, lr, embed_dim,
                 hidden_dim, noise_clip, use_clipped_noise):
        self.lr = lr
        self.use_clipped_noise = use_clipped_noise
        self.noise_clip = noise_clip

        # model
        if model == 'mlp':
            self.policy_module = MLPActionPredictor(
                input_param_dim=input_dim, state_dim=state_dim,
                action_dim=action_dim, embed_dim=embed_dim, hidden_dim=hidden_dim
            )
        elif model == 'hyper':
            self.policy_module = HyperPolicy(
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
        params = self.policy_module.init(init_key, dummy_param, dummy_state)['params']

        self.policy_state = TrainState.create(
            apply_fn=self.policy_module.apply,
            params=params,
            tx=optax.adam(lr),
        )

    def act(self, input_param, obs):
        input_param = jnp.asarray(input_param, dtype=jnp.float32)[None]
        obs = jnp.asarray(obs, dtype=jnp.float32)[None]
        action = self.policy_module.apply(
            {'params': self.policy_state.params}, input_param, obs
        )
        return np.asarray(action[0]).astype(np.float32)

    def eval(self, data_loader):
        metrics = defaultdict(lambda: 0)
        num_batches = len(data_loader)

        for batch_idx, batch in enumerate(data_loader):
            input_param, state, action, next_state, reward, discount, value = batch

            predicted_action = self.policy_module.apply(
                {'params': self.policy_state.params}, input_param, state
            )
            loss = jnp.mean((predicted_action - action)**2)

            metrics['valid/loss_action_pred'] += float(loss)
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
                predicted_action = self.policy_module.apply(
                    {'params': params}, input_param, state
                )
                return jnp.mean((predicted_action - action)**2)

            loss, grads = jax.value_and_grad(loss_fn)(self.policy_state.params)
            self.policy_state = self.policy_state.apply_gradients(grads=grads)

            metrics['train/loss_action_pred'] += float(loss)
            metrics['train/loss_total'] += float(loss)

        for k in metrics.keys():
            metrics[k] /= num_batches
        return metrics

    def save(self, model_dir, name):
        model_save_dir = Path(f'{model_dir}/step_{str(name).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        with open(f'{model_save_dir}/policy.pkl', 'wb') as f:
            pickle.dump(jax.device_get(self.policy_state.params), f)

    def load(self, model_dir, name):
        print(f"Loading the model from {model_dir}, name: {name}")
        model_load_dir = Path(f'{model_dir}/step_{str(name).zfill(8)}')

        with open(f'{model_load_dir}/policy.pkl', 'rb') as f:
            params = pickle.load(f)
            self.policy_state = self.policy_state.replace(
                params=jax.device_put(params)
            )


class MetaPolicyApproximator(PolicyApproximator):
    """
    Approximates a family of near-optimal policies.
    Uses either MAML or PEARL.
    DEFERRED: requires learn2learn dependency.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MetaPolicyApproximator (MAML/PEARL) is deferred to a later phase. "
            "This requires the learn2learn dependency which is not yet converted to JAX."
        )
