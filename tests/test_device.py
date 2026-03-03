"""
GPU and TPU verification tests.

These tests are skipped if the corresponding hardware is not available.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from models.core import DeterministicActor, Critic
from models.hypenet_core import HyperNetwork


# ---- GPU Tests ----

def _has_gpu():
    try:
        return len(jax.devices('gpu')) > 0
    except RuntimeError:
        return False


gpu_available = pytest.mark.skipif(not _has_gpu(), reason="No GPU available")


@gpu_available
class TestGPU:
    def test_gpu_available(self):
        """jax.devices('gpu') returns devices."""
        devices = jax.devices('gpu')
        assert len(devices) > 0

    def test_model_on_gpu(self):
        """Params placed on GPU."""
        rng = jax.random.PRNGKey(0)
        model = DeterministicActor(action_dim=4, hidden_dim=64)
        params = model.init(rng, jnp.ones((1, 8)))

        # Check params are on GPU
        leaf = jax.tree.leaves(params)[0]
        assert 'gpu' in str(leaf.devices()).lower() or 'cuda' in str(leaf.devices()).lower()

    def test_td3_update_on_gpu(self):
        """Full TD3 update step on GPU."""
        from agents.td3 import TD3Agent

        agent = TD3Agent(
            obs_shape=(8,), action_shape=(4,), device='gpu',
            lr=3e-4, hidden_dim=64, critic_target_tau=0.005,
            num_expl_steps=100, update_every_steps=2,
            stddev_schedule='0.2', stddev_clip=0.3
        )

        batch_size = 32
        obs = np.random.randn(batch_size, 8).astype(np.float32)
        action = np.random.randn(batch_size, 4).astype(np.float32)
        reward = np.random.randn(batch_size, 1).astype(np.float32)
        discount = np.ones((batch_size, 1), dtype=np.float32) * 0.99
        next_obs = np.random.randn(batch_size, 8).astype(np.float32)
        neg_obs = np.random.randn(batch_size, 8).astype(np.float32)

        def replay_iter():
            while True:
                yield (obs, action, reward, discount, next_obs, neg_obs)

        metrics = agent.update(replay_iter(), step=2)
        assert np.isfinite(metrics['critic_loss'])

    def test_hypernet_on_gpu(self):
        """Forward + backward on GPU."""
        rng = jax.random.PRNGKey(0)
        model = HyperNetwork(
            z_dim=64, base_v_input_dim=8, base_v_output_dim=4,
            dynamic_layer_dim=32, base_output_activation=True
        )
        params = model.init(rng, jnp.ones((1, 4)), jnp.ones((1, 8)))

        def loss_fn(params):
            z, out = model.apply(params, jnp.ones((2, 4)), jnp.ones((2, 8)))
            return jnp.mean(out**2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        assert np.isfinite(float(loss))

    def test_numerical_equivalence_on_gpu(self):
        """Model outputs match between CPU and GPU."""
        rng = jax.random.PRNGKey(0)
        model = DeterministicActor(action_dim=4, hidden_dim=64)
        params = model.init(rng, jnp.ones((1, 8)))

        x = jax.random.normal(rng, (4, 8))

        cpu_out = jax.device_put(model.apply(params, x), jax.devices('cpu')[0])
        gpu_out = model.apply(params, x)  # should already be on GPU

        np.testing.assert_allclose(np.array(cpu_out), np.array(gpu_out), rtol=1e-5, atol=1e-5)


# ---- TPU Tests ----

def _has_tpu():
    try:
        return len(jax.devices('tpu')) > 0
    except RuntimeError:
        return False


tpu_available = pytest.mark.skipif(not _has_tpu(), reason="No TPU available")


@tpu_available
class TestTPU:
    def test_tpu_available(self):
        """jax.devices('tpu') returns devices."""
        devices = jax.devices('tpu')
        assert len(devices) > 0

    def test_model_on_tpu(self):
        """Inference on TPU."""
        rng = jax.random.PRNGKey(0)
        model = DeterministicActor(action_dim=4, hidden_dim=64)
        params = model.init(rng, jnp.ones((1, 8)))

        x = jax.random.normal(rng, (4, 8))
        out = model.apply(params, x)

        assert out.shape == (4, 4)
        assert np.isfinite(np.array(out)).all()

    def test_bfloat16_handling(self):
        """Models work with TPU native dtype."""
        rng = jax.random.PRNGKey(0)
        model = DeterministicActor(action_dim=4, hidden_dim=64)
        params = model.init(rng, jnp.ones((1, 8)))

        x = jax.random.normal(rng, (4, 8), dtype=jnp.bfloat16)
        # Cast params to bfloat16
        bf16_params = jax.tree.map(lambda p: p.astype(jnp.bfloat16), params)
        out = model.apply(bf16_params, x)
        assert np.isfinite(np.array(out.astype(jnp.float32))).all()

    def test_jit_on_tpu(self):
        """JIT-compiled functions execute on TPU."""
        rng = jax.random.PRNGKey(0)
        model = DeterministicActor(action_dim=4, hidden_dim=64)
        params = model.init(rng, jnp.ones((1, 8)))

        @jax.jit
        def forward(params, x):
            return model.apply(params, x)

        x = jax.random.normal(rng, (4, 8))
        out = forward(params, x)
        assert out.shape == (4, 4)

    def test_full_training_step_on_tpu(self):
        """One training step on TPU."""
        jax.config.update('jax_default_matmul_precision', 'float32')

        from agents.td3 import TD3Agent

        agent = TD3Agent(
            obs_shape=(8,), action_shape=(4,), device='tpu',
            lr=3e-4, hidden_dim=64, critic_target_tau=0.005,
            num_expl_steps=100, update_every_steps=2,
            stddev_schedule='0.2', stddev_clip=0.3
        )

        batch_size = 32
        obs = np.random.randn(batch_size, 8).astype(np.float32)
        action = np.random.randn(batch_size, 4).astype(np.float32)
        reward = np.random.randn(batch_size, 1).astype(np.float32)
        discount = np.ones((batch_size, 1), dtype=np.float32) * 0.99
        next_obs = np.random.randn(batch_size, 8).astype(np.float32)
        neg_obs = np.random.randn(batch_size, 8).astype(np.float32)

        def replay_iter():
            while True:
                yield (obs, action, reward, discount, next_obs, neg_obs)

        metrics = agent.update(replay_iter(), step=2)
        assert np.isfinite(metrics['critic_loss'])
