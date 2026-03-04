"""
Equivalence tests for models/core.py: DeterministicActor, Critic, gaussian_logprob, squash.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
# Enable float64 so PyTorch/JAX equivalence tests run in full precision
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import torch
import torch.nn as nn

# Import JAX versions
from models.core import (
    DeterministicActor as JaxActor,
    Critic as JaxCritic,
    gaussian_logprob as jax_gaussian_logprob,
    squash as jax_squash,
)

# Import PyTorch originals
from _original.models.core import (
    DeterministicActor as TorchActor,
    Critic as TorchCritic,
    gaussian_logprob as torch_gaussian_logprob,
    squash as torch_squash,
)

from tests.conftest import (
    transfer_deterministic_actor,
    transfer_critic,
    RTOL, ATOL,
)


class TestDeterministicActor:
    def test_equivalence(self):
        """Same weights + input -> same output (float64 for numerical stability)."""
        feature_dim, action_dim, hidden_dim = 8, 4, 64
        batch_size = 16
        np.random.seed(42)
        input_np = np.random.randn(batch_size, feature_dim).astype(np.float64)

        # PyTorch (float64)
        torch_model = TorchActor(feature_dim, action_dim, hidden_dim)
        torch_model.double()
        torch_model.eval()
        with torch.no_grad():
            torch_out = torch_model(torch.tensor(input_np, dtype=torch.float64)).numpy()

        # JAX (float64; transfer gives float64 params from .double() model)
        jax_model = JaxActor(action_dim=action_dim, hidden_dim=hidden_dim)
        jax_params = transfer_deterministic_actor(torch_model)
        jax_out = jax_model.apply(jax_params, jnp.array(input_np, dtype=jnp.float64))

        np.testing.assert_allclose(np.array(jax_out), torch_out, rtol=RTOL, atol=ATOL)

    def test_output_range(self):
        """Output should be in [-1, 1] due to tanh."""
        feature_dim, action_dim, hidden_dim = 8, 4, 64
        rng = jax.random.PRNGKey(0)
        jax_model = JaxActor(action_dim=action_dim, hidden_dim=hidden_dim)
        params = jax_model.init(rng, jnp.ones((1, feature_dim)))
        input_data = jax.random.normal(rng, (32, feature_dim))
        out = jax_model.apply(params, input_data)
        assert jnp.all(out >= -1.0) and jnp.all(out <= 1.0)


class TestCritic:
    def test_equivalence(self):
        """Q1 and Q2 match (float64 for numerical stability)."""
        feature_dim, action_dim, hidden_dim = 8, 4, 64
        batch_size = 16
        np.random.seed(42)
        state_np = np.random.randn(batch_size, feature_dim).astype(np.float64)
        action_np = np.random.randn(batch_size, action_dim).astype(np.float64)

        # PyTorch (float64)
        torch_model = TorchCritic(feature_dim, action_dim, hidden_dim)
        torch_model.double()
        torch_model.eval()
        with torch.no_grad():
            torch_q1, torch_q2 = torch_model(
                torch.tensor(state_np, dtype=torch.float64),
                torch.tensor(action_np, dtype=torch.float64),
            )
            torch_q1 = torch_q1.numpy()
            torch_q2 = torch_q2.numpy()

        # JAX (float64)
        jax_model = JaxCritic(hidden_dim=hidden_dim)
        jax_params = transfer_critic(torch_model)
        jax_q1, jax_q2 = jax_model.apply(
            jax_params,
            jnp.array(state_np, dtype=jnp.float64),
            jnp.array(action_np, dtype=jnp.float64),
        )

        np.testing.assert_allclose(np.array(jax_q1), torch_q1, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.array(jax_q2), torch_q2, rtol=RTOL, atol=ATOL)

    def test_q1_only(self):
        """Standalone Q1 path matches (float64 for numerical stability)."""
        feature_dim, action_dim, hidden_dim = 8, 4, 64
        batch_size = 16
        np.random.seed(42)
        state_np = np.random.randn(batch_size, feature_dim).astype(np.float64)
        action_np = np.random.randn(batch_size, action_dim).astype(np.float64)

        # PyTorch (float64)
        torch_model = TorchCritic(feature_dim, action_dim, hidden_dim)
        torch_model.double()
        torch_model.eval()
        with torch.no_grad():
            torch_q1 = torch_model.Q1(
                torch.tensor(state_np, dtype=torch.float64),
                torch.tensor(action_np, dtype=torch.float64),
            ).numpy()

        # JAX (float64)
        jax_model = JaxCritic(hidden_dim=hidden_dim)
        jax_params = transfer_critic(torch_model)
        jax_q1 = jax_model.apply(
            jax_params,
            jnp.array(state_np, dtype=jnp.float64),
            jnp.array(action_np, dtype=jnp.float64),
            method=jax_model.Q1,
        )

        np.testing.assert_allclose(np.array(jax_q1), torch_q1, rtol=RTOL, atol=ATOL)


class TestGaussianLogprob:
    def test_equivalence(self):
        """Float64 for numerical stability."""
        np.random.seed(42)
        noise_np = np.random.randn(16, 4).astype(np.float64)
        log_std_np = np.random.randn(16, 4).astype(np.float64)

        # PyTorch
        torch_result = torch_gaussian_logprob(
            torch.tensor(noise_np, dtype=torch.float64),
            torch.tensor(log_std_np, dtype=torch.float64),
        ).numpy()

        # JAX
        jax_result = jax_gaussian_logprob(
            jnp.array(noise_np, dtype=jnp.float64),
            jnp.array(log_std_np, dtype=jnp.float64),
        )

        np.testing.assert_allclose(np.array(jax_result), torch_result, rtol=RTOL, atol=ATOL)


class TestSquash:
    def test_equivalence(self):
        """Float64 for numerical stability."""
        np.random.seed(42)
        mu_np = np.random.randn(16, 4).astype(np.float64)
        pi_np = np.random.randn(16, 4).astype(np.float64) * 0.5  # keep moderate for numerical stability
        log_pi_np = np.random.randn(16, 1).astype(np.float64)

        # PyTorch
        torch_mu, torch_pi, torch_log_pi = torch_squash(
            torch.tensor(mu_np, dtype=torch.float64),
            torch.tensor(pi_np, dtype=torch.float64),
            torch.tensor(log_pi_np, dtype=torch.float64),
        )

        # JAX
        jax_mu, jax_pi, jax_log_pi = jax_squash(
            jnp.array(mu_np, dtype=jnp.float64),
            jnp.array(pi_np, dtype=jnp.float64),
            jnp.array(log_pi_np, dtype=jnp.float64),
        )

        np.testing.assert_allclose(np.array(jax_mu), torch_mu.numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.array(jax_pi), torch_pi.numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            np.array(jax_log_pi), torch_log_pi.detach().numpy(), rtol=RTOL, atol=ATOL
        )
