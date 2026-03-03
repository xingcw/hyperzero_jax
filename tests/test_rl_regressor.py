"""
Equivalence tests for models/rl_regressor.py.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

import torch

# Import JAX versions
from models.rl_regressor import (
    HyperPolicy as JaxHyperPolicy,
    HyperRLSolution as JaxHyperRLSolution,
    MLPActionPredictor as JaxMLPActionPredictor,
    MLPRLSolution as JaxMLPRLSolution,
    MLPContextEncoder as JaxMLPContextEncoder,
)

# Import PyTorch originals
from _original.models.rl_regressor import (
    HyperPolicy as TorchHyperPolicy,
    HyperRLSolution as TorchHyperRLSolution,
    MLPActionPredictor as TorchMLPActionPredictor,
    MLPRLSolution as TorchMLPRLSolution,
    MLPContextEncoder as TorchMLPContextEncoder,
)

from tests.conftest import (
    transfer_hyper_policy,
    transfer_hyper_rl_solution,
    transfer_mlp_action_predictor,
    transfer_mlp_rl_solution,
    transfer_mlp_context_encoder,
    RTOL, ATOL, RTOL_RELAXED, ATOL_RELAXED,
)


class TestHyperPolicy:
    def test_equivalence(self):
        input_param_dim, state_dim, action_dim = 4, 8, 4
        embed_dim, hidden_dim = 64, 32
        batch_size = 8
        np.random.seed(42)
        param_np = np.random.randn(batch_size, input_param_dim).astype(np.float32)
        state_np = np.random.randn(batch_size, state_dim).astype(np.float32)

        # PyTorch
        torch_model = TorchHyperPolicy(input_param_dim, state_dim, action_dim, embed_dim, hidden_dim)
        torch_model.eval()
        with torch.no_grad():
            torch_out = torch_model(torch.tensor(param_np), torch.tensor(state_np)).numpy()

        # JAX
        jax_model = JaxHyperPolicy(
            input_param_dim=input_param_dim, state_dim=state_dim,
            action_dim=action_dim, embed_dim=embed_dim, hidden_dim=hidden_dim
        )
        jax_params = transfer_hyper_policy(torch_model)
        jax_out = jax_model.apply(jax_params, jnp.array(param_np), jnp.array(state_np))

        np.testing.assert_allclose(np.array(jax_out), torch_out,
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestHyperRLSolution:
    def _create_models(self):
        input_param_dim, state_dim, action_dim = 4, 8, 4
        embed_dim, hidden_dim = 64, 32
        batch_size = 8
        np.random.seed(42)
        param_np = np.random.randn(batch_size, input_param_dim).astype(np.float32)
        state_np = np.random.randn(batch_size, state_dim).astype(np.float32)
        action_np = np.random.randn(batch_size, action_dim).astype(np.float32)

        # PyTorch
        torch_model = TorchHyperRLSolution(input_param_dim, state_dim, action_dim, embed_dim, hidden_dim)
        torch_model.eval()

        # JAX
        jax_model = JaxHyperRLSolution(
            input_param_dim=input_param_dim, state_dim=state_dim,
            action_dim=action_dim, embed_dim=embed_dim, hidden_dim=hidden_dim
        )
        jax_params = transfer_hyper_rl_solution(torch_model)

        return torch_model, jax_model, jax_params, param_np, state_np, action_np

    def test_equivalence(self):
        """z, pred_action, q_value match."""
        torch_model, jax_model, jax_params, param_np, state_np, action_np = self._create_models()

        with torch.no_grad():
            torch_z, torch_action, torch_q = torch_model(
                torch.tensor(param_np), torch.tensor(state_np), torch.tensor(action_np)
            )

        jax_z, jax_action, jax_q = jax_model.apply(
            jax_params, jnp.array(param_np), jnp.array(state_np), jnp.array(action_np)
        )

        np.testing.assert_allclose(np.array(jax_z), torch_z.numpy(),
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        np.testing.assert_allclose(np.array(jax_action), torch_action.numpy(),
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        np.testing.assert_allclose(np.array(jax_q), torch_q.numpy(),
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_embed_predict_action(self):
        """embed_task + predict_action pipeline."""
        torch_model, jax_model, jax_params, param_np, state_np, _ = self._create_models()

        with torch.no_grad():
            torch_z = torch_model.embed_task(torch.tensor(param_np))
            torch_action = torch_model.predict_action(torch_z, torch.tensor(state_np)).numpy()

        jax_z = jax_model.apply(jax_params, jnp.array(param_np), method=jax_model.embed_task)
        jax_action = jax_model.apply(jax_params, jax_z, jnp.array(state_np),
                                      method=jax_model.predict_action)

        np.testing.assert_allclose(np.array(jax_action), torch_action,
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_predict_q_value(self):
        torch_model, jax_model, jax_params, param_np, state_np, action_np = self._create_models()

        with torch.no_grad():
            torch_z = torch_model.embed_task(torch.tensor(param_np))
            torch_q = torch_model.predict_q_value(
                torch_z, torch.tensor(state_np), torch.tensor(action_np)
            ).numpy()

        jax_z = jax_model.apply(jax_params, jnp.array(param_np), method=jax_model.embed_task)
        jax_q = jax_model.apply(jax_params, jax_z, jnp.array(state_np), jnp.array(action_np),
                                 method=jax_model.predict_q_value)

        np.testing.assert_allclose(np.array(jax_q), torch_q,
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestMLPActionPredictor:
    def test_equivalence(self):
        input_param_dim, state_dim, action_dim = 4, 8, 4
        embed_dim, hidden_dim = 64, 32
        batch_size = 8
        np.random.seed(42)
        param_np = np.random.randn(batch_size, input_param_dim).astype(np.float32)
        state_np = np.random.randn(batch_size, state_dim).astype(np.float32)

        # PyTorch
        torch_model = TorchMLPActionPredictor(input_param_dim, state_dim, action_dim, embed_dim, hidden_dim)
        torch_model.eval()
        with torch.no_grad():
            torch_out = torch_model(torch.tensor(param_np), torch.tensor(state_np)).numpy()

        # JAX
        jax_model = JaxMLPActionPredictor(
            input_param_dim=input_param_dim, state_dim=state_dim,
            action_dim=action_dim, embed_dim=embed_dim, hidden_dim=hidden_dim
        )
        jax_params = transfer_mlp_action_predictor(torch_model)
        jax_out = jax_model.apply(jax_params, jnp.array(param_np), jnp.array(state_np))

        np.testing.assert_allclose(np.array(jax_out), torch_out,
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestMLPRLSolution:
    def test_equivalence(self):
        input_param_dim, state_dim, action_dim = 4, 8, 4
        embed_dim, hidden_dim = 64, 32
        batch_size = 8
        np.random.seed(42)
        param_np = np.random.randn(batch_size, input_param_dim).astype(np.float32)
        state_np = np.random.randn(batch_size, state_dim).astype(np.float32)
        action_np = np.random.randn(batch_size, action_dim).astype(np.float32)

        # PyTorch
        torch_model = TorchMLPRLSolution(input_param_dim, state_dim, action_dim, embed_dim, hidden_dim)
        torch_model.eval()
        with torch.no_grad():
            torch_task, torch_action, torch_q = torch_model(
                torch.tensor(param_np), torch.tensor(state_np), torch.tensor(action_np)
            )

        # JAX
        jax_model = JaxMLPRLSolution(
            input_param_dim=input_param_dim, state_dim=state_dim,
            action_dim=action_dim, embed_dim=embed_dim, hidden_dim=hidden_dim
        )
        jax_params = transfer_mlp_rl_solution(torch_model)
        jax_task, jax_action, jax_q = jax_model.apply(
            jax_params, jnp.array(param_np), jnp.array(state_np), jnp.array(action_np)
        )

        np.testing.assert_allclose(np.array(jax_task), torch_task.numpy(),
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        np.testing.assert_allclose(np.array(jax_action), torch_action.numpy(),
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        np.testing.assert_allclose(np.array(jax_q), torch_q.numpy(),
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestMLPContextEncoder:
    def test_deterministic(self):
        """encode() mu/log_var match."""
        state_dim, action_dim = 8, 4
        embed_dim, hidden_dim = 32, 64
        batch_size = 8
        np.random.seed(42)
        state_np = np.random.randn(batch_size, state_dim).astype(np.float32)
        action_np = np.random.randn(batch_size, action_dim).astype(np.float32)

        # PyTorch
        torch_model = TorchMLPContextEncoder(state_dim, action_dim, embed_dim, hidden_dim)
        torch_model.eval()
        with torch.no_grad():
            torch_mu, torch_log_var = torch_model.encode(
                torch.tensor(state_np), torch.tensor(action_np)
            )

        # JAX
        jax_model = JaxMLPContextEncoder(
            state_dim=state_dim, action_dim=action_dim,
            embed_dim=embed_dim, hidden_dim=hidden_dim
        )
        jax_params = transfer_mlp_context_encoder(torch_model)
        jax_mu, jax_log_var = jax_model.apply(
            jax_params, jnp.array(state_np), jnp.array(action_np),
            method=jax_model.encode
        )

        np.testing.assert_allclose(np.array(jax_mu), torch_mu.numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.array(jax_log_var), torch_log_var.numpy(), rtol=RTOL, atol=ATOL)
