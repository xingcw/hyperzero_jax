"""
Tests for approximators/policy.py and approximators/rl_solution.py.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from approximators.policy import PolicyApproximator, MetaPolicyApproximator
from approximators.rl_solution import RLApproximator, MetaRLApproximator
from utils.dataloader import FastTensorDataLoader


def _make_synthetic_data(n=200, input_dim=4, state_dim=8, action_dim=4):
    """Create synthetic training data."""
    np.random.seed(42)
    input_param = np.random.randn(n, input_dim).astype(np.float32)
    state = np.random.randn(n, state_dim).astype(np.float32)
    action = np.tanh(np.random.randn(n, action_dim)).astype(np.float32)
    next_state = np.random.randn(n, state_dim).astype(np.float32)
    reward = np.random.randn(n, 1).astype(np.float32)
    discount = np.ones((n, 1), dtype=np.float32) * 0.99
    value = np.random.randn(n, 1).astype(np.float32)
    return input_param, state, action, next_state, reward, discount, value


class TestPolicyApproximator:
    def test_forward(self):
        """act() produces valid actions."""
        approx = PolicyApproximator(
            model='mlp', input_dim=4, state_dim=8, action_dim=4,
            device='cpu', lr=1e-3, embed_dim=64, hidden_dim=32,
            noise_clip=0.5, use_clipped_noise=False
        )
        obs = np.random.randn(8).astype(np.float32)
        input_param = np.random.randn(4).astype(np.float32)
        action = approx.act(input_param, obs)
        assert action.shape == (4,)
        assert action.dtype == np.float32
        # tanh output should be in [-1, 1]
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_forward_hyper(self):
        """act() with hyper model."""
        approx = PolicyApproximator(
            model='hyper', input_dim=4, state_dim=8, action_dim=4,
            device='cpu', lr=1e-3, embed_dim=64, hidden_dim=32,
            noise_clip=0.5, use_clipped_noise=False
        )
        obs = np.random.randn(8).astype(np.float32)
        input_param = np.random.randn(4).astype(np.float32)
        action = approx.act(input_param, obs)
        assert action.shape == (4,)

    def test_update_loss(self):
        """One update produces finite loss."""
        data = _make_synthetic_data()
        loader = FastTensorDataLoader(*data, batch_size=64)

        approx = PolicyApproximator(
            model='mlp', input_dim=4, state_dim=8, action_dim=4,
            device='cpu', lr=1e-3, embed_dim=64, hidden_dim=32,
            noise_clip=0.5, use_clipped_noise=False
        )

        metrics = approx.update(loader)
        assert 'train/loss_total' in metrics
        assert np.isfinite(metrics['train/loss_total'])

    def test_meta_not_implemented(self):
        """MetaPolicyApproximator raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            MetaPolicyApproximator(
                model='mlp', input_dim=4, state_dim=8, action_dim=4,
                device='cpu', lr=1e-3, fast_lr=1e-2, embed_dim=64,
                hidden_dim=32, noise_clip=0.5, use_clipped_noise=False,
                adaptation_steps=5, use_pearl=False, kl_lambda=0.1
            )


class TestRLApproximator:
    def test_forward(self):
        """act() and q() produce valid outputs."""
        approx = RLApproximator(
            model='mlp', input_dim=4, state_dim=8, action_dim=4,
            device='cpu', lr=1e-3, embed_dim=64, hidden_dim=32,
            noise_clip=0.5, use_clipped_noise=False,
            use_td=True, td_weight=1.0, value_weight=1.0
        )
        obs = np.random.randn(8).astype(np.float32)
        input_param = np.random.randn(4).astype(np.float32)

        action = approx.act(input_param, obs)
        assert action.shape == (4,)

        q_val = approx.q(input_param, obs, action)
        assert q_val.shape == (1,)
        assert np.isfinite(q_val).all()

    def test_forward_hyper(self):
        """act() with hyper model."""
        approx = RLApproximator(
            model='hyper', input_dim=4, state_dim=8, action_dim=4,
            device='cpu', lr=1e-3, embed_dim=64, hidden_dim=32,
            noise_clip=0.5, use_clipped_noise=False,
            use_td=False, td_weight=1.0, value_weight=1.0
        )
        obs = np.random.randn(8).astype(np.float32)
        input_param = np.random.randn(4).astype(np.float32)
        action = approx.act(input_param, obs)
        assert action.shape == (4,)

    def test_update_loss(self):
        """One update: composite loss is finite."""
        data = _make_synthetic_data()
        loader = FastTensorDataLoader(*data, batch_size=64)

        approx = RLApproximator(
            model='mlp', input_dim=4, state_dim=8, action_dim=4,
            device='cpu', lr=1e-3, embed_dim=64, hidden_dim=32,
            noise_clip=0.5, use_clipped_noise=False,
            use_td=True, td_weight=1.0, value_weight=1.0
        )

        metrics = approx.update(loader)
        assert 'train/loss_total' in metrics
        assert 'train/loss_action_pred' in metrics
        assert 'train/loss_td' in metrics
        assert np.isfinite(metrics['train/loss_total'])

    def test_td_error(self):
        """TD error computation runs without error."""
        data = _make_synthetic_data(n=32)
        input_param, state, action, next_state, reward, discount, value = [
            jnp.asarray(d) for d in data
        ]

        approx = RLApproximator(
            model='mlp', input_dim=4, state_dim=8, action_dim=4,
            device='cpu', lr=1e-3, embed_dim=64, hidden_dim=32,
            noise_clip=0.5, use_clipped_noise=False,
            use_td=True, td_weight=1.0, value_weight=1.0
        )

        task_emb = approx.rl_module.apply(
            {'params': approx.rl_state.params}, input_param,
            method=approx.rl_module.embed_task
        )
        td_err = approx._get_td_error(
            approx.rl_state.params, task_emb, next_state, reward, discount, value
        )
        assert np.isfinite(float(td_err))

    def test_meta_not_implemented(self):
        """MetaRLApproximator raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            MetaRLApproximator(
                model='mlp', input_dim=4, state_dim=8, action_dim=4,
                device='cpu', lr=1e-3, fast_lr=1e-2, embed_dim=64,
                hidden_dim=32, noise_clip=0.5, use_clipped_noise=False,
                use_td=True, td_weight=1.0, value_weight=1.0,
                adaptation_steps=5
            )
