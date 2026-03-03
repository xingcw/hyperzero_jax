"""
Tests for training scripts: smoke tests and loss decrease.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from approximators.policy import PolicyApproximator
from approximators.rl_solution import RLApproximator
from utils.dataloader import FastTensorDataLoader


def _make_synthetic_data(n=200, input_dim=4, state_dim=8, action_dim=4):
    np.random.seed(42)
    input_param = np.random.randn(n, input_dim).astype(np.float32)
    state = np.random.randn(n, state_dim).astype(np.float32)
    action = np.tanh(np.random.randn(n, action_dim)).astype(np.float32)
    next_state = np.random.randn(n, state_dim).astype(np.float32)
    reward = np.random.randn(n, 1).astype(np.float32)
    discount = np.ones((n, 1), dtype=np.float32) * 0.99
    value = np.random.randn(n, 1).astype(np.float32)
    return input_param, state, action, next_state, reward, discount, value


class TestPolicyTraining:
    def test_loss_decreases(self):
        """Loss decreases over 10 epochs on synthetic data."""
        data = _make_synthetic_data(n=200)
        loader = FastTensorDataLoader(*data, batch_size=64, shuffle=True)

        approx = PolicyApproximator(
            model='mlp', input_dim=4, state_dim=8, action_dim=4,
            device='cpu', lr=1e-3, embed_dim=64, hidden_dim=32,
            noise_clip=0.5, use_clipped_noise=False
        )

        losses = []
        for epoch in range(10):
            metrics = approx.update(loader)
            losses.append(metrics['train/loss_total'])

        # Loss should generally decrease
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"


class TestRLApproximatorTraining:
    def test_loss_decreases(self):
        """Loss decreases over 10 epochs on synthetic data."""
        data = _make_synthetic_data(n=200)
        loader = FastTensorDataLoader(*data, batch_size=64, shuffle=True)

        approx = RLApproximator(
            model='mlp', input_dim=4, state_dim=8, action_dim=4,
            device='cpu', lr=1e-3, embed_dim=64, hidden_dim=32,
            noise_clip=0.5, use_clipped_noise=False,
            use_td=False, td_weight=1.0, value_weight=1.0
        )

        losses = []
        for epoch in range(10):
            metrics = approx.update(loader)
            losses.append(metrics['train/loss_total'])

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"


class TestHyperTraining:
    def test_hyper_policy_loss_decreases(self):
        """HyperPolicy loss decreases."""
        data = _make_synthetic_data(n=200)
        loader = FastTensorDataLoader(*data, batch_size=64, shuffle=True)

        approx = PolicyApproximator(
            model='hyper', input_dim=4, state_dim=8, action_dim=4,
            device='cpu', lr=1e-3, embed_dim=64, hidden_dim=32,
            noise_clip=0.5, use_clipped_noise=False
        )

        losses = []
        for epoch in range(10):
            metrics = approx.update(loader)
            losses.append(metrics['train/loss_total'])

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
