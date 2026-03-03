"""
Tests for agents/td3.py.
"""
import sys
import os
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from agents.td3 import TD3Agent
from utils.utils import soft_update_params


class TestTD3Agent:
    def _make_agent(self):
        return TD3Agent(
            obs_shape=(8,),
            action_shape=(4,),
            device='cpu',
            lr=3e-4,
            hidden_dim=64,
            critic_target_tau=0.005,
            num_expl_steps=1000,
            update_every_steps=2,
            stddev_schedule='0.2',
            stddev_clip=0.3,
        )

    def test_act_deterministic(self):
        """Eval mode: same obs -> same action."""
        agent = self._make_agent()
        obs = np.random.randn(8).astype(np.float32)

        action1 = agent.act(obs, step=5000, eval_mode=True)
        action2 = agent.act(obs, step=5000, eval_mode=True)

        np.testing.assert_array_equal(action1, action2)
        assert action1.shape == (4,)
        assert action1.dtype == np.float32

    def test_act_exploration(self):
        """Exploration mode adds noise."""
        agent = self._make_agent()
        obs = np.random.randn(8).astype(np.float32)

        np.random.seed(42)
        action1 = agent.act(obs, step=5000, eval_mode=False)
        np.random.seed(123)
        action2 = agent.act(obs, step=5000, eval_mode=False)

        # Different seeds should give different noise
        assert not np.array_equal(action1, action2)

    def test_soft_update(self):
        """Target params match after soft update."""
        agent = self._make_agent()
        tau = agent.critic_target_tau

        old_target = jax.tree.map(jnp.copy, agent.critic_target_params)
        new_params = agent.critic_state.params

        result = soft_update_params(new_params, old_target, tau)

        # Verify the formula: tau * new + (1-tau) * old
        for key in result:
            if isinstance(result[key], dict):
                for subkey in result[key]:
                    expected = tau * np.array(new_params[key][subkey]) + \
                               (1 - tau) * np.array(old_target[key][subkey])
                    np.testing.assert_allclose(np.array(result[key][subkey]),
                                               expected, rtol=1e-6)

    def test_save_load_roundtrip(self):
        """Checkpoint save/reload preserves params."""
        agent = self._make_agent()

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir, 1000)

            # Create new agent and load
            agent2 = self._make_agent()
            agent2.load(tmpdir, 1000)

            # Compare params
            jax.tree.map(
                lambda a, b: np.testing.assert_array_equal(np.array(a), np.array(b)),
                agent.actor_state.params,
                agent2.actor_state.params
            )
            jax.tree.map(
                lambda a, b: np.testing.assert_array_equal(np.array(a), np.array(b)),
                agent.critic_state.params,
                agent2.critic_state.params
            )

    def test_single_update_step(self):
        """One full update doesn't crash and produces valid metrics."""
        agent = self._make_agent()

        # Create fake replay batch
        batch_size = 32
        obs = np.random.randn(batch_size, 8).astype(np.float32)
        action = np.random.randn(batch_size, 4).astype(np.float32)
        reward = np.random.randn(batch_size, 1).astype(np.float32)
        discount = np.ones((batch_size, 1), dtype=np.float32) * 0.99
        next_obs = np.random.randn(batch_size, 8).astype(np.float32)
        neg_obs = np.random.randn(batch_size, 8).astype(np.float32)

        # Wrap as an iterator
        def replay_iter():
            while True:
                yield (obs, action, reward, discount, next_obs, neg_obs)

        it = replay_iter()

        # Step 2 triggers both critic and actor update
        metrics = agent.update(it, step=2)

        assert 'critic_loss' in metrics
        assert 'actor_loss' in metrics
        assert 'batch_reward' in metrics
        assert np.isfinite(metrics['critic_loss'])
        assert np.isfinite(metrics['actor_loss'])
