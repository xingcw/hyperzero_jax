"""
Tests for utils/utils.py.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from utils.utils import soft_update_params, schedule, set_seed_everywhere


class TestSoftUpdateParams:
    def test_soft_update(self):
        """JAX pytree soft update matches manual computation."""
        params = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0, 4.0])}
        target_params = {'a': jnp.array([5.0, 6.0]), 'b': jnp.array([7.0, 8.0])}
        tau = 0.05

        result = soft_update_params(params, target_params, tau)

        expected_a = tau * np.array([1.0, 2.0]) + (1 - tau) * np.array([5.0, 6.0])
        expected_b = tau * np.array([3.0, 4.0]) + (1 - tau) * np.array([7.0, 8.0])

        np.testing.assert_allclose(np.array(result['a']), expected_a, rtol=1e-6)
        np.testing.assert_allclose(np.array(result['b']), expected_b, rtol=1e-6)

    def test_nested_pytree(self):
        """Soft update works on nested pytrees."""
        params = {'layer': {'kernel': jnp.ones((2, 3)), 'bias': jnp.zeros(3)}}
        target = {'layer': {'kernel': jnp.zeros((2, 3)), 'bias': jnp.ones(3)}}
        tau = 0.1

        result = soft_update_params(params, target, tau)
        np.testing.assert_allclose(np.array(result['layer']['kernel']),
                                   0.1 * np.ones((2, 3)), rtol=1e-6)
        np.testing.assert_allclose(np.array(result['layer']['bias']),
                                   0.9 * np.ones(3), rtol=1e-6)


class TestSchedule:
    def test_constant(self):
        assert schedule('0.1', 0) == 0.1
        assert schedule('0.1', 100) == 0.1

    def test_linear(self):
        val = schedule('linear(1.0,0.0,100)', 50)
        np.testing.assert_allclose(val, 0.5, rtol=1e-6)

    def test_linear_clamped(self):
        val = schedule('linear(1.0,0.0,100)', 200)
        np.testing.assert_allclose(val, 0.0, rtol=1e-6)


class TestSetSeedEverywhere:
    def test_returns_prng_key(self):
        key = set_seed_everywhere(42)
        assert key.shape == (2,) or key.shape == ()  # JAX key format
        # Verify it's a valid key by splitting
        k1, k2 = jax.random.split(key)
        assert k1 is not None
