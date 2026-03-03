"""
Tests for utils/dataloader.py.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from utils.dataloader import FastTensorDataLoader, FastTensorMetaDataLoader


class TestFastTensorDataLoader:
    def test_batch_sizes(self):
        """Correct iteration count."""
        n, d = 100, 8
        data = np.random.randn(n, d).astype(np.float32)
        labels = np.random.randn(n, 1).astype(np.float32)

        loader = FastTensorDataLoader(data, labels, batch_size=32)
        batches = list(loader)
        # ceil(100/32) = 4 batches
        assert len(batches) == 4
        assert len(loader) == 4

        # Check total samples
        total = sum(b[0].shape[0] for b in batches)
        assert total == n

    def test_shuffle_deterministic(self):
        """Same key = same order."""
        n, d = 50, 4
        data = np.random.randn(n, d).astype(np.float32)
        key = jax.random.PRNGKey(42)

        loader1 = FastTensorDataLoader(data, batch_size=10, shuffle=True,
                                        rng_key=jax.random.PRNGKey(42))
        loader2 = FastTensorDataLoader(data, batch_size=10, shuffle=True,
                                        rng_key=jax.random.PRNGKey(42))

        for b1, b2 in zip(loader1, loader2):
            np.testing.assert_array_equal(np.array(b1[0]), np.array(b2[0]))

    def test_output_types(self):
        """Outputs are jnp.ndarray."""
        n, d = 20, 4
        data = np.random.randn(n, d).astype(np.float32)
        loader = FastTensorDataLoader(data, batch_size=10)
        batch = next(iter(loader))
        assert isinstance(batch[0], jnp.ndarray)


class TestFastTensorMetaDataLoader:
    def test_task_sampling(self):
        """Test sampling from specific tasks."""
        n_tasks, n_samples, d = 5, 100, 8
        data = np.random.randn(n_tasks, n_samples, d).astype(np.float32)
        labels = np.random.randn(n_tasks, n_samples, 1).astype(np.float32)

        loader = FastTensorMetaDataLoader(data, labels, batch_size=32, shuffle=True)
        assert loader.n_tasks == n_tasks

        loader.shuffle_indices()
        batch = loader.sample(0)
        assert batch[0].shape == (32, d)
        assert batch[1].shape == (32, 1)

    def test_wraps_around(self):
        """After exhausting data, re-shuffles and resets."""
        n_tasks, n_samples, d = 3, 50, 4
        data = np.random.randn(n_tasks, n_samples, d).astype(np.float32)

        loader = FastTensorMetaDataLoader(data, batch_size=30, shuffle=True)
        loader.shuffle_indices()

        # First sample: 30 items
        batch1 = loader.sample(0)
        assert batch1[0].shape[0] == 30

        # Second sample: 20 items (remainder)
        batch2 = loader.sample(0)
        assert batch2[0].shape[0] == 20

        # Third sample: should wrap around
        batch3 = loader.sample(0)
        assert batch3[0].shape[0] == 30
