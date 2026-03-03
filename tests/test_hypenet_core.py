"""
Equivalence tests for models/hypenet_core.py.
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
from models.hypenet_core import (
    ResBlock as JaxResBlock,
    Head as JaxHead,
    Meta_Embadding as JaxMetaEmbadding,
    HyperNetwork as JaxHyperNetwork,
    DoubleHeadedHyperNetwork as JaxDoubleHeadedHyperNetwork,
)

# Import PyTorch originals
from _original.models.hypenet_core import (
    ResBlock as TorchResBlock,
    Head as TorchHead,
    Meta_Embadding as TorchMetaEmbadding,
    HyperNetwork as TorchHyperNetwork,
    DoubleHeadedHyperNetwork as TorchDoubleHeadedHyperNetwork,
)

from tests.conftest import (
    _transfer_resblock,
    _transfer_head,
    _transfer_meta_embedding,
    transfer_hypernetwork,
    transfer_double_headed_hypernetwork,
    RTOL, ATOL, RTOL_RELAXED, ATOL_RELAXED,
)


class TestResBlock:
    def test_equivalence(self):
        dim = 32
        batch_size = 8
        np.random.seed(42)
        input_np = np.random.randn(batch_size, dim).astype(np.float32)

        # PyTorch
        torch_model = TorchResBlock(dim, dim)
        torch_model.eval()
        with torch.no_grad():
            torch_out = torch_model(torch.tensor(input_np)).numpy()

        # JAX
        jax_model = JaxResBlock(out_size=dim)
        jax_params = {'params': _transfer_resblock(torch_model)}
        jax_out = jax_model.apply(jax_params, jnp.array(input_np))

        np.testing.assert_allclose(np.array(jax_out), torch_out, rtol=RTOL, atol=ATOL)


class TestHead:
    def test_equivalence(self):
        latent_dim, output_dim_in, output_dim_out = 64, 8, 4
        batch_size = 16
        stddev = 0.05
        np.random.seed(42)
        input_np = np.random.randn(batch_size, latent_dim).astype(np.float32)

        # PyTorch
        torch_model = TorchHead(latent_dim, output_dim_in, output_dim_out, stddev)
        torch_model.eval()
        with torch.no_grad():
            torch_w, torch_b = torch_model(torch.tensor(input_np))
            torch_w = torch_w.numpy()
            torch_b = torch_b.numpy()

        # JAX
        jax_model = JaxHead(latent_dim=latent_dim,
                             output_dim_in=output_dim_in,
                             output_dim_out=output_dim_out,
                             sttdev=stddev)
        jax_params = {'params': _transfer_head(torch_model)}
        jax_w, jax_b = jax_model.apply(jax_params, jnp.array(input_np))

        np.testing.assert_allclose(np.array(jax_w), torch_w, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.array(jax_b), torch_b, rtol=RTOL, atol=ATOL)

    def test_init_distribution(self):
        """Verify uniform range of initialization."""
        latent_dim, output_dim_in, output_dim_out = 64, 8, 4
        stddev = 0.05
        rng = jax.random.PRNGKey(0)
        jax_model = JaxHead(latent_dim=latent_dim,
                             output_dim_in=output_dim_in,
                             output_dim_out=output_dim_out,
                             sttdev=stddev)
        params = jax_model.init(rng, jnp.ones((1, latent_dim)))
        w1_kernel = params['params']['W1']['kernel']
        # Should be within [-stddev, stddev]
        assert np.all(np.array(w1_kernel) >= -stddev - 1e-7)
        assert np.all(np.array(w1_kernel) <= stddev + 1e-7)


class TestMetaEmbadding:
    def test_equivalence(self):
        meta_dim, z_dim = 4, 64
        batch_size = 8
        np.random.seed(42)
        input_np = np.random.randn(batch_size, meta_dim).astype(np.float32)

        # PyTorch
        torch_model = TorchMetaEmbadding(meta_dim, z_dim)
        torch_model.eval()
        with torch.no_grad():
            torch_out = torch_model(torch.tensor(input_np)).numpy()

        # JAX
        jax_model = JaxMetaEmbadding(z_dim=z_dim)
        jax_params = {'params': _transfer_meta_embedding(torch_model)}
        jax_out = jax_model.apply(jax_params, jnp.array(input_np))

        np.testing.assert_allclose(np.array(jax_out), torch_out, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_init_distribution(self):
        """Verify meta_embedding_init produces values in expected range."""
        meta_dim, z_dim = 4, 64
        rng = jax.random.PRNGKey(0)
        jax_model = JaxMetaEmbadding(z_dim=z_dim)
        params = jax_model.init(rng, jnp.ones((1, meta_dim)))
        # First Dense layer has fan_in = meta_dim = 4, bound = 1/(2*sqrt(4)) = 0.25
        first_kernel = params['params']['Dense_0']['kernel']
        bound = 1.0 / (2.0 * np.sqrt(meta_dim))
        assert np.all(np.array(first_kernel) >= -bound - 1e-7)
        assert np.all(np.array(first_kernel) <= bound + 1e-7)


class TestHyperNetwork:
    def test_equivalence(self):
        meta_v_dim, z_dim = 4, 64
        base_v_input_dim, base_v_output_dim = 8, 4
        dynamic_layer_dim = 32
        batch_size = 8
        np.random.seed(42)
        meta_v_np = np.random.randn(batch_size, meta_v_dim).astype(np.float32)
        base_v_np = np.random.randn(batch_size, base_v_input_dim).astype(np.float32)

        # PyTorch
        torch_model = TorchHyperNetwork(
            meta_v_dim, z_dim, base_v_input_dim, base_v_output_dim,
            dynamic_layer_dim, base_output_activation=None
        )
        torch_model.eval()
        with torch.no_grad():
            torch_z, torch_out = torch_model(
                torch.tensor(meta_v_np), torch.tensor(base_v_np)
            )
            torch_z = torch_z.numpy()
            torch_out = torch_out.numpy()

        # JAX
        jax_model = JaxHyperNetwork(
            z_dim=z_dim,
            base_v_input_dim=base_v_input_dim,
            base_v_output_dim=base_v_output_dim,
            dynamic_layer_dim=dynamic_layer_dim,
            base_output_activation=False
        )
        jax_params = transfer_hypernetwork(torch_model)
        jax_z, jax_out = jax_model.apply(
            jax_params, jnp.array(meta_v_np), jnp.array(base_v_np)
        )

        np.testing.assert_allclose(np.array(jax_z), torch_z, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        np.testing.assert_allclose(np.array(jax_out), torch_out, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_with_activation(self):
        """Test with tanh activation."""
        meta_v_dim, z_dim = 4, 64
        base_v_input_dim, base_v_output_dim = 8, 4
        dynamic_layer_dim = 32
        batch_size = 8
        np.random.seed(42)
        meta_v_np = np.random.randn(batch_size, meta_v_dim).astype(np.float32)
        base_v_np = np.random.randn(batch_size, base_v_input_dim).astype(np.float32)

        # PyTorch
        torch_model = TorchHyperNetwork(
            meta_v_dim, z_dim, base_v_input_dim, base_v_output_dim,
            dynamic_layer_dim, base_output_activation=torch.tanh
        )
        torch_model.eval()
        with torch.no_grad():
            torch_z, torch_out = torch_model(
                torch.tensor(meta_v_np), torch.tensor(base_v_np)
            )

        # JAX
        jax_model = JaxHyperNetwork(
            z_dim=z_dim,
            base_v_input_dim=base_v_input_dim,
            base_v_output_dim=base_v_output_dim,
            dynamic_layer_dim=dynamic_layer_dim,
            base_output_activation=True
        )
        jax_params = transfer_hypernetwork(torch_model)
        jax_z, jax_out = jax_model.apply(
            jax_params, jnp.array(meta_v_np), jnp.array(base_v_np)
        )

        np.testing.assert_allclose(np.array(jax_out), torch_out.numpy(),
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


class TestBmmEquivalence:
    def test_bmm_vs_matmul(self):
        """jnp.matmul vs torch.bmm on random 3D tensors."""
        np.random.seed(42)
        a_np = np.random.randn(4, 3, 5).astype(np.float32)
        b_np = np.random.randn(4, 5, 2).astype(np.float32)

        torch_result = torch.bmm(torch.tensor(a_np), torch.tensor(b_np)).numpy()
        jax_result = jnp.matmul(jnp.array(a_np), jnp.array(b_np))

        np.testing.assert_allclose(np.array(jax_result), torch_result, rtol=RTOL, atol=ATOL)


class TestDoubleHeadedHyperNetwork:
    def _create_models(self):
        meta_v_dim, z_dim = 4, 64
        base_v_input_dim = [8, 12]
        base_v_output_dim = [4, 1]
        dynamic_layer_dim = 32
        batch_size = 8

        np.random.seed(42)
        meta_v_np = np.random.randn(batch_size, meta_v_dim).astype(np.float32)
        base_v_1_np = np.random.randn(batch_size, base_v_input_dim[0]).astype(np.float32)
        base_v_2_np = np.random.randn(batch_size, base_v_input_dim[1]).astype(np.float32)

        # PyTorch
        torch_model = TorchDoubleHeadedHyperNetwork(
            meta_v_dim, z_dim, base_v_input_dim, base_v_output_dim,
            dynamic_layer_dim, base_output_activation=[torch.tanh, None]
        )
        torch_model.eval()

        # JAX
        jax_model = JaxDoubleHeadedHyperNetwork(
            z_dim=z_dim,
            base_v_input_dim_1=base_v_input_dim[0],
            base_v_input_dim_2=base_v_input_dim[1],
            base_v_output_dim_1=base_v_output_dim[0],
            base_v_output_dim_2=base_v_output_dim[1],
            dynamic_layer_dim=dynamic_layer_dim,
            base_output_activation_1=True,
            base_output_activation_2=False,
        )
        jax_params = transfer_double_headed_hypernetwork(torch_model)

        return torch_model, jax_model, jax_params, meta_v_np, base_v_1_np, base_v_2_np

    def test_equivalence(self):
        """Both heads + shared embedding."""
        torch_model, jax_model, jax_params, meta_v_np, base_v_1_np, base_v_2_np = self._create_models()

        with torch.no_grad():
            torch_z, torch_out1, torch_out2 = torch_model(
                torch.tensor(meta_v_np),
                torch.tensor(base_v_1_np),
                torch.tensor(base_v_2_np)
            )

        jax_z, jax_out1, jax_out2 = jax_model.apply(
            jax_params,
            jnp.array(meta_v_np),
            jnp.array(base_v_1_np),
            jnp.array(base_v_2_np)
        )

        np.testing.assert_allclose(np.array(jax_z), torch_z.numpy(),
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        np.testing.assert_allclose(np.array(jax_out1), torch_out1.numpy(),
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
        np.testing.assert_allclose(np.array(jax_out2), torch_out2.numpy(),
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_embed_only(self):
        torch_model, jax_model, jax_params, meta_v_np, _, _ = self._create_models()

        with torch.no_grad():
            torch_z = torch_model.embed(torch.tensor(meta_v_np)).numpy()

        jax_z = jax_model.apply(jax_params, jnp.array(meta_v_np), method=jax_model.embed)

        np.testing.assert_allclose(np.array(jax_z), torch_z,
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_forward_net_1_only(self):
        torch_model, jax_model, jax_params, meta_v_np, base_v_1_np, _ = self._create_models()

        with torch.no_grad():
            torch_z = torch_model.embed(torch.tensor(meta_v_np))
            torch_out1 = torch_model.forward_net_1(torch_z, torch.tensor(base_v_1_np)).numpy()
            torch_z = torch_z.numpy()

        jax_z = jax_model.apply(jax_params, jnp.array(meta_v_np), method=jax_model.embed)
        jax_out1 = jax_model.apply(jax_params, jax_z, jnp.array(base_v_1_np),
                                    method=jax_model.forward_net_1)

        np.testing.assert_allclose(np.array(jax_out1), torch_out1,
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    def test_forward_net_2_only(self):
        torch_model, jax_model, jax_params, meta_v_np, _, base_v_2_np = self._create_models()

        with torch.no_grad():
            torch_z = torch_model.embed(torch.tensor(meta_v_np))
            torch_out2 = torch_model.forward_net_2(torch_z, torch.tensor(base_v_2_np)).numpy()

        jax_z = jax_model.apply(jax_params, jnp.array(meta_v_np), method=jax_model.embed)
        jax_out2 = jax_model.apply(jax_params, jax_z, jnp.array(base_v_2_np),
                                    method=jax_model.forward_net_2)

        np.testing.assert_allclose(np.array(jax_out2), torch_out2,
                                   rtol=RTOL_RELAXED, atol=ATOL_RELAXED)
