"""
Weight transfer infrastructure for PyTorch -> Flax equivalence testing.

Handles:
- PyTorch weight (shape [out, in]) -> Flax kernel (shape [in, out]) via transpose
- PyTorch bias -> Flax bias (same shape, no transpose)
- nn.Sequential index-based naming -> Flax compact naming
- Nested module hierarchies
"""
import sys
import os
import numpy as np
import pytest

import jax
import jax.numpy as jnp

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Tolerance constants
RTOL = 1e-5
ATOL = 1e-5
# Relaxed tolerances for deep HyperNetwork paths
RTOL_RELAXED = 1e-4
ATOL_RELAXED = 1e-4


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


def transfer_weights_linear(torch_weight, torch_bias):
    """Transfer a single PyTorch Linear layer to Flax Dense params."""
    params = {'kernel': torch_weight.T.copy()}
    if torch_bias is not None:
        params['bias'] = torch_bias.copy()
    return params


def _get_torch_params(module):
    """Extract weight and bias numpy arrays from a PyTorch module."""
    weight = module.weight.detach().cpu().numpy()
    bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
    return weight, bias


def transfer_deterministic_actor(torch_model):
    """Transfer DeterministicActor weights from PyTorch to Flax param dict."""
    layers = list(torch_model.policy.children())
    # Sequential: Linear, ReLU, Linear, ReLU, Linear -> Dense_0, Dense_1, Dense_2
    linear_idx = 0
    params = {}
    for layer in layers:
        if hasattr(layer, 'weight'):
            w, b = _get_torch_params(layer)
            params[f'Dense_{linear_idx}'] = transfer_weights_linear(w, b)
            linear_idx += 1
    return {'params': params}


def transfer_critic(torch_model):
    """Transfer Critic weights from PyTorch to Flax param dict."""
    params = {}

    # Q1_net: Sequential(Linear, ReLU, Linear, ReLU, Linear)
    q1_layers = list(torch_model.Q1_net.children())
    q1_linear_idx = 0
    for layer in q1_layers:
        if hasattr(layer, 'weight'):
            w, b = _get_torch_params(layer)
            params[f'Q1_dense{q1_linear_idx + 1}'] = transfer_weights_linear(w, b)
            q1_linear_idx += 1

    # Q2_net: Sequential(Linear, ReLU, Linear, ReLU, Linear)
    q2_layers = list(torch_model.Q2_net.children())
    q2_linear_idx = 0
    for layer in q2_layers:
        if hasattr(layer, 'weight'):
            w, b = _get_torch_params(layer)
            params[f'Q2_dense{q2_linear_idx + 1}'] = transfer_weights_linear(w, b)
            q2_linear_idx += 1

    return {'params': params}


def _transfer_resblock(torch_resblock):
    """Transfer a ResBlock's weights."""
    # ResBlock.fc = Sequential(ReLU, Linear, ReLU, Linear)
    layers = list(torch_resblock.fc.children())
    linear_idx = 0
    params = {}
    for layer in layers:
        if hasattr(layer, 'weight'):
            w, b = _get_torch_params(layer)
            params[f'Dense_{linear_idx}'] = transfer_weights_linear(w, b)
            linear_idx += 1
    return params


def _transfer_head(torch_head):
    """Transfer a Head's weights."""
    w_W1, b_W1 = _get_torch_params(torch_head.W1)
    w_b1, b_b1 = _get_torch_params(torch_head.b1)
    return {
        'W1': transfer_weights_linear(w_W1, b_W1),
        'b1': transfer_weights_linear(w_b1, b_b1),
    }


def _transfer_meta_embedding(torch_meta_emb):
    """Transfer Meta_Embadding weights."""
    # hyper = Sequential(Linear, ResBlock, ResBlock, Linear, ResBlock, ResBlock, Linear, ResBlock, ResBlock)
    layers = list(torch_meta_emb.hyper.children())
    params = {}
    linear_idx = 0
    resblock_idx = 0

    for layer in layers:
        layer_type = type(layer).__name__
        if layer_type == 'Linear':
            w, b = _get_torch_params(layer)
            params[f'Dense_{linear_idx}'] = transfer_weights_linear(w, b)
            linear_idx += 1
        elif layer_type == 'ResBlock':
            params[f'ResBlock_{resblock_idx}'] = _transfer_resblock(layer)
            resblock_idx += 1

    return params


def transfer_hypernetwork(torch_model):
    """Transfer HyperNetwork weights."""
    params = {
        'hyper': _transfer_meta_embedding(torch_model.hyper),
        'layer1': _transfer_head(torch_model.layer1),
        'last_layer': _transfer_head(torch_model.last_layer),
    }
    return {'params': params}


def transfer_double_headed_hypernetwork(torch_model):
    """Transfer DoubleHeadedHyperNetwork weights."""
    params = {
        'hyper': _transfer_meta_embedding(torch_model.hyper),
        'layer1_1': _transfer_head(torch_model.layer1_1),
        'last_layer_1': _transfer_head(torch_model.last_layer_1),
        'layer1_2': _transfer_head(torch_model.layer1_2),
        'last_layer_2': _transfer_head(torch_model.last_layer_2),
    }
    return {'params': params}


def transfer_hyper_policy(torch_model):
    """Transfer HyperPolicy weights."""
    hypernet_params = transfer_hypernetwork(torch_model.hyper_policy)
    return {'params': {'hyper_policy': hypernet_params['params']}}


def transfer_hyper_rl_solution(torch_model):
    """Transfer HyperRLSolution weights."""
    dh_params = transfer_double_headed_hypernetwork(torch_model.hyper_rl_net)
    return {'params': {'hyper_rl_net': dh_params['params']}}


def transfer_mlp_action_predictor(torch_model):
    """Transfer MLPActionPredictor weights."""
    params = {}

    # Meta_Embadding
    params['embedding'] = _transfer_meta_embedding(torch_model.embedding)

    # policy_net = Sequential(Linear, ReLU, Linear)
    layers = list(torch_model.policy_net.children())
    linear_idx = 0
    for layer in layers:
        if hasattr(layer, 'weight'):
            w, b = _get_torch_params(layer)
            key = f'dense{linear_idx + 1}'
            params[key] = transfer_weights_linear(w, b)
            linear_idx += 1

    return {'params': params}


def transfer_mlp_rl_solution(torch_model):
    """Transfer MLPRLSolution weights."""
    params = {}

    # Meta_Embadding
    params['embedding'] = _transfer_meta_embedding(torch_model.embedding)

    # q_net = Sequential(Linear, ReLU, Linear)
    q_layers = list(torch_model.q_net.children())
    q_linear_idx = 0
    for layer in q_layers:
        if hasattr(layer, 'weight'):
            w, b = _get_torch_params(layer)
            params[f'q_dense{q_linear_idx + 1}'] = transfer_weights_linear(w, b)
            q_linear_idx += 1

    # policy_net = Sequential(Linear, ReLU, Linear)
    p_layers = list(torch_model.policy_net.children())
    p_linear_idx = 0
    for layer in p_layers:
        if hasattr(layer, 'weight'):
            w, b = _get_torch_params(layer)
            params[f'policy_dense{p_linear_idx + 1}'] = transfer_weights_linear(w, b)
            p_linear_idx += 1

    return {'params': params}


def transfer_mlp_context_encoder(torch_model):
    """Transfer MLPContextEncoder weights."""
    params = {}

    # fc = Sequential(Linear, ReLU, Linear, ReLU)
    fc_layers = list(torch_model.fc.children())
    linear_idx = 0
    for layer in fc_layers:
        if hasattr(layer, 'weight'):
            w, b = _get_torch_params(layer)
            params[f'fc_dense{linear_idx + 1}'] = transfer_weights_linear(w, b)
            linear_idx += 1

    # mu and log_var
    w, b = _get_torch_params(torch_model.mu)
    params['mu_layer'] = transfer_weights_linear(w, b)
    w, b = _get_torch_params(torch_model.log_var)
    params['log_var_layer'] = transfer_weights_linear(w, b)

    return {'params': params}
