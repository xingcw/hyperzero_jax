import jax
import jax.numpy as jnp
import flax.linen as nn

from models.hypenet_core import (HyperNetwork, DoubleHeadedHyperNetwork,
                                  Meta_Embadding, meta_embedding_init)


class HyperPolicy(nn.Module):
    """
    Approximates the mapping R(\phi) -> \pi^* (a|s)
    """
    input_param_dim: int
    state_dim: int
    action_dim: int
    embed_dim: int
    hidden_dim: int

    def setup(self):
        self.hyper_policy = HyperNetwork(
            z_dim=self.embed_dim,
            base_v_input_dim=self.state_dim,
            base_v_output_dim=self.action_dim,
            dynamic_layer_dim=self.hidden_dim,
            base_output_activation=True  # tanh
        )

    def __call__(self, input_param, state):
        z, action = self.hyper_policy(input_param, state)
        return action


class HyperRLSolution(nn.Module):
    """
    Baseline. Approximates the mapping R(\phi) -> Q^*(s, a), \pi^*(s)
    """
    input_param_dim: int
    state_dim: int
    action_dim: int
    embed_dim: int
    hidden_dim: int

    def setup(self):
        self.hyper_rl_net = DoubleHeadedHyperNetwork(
            z_dim=self.embed_dim,
            base_v_input_dim_1=self.state_dim,
            base_v_input_dim_2=self.state_dim + self.action_dim,
            base_v_output_dim_1=self.action_dim,
            base_v_output_dim_2=1,
            dynamic_layer_dim=self.hidden_dim,
            base_output_activation_1=True,  # tanh
            base_output_activation_2=False   # None
        )

    def __call__(self, input_param, state, action):
        state_action = jnp.concatenate([state, action], axis=-1)
        z, pred_action, q_value = self.hyper_rl_net(input_param, state, state_action)
        return z, pred_action, q_value

    def embed_task(self, input_param):
        z = self.hyper_rl_net.embed(input_param)
        return z

    def predict_action(self, z, state):
        pred_action = self.hyper_rl_net.forward_net_1(z, state)
        return pred_action

    def predict_q_value(self, z, state, action):
        state_action = jnp.concatenate([state, action], axis=-1)
        q_value = self.hyper_rl_net.forward_net_2(z, state_action)
        return q_value


class MLPActionPredictor(nn.Module):
    """
    Baseline. Approximates the mapping R(\phi) -> \pi^*(s)
    """
    input_param_dim: int
    state_dim: int
    action_dim: int
    embed_dim: int
    hidden_dim: int

    def setup(self):
        self.embedding = Meta_Embadding(z_dim=self.embed_dim)
        self.dense1 = nn.Dense(self.hidden_dim,
                               kernel_init=nn.initializers.orthogonal(),
                               bias_init=nn.initializers.zeros)
        self.dense2 = nn.Dense(self.action_dim,
                               kernel_init=nn.initializers.orthogonal(),
                               bias_init=nn.initializers.zeros)

    def __call__(self, input_param, state):
        emb = self.embedding(input_param)
        emb_states = jnp.concatenate([emb, state], axis=-1)
        x = nn.relu(self.dense1(emb_states))
        action = self.dense2(x)
        return jnp.tanh(action)


class MLPRLSolution(nn.Module):
    """
    Baseline. Approximates the mapping R(\phi) -> Q^*(s, a), \pi^*(s)
    """
    input_param_dim: int
    state_dim: int
    action_dim: int
    embed_dim: int
    hidden_dim: int

    def setup(self):
        self.embedding = Meta_Embadding(z_dim=self.embed_dim)

        # Q-network
        self.q_dense1 = nn.Dense(self.hidden_dim,
                                 kernel_init=nn.initializers.orthogonal(),
                                 bias_init=nn.initializers.zeros)
        self.q_dense2 = nn.Dense(1,
                                 kernel_init=nn.initializers.orthogonal(),
                                 bias_init=nn.initializers.zeros)

        # Policy network
        self.policy_dense1 = nn.Dense(self.hidden_dim,
                                      kernel_init=nn.initializers.orthogonal(),
                                      bias_init=nn.initializers.zeros)
        self.policy_dense2 = nn.Dense(self.action_dim,
                                      kernel_init=nn.initializers.orthogonal(),
                                      bias_init=nn.initializers.zeros)

    def __call__(self, input_param, state, action):
        task = self.embed_task(input_param)
        pred_action = self.predict_action(task, state)
        q_value = self.predict_q_value(task, state, action)
        return task, pred_action, q_value

    def embed_task(self, input_param):
        task = self.embedding(input_param)
        return task

    def predict_action(self, task, state):
        task_state = jnp.concatenate([task, state], axis=-1)
        x = nn.relu(self.policy_dense1(task_state))
        pred_action = self.policy_dense2(x)
        return jnp.tanh(pred_action)

    def predict_q_value(self, task, state, action):
        task_state_action = jnp.concatenate([task, state, action], axis=-1)
        x = nn.relu(self.q_dense1(task_state_action))
        q_value = self.q_dense2(x)
        return q_value


class MLPContextEncoder(nn.Module):
    """
    Context encoder of PEARL.
    """
    state_dim: int
    action_dim: int
    embed_dim: int
    hidden_dim: int

    def setup(self):
        self.fc_dense1 = nn.Dense(self.hidden_dim,
                                  kernel_init=nn.initializers.orthogonal(),
                                  bias_init=nn.initializers.zeros)
        self.fc_dense2 = nn.Dense(self.hidden_dim,
                                  kernel_init=nn.initializers.orthogonal(),
                                  bias_init=nn.initializers.zeros)
        self.mu_layer = nn.Dense(self.embed_dim,
                                 kernel_init=nn.initializers.orthogonal(),
                                 bias_init=nn.initializers.zeros)
        self.log_var_layer = nn.Dense(self.embed_dim,
                                      kernel_init=nn.initializers.orthogonal(),
                                      bias_init=nn.initializers.zeros)

    def encode(self, state, action):
        state_action = jnp.concatenate([state, action], axis=-1)
        z = nn.relu(self.fc_dense1(state_action))
        z = nn.relu(self.fc_dense2(z))
        return self.mu_layer(z), self.log_var_layer(z)

    def reparameterize(self, mu, log_var, rng):
        std = jnp.exp(0.5 * log_var)
        eps = jax.random.normal(rng, std.shape)
        return eps * std + mu

    def __call__(self, state, action, rng=None):
        mu, log_var = self.encode(state, action)
        if rng is None:
            rng = self.make_rng('sample')
        z = self.reparameterize(mu, log_var, rng)
        return z, mu, log_var
