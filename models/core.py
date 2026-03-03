import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


def gaussian_logprob(noise, log_std):
    """
    Compute Gaussian log probability.
    """
    residual = (-0.5 * noise**2 - log_std).sum(-1, keepdims=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.shape[-1]


def squash(mu, pi, log_pi):
    """
    Apply squashing function.
    """
    mu = jnp.tanh(mu)
    if pi is not None:
        pi = jnp.tanh(pi)
    if log_pi is not None:
        log_pi -= jnp.log(jnp.maximum(1 - pi**2, 1e-6)).sum(-1, keepdims=True)
    return mu, pi, log_pi


class DeterministicActor(nn.Module):
    """
    Original TD3 actor.
    """
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(self.hidden_dim,
                     kernel_init=nn.initializers.orthogonal(),
                     bias_init=nn.initializers.zeros)(state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim,
                     kernel_init=nn.initializers.orthogonal(),
                     bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim,
                     kernel_init=nn.initializers.orthogonal(),
                     bias_init=nn.initializers.zeros)(x)
        return jnp.tanh(x)


class Critic(nn.Module):
    """
    Original TD3 critic.
    """
    hidden_dim: int

    def setup(self):
        # Q1 architecture
        self.Q1_dense1 = nn.Dense(self.hidden_dim,
                                  kernel_init=nn.initializers.orthogonal(),
                                  bias_init=nn.initializers.zeros)
        self.Q1_dense2 = nn.Dense(self.hidden_dim,
                                  kernel_init=nn.initializers.orthogonal(),
                                  bias_init=nn.initializers.zeros)
        self.Q1_dense3 = nn.Dense(1,
                                  kernel_init=nn.initializers.orthogonal(),
                                  bias_init=nn.initializers.zeros)

        # Q2 architecture
        self.Q2_dense1 = nn.Dense(self.hidden_dim,
                                  kernel_init=nn.initializers.orthogonal(),
                                  bias_init=nn.initializers.zeros)
        self.Q2_dense2 = nn.Dense(self.hidden_dim,
                                  kernel_init=nn.initializers.orthogonal(),
                                  bias_init=nn.initializers.zeros)
        self.Q2_dense3 = nn.Dense(1,
                                  kernel_init=nn.initializers.orthogonal(),
                                  bias_init=nn.initializers.zeros)

    def __call__(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)

        q1 = nn.relu(self.Q1_dense1(sa))
        q1 = nn.relu(self.Q1_dense2(q1))
        q1 = self.Q1_dense3(q1)

        q2 = nn.relu(self.Q2_dense1(sa))
        q2 = nn.relu(self.Q2_dense2(q2))
        q2 = self.Q2_dense3(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)

        q1 = nn.relu(self.Q1_dense1(sa))
        q1 = nn.relu(self.Q1_dense2(q1))
        q1 = self.Q1_dense3(q1)
        return q1
