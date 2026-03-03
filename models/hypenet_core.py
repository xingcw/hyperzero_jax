"""
HyperNetwork implementation is based on

https://arxiv.org/abs/2106.06842
https://github.com/keynans/HypeRL
"""
import jax
import jax.numpy as jnp
import flax.linen as nn


def uniform_init(stddev):
    """Custom uniform initializer for Head layers."""
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -stddev, stddev)
    return init


def meta_embedding_init(key, shape, dtype=jnp.float32):
    """Custom initializer for Meta_Embadding linear layers: 1/(2*sqrt(fan_in))."""
    fan_in = shape[0] if len(shape) >= 2 else shape[0]
    bound = 1.0 / (2.0 * jnp.sqrt(float(fan_in)))
    return jax.random.uniform(key, shape, dtype, -bound, bound)


class ResBlock(nn.Module):
    """
    Residual block used for learnable task embeddings.
    """
    out_size: int

    @nn.compact
    def __call__(self, x):
        h = nn.relu(x)
        h = nn.Dense(self.out_size,
                     kernel_init=meta_embedding_init,
                     bias_init=nn.initializers.zeros)(h)
        h = nn.relu(h)
        h = nn.Dense(self.out_size,
                     kernel_init=meta_embedding_init,
                     bias_init=nn.initializers.zeros)(h)
        return x + h


class Head(nn.Module):
    """
    Hypernetwork head for generating weights of a single layer of an MLP.
    """
    latent_dim: int
    output_dim_in: int
    output_dim_out: int
    sttdev: float

    def setup(self):
        self.W1 = nn.Dense(self.output_dim_in * self.output_dim_out,
                           kernel_init=uniform_init(self.sttdev),
                           bias_init=nn.initializers.zeros)
        self.b1 = nn.Dense(self.output_dim_out,
                           kernel_init=uniform_init(self.sttdev),
                           bias_init=nn.initializers.zeros)

    def __call__(self, x):
        # weights, bias and scale for dynamic layer
        w = self.W1(x).reshape(-1, self.output_dim_out, self.output_dim_in)
        b = self.b1(x).reshape(-1, self.output_dim_out, 1)
        return w, b


class Meta_Embadding(nn.Module):
    """
    Hypernetwork meta embedding.
    """
    z_dim: int

    @nn.compact
    def __call__(self, meta_v):
        # First block: meta_dim -> z_dim // 4
        x = nn.Dense(self.z_dim // 4,
                     kernel_init=meta_embedding_init,
                     bias_init=nn.initializers.zeros)(meta_v)
        x = ResBlock(self.z_dim // 4)(x)
        x = ResBlock(self.z_dim // 4)(x)

        # Second block: z_dim // 4 -> z_dim // 2
        x = nn.Dense(self.z_dim // 2,
                     kernel_init=meta_embedding_init,
                     bias_init=nn.initializers.zeros)(x)
        x = ResBlock(self.z_dim // 2)(x)
        x = ResBlock(self.z_dim // 2)(x)

        # Third block: z_dim // 2 -> z_dim
        x = nn.Dense(self.z_dim,
                     kernel_init=meta_embedding_init,
                     bias_init=nn.initializers.zeros)(x)
        x = ResBlock(self.z_dim)(x)
        x = ResBlock(self.z_dim)(x)

        z = x.reshape(-1, self.z_dim)
        return z


class HyperNetwork(nn.Module):
    """
    A hypernetwork that creates another neural network of
    base_v_input_dim -> base_v_output_dim using z_dim.
    """
    z_dim: int
    base_v_input_dim: int
    base_v_output_dim: int
    dynamic_layer_dim: int
    base_output_activation: bool = False  # True for tanh, False for None

    def setup(self):
        self.hyper = Meta_Embadding(z_dim=self.z_dim)
        self.layer1 = Head(latent_dim=self.z_dim,
                           output_dim_in=self.base_v_input_dim,
                           output_dim_out=self.dynamic_layer_dim,
                           sttdev=0.05)
        self.last_layer = Head(latent_dim=self.z_dim,
                               output_dim_in=self.dynamic_layer_dim,
                               output_dim_out=self.base_v_output_dim,
                               sttdev=0.008)

    def __call__(self, meta_v, base_v):
        # produce dynamic weights
        z = self.hyper(meta_v)
        w1, b1 = self.layer1(z)
        w2, b2 = self.last_layer(z)

        # dynamic network pass
        out = nn.relu(jnp.matmul(w1, base_v[:, :, None]) + b1)
        out = jnp.matmul(w2, out) + b2
        if self.base_output_activation:
            out = jnp.tanh(out)

        batch_size = out.shape[0]
        return z, out.reshape(batch_size, -1)


class DoubleHeadedHyperNetwork(nn.Module):
    """
    A hypernetwork that creates two neural networks of
    base_v_input_dim[i] -> base_v_output_dim[i] using z_dim.
    """
    z_dim: int
    base_v_input_dim_1: int
    base_v_input_dim_2: int
    base_v_output_dim_1: int
    base_v_output_dim_2: int
    dynamic_layer_dim: int
    base_output_activation_1: bool = False  # True for tanh, False for None
    base_output_activation_2: bool = False

    def setup(self):
        self.hyper = Meta_Embadding(z_dim=self.z_dim)

        # main networks
        self.layer1_1 = Head(latent_dim=self.z_dim,
                             output_dim_in=self.base_v_input_dim_1,
                             output_dim_out=self.dynamic_layer_dim,
                             sttdev=0.05)
        self.last_layer_1 = Head(latent_dim=self.z_dim,
                                 output_dim_in=self.dynamic_layer_dim,
                                 output_dim_out=self.base_v_output_dim_1,
                                 sttdev=0.008)

        self.layer1_2 = Head(latent_dim=self.z_dim,
                             output_dim_in=self.base_v_input_dim_2,
                             output_dim_out=self.dynamic_layer_dim,
                             sttdev=0.05)
        self.last_layer_2 = Head(latent_dim=self.z_dim,
                                 output_dim_in=self.dynamic_layer_dim,
                                 output_dim_out=self.base_v_output_dim_2,
                                 sttdev=0.008)

    def __call__(self, meta_v, base_v_1, base_v_2):
        z = self.hyper(meta_v)
        out_1 = self.forward_net_1(z, base_v_1)
        out_2 = self.forward_net_2(z, base_v_2)
        return z, out_1, out_2

    def embed(self, meta_v):
        z = self.hyper(meta_v)
        return z

    def forward_net_1(self, z, base_v_1):
        # produce dynamic weights for network #1
        w1_1, b1_1 = self.layer1_1(z)
        w2_1, b2_1 = self.last_layer_1(z)

        # dynamic network 1 pass
        out_1 = nn.relu(jnp.matmul(w1_1, base_v_1[:, :, None]) + b1_1)
        out_1 = jnp.matmul(w2_1, out_1) + b2_1
        if self.base_output_activation_1:
            out_1 = jnp.tanh(out_1)

        batch_size = out_1.shape[0]
        return out_1.reshape(batch_size, -1)

    def forward_net_2(self, z, base_v_2):
        # produce dynamic weights for network #2
        w1_2, b1_2 = self.layer1_2(z)
        w2_2, b2_2 = self.last_layer_2(z)

        # dynamic network 2 pass
        out_2 = nn.relu(jnp.matmul(w1_2, base_v_2[:, :, None]) + b1_2)
        out_2 = jnp.matmul(w2_2, out_2) + b2_2
        if self.base_output_activation_2:
            out_2 = jnp.tanh(out_2)

        batch_size = out_2.shape[0]
        return out_2.reshape(batch_size, -1)
