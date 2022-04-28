import jax.numpy as jnp

import jax
from jax import vmap, jit
from jax import random
from jax.scipy.stats import norm as jax_normal
from jax.tree_util import tree_map, tree_reduce

import jax.lax as lax
import jax.experimental.stax as stax
from jax.experimental.stax import Dense, Sigmoid, LeakyRelu, Relu, Tanh

from jax.nn.initializers import normal

# from jax.ops import index, index_update

from ..utils.graph import graph_to_mat, mat_to_graph
from ..utils.tree import tree_shapes
from functools import partial

def DenseNoBias(out_dim, W_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer _without_ bias"""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        W = W_init(rng, (input_shape[-1], out_dim))
        return output_shape, (W, )

    def apply_fun(params, inputs, **kwargs):
        W, = params
        return jnp.dot(inputs, W)

    return init_fun, apply_fun


def makeDenseNet(*, hidden_layers, sig_weight, sig_bias, bias=True, activation='relu'):
    """
    Generates functions defining a fully-connected NN
    with Gaussian initialized parameters

    Args:
        hidden_layers (list): list of ints specifying the dimensions of the hidden sizes
        sig_weight: std dev of weight initialization
        sig_bias: std dev of weight initialization

    Returns:
        stax.serial neural net object
    """

    def elementwise(fun, **fun_kwargs):
        """Layer that applies a scalar function elementwise on its inputs."""
        init_fun = lambda rng, input_shape: (input_shape, ())
        apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
        return init_fun, apply_fun


    # features: [hidden_layers[0], hidden_layers[0], ..., hidden_layers[-1], 1]
    if activation == 'sigmoid':
        f_activation = Sigmoid
    elif activation == 'tanh':
        f_activation = Tanh
    elif activation == 'relu':
        f_activation = Relu
    elif activation == 'leakyrelu':
        f_activation = LeakyRelu
    elif activation == 'sine':
        f_activation = elementwise(jnp.sin)
    else:
        raise KeyError(f'Invalid activation function `{activation}`')

    modules = []
    if bias:
        for dim in hidden_layers:
            modules += [
                Dense(dim, W_init=normal(stddev=sig_weight),
                        b_init=normal(stddev=sig_bias)),
                f_activation
            ]
        modules += [Dense(1, W_init=normal(stddev=sig_weight),
                            b_init=normal(stddev=sig_bias))]
    else:
        for dim in hidden_layers:
            modules += [
                DenseNoBias(dim, W_init=normal(stddev=sig_weight)),
                f_activation
            ]
        modules += [DenseNoBias(1, W_init=normal(stddev=sig_weight))]

    return stax.serial(*modules)


class DenseNonlinearGaussianJAX:
    """
    Non-linear Gaussian BN with interactions modeled by a fully-connected neural net
    See: https://arxiv.org/abs/1909.13189
    """

    def __init__(self, *, obs_noise, sig_param, hidden_layers, g_dist=None, verbose=False, activation='relu', bias=True):
        super(DenseNonlinearGaussianJAX, self).__init__()

        self.obs_noise = jnp.array(obs_noise)
        self.sig_param = sig_param
        self.hidden_layers = hidden_layers
        self.g_dist = g_dist
        self.verbose = verbose
        self.activation = activation
        self.bias = bias

        # init single neural net function for one variable with jax stax
        self.nn_init_random_params, nn_forward = makeDenseNet(
            hidden_layers=self.hidden_layers,
            sig_weight=self.sig_param,
            sig_bias=self.sig_param,
            activation=self.activation,
            bias=self.bias)

        # [?], [N, d] -> [N,]
        self.nn_forward = lambda theta, x: nn_forward(theta, x).squeeze(-1)

        # vectorize init and forward functions
        self.eltwise_nn_init_random_params = vmap(self.nn_init_random_params, (0, None), 0)
        self.double_eltwise_nn_init_random_params = vmap(self.eltwise_nn_init_random_params, (0, None), 0)
        self.triple_eltwise_nn_init_random_params = vmap(self.double_eltwise_nn_init_random_params, (0, None), 0)

        # [d2, ?], [N, d] -> [N, d2]
        self.eltwise_nn_forward = vmap(self.nn_forward, (0, None), 1)

        # [d2, ?], [d2, N, d] -> [N, d2]
        self.double_eltwise_nn_forward = vmap(self.nn_forward, (0, 0), 1)


    def get_theta_shape(self, *, n_vars):
        """ Returns tree shape of the parameters of the neural networks
        Args:
            n_vars

        Returns:
            PyTree of parameter shape
        """

        dummy_subkeys = jnp.zeros((n_vars, 2), dtype=jnp.uint32)
        _, theta = self.eltwise_nn_init_random_params(dummy_subkeys, (n_vars, )) # second arg is `input_shape` of NN forward pass

        theta_shape = tree_shapes(theta)
        return theta_shape

    def init_parameters(self, *, key, n_vars, n_particles, batch_size=0):
        """Samples batch of random parameters given dimensions of graph, from p(theta | G)
        Args:
            key: rng
            n_vars: number of variables in BN
            n_particles: number of parameter particles sampled
            batch_size: number of batches of particles being sampled

        Returns:
            theta : PyTree with leading dimension of `n_particles`
        """

        if batch_size == 0:
            subkeys = random.split(key, n_particles * n_vars).reshape(n_particles, n_vars, -1)
            _, theta = self.double_eltwise_nn_init_random_params(subkeys, (n_vars, ))
        else:
            subkeys = random.split(key, batch_size * n_particles * n_vars).reshape(batch_size, n_particles, n_vars, -1)
            _, theta = self.triple_eltwise_nn_init_random_params(subkeys, (n_vars, ))

        # to float64
        theta = tree_map(lambda arr: arr.astype(jnp.float64), theta)
        return theta

    def sample_parameters(self, *, key, n_vars):
        """Samples parameters for neural network. Here, g is ignored.
        Args:
            g (igraph.Graph): graph
            key: rng

        Returns:
            theta : list of (W, b) tuples, dependent on `hidden_layers`
        """

        subkeys = random.split(key, n_vars)
        _, theta = self.eltwise_nn_init_random_params(subkeys, (n_vars, ))

        return theta

    @partial(jit, static_argnames=('self', 'onehot'))
    def fast_sample_obs(self, x, z, g_mat, theta, node, toporder, onehot=False):
        """
        Samples `n_samples` observations by doing single forward passes in topological order
        Args:
            key: rng
            n_samples (int): number of samples
            g_mat: adjacency matrix
            toporder: topopoligcal order of the graph nodes
            theta : PyTree of parameters
            interv: {intervened node : clamp value}
        Returns:
            x : [n_samples, d]
        """

        if onehot:
            t= toporder
            x = lax.fori_loop(
                0,
                len(toporder),
                lambda j, arr:
                    arr.at[:, t[j]].set(
                        (self.eltwise_nn_forward(theta, arr * g_mat[:, t[j]])[:, t[j]] + z[:, t[j]])*(1.0-node[t[j]]) + node[t[j]]*arr[:, t[j]]
                    ),
                x
            )
        else:
            t= toporder
            x = lax.fori_loop(
                0,
                len(toporder),
                lambda j, arr:
                    jnp.where(t[j] == node,
                    arr.at[:, node].set(arr[:, node]),
                    arr.at[:, t[j]].set(
                            self.eltwise_nn_forward(theta, arr * g_mat[:, t[j]])[:, t[j]] + z[:, t[j]]
                        )
                    ),
                x
            )

        return x


    def sample_obs(self, *, key, n_samples, g, theta, toporder=None, node=None, value_sampler=None, deterministic=False):
        """
        Samples `n_samples` observations by doing single forward passes in topological order
        Args:
            key: rng
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta : PyTree of parameters
            interv: {intervened node : clamp value}
        Returns:
            x : [n_samples, d]
        """
        g_mat = graph_to_mat(g)
        n_vars = g_mat.shape[0]

        z = self.obs_noise * random.normal(key, shape=(n_samples, n_vars)) # additive gaussian noise on the z
        if deterministic:
            z = 0*z
        x = jnp.zeros((n_samples, n_vars))
        if node is not None:
            values = value_sampler.sample(n_samples)
            x = x.at[:, node].set(values)
        # find topological order for ancestral sampling
        if toporder is None:
            toporder = g.topological_sorting()

        toporder = jnp.array(toporder)

        return self.fast_sample_obs(
            x=x,
            z=z,
            g_mat=g_mat,
            theta=theta,
            node=node,
            toporder=toporder)

    @partial(jit, static_argnames=('self', 'n_samples', 'deterministic', 'onehot'))
    def new_sample_obs(self, *, key, g_mat, theta, toporder, n_samples, nodes=None, values=None, deterministic=False, onehot=False):
        n_vars = g_mat.shape[0]
        B = nodes.shape[0]

        x = jnp.zeros((B, n_samples, n_vars))
        z = self.obs_noise * random.normal(key, shape=(B, n_samples, n_vars)) # additive gaussian noise on the z
        if deterministic:
            z = 0*z
        if nodes is not None:
            if hasattr(values, 'sample'):
                values = values.sample(n_samples)
            if onehot:
                x = (nodes*values[:, None].repeat(n_vars, -1))[:, None] + ((1.0-nodes)[:, None]*x)
            else:
                fn = lambda arr, idx, vals: arr.at[:, idx].set(vals)
                x = jax.vmap(fn)(x, jnp.int32(nodes), values)
        x = jax.vmap(
            self.fast_sample_obs,
            (0, 0, None, None, 0, 0, None)
        )(x, z, g_mat, theta, nodes, toporder, onehot)

        return x

    def old_sample_obs(self, *, key, n_samples, g, theta, toporder=None, node = None, value_sampler = None, deterministic=False):
        """
        Samples `n_samples` observations by doing single forward passes in topological order
        Args:
            key: rng
            n_samples (int): number of samples
            g (igraph.Graph): graph
            theta : PyTree of parameters
            interv: {intervened node : clamp value}
        Returns:
            x : [n_samples, d]
        """

        # find topological order for ancestral sampling
        if toporder is None:
            toporder = g.topological_sorting()

        n_vars = len(g.vs)
        x = jnp.zeros((n_samples, n_vars))
        z = jnp.zeros((n_samples, n_vars))

        if not deterministic:
            for i in range(n_vars):
                key, subk = random.split(key)
                # z = index_update(z, index[:, i], self.obs_noise[i] * random.normal(subk, shape=(n_samples,)))
                z = z.at[:, i].set(self.obs_noise[i] * random.normal(subk, shape=(n_samples,)))

        g_mat = graph_to_mat(g)

        # ancestral sampling
        # for simplicity, does d full forward passes for simplicity, which avoids indexing into python list of parameters
        for j in toporder:

            # intervention
            if j == node:
                if hasattr(value_sampler, 'sample'):
                    values = value_sampler.sample(n_samples)
                else:
                    values = value_sampler
                # x = index_update(x, index[:, j], values)
                x = x.at[:, j].set(values)
                continue

            # regular ancestral sampling
            parents = g_mat[:, j].reshape(1, -1)

            has_parents = parents.sum() > 0

            x_msk = x * parents
            means = self.eltwise_nn_forward(theta, x_msk)
            x = x.at[:, j].set(means[:, j] + z[:, j])

        return x

    def log_prob_parameters(self, *, theta, w):
        """log p(theta | g)
        Assumes N(mean_edge, sig_edge^2) distribution for any given edge

        Args:
            theta: parmeter PyTree
            w: adjacency matrix of graph [n_vars, n_vars]

        Returns:
            logprob [1,]
        """
        # compute log prob for each weight
        logprobs = tree_map(lambda leaf_theta: jax_normal.logpdf(x=leaf_theta, loc=0.0, scale=self.sig_param), theta)

        # mask logprobs of first layer weight matrix [0][0] according to graph
        # [d, d, dim_first_layer] = [d, d, dim_first_layer] * [d, d, 1]
        if self.bias:
            first_weight_logprobs, first_bias_logprobs = logprobs[0]
            logprobs[0] = (first_weight_logprobs * w.T[:, :, None], first_bias_logprobs)
        else:
            first_weight_logprobs,  = logprobs[0]
            logprobs[0] = (first_weight_logprobs * w.T[:, :, None],)

        # sum logprobs of every parameter tensor and add all up
        return tree_reduce(jnp.add, tree_map(jnp.sum, logprobs))


    def log_likelihood(self, *, data, theta, w, interv_targets):
        """log p(x | theta, G)
        Assumes N(mean_obs, obs_noise^2) distribution for any given observation

        Args:
            data: observations [N, d]
            theta: parameter PyTree
            w: adjacency matrix [n_vars, n_vars]
            interv_targets: boolean indicator of intervention locations [n_vars, ]

        Returns:
            logprob [1, ]
        """

        # [d2, N, d] = [1, N, d] * [d2, 1, d] mask non-parent entries of each j
        all_x_msk = data[None] * w.T[:, None]

        # [N, d2] NN forward passes for parameters of each param j
        all_means = self.double_eltwise_nn_forward(theta, all_x_msk)

        # sum scores for all nodes and data
        return jnp.sum(
            jnp.where(
                # [1, n_vars]
                interv_targets[None, ...],
                0.0,
                # [n_observations, n_vars]
                jax_normal.logpdf(x=data, loc=all_means, scale=self.obs_noise)
            )
        )

    def log_likelihood_single(self, *, data, theta, w, interv_targets):
        """log p(x | theta, G)
        Assumes N(mean_obs, obs_noise^2) distribution for any given observation

        Args:
            data: observations [N, d]
            theta: parameter PyTree
            w: adjacency matrix [n_vars, n_vars]
            interv_targets: boolean indicator of intervention locations [n_vars, ]

        Returns:
            logprob [1, ]
        """

        # [d2, N, d] = [1, N, d] * [d2, 1, d] mask non-parent entries of each j
        all_x_msk = data[None] * w.T[:, None]

        # [N, d2] NN forward passes for parameters of each param j
        all_means = self.double_eltwise_nn_forward(theta, all_x_msk)

        # sum scores for all nodes and data
        return jnp.sum(jnp.where(
                # [1, n_vars]
                interv_targets[None, ...],
                0.0,
                # [n_observations, n_vars]
                jax_normal.logpdf(x=data, loc=all_means, scale=self.obs_noise)
            ), axis = -1)
