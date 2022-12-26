import jax.numpy as jnp
from jax import jit, vmap
from jax import random

from .inference import JointDiBS
from .kernel import JointAdditiveFrobeniusSEKernel

from .eval.target import make_linear_gaussian_model, make_nonlinear_gaussian_model
from .eval.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
            
from .utils.graph import elwise_acyclic_constr_nograd as constraint
from .utils.func import joint_dist_to_marginal, particle_joint_empirical, particle_joint_mixture
from .utils.visualize import visualize, visualize_ground_truth

from .config.example import DiBSExampleSettings

key = random.PRNGKey(123)

n_vars = 10 # number of variables in BN
graph_prior_str = 'sf' # 'er': Erdos-Renyi  'sf': scale-free (power-law degree distribution)

key, subk = random.split(key)
print(key, subk)
target = make_linear_gaussian_model(key=subk, n_vars=n_vars, graph_prior_str=graph_prior_str)
# target = make_nonlinear_gaussian_model(key=subk, n_vars=n_vars, graph_prior_str=graph_prior_str)

# train and held-out observations
x = jnp.array(target.x)
x_ho = jnp.array(target.x_ho)

model = target.inference_model
print(type(model).__name__)

no_interv_targets = jnp.zeros(n_vars).astype(bool) # observational data

def log_prior(single_w_prob):
    """log p(G) using edge probabilities as G"""
    return target.graph_model.unnormalized_log_prob_soft(soft_g=single_w_prob)

def log_likelihood(single_w, single_theta, single_sigma, x, interv_targets, rng):
        log_prob_theta = target.inference_model.log_prob_parameters(
            theta=single_theta, w=single_w
        )
        log_lik = target.inference_model.log_likelihood(
            w=single_w, theta=single_theta, sigma= single_sigma, data=x, interv_targets=interv_targets
        )
        return log_prob_theta + log_lik

eltwise_log_prob = vmap(
        lambda g, theta, sigma, x, interv_targets: log_likelihood(
            g, theta, sigma, x, interv_targets, None
        ),
        (0, 0, 0, None, None),
        0,
    )

n_particles = 20
n_steps = 1000
hparams = DiBSExampleSettings()

# initialize kernel and algorithm
kernel = JointAdditiveFrobeniusSEKernel(
    h_latent=hparams.h_latent,
    h_theta=hparams.h_theta,
    h_sigma=1.0)

dibs = JointDiBS(
    kernel=kernel, 
    target_log_prior=log_prior,
    target_log_joint_prob=log_likelihood,
    alpha_linear=hparams.alpha_linear)
        
# initialize particles
key, subk = random.split(key)
init_particles_z, init_particles_theta, init_particles_sigma = dibs.sample_initial_random_particles(
    key=subk, model=model, n_particles=n_particles, n_vars=n_vars)


key, subk = random.split(key)
interv_targets = jnp.zeros((x.shape[0], n_vars)).astype(bool)
particles_z, particles_theta, particles_sigma = dibs.sample_particles(key=subk, n_steps=n_steps, 
    init_particles_z=init_particles_z, init_particles_theta=init_particles_theta, init_particles_sigma=init_particles_sigma, data=x, interv_targets=interv_targets,
    callback_every=50, callback=None)


particles_g = dibs.particle_to_g_lim(particles_z)
dibs_empirical = particle_joint_empirical(particles_g, particles_theta, particles_sigma)
dibs_mixture = particle_joint_mixture(particles_g, particles_theta, particles_sigma, eltwise_log_prob, x, interv_targets)

for descr, dist in [('DiBS ', dibs_empirical), ('DiBS+', dibs_mixture)]:
    dist_marginal = joint_dist_to_marginal(dist)
    eshd = expected_shd(dist=dist_marginal, g=target.g)        
    auroc = threshold_metrics(dist=dist_marginal, g=target.g)['roc_auc']
    negll = neg_ave_log_likelihood(dist=dist, eltwise_log_joint_target=eltwise_log_prob, x=x_ho, interv_targets=interv_targets)
    
    print(f'{descr} |  E-SHD: {eshd:4.1f}    AUROC: {auroc:5.2f}    neg. LL {negll:5.2f}')