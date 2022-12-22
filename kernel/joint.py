
import jax.numpy as jnp
import jax.lax as lax

from ..utils.func import squared_norm_pytree


class JointMultiplicativeFrobeniusSEKernel:
    """
    Squared exponential kernel, that simply computes the 
    exponentiated quadratic of the difference in Frobenius norms

    k([Z, theta], [Z', theta']) = 
        scale * exp(- 1/h_z   ||Z - Z'||^2_F  
                    - 1/h_th  ||theta - theta'||^2_F )

    """

    def __init__(self, *, h_latent, h_theta, h_sigma, scale=1.0):
        super(JointMultiplicativeFrobeniusSEKernel, self).__init__()

        self.h_latent = h_latent
        self.h_theta = h_theta
        self.h_sigma = h_sigma
        self.scale = scale
        
        
    def eval(self, *, x_latent, x_theta, x_sigma, y_latent, y_theta, y_sigma, h_latent=-1.0, h_theta=-1.0, h_sigma=-1.0):
        """Evaluates kernel function k(x, y) 
        
        Args:
            x_latent: [...]
            x_theta: PyTree 
            y_latent: [...]
            y_theta: PyTree 
            h_latent: bandwidth for Z term; h_latent == -1 indicates class setting is used
            h_theta: bandwidth for Z term; h_theta == -1 indicates class setting is used
        
        Returns: 
            [1, ]
        """
        # bandwidth (jax-consistent checking which h is used)
        h_latent_ = lax.cond(
            h_latent == -1.0,
            lambda _: self.h_latent,
            lambda _: h_latent,
            operand=None)

        h_theta_ = lax.cond(
            h_theta == -1.0,
            lambda _: self.h_theta,
            lambda _: h_theta,
            operand=None)

        h_sigma_ = lax.cond(
            h_sigma == -1.0,
            lambda _: self.h_sigma,
            lambda _: h_sigma,
            operand=None)
        # compute norm
        latent_squared_norm = jnp.sum((x_latent - y_latent) ** 2.0)
        theta_squared_norm = squared_norm_pytree(x_theta, y_theta)
        sigma_squared_norm = jnp.sum((x_sigma - y_sigma) ** 2.0)
        # compute kernel
        return self.scale * jnp.exp(- (latent_squared_norm / h_latent_) - (theta_squared_norm / h_theta_)- (sigma_squared_norm / h_sigma_))


class JointAdditiveFrobeniusSEKernel:
    """
    Squared exponential kernel, that simply computes the 
    exponentiated quadratic of the difference in Frobenius norms

    k([Z, theta], [Z', theta']) = 
        scale_z     * exp(- 1/h_z  ||Z - Z'||^2_F)
      + scale_theta * exp(- 1/h_th ||theta - theta'||^2_F )

    """

    def __init__(self, *, h_latent, h_theta, h_sigma, scale_latent=1.0, scale_theta=1.0, scale_sigma=1.0):
        super(JointAdditiveFrobeniusSEKernel, self).__init__()

        self.h_latent = h_latent
        self.h_theta = h_theta
        self.h_sigma = h_sigma
        self.scale_latent = scale_latent
        self.scale_theta = scale_theta
        self.scale_sigma = scale_sigma

    def eval(self, *, x_latent, x_theta, x_sigma, y_latent, y_theta, y_sigma, h_latent=-1.0, h_theta=-1.0, h_sigma=-1.0):
        """Evaluates kernel function k(x, y) 
        
        Args:
            x_latent: [...]
            x_theta: PyTree 
            y_latent: [...]
            y_theta: PyTree 
            h_latent: bandwidth for Z term; h_latent == -1 indicates class setting is used
            h_theta: bandwidth for Z term; h_theta == -1 indicates class setting is used
        
        Returns: 
            [1, ]
        """
        # bandwidth (jax-consistent checking which h is used)
        h_latent_ = lax.cond(
            h_latent == -1.0,
            lambda _: self.h_latent,
            lambda _: h_latent,
            operand=None)

        h_theta_ = lax.cond(
            h_theta == -1.0,
            lambda _: self.h_theta,
            lambda _: h_theta,
            operand=None)

        h_sigma_ = lax.cond(
            h_sigma == -1.0,
            lambda _: self.h_sigma,
            lambda _: h_sigma,
            operand=None)
        # compute norm
        latent_squared_norm = jnp.sum((x_latent - y_latent) ** 2.0)
        theta_squared_norm = squared_norm_pytree(x_theta, y_theta)
        sigma_squared_norm = jnp.sum((x_sigma - y_sigma) ** 2.0)
        # compute kernel
        return (self.scale_latent * jnp.exp(- latent_squared_norm / h_latent_)
              + self.scale_theta  * jnp.exp(- theta_squared_norm  / h_theta_ )
              + self.scale_sigma  * jnp.exp(- sigma_squared_norm  / h_sigma_ ))

