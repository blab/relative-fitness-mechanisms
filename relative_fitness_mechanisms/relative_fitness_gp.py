from functools import partial
from typing import Dict, Optional
import evofr as ef
import jax
import jaxopt
import numpy as np
import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from tinygp import kernels, GaussianProcess

# TODO: Compare to a quick and dirty model where we
# TODO: fit to log(x_{v,pivot}) over time and then estimate marginal likelihood over parameters


# Abstract EvofrGP class
# class EvofrGP:


class SquaredExpCov:
    def __init__(
        self, alpha: Optional[float] = None, rho: Optional[float] = None
    ):
        self.alpha = alpha
        self.rho = rho

    def model_numpyro(self):
        if self.alpha is None:
            alpha = numpyro.sample("alpha", dist.HalfNormal(0.1))
        else:
            alpha = self.alpha

        if self.rho is None:
            rho = numpyro.sample("rho", dist.LogNormal(loc=3, scale=1))
        else:
            rho = self.rho
        return self.kernel(rho, alpha)

    def model_qd(self):
        alpha = 0.1 if self.alpha is None else self.alpha
        rho = 100 if self.rho is None else self.rho
        return self.kernel(rho, alpha)

    @staticmethod
    def kernel(rho, alpha):
        return alpha * kernels.ExpSquared(rho)

    # Basically jsut need arithemetic to act on kernels from model_numpyro
    def build(self, theta, x, diag):
        kernel = self.kernel(
            rho=jnp.exp(theta["log_scale_expsq"]),
            alpha=jnp.exp(theta["log_amp_expsq"]),
        )
        return GaussianProcess(kernel, x, diag=diag)

    def init_params(self):
        alpha = 0.05 if self.alpha is None else self.alpha
        rho = 100 if self.rho is None else self.rho
        return {
            "log_scale_expsq": jnp.log(rho),
            "log_amp_expsq": jnp.log(alpha),
        }


def relative_fitness_gp_numpyro(
    seq_counts, N, gp_kernel, tau=None, pred=False, var_names=None
):
    N_time, N_variants = seq_counts.shape

    # Sample N-1 GPs for variant fitnesses
    # gp_kernel = SquaredExpCov() if gp_kernel is None else gp_kernel
    gp = GaussianProcess(
        gp_kernel.model_numpyro(), jnp.arange(N_time), diag=1e-5
    )

    with numpyro.plate("variant", N_variants - 1):
        _init_logit = numpyro.sample("init_logit", dist.Normal(0, 5.0))
        _fitness = numpyro.sample(
            "_delta",
            dist.MultivariateNormal(gp.loc, scale_tril=gp.solver.scale_tril),
        )
    numpyro.deterministic("delta", _fitness.T)
    fitness = jnp.vstack((_fitness, jnp.zeros(N_time))).T

    # Sum fitness to get dynamics over time
    init_logit = jnp.append(_init_logit, 0.0)
    logits = jnp.cumsum(fitness.at[0, :].set(0), axis=0) + init_logit

    # Evaluate likelihood
    obs = None if pred else np.nan_to_num(seq_counts)
    numpyro.sample(
        "seq_counts",
        dist.MultinomialLogits(logits=logits, total_count=np.nan_to_num(N)),
        obs=obs,
    )

    # Compute frequency
    numpyro.deterministic("freq", softmax(logits, axis=-1))

    # Compute growth advantage from model
    if tau is not None:
        numpyro.deterministic(
            "ga", jnp.exp(fitness[:, :-1] * tau)
        )  # Last row corresponds to linear predictor / growth advantage


# TODO: Set default gp_kernel outside of functions?
def relative_fitness_gp_tinygp(
    seq_counts, N, pseudocount, gp_kernel, tau=None
):
    N_time, N_variant = seq_counts.shape

    # Compute relative frequency
    relative_frequency = (seq_counts[:, :-1] + pseudocount) / (
        (seq_counts[:, -1] + pseudocount)[:, None]
    )
    # empirical_fitness = np.diff(jnp.log(relative_frequency), axis=0)
    time = np.arange(N_time)
    y = jnp.log(relative_frequency)  # or empirical_fitness

    # Get GaussianProcess
    gp_kernel = SquaredExpCov() if gp_kernel is None else gp_kernel
    # gp = GaussianProcess(gp_kernel.model_qd(), time[1:], diag=1e-5)

    # We want to marginalize if no parameters provided, but otherwise not
    def log_prob(theta, x, y):
        _gp = gp_kernel.build(theta, x, 1 / (4 * N))
        logps = jax.vmap(_gp.log_probability, in_axes=-1)(y)
        return -logps.sum()

    # Find theta_init and run solver
    theta_init = gp_kernel.init_params()
    solver = jaxopt.ScipyMinimize(fun=log_prob, method="CG", maxiter=5000)
    soln = solver.run(theta_init, x=time, y=y)
    print(soln)
    print(f"Final negative log likelihood: {soln.state.fun_val}")

    # Build gps with found parameters and condition on data
    gp = gp_kernel.build(soln.params, time)
    cond_gps = [gp.condition(y[:, v], time)[-1] for v in range(N_variant - 1)]

    # Now process these GPs to generate frequencies and delta over time

    # Start with samples since we know how to wokr with them
    samples = dict()
    log_rel_freq = [
        cdgp.sample(jax.random.PRNGKey(10), shape=(100,)) for cdgp in cond_gps
    ]
    log_rel_freq = jnp.stack(log_rel_freq, axis=-1)
    rel_freq = jnp.concatenate(
        (jnp.exp(log_rel_freq), jnp.ones((100, N_time, 1))),
        axis=-1,
    )
    samples["freq"] = rel_freq / rel_freq.sum(axis=-1)[:, :, None]
    samples["delta"] = jnp.diff(log_rel_freq, axis=1)
    return cond_gps, y, samples

    # cond_gps = [
    #     gp.condition(empirical_fitness[:, v])[-1] for v in range(N_variant - 1)
    # ]
    # return [cond_gps, empirical_fitness]

    # return [
    #     (gp.condition(empirical_fitness[:, v])[-1], empirical_fitness[:, v])
    #     for v in range(N_variant - 1)
    # ]
    # return vmap(gp.condition, in_axes=-1, out_axes=-1)(empirical_fitness)

    # return gp.condition(empirical_fitness)

    # TODO: Make marginalizing easy


class RelativeFitnessGP(ef.ModelSpec):
    def __init__(
        self,
        gp: Optional[GaussianProcess] = None,
        tau: Optional[float] = None,
        pseudocount: float = 1,
    ):
        self.gp = gp if gp is not None else SquaredExpCov()
        self.tau = tau
        self.pseudocount = pseudocount
        self.model_fn = partial(
            relative_fitness_gp_numpyro, gp_kernel=self.gp, tau=self.tau
        )

    def augment_data(self, data: dict) -> None:
        return None

    def fit_mcmc(
        self,
        data: ef.VariantFrequencies,
        num_warmup: int = 100,
        num_samples: int = 100,
    ) -> ef.PosteriorHandler:
        """
        Abstract away NUTS stuff in Evofr and numpyro for quick usage.
        """
        inference_method = ef.InferNUTS(
            num_warmup=num_warmup, num_samples=num_samples
        )
        return inference_method.fit(self, data)

    def fit_qd(self, data: ef.VariantFrequencies):
        """
        Fit quick and dirty model using pseudocounts and relative frequencies
        """
        return relative_fitness_gp_tinygp(
            data.seq_counts,
            data.seq_counts.sum(axis=-1),
            self.pseudocount,
            self.gp,
            self.tau,
        )

    def forecast_mcmc(self, samples, forecast_L):

        # TODO: Add options for computing forecast date .etc.

        # Create time points to forecast
        N_samples, N_time, _ = samples["delta"].shape
        ts = jnp.arange(N_time)
        pred_ts = np.arange(1, forecast_L + 1) + N_time

        # Define GP
        # Need to condition on samples though right?
        gp = GaussianProcess(self.gp.model_numpyro(), ts, diag=1e-5)

        # Forecast relative fitness
        def _condition_sample(fitness, xs, key):
            gp_cond = gp.condition(fitness, xs)
            return gp_cond.gp.sample(key)

        ## Map over sample and variant
        condition_sample = jax.vmap(
            jax.vmap(_condition_sample, in_axes=(1, None, None), out_axes=1),
            in_axes=(0, None, 0),
            out_axes=0,
        )

        keys = jax.random.PRNGKey(1)
        keys = jax.random.split(keys, N_samples)
        samples["delta_forecast"] = condition_sample(
            samples["delta"], pred_ts, keys
        )

        # Forecast frequency
        def _forecast_freq(fitness, frequency):
            init_freq = jnp.log(frequency[-1, :])  # Last known frequency
            _fitness = jnp.concatenate(
                (fitness, jnp.zeros((forecast_L, 1))), axis=-1
            )
            cum_fitness = jnp.cumsum(_fitness.at[0, :].set(0), axis=0)
            return softmax(cum_fitness + init_freq, axis=-1)

        vmap_forecast_freq = jax.vmap(
            _forecast_freq, in_axes=(0, 0), out_axes=0
        )
        samples["freq_forecast"] = vmap_forecast_freq(
            samples["delta_forecast"], samples["freq"]
        )
        return samples
