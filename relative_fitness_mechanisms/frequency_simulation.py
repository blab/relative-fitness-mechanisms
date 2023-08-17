import jax.numpy as jnp
from jax import jit, lax
from functools import partial


@staticmethod
def freq_step(freq, delta):
    return jnp.dot(delta[:, None] - delta, freq) * freq


class KnownRelativeFitness:
    def __init__(self):
        pass

    @staticmethod
    def simulate_frequencies(freq0, delta):
        @jit
        def _freq_scan(freq, delta):
            freq_next = freq + freq_step(freq, delta)
            freq_next = jnp.clip(freq_next, 0.0, 1.0)
            return freq_next, freq_next

        _, freq = lax.scan(_freq_scan, init=freq0, xs=delta)
        return freq


class UnknownRelativeFitness:
    def __init__(self, compute_delta, update_state):
        self.compute_delta = compute_delta
        self.update_state = update_state

    @staticmethod
    @partial(
        jit, static_argnames=["compute_delta", "update_state", "num_steps"]
    )
    def simulate_frequencies(
        freq0, state0, compute_delta, update_state, num_steps
    ):
        # def _make_freq_scan(compute_delta, update_state):
        def _freq_scan(carry, _):
            freq, state = carry
            # Compute relative fitness
            delta = compute_delta(state, freq)

            # Update frequencies
            freq_next = freq + freq_step(freq, delta)
            freq_next = jnp.clip(freq_next, 0.0, 1.0)

            # Update hidden state
            state_next = update_state(state, freq)
            return (freq_next, state_next), (freq_next, state_next)

        #     return _freq_scan
        #
        # _freq_scan = jit(_make_freq_scan(compute_delta, update_state))
        _, (freq, state) = lax.scan(
            _freq_scan, init=(freq0, state0), xs=None, length=num_steps
        )
        return freq, state


# We then want to wrap this for the escape computation
def simulate_escape_transmission(freq0, beta, escape, S, phi):
    delta = beta * S[:, None] + beta * escape * phi[:, None]
    delta_norm = delta - delta[:, -1]  # should be (T, V)
    return KnownRelativeFitness.simulate_frequencies(freq0, delta_norm)


def simulate_escape(freq0, beta, escape_mat, phi):
    # escape_mat: [V, S], phi: [T, S]
    # beta: float, freq0: [V]
    delta = beta * jnp.einsum("vs, ts -> vt", escape_mat @ phi)
    delta_norm = delta - delta[:, -1]
    return KnownRelativeFitness.simulate_frequencies(freq0, delta_norm)
