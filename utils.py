import jax.numpy as jnp
from jax import random, jit, lax
import numpy as np
from scipy.fft import fft, fftfreq

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from functools import partial


@jit
def step_function(carry, input) -> tuple:
    key, x, observation_interval, H, Q, R, model_step, counter = carry
    n = len(x)
    key, subkey = random.split(key)
    x_j = model_step(x)

    # Add process noise Q only at observation times using a conditional operation
    def update_observation() -> tuple:
        x_noise = x_j + random.multivariate_normal(key, jnp.zeros(n), Q)
        obs_state = jnp.dot(H, x_noise)
        # Adjust noise dimension to the number of observed states
        obs_noise = random.multivariate_normal(subkey, jnp.zeros(H.shape[0]), R)

        return x_noise, obs_state + obs_noise

    def no_update() -> tuple:
        # Return a vector of NaNs matching the number of observed states
        return x_j, jnp.nan * jnp.ones(H.shape[0])

    # Conditional update based on the observation interval
    x_j, obs = lax.cond(counter % observation_interval == 0, update_observation, no_update)
    counter += 1
    carry = (key, x_j, observation_interval, H, Q, R, model_step, counter)
    output = (x_j, obs)

    return carry, output


@partial(jit, static_argnums=(1, 2, 7))
def generate_true_states(key, num_steps, n, x0, H, Q, R, model_step, observation_interval) -> tuple:
    initial_carry = (key, x0, observation_interval, H, Q, R, model_step, 1)
    _, (xs, observations) = lax.scan(step_function, initial_carry, None, length=num_steps - 1)
    key, subkey = random.split(key)
    initial_observation = H @ x0 + random.multivariate_normal(subkey, jnp.zeros(H.shape[0]), R)
    xs = jnp.vstack([x0[jnp.newaxis, :], xs])
    observations = jnp.vstack([initial_observation[jnp.newaxis, :], observations])

    return observations, xs


def visualize_observations(observations) -> None:
    observation_values = observations.T  # Transpose for plotting
    cmap = LinearSegmentedColormap.from_list("CustomColormap", [(0, "blue"), (0.5, "white"), (1, "red")])
    plt.figure(figsize=(12, 6))
    plt.imshow(
        observation_values,
        cmap=cmap,
        aspect="auto",
        extent=[0, observations.shape[0], 0, observations.shape[1]],
    )
    plt.colorbar(label="Observation Value")
    plt.xlabel("Time Step")
    plt.ylabel("State/Variable Number")
    plt.title("Observations Over Time")
    plt.show()


def plot_ensemble_mean_and_variance(states, observations, state_index, observation_interval, title_suffix="") -> None:
    time_steps = jnp.arange(states.shape[0])
    state_mean = jnp.mean(states[:, :, state_index], axis=1)
    state_std = jnp.std(states[:, :, state_index], axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps, state_mean, label="State Mean", color="orange")
    plt.fill_between(
        time_steps,
        state_mean - 1.96 * state_std,
        state_mean + 1.96 * state_std,
        color="orange",
        alpha=0.3,
        label="95% Confidence Interval",
    )
    observed_time_steps = jnp.arange(0, len(observations), observation_interval)
    observed_values = observations[observed_time_steps, state_index]
    plt.scatter(
        observed_time_steps,
        observed_values,
        label="Observation",
        color="red",
        marker="x",
    )

    # plt.title(f'State {state_index+1} Ensemble Mean and Variance {title_suffix}')
    plt.xlabel("Time Step")
    # plt.ylabel(f'State {state_index+1} Value')
    plt.legend()
    plt.show()


@partial(jit, static_argnums=(0))
def generate_localization_matrix(n, localization_radius) -> jnp.ndarray:
    """
    Generate a localization matrix with given radius
    """
    i = jnp.arange(n)[:, None]
    j = jnp.arange(n)
    min_modulo_distance = jnp.minimum(jnp.abs(i - j), n - jnp.abs(i - j))
    r = min_modulo_distance / localization_radius
    localization_matrix = jnp.exp(-(r**2))
    return localization_matrix


def spectrum_log(field: np.ndarray, res: float, title: str = None, save_path: str = None) -> None:
    """Logarithmic spectrum plotter"""

    Nx = field.shape[-1]

    field_freq = 2 * jnp.pi * fftfreq(Nx, d=res)[: Nx // 2]
    field_amp = 2 * np.abs(fft(field, norm="forward"))[: Nx // 2]
    field_amp[0] = 0.5 * field_amp[0]  # DC correction

    # decades = [(10**i) for i in range(0, -16)]  # Adjust range as needed
    fig, ax = plt.subplots()
    ax.plot(field_freq, field_amp, color="#325cab")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$A$")
    ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, transparent=False, bbox_inches="tight")
        plt.close(fig)


def spectrum_log_output(field: np.ndarray, res: float, title: str = None, save_path: str = None) -> None:
    """Logarithmic spectrum plotter"""

    Nx = field.shape[-1]

    field_freq = 2 * jnp.pi * fftfreq(Nx, d=res)[: Nx // 2]
    field_amp = 2 * np.abs(fft(field, norm="forward"))[: Nx // 2]
    field_amp[0] = 0.5 * field_amp[0]  # DC correction

    # decades = [(10**i) for i in range(0, -16)]  # Adjust range as needed
    # fig, ax = plt.subplots()
    # ax.plot(field_freq, field_amp, color="#325cab")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_xlabel("$k$")
    # ax.set_ylabel("$A$")
    # ax.set_title(title)
    # if save_path is not None:
    #     fig.savefig(save_path, dpi=300, transparent=False, bbox_inches="tight")
    #     plt.close(fig)

    return field_amp


def spectrum_linear(field: np.ndarray, res: float, title: str = None, save_path: str = None) -> None:
    """Standard spectrum plotter"""

    Nx = field.shape[-1]
    field_freq = 2 * np.pi * fftfreq(Nx, d=res)[: Nx // 2]
    field_amp = 2 * np.abs(fft(field, norm="forward")[: Nx // 2])
    field_amp[0] = 0.5 * field_amp[0]  # DC correction

    fig, ax = plt.subplots()
    ax.plot(field_freq, field_amp, color="#325cab")
    ax.set_xlim(
        0,
    )
    ax.set_ylim(
        0,
    )
    ax.set_xlabel("$k$")
    ax.set_ylabel("$A$")
    ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, transparent=False, bbox_inches="tight")
        plt.close(fig)
