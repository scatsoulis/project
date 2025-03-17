import sys
import numpy as np
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

import matplotlib.pyplot as plt


# Plotting parameters
plt.close("all")
params = {
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.sans-serif": "Calibri",
    "font.size": 24,
    "legend.fontsize": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "lines.linewidth": 2,
    "text.usetex": True,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": False,
    "legend.loc": "best",
    "legend.edgecolor": "1.0",
    "lines.markersize": 4.0,
    "figure.constrained_layout.use": True,
    "axes.formatter.use_mathtext": True,
    "savefig.dpi": 300,
}
plt.rcParams.update(params)


@jit
def kuramoto_sivashinsky_step(u: jnp.ndarray, dt: float, E, E2, Q, f1, f2, f3, g) -> jnp.ndarray:
    u_fft = jnp.fft.fft(u)

    Nv = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(u_fft)) ** 2)
    a = E2 * u_fft + Q * Nv

    Na = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(a)) ** 2)
    b = E2 * u_fft + Q * Na

    Nb = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(b)) ** 2)
    c = E2 * a + Q * (2 * Nb - Nv)

    Nc = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(c)) ** 2)

    u_fft_next = E * u_fft + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3  #  + E_forced*mu * jnp.fft.fft(yobs - v)
    u_next = jnp.real(jnp.fft.ifft(u_fft_next))

    return u_next


# Models as classes
class BaseModel:
    def __init__(self, dt=0.01):
        self.dt = dt

    def step(self, x):
        raise NotImplementedError("The step method must be implemented by subclasses.")


class KuramotoSivashinsky(BaseModel):
    def __init__(self, dt: float, Nx: int, L: int, modes: int, lambda_param: float) -> None:
        super().__init__(dt)
        self.Nx, self.L, self.modes, self.lambda_param = (Nx, L, modes, lambda_param)  # discr. points, domain length, exponential time differencing points (modes)
        self.k, self.E, self.E2, self.Q, self.f1, self.f2, self.f3, self.g = self.precompute_constants()

    def precompute_constants(self) -> tuple:
        k = (2 * jnp.pi / self.L) * jnp.concatenate(
            [
                jnp.arange(0, self.Nx // 2),
                jnp.array([0]),
                jnp.arange(-self.Nx // 2 + 1, 0),
            ]
        )
        Lk = self.lambda_param * k**2 - k**4
        E = jnp.exp(self.dt * Lk)
        E2 = jnp.exp(self.dt * Lk / 2)
        r = jnp.exp(1j * jnp.pi * (jnp.arange(1, self.modes + 1) - 0.5) / self.modes)
        LR = self.dt * jnp.tile(Lk, (self.modes, 1)).T + jnp.tile(r, (self.Nx, 1))
        Q = self.dt * jnp.real(jnp.mean((jnp.exp(LR / 2) - 1) / LR, axis=1))

        f1 = self.dt * jnp.real(jnp.mean((-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
        f2 = self.dt * jnp.real(jnp.mean((2 + LR + jnp.exp(LR) * (-2 + LR)) / LR**3, axis=1))
        f3 = self.dt * jnp.real(jnp.mean((-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / LR**3, axis=1))
        g = -0.5j * k

        return k, E, E2, Q, f1, f2, f3, g

    def step(self, u: jnp.ndarray) -> jnp.ndarray:
        return kuramoto_sivashinsky_step(u, self.dt, self.E, self.E2, self.Q, self.f1, self.f2, self.f3, self.g)


def u_rand(L: float) -> callable:
    nwaves = 5
    a = np.arange(nwaves) + 1
    b = np.random.rand(nwaves)
    d = np.random.rand(nwaves)

    xi = np.linspace(0, L, 1000)
    fx = xi * 0
    for i in range(nwaves):
        fx = fx + b[i] * np.sin(2 * a[i] * np.pi * (xi - d[i]) / L)

    mina = np.min(fx)
    maxa = np.max(fx)

    maxx = max(jnp.abs(mina), maxa)

    def f(x: np.ndarray) -> np.ndarray:
        fx = x * 0
        for i in range(nwaves):
            fx = fx + b[i] * np.sin(2 * a[i] * np.pi * (x - d[i]) / L)

        return fx / maxx

    return f


if __name__ == "__main__":
    N = 100
    Nx = 256  # Spatial grid points
    L = 2 * np.pi * 16  # Domain length
    dt = 0.25  # Time-step size
    modes = 16
    lambda_param_true = 1

    iterations = 2000  # Total simulation time-steps
    warmup = 400
    obs_interval = 1  # Observation assimilation interval
    J = (iterations - warmup) // obs_interval

    # Stochastics
    noise_switch = 1
    obs_noise = 0.1  # Observation noise level
    rng_key = np.random.seed(0)  # random key for reproducibility

    # Instantiate KS solver
    ks_dynamics_true = KuramotoSivashinsky(dt, Nx, L, modes, lambda_param_true)  # Kuramoto-Sivashinsky model

    # Generate a different IC for each particle & simulate the truth
    u_truth = np.zeros((iterations + 1, N, Nx))
    for n in tqdm(range(N)):
        u_truth[0, n] = u_rand(ks_dynamics_true.L)(np.linspace(0, L, Nx))  # Initial condition for truth
        for itera in range(iterations):
            u_truth[itera + 1, n] = ks_dynamics_true.step(u_truth[itera, n])

    # Remove burn-in period
    u_truth = u_truth[400:]

    # Initialize u_ensemble with ICs at burn-in
    u_ensemble = np.zeros((J + 1, N, Nx))
    u_ensemble[0] = [u_truth[0, n] for n in range(N)]  # Initial ensemble members
    # u_ensemble[0, n] += np.sqrt(1) * np.random.normal(0, 1, size=(N, Nx))  # Add noise to the initial ensemble members

    # Generate new truth for assimilation
    u_truth = np.zeros((iterations + 1, Nx))
    u_truth[0] = u_rand(ks_dynamics_true.L)(np.linspace(0, L, Nx))
    for itera in range(iterations):
        u_truth[itera + 1] = ks_dynamics_true.step(u_truth[itera])  # Advance the truth
    u_truth = u_truth[warmup:]  # Truncate burn-in period

    # Initialize y_obs
    y_obs = np.zeros((J + 1, N, Nx))

    # Assimilation loop
    def enkf(lambda_input: float, y_obs: np.ndarray, N: int, obs_noise: float) -> np.ndarray:
        ks_dynamics_test = KuramotoSivashinsky(dt, Nx, L, modes, lambda_input)  # Kuramoto-Sivashinsky model

        for itera in tqdm(range(iterations - warmup)):
            if itera % obs_interval == 0:  # Assimilate every obs_interval steps
                # Prediction step
                for n in range(N):
                    u_ensemble[(itera // obs_interval) + 1, n] = ks_dynamics_test.step(u_ensemble[itera // obs_interval, n])  # Advance each particle by an assimilation step

                mean = np.mean(u_ensemble[(itera // obs_interval) + 1], axis=0, keepdims=True)  # Mean of the ensemble
                cov = np.cov(u_ensemble[(itera // obs_interval) + 1] - mean, rowvar=True)  # (N, N) matrix

                # Analysis step
                y_obs[(itera // obs_interval) + 1] = u_truth[(itera // obs_interval) + 1] + np.sqrt(obs_noise) * np.random.normal(0, 1, size=(N, Nx))  # Add noise to the true state
                S = cov + obs_noise * np.ones((N, N))  # (N, N)
                K = np.matmul(cov, np.linalg.inv(S))  # (N, N)
                u_ensemble[(itera // obs_interval) + 1] = u_ensemble[(itera // obs_interval) + 1] + K @ (y_obs[(itera // obs_interval) + 1] - u_ensemble[(itera // obs_interval) + 1])  # (N, Nx)

        return u_ensemble

    def param_estimation_whole(u_ensemble: np.ndarray, u_truth: np.ndarray) -> float:
        u_ensemble_mean = np.mean(u_ensemble, axis=1)  # Mean of the ensemble

        ad = np.abs(u_ensemble_mean - u_truth)  # Mean absolute deviation
        min_u, max_u = np.min(u_truth), np.max(u_truth)
        nad = ad / (max_u - min_u)  # Normalized absolute deviation
        mnad = np.mean(nad, axis=1)  # Mean normalized absolute deviation
        tmnad = np.mean(mnad)  # Mean normalized absolute deviation

        return tmnad

    def param_estimation_last(u_ensemble: np.ndarray, u_truth: np.ndarray) -> float:
        u_ensemble_mean = np.mean(u_ensemble, axis=1)  # Mean of the ensemble

        ad = np.abs(u_ensemble_mean - u_truth)  # Mean absolute deviation
        min_u, max_u = np.min(u_truth), np.max(u_truth)
        nad = ad / (max_u - min_u)  # Normalized absolute deviation
        mnad = np.mean(nad, axis=1)  # Mean normalized absolute deviation
        tmnad = mnad[-1]  # Mean normalized absolute deviation

        return tmnad

    t_mnad_hist_whole = []
    t_mnad_hist_last = []
    lambda_space = np.arange(0.0, 2.2, 0.2)
    for lambda_input in tqdm(lambda_space):
        print(f"Testing lambda_input: {lambda_input}")
        u_ensemble = enkf(lambda_input, y_obs, N, obs_noise)

        tmnad_whole = param_estimation_whole(u_ensemble, u_truth)
        t_mnad_hist_whole.append(tmnad_whole)

        tmnad_last = param_estimation_last(u_ensemble, u_truth)
        t_mnad_hist_last.append(tmnad_last)

    optimal_lambda_whole = lambda_space[np.argmin(t_mnad_hist_whole)]
    optimal_lambda_last = lambda_space[np.argmin(t_mnad_hist_last)]

    plt.figure(figsize=(12, 6))
    plt.plot(lambda_space, t_mnad_hist_whole, marker="x", linestyle="--", color="blue", markersize=12, label="Entire horizon")
    plt.plot(lambda_space, t_mnad_hist_last, marker="x", linestyle="--", color="deepskyblue", markersize=12, label="Last time-step")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$tMNAD(u, \hat{u})$")
    plt.xticks(np.arange(0.0, 2.2, 0.2))
    plt.xlim(0, 2.0)
    plt.ylim(
        0,
    )
    plt.legend()
    plt.savefig("lambda.pdf", dpi=300, bbox_inches="tight", transparent=False)

    sys.exit()

    # def em_algorithm(y_obs: np.ndarray, J: int, N: int, max_iters: int = 50, tol: float = 1e-3) -> float:
    #     theta = 1.50  # Initial guess
    #     theta_log = np.zeros(max_iters)
    #     theta_log[0] = theta
    #     for iteration in range(max_iters):
    #         print(f"Iteration {iteration}: theta = {theta:.4f}")
    #         v_ens = enkf(theta, y_obs, N, J)

    #         num = 0
    #         den = 0
    #         for n in range(N):
    #             for j in range(J):
    #                 num += v_ens[j + 1, n] * v_ens[j, n]
    #                 den += v_ens[j, n] ** 2
    #         theta_new = num / den

    #         if abs(theta_new - theta) < tol:
    #             break
    #         theta = theta_new

    #         theta_log[iteration + 1] = theta

    #     return theta, theta_log, iteration
    N_space = [1, 5]  # , 100, 200]
    u_ens_n = np.zeros((2, J + 1, N, Nx))
    for i, n in enumerate(N_space):
        u_ensemble = enkf(lambda_param_true, y_obs, n, obs_noise)  # Run the EnKF with the true lambda
        u_ens_n[i] = u_ensemble  # Store the ensemble for each N

    u_ens_n_mean = param
    plt.figure(figsize=(12, 6))
    plt.boxplot(u_ens_n)
    plt.colorbar(cmap="plasma", label=r"$u(x, t)$", extend="both")
    plt.xlabel(r"$t/\tau_L$")
    plt.ylabel(r"$x$")
    plt.savefig("u_truth.pdf", dpi=300, bbox_inches="tight", transparent=False)

    # Plotting
    grid_time = np.linspace(0, J, J + 1)
    u_ensemble_mean = np.mean(u_ensemble, axis=1)  # Mean of the ensemble
    u_ensemble_std = np.std(u_ensemble, axis=1)  # Standard deviation of the ensemble

    ad = np.abs(u_ensemble_mean - u_truth)  # Mean absolute deviation
    min_u, max_u = np.min(u_truth), np.max(u_truth)
    nad = ad / (max_u - min_u)  # Normalized absolute deviation
    mnad = np.mean(nad, axis=1)  # Mean normalized absolute deviation

    tmnad = np.mean(mnad)  # Mean normalized absolute deviation

    print("nad shape:", nad.shape)
    print("mnad shape:", mnad.shape)
    print("tmnad shape:", tmnad.shape)

    # u field
    plt.figure(figsize=(12, 6))
    plt.imshow(
        u_truth.T,
        cmap="plasma",
        aspect="auto",
        extent=[0, u_truth.shape[0] * dt / 11.46, 0, L],
    )
    plt.colorbar(cmap="plasma", label=r"$u(x, t)$", extend="both")
    plt.xlabel(r"$t/\tau_L$")
    plt.ylabel(r"$x$")
    plt.savefig("u_truth.pdf", dpi=300, bbox_inches="tight", transparent=False)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        u_ensemble_mean.T,
        cmap="plasma",
        aspect="auto",
        extent=[0, u_truth.shape[0] * dt / 11.46, 0, L],
    )
    plt.colorbar(cmap="plasma", label=r"$u(x, t)$", extend="both")
    plt.xlabel(r"$t/\tau_L$")
    plt.ylabel(r"$x$")
    plt.savefig("u_pred.pdf", dpi=300, bbox_inches="tight", transparent=False)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        nad.T,
        cmap="Reds",
        aspect="auto",
        extent=[0, u_truth.shape[0] * dt / 11.46, 0, L],
    )
    plt.colorbar(label=r"NAD", extend="both")
    plt.xlabel(r"$t/\tau_L$")
    plt.ylabel(r"$x$")
    plt.savefig("nad.pdf", dpi=300, bbox_inches="tight", transparent=False)

    u_ens_std_mean = np.mean(u_ensemble_std, axis=1) / (max_u - min_u)
    log_lower_bound = np.log(np.maximum(mnad - 1.96 * u_ens_std_mean, 1e-10))
    log_upper_bound = np.log(mnad + 1.96 * u_ens_std_mean)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(grid_time, np.log(mnad), color="blue", label=r"$MNAD$", linewidth=2)
    plt.fill_between(grid_time[J / 5 :], log_lower_bound[J / 5 :], log_upper_bound[J / 5 :], color="blue", alpha=0.2)
    plt.xlabel(r"$t/\tau_L$")
    plt.ylabel(r"log(MNAD)")
    plt.xlim(0, u_truth.shape[0] * dt / 11.46)
    plt.savefig("mnad.pdf", dpi=300, bbox_inches="tight", transparent=False)

    sys.exit()
