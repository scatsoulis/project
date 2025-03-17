import numpy as np
from KS_solver import KS_step, KS_step_nudge
from jax_models import KuramotoSivashinsky
import matplotlib.pyplot as plt


def main():
    path = "./"

    # define our KS domain and system
    dt = 0.25
    s = 256  # number of points in domain
    M = 16
    l = 2 * np.pi * 16  # domain length
    dt = 0.25  # time step size
    train_period = 2000  # total simulation time
    KS = KuramotoSivashinsky(dt=dt, s=s, l=l, M=M)

    # Define parameters for nudging
    eta = 1e-1  # standard deviation of observation gaussian noise relative to std of KS output
    mu = 10 ** (-1.05)

    # Define KS characteristics for low-resolution system
    cf = 2  # Coarsen factor
    KS_low = KuramotoSivashinsky(dt=dt, s=s // cf, l=l, M=M, mu=mu)
    ndata = 1  # number of ensemble members to generate data for

    x = np.arange(0, l, l / s)
    # Generate our high resolution simulation data
    data = np.zeros((ndata, train_period + 1, s))
    for i in range(ndata):
        f = f_rand(KS.l)
        data[i, :, :] = KS_step(KS, train_period, f(x))

    # Cut off initial burn-in period
    data = data[:, 400:]

    std = np.std(data[0, 0])

    # Generate low-resolution data from same starting point as our high-resolution data
    data_low = np.zeros((ndata, len(data[0]), s // cf))
    data_low_nudge = np.zeros((ndata, len(data[0]), s // cf))
    for i in range(ndata):
        obs = data[i, :, ::cf]
        obs = obs + np.random.normal(scale=eta * std, size=obs.shape)
        data_low[i, :, :] = KS_step(KS_low, len(data[i]) - 1, data[i, 0, ::cf])
        data_low_nudge[i, :, :] = KS_step_nudge(KS_low, len(data[i]) - 1, data[i, 0, ::cf], obs)

    print("nudged error", np.mean(np.abs(data[0, -1, ::cf] - data_low_nudge[0, -1])))
    print("resolution error", np.mean(np.abs(data_low[0, -1] - data[0, -1, ::cf])))

    # Plotting
    x_plot = np.linspace(0, l, s + 1)
    x_low_plot = np.linspace(0, l, s // cf + 1)
    t_plot = np.arange(0, dt * (len(data[0]) + 1), dt)

    # Only look at first ensemble member
    data = data[0]
    data_low = data_low[0]
    data_low_nudge = data_low_nudge[0]

    fig, ax = plt.subplots(3, 2, figsize=(16, 16))

    # Plot high resolution at top of each column of axes
    cm1 = ax[0, 0].pcolormesh(t_plot, x_plot, data.T, cmap="plasma")
    plt.colorbar(cm1, ax=ax[0, 0])
    ax[0, 1].pcolormesh(t_plot, x_plot, data.T, cmap="plasma")
    plt.colorbar(cm1, ax=ax[0, 1])

    # Plot low resolution results and errors
    ax[1, 0].pcolormesh(t_plot, x_low_plot, data_low.T, cmap="plasma")
    plt.colorbar(cm1, ax=ax[1, 0])
    dif = data_low - data[:, ::cf]
    cm = ax[2, 0].pcolormesh(t_plot, x_low_plot, dif.T, cmap="bwr")
    plt.colorbar(cm, ax=ax[2, 0])

    # Plot low resolution nudged results and errors
    ax[1, 1].pcolormesh(t_plot, x_low_plot, data_low_nudge.T, cmap="plasma")
    plt.colorbar(cm1, ax=ax[1, 1])
    dif = data_low_nudge - data[:, ::cf]
    cm = ax[2, 1].pcolormesh(t_plot, x_low_plot, dif.T, cmap="bwr")
    plt.colorbar(cm, ax=ax[2, 1])

    # Other plot characteristics
    ax[0, 0].set_title("Full Resolution")
    ax[0, 1].set_title("Full Resolution")
    ax[1, 0].set_title(f"1/{cf} Resolution")
    ax[2, 0].set_title("Difference")
    ax[2, 1].set_title("Difference")
    ax[1, 1].set_title(f"1/{cf} Resolution Nudged")
    ax[2, 0].set_xlabel("Time [s]")
    ax[2, 1].set_xlabel("Time [s]")
    for axi in ax[:, 0]:
        axi.set_ylabel("x")

    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    plt.savefig(f"{path}/fig.png", bbox_inches="tight")


def f_rand(L):
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
    maxx = max(np.abs(mina), maxa)

    def f(x):
        fx = x * 0
        for i in range(nwaves):
            fx = fx + b[i] * np.sin(2 * a[i] * np.pi * (x - d[i]) / L)

        return fx / maxx

    return f


if __name__ == "__main__":
    main()
