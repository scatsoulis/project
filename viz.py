import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.fft import fftfreq
from matplotlib.colors import LogNorm


from utils import spectrum_log, spectrum_linear, spectrum_log_output


# Plotting parameters
plt.close("all")
params = {
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.sans-serif": "Calibri",
    "font.size": 18,
    "legend.fontsize": 18,
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

# data = np.load("data/KSdata_dt0.25_Nx256_L100.53_M16_time2000_samples2_lambda2.npy")
data = np.load("data/KSdata_dt0.25_Nx256_L100.53096491487338_M16_time2000_samples2_lambda1.npy")

u = data[0, :, :]

grid_x = jnp.linspace(0.0, 22.0, u.shape[1])
u0 = u[0]
u_burnin = u[400]
u_last = u[-1]

_, ax = plt.subplots()
ax.plot(grid_x, u[0].squeeze(), label=r"$u_0$", color="blue")
ax.set_xlim(0.0, 22.0)
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

_, ax = plt.subplots()
ax.plot(grid_x, u_burnin.squeeze(), label=r"$u_0$", color="blue")
ax.set_xlim(0.0, 22.0)
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

_, ax = plt.subplots()
ax.plot(grid_x, u[-1].squeeze(), label=r"$u_0$", color="blue")
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

spectrum_linear(u0, res=2 * np.pi * 16 / 256)
spectrum_log(u0, res=2 * np.pi * 16 / 256)
spectrum_log(u_burnin, res=2 * np.pi * 16 / 256)
spectrum_log(u_last, res=2 * np.pi * 16 / 256)


spec = np.zeros((u.shape[0], u.shape[1] // 2))
for time in range(u.shape[0]):
    spec[time, :] = spectrum_log_output(u[time, :], res=2 * np.pi * 16 / 256)

# spec = spec[:400]



wavenumber_axis = 2 * jnp.pi * fftfreq(u.shape[1], d=2 * np.pi * 16 / 256)[: u.shape[1] // 2]
norm = LogNorm(vmin=spec.min()+10**(-19), vmax=spec.max())  # Avoid log(0) issues
# u spectrum log
plt.figure(figsize=(12, 6))
im = plt.imshow(
    spec.T,
    cmap="plasma",
    norm=norm,
    aspect="auto",
    extent=[0, spec.shape[0], wavenumber_axis[-1], wavenumber_axis[0]],
)
plt.colorbar(im, label="E(k, t)", extend="both")
plt.yscale("log")
plt.xlabel(r"$t$")
plt.ylabel(r"$k$")
plt.savefig("spectrum_log_full.pdf", dpi=300, transparent=False, bbox_inches="tight")

# u spectrum
plt.figure(figsize=(12, 6))
plt.imshow(
    spec.T,
    cmap="plasma",
    aspect="auto",
    extent=[0, spec.shape[0], wavenumber_axis[-1], wavenumber_axis[0]],
)
plt.colorbar(cmap="seismic", label="E(k, t)", extend="both")
plt.xlabel(r"$t$")
plt.ylabel(r"$k$")
plt.savefig("spectrum_linear_full.pdf", dpi=300, transparent=False, bbox_inches="tight")

# u field
plt.figure(figsize=(12, 6))
plt.imshow(
    u.T,
    cmap="plasma",
    aspect="auto",
    extent=[0, u.shape[0], 0, u.shape[1]],
)
plt.colorbar(cmap="plasma", label="u(x, t)", extend="both")
plt.xlabel(r"$t$")
plt.ylabel(r"$x$")


sys.exit()


grid_x = jnp.linspace(0.0, 22.0, 128)
for tiempo in range(data.shape[1]):
    _, ax = plt.subplots()
    ax.plot(grid_x, data[0, tiempo, :].squeeze(), linestyle="dashdot", label=r"$u_0$", color="blue")
    ax.plot(grid_x, np.zeros_like(grid_x), "--", color="black")
    ax.set_xlim(0.0, 22.0)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.savefig(f"vid/sample_{str(tiempo).zfill(2)}.png", dpi=300, transparent=False, bbox_inches="tight")


sys.exit()
