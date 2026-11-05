"""Fig. 2 - dynamical data collapse of eps_g(t) near alpha = 1 (lambda = 0, gamma = 0.5).

eps_g(t) |1 - alpha| plotted against x = (1 - alpha)^2 gamma t / 4 collapses onto F(x), Eq. (12).
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from _common import panel_labels, parse_args

from ddcrit import stacked, theory

args = parse_args("fig2.pdf")
gamma, dt, n_steps = 0.5, 0.1, 100000
under = [0.90, 0.92, 0.94, 0.96, 0.98]
over = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06]
ddir = os.path.join(args.data, "fig2")
i0, step = 20, 2000          # skip the first few steps, then subsample for plotting

blues = [matplotlib.colormaps["Blues"](v) for v in np.linspace(0.5, 1, 5)]
greys = [matplotlib.colormaps["Greys"](v) for v in np.linspace(0.5, 1, 6)]
fig, ax = plt.subplots(2, 2, figsize=(15, 12), dpi=200)
plt.subplots_adjust(wspace=0.35, hspace=0.4)

for row, (alist, cols, fmt) in enumerate([(under, blues, "o"), (over, greys, "D")]):
    for a, c in zip(alist, cols):
        eg = stacked(ddir, a, gamma, 0.0, "err_g")[:, :n_steps]     # (runs, steps)
        t = np.arange(eg.shape[1]) * dt
        scaled = eg * abs(1 - a)
        ax[row][0].errorbar(t[::step], eg.mean(0)[::step], yerr=eg.std(0, ddof=1)[::step],
                            color=c, fmt=fmt, label=rf"$\alpha={a}$")
        sl = slice(i0, None, step)
        x = (abs(1 - a) * np.sqrt(gamma * t[sl]) / 2) ** 2
        ax[row][1].errorbar(x, scaled.mean(0)[sl], yerr=scaled.std(0, ddof=1)[sl],
                            color=c, fmt=fmt, label=rf"$\alpha={a}$", zorder=1)

xs = [np.linspace(0, 13, 1000), np.linspace(0, 5, 1000)]
for row in range(2):
    ax[row][1].plot(xs[row], theory.F_scaling(xs[row]), color="red", label="F(x)", lw=3.0, zorder=2)
    ax[row][0].set_xlabel("t", fontsize=25)
    ax[row][0].set_ylabel(r"$\varepsilon_g(t)$", fontsize=25)
    ax[row][0].legend(fontsize=13.5, loc="upper left")
    for col in range(2):
        ax[row][col].tick_params(axis="both", labelsize=25)
ax[0][1].set_xlabel(r"$(1-\alpha)^2{\gamma t}/4$", fontsize=25)
ax[0][1].set_ylabel(r"$\varepsilon_g (1-\alpha)$", fontsize=25)
ax[1][1].set_xlabel(r"$(\alpha-1)^2{\gamma t}/4$", fontsize=25)
ax[1][1].set_ylabel(r"$\varepsilon_g (\alpha-1)$", fontsize=25)

panel_labels(fig, [(0.08, 0.53), (0.50, 0.53), (0.08, 0.08), (0.50, 0.08)])
plt.savefig(args.out, bbox_inches="tight")
print("saved", args.out)
