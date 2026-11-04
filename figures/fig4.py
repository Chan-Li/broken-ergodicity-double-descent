"""Fig. 4 - regularisation rounds the divergence of eps_g (gamma = 1, r = 0.1, 50 runs).

(a) eps_g(t->inf) vs alpha for several lambda, with Eq. (15).
(b) peak height eps_gmax vs 1/sqrt(lambda); inset: peak shift Delta alpha_c = 1 - alpha_peak vs lambda.
    The peak is located by a cubic spline through the run-averaged eps_g(alpha), and its
    uncertainty is estimated by bootstrap resampling of the runs (B = 1000).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from _common import panel_labels, parse_args
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import CubicSpline

from ddcrit import final_values, theory

args = parse_args("fig4.pdf")
gamma, r, D, n_runs, B = 1.0, 0.1, 1.0, 50, 1000
alphas = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05])
lambdas = np.array([0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01])
ddir = os.path.join(args.data, "fig4")
rng = np.random.default_rng(0)


def runs(lamb):
    """Final eps_g of every run: array (n_alpha, n_runs)."""
    return np.array([final_values(ddir, a, gamma, lamb)[:n_runs] for a in alphas])


def peak(curve):
    grid = np.linspace(0.95, 1.05, 10000)
    vals = CubicSpline(alphas, curve)(grid)
    return grid[np.argmax(vals)], vals.max()


def bootstrap_peak(data):
    shift, height = [], []
    for _ in range(B):
        resampled = np.array([rng.choice(row, size=row.size, replace=True) for row in data])
        a_max, e_max = peak(resampled.mean(1))
        shift.append(1 - a_max)
        height.append(e_max)
    return np.mean(shift), np.std(shift, ddof=1), np.mean(height), np.std(height, ddof=1)


fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=200)
plt.subplots_adjust(wspace=0.25, hspace=0.4)

# ------------------------------------------------------------------ panel (a)
colors, markers = ["#5B7B9D", "#6C8B74", "#A1656A"], ["o", "s", "^"]
for i, l in enumerate(lambdas[:3]):
    d = runs(l)
    ax[0].errorbar(alphas, d.mean(1), yerr=d.std(1, ddof=1), color=colors[i], label=rf"$\lambda={l:.3f}$",
                   fmt=markers[i], ms=6, capsize=3, capthick=1.0, ecolor="#9A9A9A", elinewidth=1.0, zorder=3)
    ax[0].plot(alphas, theory.eg_regularised(alphas, l, gamma, r, D), color=colors[i], lw=2.0, alpha=0.9, zorder=2)
ax[0].legend(fontsize=15, frameon=False, loc="upper left")
ax[0].tick_params(axis="both", labelsize=20)
ax[0].set_xlabel(r"$\alpha$", fontsize=25)
ax[0].set_ylabel(r"$\varepsilon_g(t\to\infty)$", fontsize=25)
ax[0].set_ylim(7, 13.5)

# ------------------------------------------------------------------ panel (b)
stats = np.array([bootstrap_peak(runs(l)) for l in lambdas])     # (n_lambda, 4)
shift_av, shift_sd, hmax_av, hmax_sd = stats.T
# Error bars: bootstrap standard deviation divided by sqrt(B), as in the published figure.
x = 1 / np.sqrt(lambdas)
fit = np.poly1d(np.polyfit(x, hmax_av, 1))
xf = np.linspace(x.min(), x.max(), 100)
ax[1].plot(xf, fit(xf), "--", color="#B35C44", lw=2)
ax[1].errorbar(x, hmax_av, yerr=hmax_sd / np.sqrt(B), fmt="s", ms=6, capsize=4, color="#6E6E6E",
               ecolor="#B0B0B0", elinewidth=1.2, markerfacecolor="#82A0AA")
ax[1].set_xlabel(r"$1/\sqrt{\lambda}$", fontsize=25)
ax[1].set_ylabel(r"$\langle \varepsilon_{\mathrm{gmax}}\rangle$", fontsize=25)
ax[1].tick_params(axis="both", which="both", labelsize=20)
ax[1].set_ylim(5, 16)

axins = inset_axes(ax[1], width="45%", height="45%", loc="upper right",
                   bbox_to_anchor=(-0.33, -0.055, 0.9, 1), bbox_transform=ax[1].transAxes, borderpad=0)
lam0 = np.concatenate([[0.0], lambdas])       # lambda = 0: peak exactly at alpha_c = 1
sh0 = np.concatenate([[0.0], shift_av])
sd0 = np.concatenate([[0.0], shift_sd])
axins.errorbar(lam0, sh0, yerr=sd0 / np.sqrt(B), fmt="o", ms=6, capsize=4, color="#4B6587",
               ecolor="#9A9A9A", elinewidth=1.2, markerfacecolor="#82A0AA")
coef = np.polyfit(lam0, sh0, 1)
xf = np.linspace(0, lam0.max(), 100)
axins.plot(xf, np.poly1d(coef)(xf), "--", color="#B35C44", lw=2)
axins.set_xlabel(r"$\lambda$", fontsize=15)
axins.set_ylabel(r"$\langle \Delta \alpha_c \rangle $", fontsize=15)
axins.tick_params(axis="both", labelsize=15)
axins.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
print(f"slope d(Delta alpha_c)/d lambda = {coef[0]:.3f}")

panel_labels(fig, [(0.08, 0.03), (0.50, 0.03)])
plt.savefig(args.out, bbox_inches="tight")
print("saved", args.out)
