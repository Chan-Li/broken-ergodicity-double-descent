"""Fig. 3 - static and dynamical scaling with weak l2 regularisation (gamma = 1).

(a, c) eps_g(t->inf) |1-alpha| vs x = 4 lambda / (gamma (1-alpha)^2)  ->  F~(x) = 1/sqrt(1+x)
(b, d) eps_g(t) |1-alpha| sqrt(1 + x) vs the rescaled time            ->  F(x), Eq. (12)

Data: data/fig3_static (T = 1000) and data/fig3_dynamic (T = 10^4).
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from _common import panel_labels, parse_args

from ddcrit import final_values, stacked, theory

args = parse_args("fig3.pdf")
gamma, D, dt, n_steps = 1.0, 1.0, 0.1, 100000
sdir = os.path.join(args.data, "fig3_static")
ddir = os.path.join(args.data, "fig3_dynamic")

lam_full = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
static_sets = [  # (alphas, lambdas per alpha)
    {a: lam_full for a in [0.90, 0.92, 0.94, 0.96, 0.98]} | {0.99: lam_full[:16]},
    {a: lam_full[:14] for a in [1.01, 1.02, 1.03, 1.04, 1.05]},
]
dyn_sets = [[0.96, 0.98], [1.01, 1.02]]
dyn_lams = [0.0001, 0.0003, 0.0005]

blues = [matplotlib.colormaps["Blues"](v) for v in np.linspace(0.5, 1, 7)]
greys = [matplotlib.colormaps["Greys"](v) for v in np.linspace(0.5, 1, 7)]
fig, ax = plt.subplots(2, 2, figsize=(18, 12.5), dpi=200)
plt.subplots_adjust(wspace=0.35, hspace=0.4)

# ------------------------------------------------------ (a), (c) static collapse
for row, (sset, fmt, xmin) in enumerate([(static_sets[0], "o", 2.0), (static_sets[1], "D", 5.0)]):
    cols = blues if row == 0 else greys[::-1]
    xmax = 0
    for k, (a, lams) in enumerate(sset.items()):
        y = np.array([final_values(sdir, a, gamma, l) * abs(1 - a) for l in lams])  # (n_lambda, runs)
        x = 4 * np.array(lams) / (gamma * abs(1 - a) ** 2)
        xmax = max(xmax, x.max())
        ax[row][0].errorbar(x, y.mean(1), yerr=y.std(1, ddof=1), color=cols[k], fmt=fmt,
                            label=rf"$\alpha={a}$", zorder=1)
    xx = np.linspace(xmin, xmax, 20000)
    ax[row][0].plot(xx, theory.F_static(xx, D), color="red", lw=3.0,
                    label=r"$\tilde{F}(x) = \frac{1}{\sqrt{1+x}}$", zorder=2)
    ax[row][0].legend(fontsize=18)
    ax[row][0].tick_params(axis="both", labelsize=25)
sym = [r"1-\alpha", r"\alpha-1"]
for row in range(2):
    ax[row][0].set_xlabel(rf"$\frac{{4\lambda}}{{\gamma({sym[row]})^2}}$", fontsize=30)
    ax[row][0].set_ylabel(rf"$\varepsilon_g(t\to\infty)(|{sym[row]}|)$", fontsize=25)

# ------------------------------------------------------ (b), (d) dynamical collapse
# The rescaling uses 2 alpha (1 + alpha) in place of 4 gamma; the two agree at alpha = 1.
sl = slice(20, None, 2000)
for row, (alist, fmt) in enumerate([(dyn_sets[0], "o"), (dyn_sets[1], "D")]):
    cols = blues if row == 0 else greys
    k = 0
    for a in alist:
        for l in dyn_lams:
            c = 2 * a * (1 + a)
            eg = stacked(ddir, a, gamma, l, "err_g")[:, :n_steps]
            t = np.arange(eg.shape[1]) * dt
            eg = eg * abs(1 - a) * np.sqrt(1 + c * l / (1 - a) ** 2)
            x = ((1 - a) ** 2 / c + l) * t[sl]
            ax[row][1].errorbar(x, eg.mean(0)[sl], yerr=eg.std(0, ddof=1)[sl], color=cols[k], fmt=fmt,
                                label=rf"$\alpha={a}, \lambda = {l}$", zorder=1)
            k += 1
    xx = np.linspace(0, 10 if row == 0 else 6.6, 20000)
    ax[row][1].plot(xx, theory.F_scaling(xx), color="red", lw=3.0, label=r"${F}(x)$", zorder=2)
    ax[row][1].legend(fontsize=18)
    ax[row][1].tick_params(axis="both", labelsize=25)
    ax[row][1].set_xlabel(rf"${{\left(\frac{{\gamma({sym[row]})^2}}{{4\lambda}}+1\right)\lambda t}}$", fontsize=30)
    ax[row][1].set_ylabel(rf"$\varepsilon_g(t)(|{sym[row]}|)\sqrt{{1+\frac{{4\lambda}}{{\gamma({sym[row]})^2}}}}$",
                          fontsize=20)

panel_labels(fig, [(0.08, 0.52), (0.50, 0.52), (0.08, 0.08), (0.50, 0.08)])
plt.savefig(args.out, bbox_inches="tight")
print("saved", args.out)
