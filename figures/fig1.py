"""Fig. 1 - SGD simulations vs DMFT / analytic predictions (lambda = 0).

(a) steady-state eps_g and eps_t vs alpha (data/fig1a),
(b) eps_g(t), eps_t(t) at early times for alpha = 0.5, 1, 4 (data/fig1b, data/fig1b_dmft.npz).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from _common import panel_labels, parse_args

from ddcrit import final_values, stacked, theory

args = parse_args("fig1.pdf")
D, r, beta0_var = 1.0, 1.0, 1.0
C00 = r + beta0_var                     # C(0,0) = <betahat(0)^2> + r

# ------------------------------------------------------------------ panel (a)
alphas = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96,
                   1.04, 1.06, 1.1, 1.3, 1.5, 1.8, 2.0, 3.0, 4.0, 5.0])
d1a = os.path.join(args.data, "fig1a")
test = np.array([final_values(d1a, a, 0.5, 0.0, "err_g") for a in alphas])
train = np.array([final_values(d1a, a, 0.5, 0.0, "err_t") for a in alphas])

fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=200)
below = alphas < 1
fit = np.linspace(1.04, 5, 100)
ax[0].errorbar(alphas, test.mean(1), yerr=test.std(1, ddof=1), color="black", fmt="o", ms=5,
               label="Simulation: generalization error")
ax[0].errorbar(alphas, train.mean(1), yerr=train.std(1, ddof=1), color="blue", fmt="o", ms=5,
               label="Simulation: training error")
ax[0].plot(alphas[below], theory.eg_under(alphas[below], D), color="black", label=r"$\frac{D}{1-\alpha}$")
ax[0].plot(fit, theory.eg_over(fit, C00, D), color="grey",
           label=r"$\frac{(\alpha-1)C(0,0)}{\alpha} + \frac{D\alpha}{\alpha-1}$")
ax[0].plot(alphas, theory.et_under(alphas, D), color="blue", label=r"$D(1-\alpha)H(1-\alpha)$")
ax[0].legend(fontsize=14.5)
ax[0].set_xlabel(r"$\alpha = \frac{d}{n}$", fontsize=30)
ax[0].set_ylabel("Error Value", fontsize=25)
ax[0].tick_params(axis="both", labelsize=25)

# ------------------------------------------------------------------ panel (b)
dyn_alphas = [0.5, 1.0, 4.0]
gamma_b = 0.5
d1b = os.path.join(args.data, "fig1b")
dmft = np.load(os.path.join(args.data, "fig1b_dmft.npz"))
n = dmft["eps_g"].shape[1]
it = np.arange(n)
step = 10
c_g, c_t, markers = ["blue", "red", "dimgrey"], ["darkslateblue", "indianred", "darkgrey"], ["o", "D", "s"]

h_t, h_g = [], []
h_t.append(ax[1].plot(it, dmft["eps_t"][0], color="orchid", lw=2, label=r"$\varepsilon_t$: DMFT")[0])
h_g.append(ax[1].plot(it, dmft["eps_g"][0], color="purple", lw=2, label=r"$\varepsilon_g$: DMFT")[0])
for j, a in enumerate(dyn_alphas):
    et = stacked(d1b, a, gamma_b, 0.0, "err_t", n)
    eg = stacked(d1b, a, gamma_b, 0.0, "err_g", n)
    h_t.append(ax[1].errorbar(it[::step], et.mean(0)[::step], yerr=et.std(0, ddof=1)[::step], color=c_t[j],
                              fmt=markers[j], ls="none", label=rf"$SGD:\alpha = {a}$"))
    h_g.append(ax[1].errorbar(it[::step], eg.mean(0)[::step], yerr=eg.std(0, ddof=1)[::step], color=c_g[j],
                              fmt=markers[j], ls="none", label=rf"$SGD:\alpha = {a}$"))
    ax[1].plot(it, dmft["eps_t"][j], lw=2, color="orchid")
    ax[1].plot(it, dmft["eps_g"][j], lw=2, color="purple")
ax[1].set_xlabel("Iterations", fontsize=25, color="blue")
ax[1].set_ylabel("Error Value", fontsize=25)
ax[1].tick_params(axis="both", labelsize=25)
ax[1].set_ylim(-0.1, 5.7)
handles = [h for pair in zip(h_t, h_g) for h in pair]
ax[1].legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=15, ncol=2)

panel_labels(fig, [(0.08, 0.08), (0.495, 0.08)])
plt.savefig(args.out, bbox_inches="tight")
print("saved", args.out)
