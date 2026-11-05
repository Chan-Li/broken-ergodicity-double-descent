# Broken ergodicity and double descent

[![arXiv](https://img.shields.io/badge/arXiv-2607.04135-b31b1b.svg)](https://arxiv.org/abs/2607.04135)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code for the paper

> **Broken Ergodicity and the Violation of the Fluctuation–Dissipation Theorem Lead to Generalization Beyond Overfitting in Machine Learning**
> Chan Li and Nigel Goldenfeld
> *Physical Review Letters* (accepted, 2026) · [arXiv:2607.04135](https://arxiv.org/abs/2607.04135)

We study double descent in teacher–student linear regression trained by stochastic
gradient descent, using dynamical mean-field theory (DMFT). The interpolation
threshold α = d/N = 1 is a dynamical phase transition. Below it, the training
dynamics is ergodic and obeys the fluctuation–dissipation theorem. Above it,
ergodicity is broken and the response function acquires a persistent part
(α − 1)/α, analogous to the London rigidity of a superconductor. This repository
contains:

* SGD simulations of the teacher–student model (Eqs. 1–3),
* a numerical solver for the self-consistent DMFT equations (Eqs. 4–5, 8),
* the analytic predictions (Eqs. 6, 7, 9, 12, 13, 15),
* scripts that regenerate the data and every figure of the paper.

Equation numbers refer to the published Letter (Eqs. 12–15 are in the End Matter); "SM" is its
Supplemental Material.

## Installation

```bash
git clone https://github.com/Chan-Li/broken-ergodicity-double-descent.git
cd broken-ergodicity-double-descent
pip install -r requirements.txt
```

Python ≥ 3.9. A CUDA GPU is strongly recommended for the N = 10⁴ simulations.
Simulations fall back to the CPU automatically.

## Repository layout

```
ddcrit/
  sgd.py        LinearRegressionSGD: SGD / gradient flow for teacher-student linear regression
  dmft.py       DMFTSolver: fixed-point solution of the DMFT equations for R, Q, C
  theory.py     closed-form results: Eqs. (6), (7), (9), (12), (15)
  io.py         saving / loading runs
scripts/
  run_sgd.py           CLI for SGD simulations (one .npz per run)
  run_dmft.py          CLI for the DMFT solver
  reproduce_paper.sh   exact parameters used for every figure
figures/
  fig1.py ... fig4.py  plotting scripts (read from data/)
```

## Quick start

```python
import torch
from ddcrit.sgd import LinearRegressionSGD, gaussian_vector
from ddcrit import theory

N, alpha, gamma, D, r = 1000, 0.5, 0.5, 1.0, 1.0
P = int(alpha * N)                                    # d = alpha N weights
model = LinearRegressionSGD(P, N, D, gamma, lamb=0.0, n_steps=10_000,
                            beta_s=gaussian_vector(P, r), beta0=torch.randn(P, 1))
res = model.train(dt=0.1)                             # zeta = 0.1, T = 1000
print(res["err_g"][-1], theory.eg_under(alpha, D))   # simulation vs D/(1-alpha) = 2
```

## Reproducing the figures

The raw simulation data (several GB) is not included in this repository. Regenerate it with

```bash
bash scripts/reproduce_paper.sh fig1    # fig2 / fig3 / fig4 / all
python figures/fig1.py                  # -> figures/fig1.pdf
```

| Figure | Content | Parameters | Data |
|---|---|---|---|
| 1(a) | steady-state ε_g, ε_t vs α | N = 1000, γ = 0.5, ζ = 0.1, T = 10⁴ for 0.9 ≤ α ≤ 1.1 and T = 10³ otherwise, 10 runs | `data/fig1a` |
| 1(b) | ε_g(t), ε_t(t): SGD vs DMFT | N = 1000, γ = 0.5, α ∈ {0.5, 1, 4}, 200 steps | `data/fig1b`, `data/fig1b_dmft.npz` |
| 2 | dynamical data collapse, λ = 0 | N = 10⁴, γ = 0.5, T = 10⁴, 10 runs | `data/fig2` |
| 3 | static and dynamical collapse with λ > 0 | N = 10⁴, γ = 1, 10 runs; (a, c) λ ∈ [0.01, 1], T = 10³; (b, d) λ ∈ {1, 3, 5} × 10⁻⁴, T = 10⁴ | `data/fig3_static`, `data/fig3_dynamic` |
| 4 | peak height and shift of ε_g | N = 10⁴, γ = 1, r = 0.1, 50 runs, bootstrap B = 1000 | `data/fig4` |

All runs use D = 1 and ζ = 0.1. Every parameter point is an independent command
in `scripts/reproduce_paper.sh`, so the work can be split across cluster jobs.
`--skip-existing` makes interrupted jobs resumable.

### Notes on the simulations

* **Notation.** In the code, `N` is the number of training samples and `P = d` is the input
  dimension, so α = P/N. Inputs are drawn from x ~ N(0, I) and predictions carry the 1/√d factor.
* **SGD.** At every step each sample is included independently with probability γ (Eq. 2).
  γ = 1 is full-batch gradient descent. Continuous time is t = (step) × ζ.
* **Recorded observables** (one value per step, before the update): `err_g` on a fixed held-out
  set of 10⁴ samples, `err_t` on the entire training set, `C = |β̂|²/d` and `Q = β̂·β/d`.
* **Teacher.** The teacher β is shared by all runs at a given (α, λ). The training set, test
  set and initialisation β̂(0) ~ N(0, I) are redrawn for every run.
* **Fig. 4** resolves a small peak shift Δα_c. To suppress finite-size fluctuations, the
  teacher and label noise are rejection-sampled so that their empirical variances equal r and D
  (`--exact-teacher --exact-noise`). Each run continues until ε_g stops changing
  (`--early-stop-tol`), and the final ε_g is averaged over 20 fresh test sets (`--final-test`).
  The peak height and position are bootstrapped (B = 1000); as described in SM Sec. III.B, the
  error bars are the standard deviation of the mean, std/√B.
* **Fig. 3** uses the leading-order scaling variables of Eq. (13) (factor 4). The exact form
  (SM Sec. V.A.2) has 2α(1 + α) in place of 4; the two agree at α = 1.
* **Fig. 1(b) DMFT.** The response-function iteration starts from i.i.d. standard Gaussian entries
  below the diagonal, and Ĉ(0,0), Q(0) and r are set from the same random draw as the SGD run.

## Using the DMFT solver

```python
import numpy as np
from ddcrit.dmft import DMFTSolver

beta_s, beta0 = np.random.normal(0, 1, (1000, 1)), np.random.normal(0, 1, (1000, 1))
solver = DMFTSolver(T=20, dt=0.1, D=1.0, alpha=4.0, gamma=1.0, lamb=0.0, beta_s=beta_s, beta0=beta0)
eps_g, eps_t = solver.solve()
```

The solver iterates the response function R(t, t′) to its fixed point, then the
overlap Q(t), then the correlation function C(t, t′). The cost grows as
O(n_t³) per iteration, so it is meant for short time windows (n_t = T/dt ≈ 200).

## Citation

If you use this code, please cite

```bibtex
@article{li2026broken,
  title         = {Broken Ergodicity and the Violation of the Fluctuation--Dissipation Theorem
                   Lead to Generalization Beyond Overfitting in Machine Learning},
  author        = {Li, Chan and Goldenfeld, Nigel},
  journal       = {Physical Review Letters},
  note          = {Accepted},
  year          = {2026},
  eprint        = {2607.04135},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2607.04135}
}
```

GitHub also reads [`CITATION.cff`](CITATION.cff) ("Cite this repository" button).

## License

MIT, see [LICENSE](LICENSE).
