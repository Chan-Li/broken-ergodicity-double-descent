"""Numerical solution of the self-consistent DMFT equations for SGD.

The effective single-weight process (Eq. 4 of the paper) is closed by
self-consistent equations for

* ``R(t, t')`` - response function,
* ``Q(t)``     - teacher-student overlap  <betahat(t) beta> ,
* ``C(t, t')`` - weight correlation function <betahat(t) betahat(t')> ,

on a discrete time grid ``t_k = k * dt``, k = 0..nTime-1. The memory kernel
``(gamma R + delta)^{-1}`` is represented by ``invR = (gamma R + I/dt)^{-1}``.
Each object is obtained by fixed-point iteration (first R, then Q, then C). The
errors are

* eps_g(t) = r + D + C(t, t) - 2 Q(t),
* eps_t(t) = [ (gamma R + delta)^{-1} * (D + C~) * (gamma R^T + delta)^{-1} ](t, t)   (Eq. 8),

with C~(t, t') = r + C(t, t') - Q(t) - Q(t') the correlation of betahat - beta.
See Sec. I of the Supplemental Material for the derivation.
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm


def _random_strictly_lower(n, sigma=1.0, rng=None):
    """Random causal initial guess for R (non-zero only for t > t')."""
    rng = np.random.default_rng() if rng is None else rng
    return np.tril(rng.normal(0.0, sigma, (n, n)), k=-1)


def _fixed_point(update, x0, max_iter=1000, tol=1e-10, label=""):
    """Iterate ``x <- update()`` until the squared change drops below ``tol``."""
    prev = x0.copy()
    for i in tqdm(range(max_iter), desc=label, leave=False):
        cur = update()
        diff = np.sum((cur - prev) ** 2)
        if diff < tol:
            return cur, i
        prev = cur.copy()
    print(f"[{label}] not converged after {max_iter} iterations (diff={diff:.3e})")
    return cur, max_iter


class DMFTSolver:
    """DMFT for SGD on teacher-student linear regression.

    Parameters
    ----------
    T, dt : float
        Total training time and time step (nTime = T / dt grid points).
    D : float
        Label-noise variance.
    alpha : float
        Model capacity d / N.
    gamma : float
        Sample-selection probability of SGD.
    lamb : float
        l2 regularisation strength.
    beta_s, beta0 : ndarray, shape (P, 1)
        Teacher weights and student initialisation. They only enter through
        r = <beta^2>, Q(0) = <beta0 beta> and C(0, 0) = <beta0^2>.
    """

    def __init__(self, T, dt, D, alpha, gamma, lamb, beta_s, beta0, seed=None):
        self.T, self.dt = T, dt
        self.D, self.alpha, self.g, self.lamb = D, alpha, gamma, lamb
        P = beta_s.shape[0]
        self.r = float(np.mean(beta_s ** 2))
        self.n = n = int(round(T / dt))

        rng = np.random.default_rng(seed)
        self.R = _random_strictly_lower(n, 1.0, rng)
        self.Q = np.zeros((n, 1))
        self.Q[0] = (beta0.T @ beta_s / P).item()
        self.C = np.zeros((n, n))
        self.C[0, 0] = (beta0.T @ beta0 / P).item()
        self._update_kernels()

    # ----------------------------------------------------------- utilities
    def _update_kernels(self):
        I = np.identity(self.n)
        self.invR = np.linalg.inv(self.g * self.R + I / self.dt)
        self.invRT = np.linalg.inv(self.g * self.R.T + I / self.dt)

    def _Ctilde(self):
        """Correlation of (betahat - beta) plus label noise: r + D + C - Q - Q^T."""
        n = self.n
        return self.r + self.D + self.C - self.Q.reshape(1, n) - self.Q.reshape(n, 1)

    # -------------------------------------------------------- update rules
    def update_R(self):
        self._update_kernels()
        n, g, a = self.n, self.g, self.alpha
        grad = (-g / a * (self.invR @ self.R) + np.identity(n)) - self.lamb * self.R * self.dt
        for t1 in range(n - 1):
            for t2 in range(n):
                self.R[t1 + 1, t2] = self.R[t1, t2] + grad[t1, t2]
        return self.R

    def update_Q(self):
        self._update_kernels()
        n, g, a = self.n, self.g, self.alpha
        for t1 in range(n - 1):
            self.Q[t1 + 1] = (self.Q[t1]
                              - g / a * (self.invR @ (self.Q - self.r)).reshape(n, 1)[t1]
                              - self.lamb * self.Q[t1] * self.dt)
        return self.Q

    def update_C(self):
        self._update_kernels()
        n, g, a = self.n, self.g, self.alpha
        grad = (-g / a * (self.invR @ (self.C - self.Q.reshape(1, n)))
                + g * g / a * (self.invR @ self._Ctilde() @ self.invRT @ self.R.T)
                - self.lamb * self.C * self.dt)
        for t1 in range(n - 1):
            for t2 in range(t1 + 2):
                self.C[t1 + 1, t2] = self.C[t1, t2] + grad[t1, t2]
                self.C[t2, t1 + 1] = self.C[t1 + 1, t2]
        return self.C

    # -------------------------------------------------------------- solve
    def solve(self, max_iter=1000, tol=1e-10):
        _fixed_point(self.update_R, self.R, max_iter, tol, "R")
        _fixed_point(self.update_Q, self.Q, max_iter, tol, "Q")
        _fixed_point(self.update_C, self.C, max_iter, tol, "C")
        return self.errors()

    def errors(self):
        """Return (eps_g(t), eps_t(t)) on the time grid."""
        self._update_kernels()
        eps_g = (self.r + self.D + np.diag(self.C) - 2 * self.Q.ravel())
        et = self.dt ** 2 * (self.invR / self.dt ** 2) @ self._Ctilde() @ (self.invRT / self.dt ** 2)
        eps_t = np.diag(et)
        return eps_g, eps_t
