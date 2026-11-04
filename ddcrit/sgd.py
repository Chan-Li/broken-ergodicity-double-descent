"""Teacher-student linear regression trained by (stochastic) gradient descent.

Model (Eqs. 1-3 of the paper)
-----------------------------
* inputs        x^mu ~ N(0, I_d),               mu = 1..N
* teacher       y^mu = beta . x^mu / sqrt(d) + eps^mu,   eps^mu ~ N(0, D)
* student       yhat^mu = betahat . x^mu / sqrt(d)
* loss          L = 1/2 sum_mu s^mu (yhat^mu - y^mu)^2 + lambda/2 |betahat|^2
* SGD           betahat <- betahat - zeta * dL/dbetahat,  s^mu ~ Bernoulli(gamma)
                drawn independently for every sample at every step.

``gamma = 1`` gives full-batch gradient descent. In the limit ``zeta -> 0`` the
dynamics is the stochastic gradient flow (Eq. 3) with time ``t = n_steps * zeta``.

Notation: in the code ``N`` is the number of training samples and ``P = d`` the
input dimension / number of trainable weights, so ``alpha = P / N``.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def gaussian_vector(P, r, device="cpu"):
    """Draw a (P, 1) vector with i.i.d. N(0, r) entries."""
    return torch.randn(P, 1, device=device) * torch.sqrt(torch.tensor(r, device=device))


def exact_norm_vector(P, r, tol=1e-4, mean_tol=1e-3, device="cpu"):
    """Rejection-sample a N(0, r) vector whose empirical second moment equals ``r``
    and whose empirical mean vanishes, up to ``tol`` and ``mean_tol``.

    This removes finite-size fluctuations of r = |beta|^2 / d (and of the
    empirical noise variance D), which matters when resolving the small peak
    shift Delta alpha_c in Fig. 4.
    """
    r_tensor = torch.tensor(r, device=device)
    while True:
        x = torch.randn(P, 1, device=device) * torch.sqrt(r_tensor)
        if abs((x.T @ x / P).item() - r) < tol and abs(x.mean().item()) < mean_tol:
            return x


class LinearRegressionSGD:
    """Single-layer linear student trained on a fixed (offline) data set.

    Parameters
    ----------
    P, N : int
        Input dimension d (= number of weights) and number of training samples.
    D : float
        Label-noise variance.
    gamma : float
        Probability that a sample is included in a given SGD step.
    lamb : float
        l2 regularisation strength lambda.
    n_steps : int
        Number of gradient steps.
    beta_s, beta0 : torch.Tensor, shape (P, 1)
        Teacher weights and student initialisation.
    exact_noise : bool
        If True, training/test label noise is drawn with :func:`exact_norm_vector`.
    n_test : int
        Size of the held-out test set used to monitor eps_g(t).
    """

    def __init__(self, P, N, D, gamma, lamb, n_steps, beta_s, beta0,
                 exact_noise=False, n_test=10000, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.P, self.N, self.D = P, N, D
        self.gamma, self.lamb, self.n_steps = gamma, lamb, n_steps
        self.alpha = P / N
        self.n_test = n_test
        self._noise = exact_norm_vector if exact_noise else gaussian_vector

        self.beta_s = beta_s.clone().detach().float().to(self.device)
        self.beta = beta0.clone().detach().float().to(self.device)
        self.sqrt_P_inv = 1.0 / torch.sqrt(torch.tensor(P, device=self.device))

        # Fixed training set (offline setting).
        self.train_data = torch.randn(N, P, device=self.device)
        self.train_noise = self._noise(N, D, device=self.device)
        self.train_label = self.sqrt_P_inv * (self.train_data @ self.beta_s) + self.train_noise
        self.r = (self.beta_s.T @ self.beta_s / P).item()

    # ------------------------------------------------------------------ data
    def _test_set(self):
        x = torch.randn(self.n_test, self.P, device=self.device)
        y = self.sqrt_P_inv * x @ self.beta_s + self._noise(self.n_test, self.D, device=self.device)
        return x, y

    def _minibatch(self):
        """Apply the Bernoulli(gamma) selection variables s^mu (Eq. 2)."""
        mask = torch.bernoulli(torch.full((self.N,), self.gamma, device=self.device)).unsqueeze(1)
        return self.train_data * mask, self.train_label * mask

    def final_test_error(self, n_sets=20):
        """eps_g averaged over ``n_sets`` independent fresh test sets."""
        total = 0.0
        for _ in range(n_sets):
            x, y = self._test_set()
            total += torch.mean((y - self.sqrt_P_inv * x @ self.beta).pow(2)).item()
        return total / n_sets

    # -------------------------------------------------------------- training
    def train(self, dt, early_stop_tol=None, final_test=False, progress=True):
        """Run SGD with learning rate ``dt`` (= zeta).

        Observables are recorded *before* each update:

        * ``err_g`` - generalisation error on a fixed held-out test set,
        * ``err_t`` - training error on the *entire* training set,
        * ``C``     - |betahat|^2 / d,
        * ``Q``     - betahat . beta / d (teacher-student overlap).

        If ``early_stop_tol`` is given, training stops once consecutive values of
        ``err_g`` differ by less than this tolerance (i.e. the dynamics has
        reached its fixed point to machine precision). If ``final_test`` is True,
        :meth:`final_test_error` is appended as the last entry of ``err_g``.
        """
        err_g, err_t, C, Q = [], [], [], []
        test_x, test_y = self._test_set()
        X, Y = self.train_data, self.train_label

        for i in tqdm(range(self.n_steps), disable=not progress,
                      desc=f"alpha={self.alpha:.3f}"):
            err_t.append(torch.mean((Y - self.sqrt_P_inv * X @ self.beta).pow(2)).item())
            err_g.append(torch.mean((test_y - self.sqrt_P_inv * test_x @ self.beta).pow(2)).item())
            if early_stop_tol is not None and i > 20 and abs(err_g[-1] - err_g[-2]) < early_stop_tol:
                break
            Q.append((self.beta.T @ self.beta_s / self.P).item())
            C.append((self.beta.T @ self.beta / self.P).item())

            Xb, Yb = self._minibatch()
            pred = self.sqrt_P_inv * Xb @ self.beta
            grad = self.sqrt_P_inv * Xb.T @ (pred - Yb) + self.lamb * self.beta
            self.beta -= dt * grad

        if final_test:
            err_g.append(self.final_test_error())
        return {k: np.asarray(v) for k, v in
                dict(err_g=err_g, err_t=err_t, C=C, Q=Q).items()}
