"""Closed-form predictions from the paper (used for the solid lines in the figures)."""
from __future__ import annotations

import numpy as np
from scipy.special import erf


# ----------------------------------------------------------------------------
# Unregularised steady state (lambda = 0)
# ----------------------------------------------------------------------------
def eg_under(alpha, D):
    """Eq. (6): eps_g(t->inf) = D / (1 - alpha) for alpha < 1."""
    return D / (1.0 - np.asarray(alpha))


def et_under(alpha, D):
    """Eq. (9): eps_t(t->inf) = D (1 - alpha) for alpha < 1 (zero for alpha > 1)."""
    alpha = np.asarray(alpha)
    return np.where(alpha < 1, D * (1.0 - alpha), 0.0)


def eg_over(alpha, C00, D):
    """Eq. (7): eps_g(t->inf) = (alpha-1) C(0,0)/alpha + D alpha/(alpha-1) for alpha > 1.

    ``C00 = <betahat(0)^2> + r`` is the initial squared distance to the teacher.
    """
    alpha = np.asarray(alpha)
    return (alpha - 1.0) * C00 / alpha + D * alpha / (alpha - 1.0)


# ----------------------------------------------------------------------------
# Universal scaling functions near alpha = 1
# ----------------------------------------------------------------------------
def F_scaling(x, D=1.0):
    """Eq. (12): dynamical scaling function F(x), with F(0) = 0 and F(inf) = D."""
    x = np.asarray(x, dtype=float)
    return (-D * 2 * np.sqrt(x) / np.sqrt(np.pi) * np.exp(-2 * x) * (np.sqrt(2) - 2 * np.exp(x))
            + D * ((2 + 4 * x) * erf(np.sqrt(x)) - (1 + 4 * x) * erf(np.sqrt(2 * x))))


def F_static(x, D=1.0):
    """Static collapse curve D / F~(x) = D / sqrt(1 + x), with F~(x) = sqrt(1 + x) and x = 4 lambda / (gamma (1-alpha)^2)."""
    return D / np.sqrt(1.0 + np.asarray(x))


# ----------------------------------------------------------------------------
# Regularised steady state (lambda > 0), Eq. (15)
# ----------------------------------------------------------------------------
def lam_chi(alpha, lamb, gamma):
    """lambda * chi_lambda, the regularised susceptibility times lambda."""
    a, l, g = np.asarray(alpha), lamb, gamma
    num = -(a * l + g * (1 - a)) + np.sqrt((a * l + g * (1 - a)) ** 2 + 4 * a ** 2 * g * l)
    return num / (2 * a * g)


def eg_regularised(alpha, lamb, gamma, r, D):
    """Eq. (15): eps_g(t->inf) = (D + r (lambda chi)^2) / (1 - alpha (lambda chi - 1)^2)."""
    f = lam_chi(alpha, lamb, gamma)
    return (D + r * f ** 2) / (1 - np.asarray(alpha) * (f - 1) ** 2)
