#!/usr/bin/env python
"""Run SGD simulations of teacher-student linear regression.

One ``.npz`` file per (alpha, gamma, lambda, run) is written to ``--outdir``,
containing the time series ``err_g``, ``err_t``, ``C``, ``Q`` sampled at every step.

Example (one point of Fig. 2):

    python scripts/run_sgd.py --N 10000 --alphas 0.96 --gamma 0.5 --T 10000 \
        --runs 10 --final-test --outdir data/fig2
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ddcrit.io import run_path, save_run  # noqa: E402
from ddcrit.sgd import LinearRegressionSGD, exact_norm_vector, gaussian_vector  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=10000, help="number of training samples")
    p.add_argument("--alphas", type=float, nargs="+", required=True, help="alpha = d / N values")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0], help="l2 strengths lambda")
    p.add_argument("--gamma", type=float, default=1.0, help="sample-selection probability (1 = full batch)")
    p.add_argument("--D", type=float, default=1.0, help="label-noise variance")
    p.add_argument("--r", type=float, default=1.0, help="teacher signal strength |beta|^2 / d")
    p.add_argument("--T", type=float, default=10000.0, help="total training time (n_steps = T / dt)")
    p.add_argument("--dt", type=float, default=0.1, help="learning rate zeta")
    p.add_argument("--runs", type=int, default=10, help="independent runs per parameter point")
    p.add_argument("--exact-teacher", action="store_true",
                   help="rejection-sample the teacher so that |beta|^2/d = r exactly (Fig. 4)")
    p.add_argument("--teacher-tol", type=float, default=1e-5)
    p.add_argument("--exact-noise", action="store_true",
                   help="rejection-sample label noise so its empirical variance equals D (Fig. 4)")
    p.add_argument("--early-stop-tol", type=float, default=None,
                   help="stop when |err_g(t+1) - err_g(t)| < tol")
    p.add_argument("--final-test", action="store_true",
                   help="append eps_g averaged over 20 fresh test sets to err_g")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--skip-existing", action="store_true", help="do not recompute existing runs")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    n_steps = int(round(args.T / args.dt))

    for lamb in args.lambdas:
        for alpha in args.alphas:
            P = int(round(alpha * args.N))
            # The teacher is shared by all runs at a given (alpha, lambda); the training
            # set, test set and student initialisation are redrawn for every run.
            if args.exact_teacher:
                beta_s = exact_norm_vector(P, args.r, tol=args.teacher_tol, mean_tol=args.teacher_tol)
            else:
                beta_s = gaussian_vector(P, args.r)
            for run in range(args.runs):
                path = run_path(args.outdir, alpha, args.gamma, lamb, run)
                if args.skip_existing and os.path.exists(path):
                    continue
                beta0 = torch.randn(P, 1)
                model = LinearRegressionSGD(P, args.N, args.D, args.gamma, lamb, n_steps, beta_s, beta0,
                                            exact_noise=args.exact_noise, device=args.device)
                res = model.train(args.dt, early_stop_tol=args.early_stop_tol, final_test=args.final_test)
                save_run(path, res, N=args.N, P=P, alpha=alpha, gamma=args.gamma, lamb=lamb,
                         D=args.D, r=model.r, dt=args.dt)
                print(f"alpha={alpha:g} lambda={lamb:g} run={run}: final eps_g = {res['err_g'][-1]:.5f}")


if __name__ == "__main__":
    main()
