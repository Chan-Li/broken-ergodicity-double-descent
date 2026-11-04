#!/usr/bin/env python
"""Solve the DMFT equations and save eps_g(t), eps_t(t) for several alpha.

Example (DMFT curves of Fig. 1b):

    python scripts/run_dmft.py --alphas 0.5 1.0 4.0 --T 20 --dt 0.1 --out data/fig1b_dmft.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ddcrit.dmft import DMFTSolver  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alphas", type=float, nargs="+", required=True)
    p.add_argument("--T", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--D", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lamb", type=float, default=0.0)
    p.add_argument("--r", type=float, default=1.0, help="teacher strength <beta^2>")
    p.add_argument("--P", type=int, default=1000,
                   help="size of the vectors used to draw r, Q(0), C(0,0)")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    eps_g, eps_t = [], []
    for alpha in args.alphas:
        beta_s = rng.normal(0.0, np.sqrt(args.r), (args.P, 1))
        beta0 = rng.normal(0.0, 1.0, (args.P, 1))
        solver = DMFTSolver(args.T, args.dt, args.D, alpha, args.gamma, args.lamb, beta_s, beta0,
                            seed=None if args.seed is None else args.seed + 1)
        g, t = solver.solve(max_iter=args.max_iter)
        eps_g.append(g)
        eps_t.append(t)
        print(f"alpha={alpha:g}: eps_g(T) = {g[-1]:.4f}, eps_t(T) = {t[-1]:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, alphas=np.array(args.alphas), eps_g=np.array(eps_g), eps_t=np.array(eps_t),
             time=np.arange(int(round(args.T / args.dt))) * args.dt, D=args.D, gamma=args.gamma, r=args.r)


if __name__ == "__main__":
    main()
