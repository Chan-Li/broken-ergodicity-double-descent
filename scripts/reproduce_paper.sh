#!/usr/bin/env bash
# Generate the simulation data behind every figure of the paper.
#
#   bash scripts/reproduce_paper.sh fig1     # or fig2 / fig3 / fig4 / all
#
# Runs for N = 10,000 are expensive (10^5 steps each); a GPU is strongly
# recommended. Every command is independent, so the loops can be split across jobs.
# Pass --skip-existing (default here) to resume interrupted runs.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
SGD="python scripts/run_sgd.py --skip-existing"

fig1() {
  # (a) steady-state errors vs alpha: N = 1000, gamma = 0.5, zeta = 0.1.
  #     T = 10^4 (10^5 steps) for 0.9 <= alpha <= 1.1, T = 10^3 (10^4 steps) elsewhere.
  $SGD --N 1000 --gamma 0.5 --T 1000 --dt 0.1 --runs 10 --outdir "$DATA/fig1a" \
       --alphas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.3 1.5 1.8 2.0 3.0 4.0 5.0
  $SGD --N 1000 --gamma 0.5 --T 10000 --dt 0.1 --runs 10 --outdir "$DATA/fig1a" \
       --alphas 0.9 0.92 0.94 0.96 1.04 1.06 1.1
  # (b) early-time dynamics: N = 1000, gamma = 0.5, 200 steps (T = 20), SGD vs DMFT
  $SGD --N 1000 --gamma 0.5 --T 20 --dt 0.1 --runs 10 --outdir "$DATA/fig1b" --alphas 0.5 1.0 4.0
  python scripts/run_dmft.py --alphas 0.5 1.0 4.0 --gamma 0.5 --T 20 --dt 0.1 --out "$DATA/fig1b_dmft.npz"
}

fig2() {
  # Dynamical data collapse, lambda = 0: N = 10^4, gamma = 0.5, T = 10^4
  $SGD --N 10000 --gamma 0.5 --T 10000 --dt 0.1 --runs 10 --final-test --outdir "$DATA/fig2" \
       --alphas 0.90 0.92 0.94 0.96 0.98 1.01 1.02 1.03 1.04 1.05 1.06
}

fig3() {
  # (a, c) static collapse: steady state for lambda in [0.01, 1], T = 1000, full-batch GD
  local L="0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5"
  $SGD --N 10000 --gamma 1.0 --T 1000 --dt 0.1 --runs 10 --outdir "$DATA/fig3_static" \
       --alphas 0.90 0.92 0.94 0.96 0.98 0.99 1.01 1.02 1.03 1.04 1.05 --lambdas $L
  $SGD --N 10000 --gamma 1.0 --T 1000 --dt 0.1 --runs 10 --outdir "$DATA/fig3_static" \
       --alphas 0.99 --lambdas 0.6 0.7
  $SGD --N 10000 --gamma 1.0 --T 1000 --dt 0.1 --runs 10 --outdir "$DATA/fig3_static" \
       --alphas 0.90 0.92 0.94 0.96 0.98 --lambdas 0.6 0.7 0.8 0.9 1.0
  # (b, d) dynamical collapse with weak regularisation, T = 10^4
  $SGD --N 10000 --gamma 1.0 --T 10000 --dt 0.1 --runs 10 --outdir "$DATA/fig3_dynamic" \
       --alphas 0.96 0.98 1.01 1.02 --lambdas 0.0001 0.0003 0.0005
}

fig4() {
  # Peak of eps_g and its shift: r = 0.1, 50 runs, exact-norm teacher and noise,
  # run to convergence (early stop) and evaluate eps_g on 20 fresh test sets.
  $SGD --N 10000 --gamma 1.0 --r 0.1 --T 10000 --dt 0.1 --runs 50 --outdir "$DATA/fig4" \
       --exact-teacher --exact-noise --early-stop-tol 1e-50 --final-test \
       --alphas 0.95 0.96 0.97 0.98 0.99 1.0 1.01 1.02 1.03 1.04 1.05 \
       --lambdas 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01
}

case "${1:-all}" in
  fig1) fig1 ;; fig2) fig2 ;; fig3) fig3 ;; fig4) fig4 ;;
  all) fig1; fig2; fig3; fig4 ;;
  *) echo "usage: $0 [fig1|fig2|fig3|fig4|all]"; exit 1 ;;
esac
