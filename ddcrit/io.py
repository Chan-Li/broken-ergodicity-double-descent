"""Helpers to name, save and load simulation runs."""
from __future__ import annotations

import glob
import os

import numpy as np


def run_path(outdir, alpha, gamma, lamb, run):
    return os.path.join(outdir, f"alpha{alpha:g}_gamma{gamma:g}_lambda{lamb:g}_run{run:02d}.npz")


def save_run(path, result, **meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **result, **{k: np.asarray(v) for k, v in meta.items()})


def load_runs(outdir, alpha, gamma, lamb, key="err_g"):
    """Return a list with ``key`` from every run of a given (alpha, gamma, lambda)."""
    pattern = os.path.join(outdir, f"alpha{alpha:g}_gamma{gamma:g}_lambda{lamb:g}_run*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no runs match {pattern} - run the simulation first (see README)")
    return [np.load(f)[key] for f in files]


def final_values(outdir, alpha, gamma, lamb, key="err_g"):
    """Last recorded value of ``key`` for every run (steady-state estimate)."""
    return np.array([x[-1] for x in load_runs(outdir, alpha, gamma, lamb, key)])


def stacked(outdir, alpha, gamma, lamb, key="err_g", length=None):
    """Time series of every run stacked into an array of shape (n_runs, length)."""
    runs = load_runs(outdir, alpha, gamma, lamb, key)
    length = length or min(len(x) for x in runs)
    return np.stack([x[:length] for x in runs])
