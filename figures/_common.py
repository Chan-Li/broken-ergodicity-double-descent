import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def parse_args(default_out):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data"), help="data directory")
    p.add_argument("--out", default=os.path.join(ROOT, "figures", default_out))
    return p.parse_args()


def panel_labels(fig, positions, fontsize=30):
    for label, (x, y) in zip("abcd", positions):
        fig.text(x, y, f"({label})", fontsize=fontsize, va="top", ha="left")
