"""Small plotting helpers shared by Phase 0B-R2 report scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_timeseries(path: Path, x: np.ndarray, series: Mapping[str, np.ndarray],
                    ylabel: str, title: str) -> None:
    """Write a deterministic, headless line plot."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values in series.items():
        ax.plot(x, values, label=label)
    ax.set(xlabel="outer step", ylabel=ylabel, title=title)
    ax.grid(alpha=.25)
    if series:
        ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
