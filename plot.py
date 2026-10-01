"""Save heading traces without opening another GUI window."""

from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DEFAULT_PLOT_DIR = Path(__file__).resolve().parent / "plot"


def plot_heading(
    times,
    current_headings,
    command_headings=None,
    output_path=None,
    *,
    angular_velocities=None,
) -> Path:
    """Save absolute world headings against elapsed seconds and return the path.

    Inputs are absolute headings, including the captured target of a relative
    turn. Headings are normalized to [-180, 180). Use None or NaN for samples
    without a heading command. Lines break at angle wrap boundaries.
    By default the filename includes the local save time down to microseconds.
    When angular_velocities is supplied (degrees/second), add a separate yaw
    angular-velocity subplot sharing the time axis in the same saved image.
    """
    times = np.asarray(times, dtype=float)
    current = np.asarray(current_headings, dtype=float)
    if times.ndim != 1 or current.shape != times.shape:
        raise ValueError("times and current_headings must be matching 1D sequences")
    if not np.isfinite(times).all() or not np.isfinite(current).all():
        raise ValueError("times and current headings must be finite")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be non-decreasing")
    commands = None
    angular = None
    if angular_velocities is not None:
        angular = np.asarray(angular_velocities, dtype=float)
        if angular.shape != times.shape or not np.isfinite(angular).all():
            raise ValueError("angular velocities must be finite and match times")
    if command_headings is not None:
        commands = np.asarray(command_headings, dtype=float)
        if commands.shape != times.shape or np.isinf(commands).any():
            raise ValueError(
                "commands must match times and contain finite angles or NaN"
            )

    def trace(angles):
        normalized = (angles + 180.0) % 360.0 - 180.0
        breaks = np.flatnonzero(np.abs(np.diff(normalized)) > 180.0) + 1
        return np.insert(times, breaks, np.nan), np.insert(normalized, breaks, np.nan)

    if output_path is None:
        timestamp = datetime.now(tz=UTC).astimezone().strftime("%Y%m%d_%H%M%S_%f")
        output_path = DEFAULT_PLOT_DIR / f"heading_plot_{timestamp}.png"
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if angular is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        angular_ax = None
    else:
        fig, (ax, angular_ax) = plt.subplots(nrows=2, sharex=True, figsize=(10, 7))
    try:
        ax.plot(*trace(current), label="Current absolute heading")
        if commands is not None and np.isfinite(commands).any():
            ax.step(
                *trace(commands),
                where="post",
                linestyle="--",
                label="Absolute heading command",
            )
        ax.set(
            xlabel="Time (seconds)",
            ylabel="Absolute heading (degrees)",
            ylim=(-180, 180),
        )
        ax.set_yticks(np.arange(-180, 181, 60))
        ax.grid(True, alpha=0.3)
        ax.legend()
        if angular_ax is not None:
            angular_ax.plot(times, angular, label="Current yaw angular velocity")
            angular_ax.set(
                xlabel="Time (seconds)", ylabel="Angular velocity (degrees/second)"
            )
            angular_ax.axhline(0, color="grey", linewidth=0.8)
            angular_ax.grid(True, alpha=0.3)
            angular_ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return path
