"""Plot one-second electrical-reference means with within-second variation.

The ribbon is the standard deviation of ten 0.1-second mean delays within
each one-second interval. It is not a 30-second rolling statistic or a
confidence interval for the one-second mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HERE.parent
    / "20260924_Tx1_RB4_Characterization"
    / "023_50MHz_900s_1clock_sep.h5"
)
DEFAULT_OUTPUT = HERE / "023_ref_delay_1s_ribbon.png"
PS_PER_SECOND = 1_000_000_000_000
SUBINTERVALS_PER_SECOND = 10
PS_PER_SUBINTERVAL = PS_PER_SECOND // SUBINTERVALS_PER_SECOND


def channel_events(file, time_key, channel_key, channel, chunk_size):
    """Yield selected timestamps without loading the whole HDF5 file."""
    times = file[time_key]
    channels = file[channel_key]
    for start in range(0, len(times), chunk_size):
        stop = min(start + chunk_size, len(times))
        mask = channels[start:stop] == channel
        if np.any(mask):
            yield times[start:stop][mask]


def one_second_delay_stats(path, chunk_size, pair_tolerance_ps):
    """Pair pulses by index and aggregate their delays by TTX second."""
    with h5py.File(path, "r") as file:
        n_seconds = int(
            max(file["t1_ps"][-1], file["t2_ps"][-1]) // PS_PER_SECOND
        ) + 1
        counts = np.zeros(n_seconds, dtype=np.int64)
        sums = np.zeros(n_seconds, dtype=np.float64)
        sums_of_squares = np.zeros(n_seconds, dtype=np.float64)
        n_subintervals = n_seconds * SUBINTERVALS_PER_SECOND
        sub_counts = np.zeros(n_subintervals, dtype=np.int64)
        sub_sums = np.zeros(n_subintervals, dtype=np.float64)

        ttx = channel_events(file, "t1_ps", "ch1", 3, chunk_size)
        hh = channel_events(file, "t2_ps", "ch2", 7, chunk_size)
        ttx_buffer = next(ttx, None)
        hh_buffer = next(hh, None)

        while ttx_buffer is not None and hh_buffer is not None:
            n = min(len(ttx_buffer), len(hh_buffer))
            ttx_times = ttx_buffer[:n]
            delays = hh_buffer[:n] - ttx_times
            if np.any(np.abs(delays) >= pair_tolerance_ps):
                raise ValueError(
                    "Pulse pairing failed: a ch3/ch7 delay exceeds "
                    f"{pair_tolerance_ps} ps. Check for missing pulses."
                )

            seconds = ttx_times // PS_PER_SECOND
            counts += np.bincount(seconds, minlength=n_seconds)
            sums += np.bincount(seconds, weights=delays, minlength=n_seconds)
            sums_of_squares += np.bincount(
                seconds, weights=delays.astype(np.float64) ** 2,
                minlength=n_seconds,
            )
            subintervals = ttx_times // PS_PER_SUBINTERVAL
            sub_counts += np.bincount(subintervals, minlength=n_subintervals)
            sub_sums += np.bincount(
                subintervals, weights=delays, minlength=n_subintervals
            )

            ttx_buffer = ttx_buffer[n:] if n < len(ttx_buffer) else next(ttx, None)
            hh_buffer = hh_buffer[n:] if n < len(hh_buffer) else next(hh, None)

        if ttx_buffer is not None or hh_buffer is not None:
            raise ValueError("The two reference channels have different pulse counts.")

    # Exclude an incomplete final second.
    complete = counts >= 90_000
    seconds = np.flatnonzero(complete)
    n = counts[complete]
    mean = sums[complete] / n
    sample_variance = (
        sums_of_squares[complete] - sums[complete] ** 2 / n
    ) / (n - 1)
    pulse_sigma = np.sqrt(np.maximum(sample_variance, 0))
    mean_se = pulse_sigma / np.sqrt(n)
    sub_counts = sub_counts.reshape(-1, SUBINTERVALS_PER_SECOND)[complete]
    sub_sums = sub_sums.reshape(-1, SUBINTERVALS_PER_SECOND)[complete]
    if np.any(sub_counts == 0):
        raise ValueError("A complete second contains an empty 0.1-s subinterval.")
    sub_means = sub_sums / sub_counts
    within_second_sd = np.std(sub_means, axis=1, ddof=1)
    return seconds, mean, within_second_sd, pulse_sigma, mean_se, n


def plot_delay(seconds, mean, within_second_sd, output):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(
        seconds,
        mean - within_second_sd,
        mean + within_second_sd,
        color="tab:blue",
        alpha=0.25,
        label="Within each second: ±1 SD of 0.1-s means",
    )
    ax.plot(seconds, mean, color="tab:blue", lw=0.9, label="1-s mean delay")
    ax.set(
        xlabel="Recording time (s)",
        ylabel="Delay, HH ch7 − TTX ch3 (ps)",
        title="Electrical reference: one-second means and within-second variation",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--pair-tolerance-ps", type=int, default=1_000)
    args = parser.parse_args()

    seconds, mean, within_second_sd, pulse_sigma, mean_se, counts = one_second_delay_stats(
        args.input, args.chunk_size, args.pair_tolerance_ps
    )
    plot_delay(seconds, mean, within_second_sd, args.output)
    print(f"Saved {args.output}")
    print(f"Complete 1-s intervals: {len(seconds)}")
    print(f"Median pulses per interval: {np.median(counts):.0f}")
    print(f"Median single-pulse sigma: {np.median(pulse_sigma):.3f} ps")
    print(f"Median 1-s mean standard error: {np.median(mean_se):.3f} ps")
    print(f"Median within-second 0.1-s mean SD: {np.median(within_second_sd):.3f} ps")


if __name__ == "__main__":
    main()
