"""Compare timing precision versus acquisition time for one and two clocks."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from qcs_analyzer.results import load_dict_npz


DATA_DIR = Path(r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization")
SWEEPS = (
    ("1 clock", DATA_DIR / "023_50MHz_900s_1clock_sep", "tab:blue"),
    ("2 clocks", DATA_DIR / "027_50MHz_900s_2clock_sep", "tab:red"),
)
ZERO_POINT_ONE_CUTOFF_PS = 850.0e12


def plain_decimal_tick(value, _position):
    """Show log-axis tick values without scientific notation."""
    return np.format_float_positional(value, trim="-")


def precision_and_uncertainty(values, label, filename):
    """Return sample standard deviation and its nominal standard error."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        raise ValueError(f"{filename}: {label} needs at least two finite samples.")

    values = values - np.mean(values)
    precision = np.std(values, ddof=1)
    uncertainty = precision / np.sqrt(2 * (n - 1))
    return precision, uncertainty


def load_sweep(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Sweep folder does not exist: {folder}")

    files = []
    for filename in folder.glob("*.npz"):
        try:
            ta_s = float(filename.stem.replace("p", "."))
        except ValueError:
            continue
        if np.isfinite(ta_s) and ta_s > 0:
            files.append((ta_s, filename))
    files.sort(key=lambda item: item[0])
    if not files:
        raise FileNotFoundError(f"No Ta-labelled NPZ files found in {folder}")

    ta_values = []
    tau_precision = []
    tau_uncertainty = []
    rt_precision = []
    rt_uncertainty = []

    for ta_s, filename in files:
        results = load_dict_npz(str(filename))
        # tau_values = results["tau_diff"]
        tau_values = results["tau"]
        rt_values = results["tau_rt"]
        if filename.stem == "0p1":
            centers = np.asarray(results["center"]).ravel()
            tau_values = np.asarray(tau_values).ravel()
            rt_values = np.asarray(rt_values).ravel()
            if len(centers) != len(tau_values) or len(centers) != len(rt_values):
                raise ValueError(f"{filename}: center, tau_diff, and tau_rt lengths differ.")
            keep = np.isfinite(centers) & (centers < ZERO_POINT_ONE_CUTOFF_PS)
            tau_values = tau_values[keep]
            rt_values = rt_values[keep]

        tau_s, tau_se = precision_and_uncertainty(
            tau_values, "tau_diff", filename
        )
        rt_s, rt_se = precision_and_uncertainty(
            rt_values, "tau_rt", filename
        )
        ta_values.append(ta_s)
        tau_precision.append(tau_s)
        tau_uncertainty.append(tau_se)
        rt_precision.append(rt_s)
        rt_uncertainty.append(rt_se)

    return {
        "Ta": np.asarray(ta_values),
        "tau_precision": np.asarray(tau_precision),
        "tau_uncertainty": np.asarray(tau_uncertainty),
        "rt_precision": np.asarray(rt_precision),
        "rt_uncertainty": np.asarray(rt_uncertainty),
    }


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for label, folder, color in SWEEPS:
        data = load_sweep(folder)
        for precision_key, uncertainty_key, linestyle, marker, alpha, curve_label in (
            (
                "tau_precision", "tau_uncertainty", "-", "o", 0.22,
                label + r": $\tau-\tau_{\mathrm{ref}}$",
            ),
            (
                "rt_precision", "rt_uncertainty", "--", "s", 0.10,
                label + r": $\tau_{\mathrm{rt}}$",
            ),
        ):
            precision = data[precision_key]
            uncertainty = data[uncertainty_key]
            ax.fill_between(
                data["Ta"],
                precision - uncertainty,
                precision + uncertainty,
                color=color,
                alpha=alpha,
                linewidth=0,
            )
            ax.plot(
                data["Ta"], precision,
                linestyle=linestyle, marker=marker, color=color,
                label=curve_label, zorder=3,
            )

    ax.set_ylabel("Timing precision (ps)")
    ax.set_xlabel(r"Acquisition time $T_a$ (s)")
    ax.legend(title="Shading: +/- 1 standard error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(plain_decimal_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_decimal_tick))
    ax.grid(True, which="both", alpha=0.5)

    plt.xticks([0.1, 1, 10, 60], ["0.1", "1", "10", "60"], fontsize=12)
    plt.yticks([0.2, 0.5, 1, 5], ["0.2", "0.5", "1", "5"], fontsize=12)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
