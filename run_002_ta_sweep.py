"""Run the current QCS analysis settings for the requested 1200 s Ta sweep."""

import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qcs_analyzer.io import load_timetag
from qcs_analyzer.analysis import timetag_correction
from qcs_analyzer.qcs_kernal import qcs_kernal
from qcs_analyzer.results import save_dict_npz


INPUT = Path(
    r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization"
    r"\011_50MHz_1200s_truncated_560s.h5"
)
OUTPUT_DIR = INPUT.with_suffix("")
# TA_LABELS = (
#     "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8", "1.0",
#     "1.5", "2.0", "2.5", "3.0", "4.0", "5.0", "7.5",
#     "10.0",
# )
# TA_LABELS = (
#     "15.0", "20.0", "25.0", "30.0",
# )
# TA_LABELS = (
#     "0.01", "0.02", "0.05",
# )
TA_LABELS = (
    "1", "2", "5",
)


def close_shown_figures(*args, **kwargs):
    plt.close("all")


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(OUTPUT_DIR)
    for label in TA_LABELS:
        output = OUTPUT_DIR / f"{label.replace('.', 'p')}.npz"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")

    plt.show = close_shown_figures
    print(f"Loading {INPUT}", flush=True)
    t_ps, ch = load_timetag(str(INPUT))
    print(f"Loaded {len(t_ps):,} events", flush=True)
    t_ps, ch, details = timetag_correction(
        t_ps, ch,
        ch_qutag=3,
        ch_hh=7,
        return_parameters=True,
        correction_mode="global",
    )
    print(
        f"Corrected {len(t_ps):,} events; removed "
        f"{details['events_removed_outside_shared_interval']:,}",
        flush=True,
    )

    for label in TA_LABELS:
        ta_s = float(label)
        output = OUTPUT_DIR / f"{label.replace('.', 'p')}.npz"
        print(f"Starting Ta_s={label} s", flush=True)
        result = qcs_kernal(
            t_ps,
            ch,
            Ta_s_init=ta_s,
            tau_ps_init=62200,
            tau_rt_ps_init=107400,
            fit_curve="g2calib_2",
            time_bin_ps_init=100,
            Ta_s_max=ta_s,
            Ta_ref_s=ta_s,
            Freq=50e6,
        )
        plt.close("all")
        saved = save_dict_npz(result, output_filename=str(output))
        print(f"Saved Ta_s={label} s: {saved}", flush=True)
        del result
        gc.collect()


if __name__ == "__main__":
    main()
