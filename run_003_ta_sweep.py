"""Run the current QCS analysis settings for the requested 1200 s Ta sweep."""

import gc
from pathlib import Path

import matplotlib

from qcs_analyzer._implementation import timetag_correction_sep

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qcs_analyzer.io import load_timetag_sep
from qcs_analyzer.analysis import timetag_correction_sep
from qcs_analyzer.qcs_kernal import qcs_kernal_sep
from qcs_analyzer.results import save_dict_npz


INPUT = Path(
    r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization"
    r"\023_50MHz_900s_1clock_sep.h5"
)
OUTPUT_DIR = INPUT.with_suffix("")
TA_LABELS = (
    "0.1", "0.2", "0.3", "0.4", "0.5",
    "0.6", "0.8", "1.0", "1.5", "2.0",
    "2.5", "3.0", "4.0", "5.0", "7.5",
    "10.0", "15.0", "20.0", "30.0", "45.0"
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
    t1_ps, ch1, t2_ps, ch2, filename = load_timetag_sep(str(INPUT))
    # print(f"Loaded {len(t1_ps):,} events", flush=True)
    t1_ps, ch1, t2_ps, ch2, details = timetag_correction_sep(
        t1_ps, ch1, t2_ps, ch2,
        ch_qutag=3,
        ch_hh=7,
        return_parameters=True,
        # correction_mode="global",
        correction_mode="first",
    )

    for label in TA_LABELS:
        ta_s = float(label)
        output = OUTPUT_DIR / f"{label.replace('.', 'p')}.npz"
        print(f"Starting Ta_s={label} s", flush=True)
        result = qcs_kernal_sep(
            t1_ps,
            ch1,
            t2_ps,
            ch2,
            Ta_s_init=ta_s,
            tau_ps_init=62200,
            tau_rt_ps_init=107400,
            fit_curve="g2calib_2",
            time_bin_ps_init=20,
            Ta_s_max=ta_s,
            Ta_ref_s=ta_s,
            Ta_rt_s_max=ta_s,
            Freq=50e6,
        )
        plt.close("all")
        saved = save_dict_npz(result, output_filename=str(output))
        print(f"Saved Ta_s={label} s: {saved}", flush=True)
        del result
        gc.collect()


if __name__ == "__main__":
    main()
