"""Backward-compatible entry point for the QCS analyzer.

For new code, import from the focused ``qcs_analyzer`` modules instead:

* ``qcs_analyzer.io`` — load, merge, and truncate timetags
* ``qcs_analyzer.analysis`` — correction, g2 processing, and fitting
* ``qcs_analyzer.plotting`` — visualisations
* ``qcs_analyzer.results`` — save and load processed results
* ``qcs_analyzer.config`` — configure the default file-dialog folder
"""

# Re-export every public function from qcs_analyzer's focused submodules. This
# lets an existing notebook use one import, for example:
#     from QCS_delay_estimator_v4 import *
# Private helpers (names beginning with _) intentionally remain internal.
import sys
from qcs_analyzer import *
from qcs_analyzer import __all__ as _QCS_ANALYZER_PUBLIC_API
from qcs_analyzer.analysis import g2_process_time_dependent, timetag_correction, timetag_channel_time_difference
from qcs_analyzer.io import load_timetag
from qcs_analyzer.results import save_time_dependent_results
from qcs_analyzer.qcs_kernal import qcs_kernal
from qcs_analyzer.allan_deviation import allan_deviation_estimator

__all__ = _QCS_ANALYZER_PUBLIC_API


if __name__ == "__main__":
    # load_time_dependent_results()
    # timetag_to_h5()
    # timetag_to_h5_fast()
    # trancate_timetag(30)

    # import matplotlib.pyplot as plt
    # import numpy as np
    # from qcs_analyzer._implementation import _g2calib_2, _g2calib_1, _g2calib_2_Hfunc
    # x = np.linspace(-10000, 10000, 10000)
    # y = _g2calib_2(
    #     x,
    #     height=1.0,
    #     center=0.0,
    #     # sigma=1000.0,
    #     sigma=0.0,
    #     offset=0.0,
    # )
    # plt.plot(x, y)
    # plt.xlabel("Delay (ps)")
    # plt.ylabel(r"$g^{(2)}$")
    # plt.grid(True)
    # plt.show()

    # results = load_g2_results()
    # fit = fit_g2(
    #     results,
    #     curve="expogaussconv",
    #     parameters={
    #         "center": 4.3,
    #         "height": 50,
    #         "gamma": 0.7,
    #         "sigma": 0.05,
    #         "offset": 0.01,
    #     },
    #     fit_window=15,
    #     use_rate=100e6,
    #     show=True,
    # )
    # print(f"sigma: {fit["parameters"]["sigma"]:.8g}", f"gamma: {fit["parameters"]["gamma"]:.8g}")

    t_ps, ch, filename = load_timetag(return_filename=True)

    # allan_deviation_estimator(t_ps, ch, number_of_tau_points=100)
    # sys.exit()

    t_ps, ch, details = timetag_correction(
        t_ps, ch,
        ch_qutag=2,
        ch_hh=7,
        return_parameters=True,
        correction_mode="global"
        # correction_mode="per_trigger"
    )

    Ta_s = 0.02
    qcs_kernal(
        t_ps,
        ch,

        Ta_s_init=Ta_s,

        fit_curve="g2calib_2",
        time_bin_ps_init=100,
        # Ta_s_max=0.5,
        Ta_s_max=Ta_s,

        Ta_ref_s=0.1
        # show_fit=True,
    )

    # difference_ps, t_a, t_b, figure, axis = timetag_channel_time_difference(
    #     t_ps, ch,
    #     ch_a=7,
    #     ch_b=2,
    #     bin_s=0.1,
    #     # show=False,
    # )

    sys.exit()

    # results = g2_process(
    #     t_ps,
    #     ch,
    #     # t_center=4.4,
    #     t_center=24601.2,
    #     t_stop=0.1,
    #     t_span=2000,
    #     # time_bin=0.02,
    #     time_bin=2,
    #     sig_half=5.0,
    #     freq=100e6,
    #     chA=1,
    #     chB=6,
    #     filename=filename,
    # )
    # g2_plot(results=results)
    # save_g2_results(results)

    time_dependent_results = g2_process_time_dependent(
        t_ps,
        ch,
        t_stop_start=0.1,
        t_stop_stop=3600,
        t_stop_step=0.1,
        # fit_curve="expogaussconv",
        fit_curve="g2calib_1",
        fit_parameters={
            "height": 40 / 0.1,
            # "gamma": 0.7,
            "sigma": 0.2,
            "offset": 0.01,
        },
        # t_center=4.4,
        t_center=24601.8,
        t_span=80,
        # time_bin=0.05,
        time_bin=0.5,
        sig_half=5.0,
        fit_window=50.0,
        adaptive_t_center=True,
        display_values=(
            "center",
            "fwhm",
            # "center_uncertainty",
            # "fwhm_uncertainty",
            "coincidences",
        ),
        freq=100e6,
        chA=1,
        chB=6,
        # show_fit=True,
    )
    save_time_dependent_results(time_dependent_results)

    # load_time_dependent_results()
