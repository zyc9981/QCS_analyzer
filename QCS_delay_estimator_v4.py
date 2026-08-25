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
import numpy as np
import matplotlib.pyplot as plt
from qcs_analyzer import *
from qcs_analyzer import __all__ as _QCS_ANALYZER_PUBLIC_API
from qcs_analyzer.analysis import g2_process_time_dependent, timetag_correction, timetag_channel_time_difference
from qcs_analyzer.io import load_timetag
from qcs_analyzer.results import save_time_dependent_results, save_dict_npz, load_dict_npz
from qcs_analyzer.qcs_kernal import qcs_kernal
from qcs_analyzer.allan_deviation import allan_deviation_estimator, allan_deviation_estimator_tau_input

__all__ = _QCS_ANALYZER_PUBLIC_API


if __name__ == "__main__":
    # timetag_to_h5(ch_trig_time_tagger=3, sort_by_timestamp=False)
    # timetag_to_h5_TTXonly()
    # trancate_timetag(60)

    # results = load_g2_results()
    # fit = fit_g2(
    #     results,
    #     curve="expogaussconv",
    #     parameters={
    #         "center": 128.520,
    #         "height": 50,
    #         "gamma": 0.7,
    #         "sigma": 0.05,
    #         "offset": 0.01,
    #     },
    #     fit_window=27,
    #     use_rate=50e6,
    #     show=True,
    # )
    # print(f"sigma: {fit["parameters"]["sigma"]:.8g}", f"gamma: {fit["parameters"]["gamma"]:.8g}")

    t_ps, ch, filename = load_timetag(return_filename=True)
    print(r"Data import complete.")

    # adev_result = allan_deviation_estimator(t_ps, ch, number_of_tau_points=500, ch_clk_0=3, ch_clk_1=7)
    # save_dict_npz(adev_result)

    # qcs_result = load_dict_npz()
    # adev_result = allan_deviation_estimator_tau_input(qcs_result["tau_ref"], qcs_result["center"], number_of_tau_points=200)
    # # save_dict_npz(adev_result)
    # adev_result = allan_deviation_estimator_tau_input(qcs_result["tau"], qcs_result["center"], number_of_tau_points=200)
    # # save_dict_npz(adev_result)
    # sys.exit()

    t_ps, ch, details = timetag_correction(
        t_ps, ch,
        ch_qutag=3,
        # ch_hh=4,
        ch_hh=7,
        return_parameters=True,
        # correction_mode="first",
        correction_mode="global",
        # correction_mode="per_trigger",
    )
    
    # t_ps0 = t_ps[ch==3]
    # t_ps = t_ps - t_ps0[0]

    # t_ps0 = t_ps[ch==3]
    # t_psA = t_ps0
    # # print((t_psA[-1] - t_psA[0])*1e-12)
    # # t_ps1 = t_ps0
    # t_ps0 = np.asarray(t_ps0, dtype=float)
    # slope_ps_per_sample = (t_ps0[-1] - t_ps0[0]) / (len(t_ps0) - 1)
    # t_ps1 = t_ps0 - slope_ps_per_sample * np.arange(len(t_ps0))
    # t_ps0 = t_ps[ch==7]
    # t_psB = t_ps0
    # # print((t_psB[-1] - t_psB[0])*1e-12)
    # # t_ps2 = t_ps0
    # t_ps0 = np.asarray(t_ps0, dtype=float)
    # slope_ps_per_sample = (t_ps0[-1] - t_ps0[0]) / (len(t_ps0) - 1)
    # t_ps2 = t_ps0 - slope_ps_per_sample * np.arange(len(t_ps0))
    # nidx = np.min([np.size(t_psA), np.size(t_psB)])
    # dt_ps = t_psA[:nidx] - t_psB[:nidx]
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(np.size(t_ps1)) * 1e-5, t_ps1, label="t_ps1")
    # ax.plot(np.arange(np.size(t_ps2)) * 1e-5, t_ps2 + 1000, label="t_ps2")
    # ax.legend()
    # # fig, ax = plt.subplots()
    # # ax.plot(np.arange(nidx) * 1e-5, dt_ps)
    # # plt.show()

    # difference_ps, t_a, t_b, figure, axis = timetag_channel_time_difference(
    #     t_ps, ch,
    #     # ch_a=3,
    #     ch_a=3,
    #     # ch_b=4,
    #     ch_b=7,
    #     # bin_s=0.1,
    #     bin_s=0.01,
    #     # show=False,
    # )

    Ta_s = 0.2
    qcs_result = qcs_kernal(
        t_ps,
        ch,

        Ta_s_init=Ta_s,
        tau_ps_init=112390,
        tau_rt_ps_init=57460,

        fit_curve="g2calib_2",
        time_bin_ps_init=200,
        Ta_s_max=5,
        # Ta_s_max=Ta_s,
        # tau_span_ps_init=120e3,

        Ta_ref_s=0.1,
        # show_fit=True,
        second_order_fit=True,

        Freq = 50e6,
    )
    save_dict_npz(qcs_result)

    # qcs_results = load_dict_npz()
    # fig1, axes = plt.subplots(3, 1, sharex=True, squeeze=False)
    # axes = axes[:, 0]
    # ax1 = axes[0]
    # tau = qcs_results["tau"]
    # ax1.plot(qcs_results["center"] * 1e-12, tau - tau[0], "-", linewidth=2.0)
    # ax1.set_ylabel(r"$\tau$ (ps)")
    # ax1.grid(True)
    # ax2 = axes[1]
    # tau_ref = qcs_results["tau_ref"]
    # ax2.plot(qcs_results["center"] * 1e-12, tau_ref - tau_ref[0], "-", linewidth=2.0)
    # ax2.set_ylabel(r"$\tau$ (ps)")
    # ax2.grid(True)
    # ax3 = axes[2]
    # tau_diff = qcs_results["tau_diff"]
    # ax3.plot(qcs_results["center"] * 1e-12, tau_diff - np.mean(tau_diff), "-", linewidth=2.0)
    # ax3.set_ylabel(r"$\tau$ (ps)")
    # ax3.grid(True)
    # axes[-1].set_xlabel("Record Points (s)")
    # fig1.tight_layout()
    # plt.show()

    # sys.exit()

    # results = g2_process(
    #     t_ps,
    #     ch,
    #     t_center=112.4,
    #     # t_center=50,
    #     # t_center=24601.2,
    #     t_stop=20,
    #     t_span=200,
    #     time_bin=0.02,
    #     # time_bin=2,
    #     sig_half=5.0,
    #     freq=50e6,
    #     chA=1,
    #     chB=6,
    #     filename=filename,
    # )
    # g2_plot(results=results)
    # save_g2_results(results)

    # time_dependent_results = g2_process_time_dependent(
    #     t_ps,
    #     ch,
    #     t_stop_start=0.1,
    #     t_stop_stop=3600,
    #     t_stop_step=0.1,
    #     # fit_curve="expogaussconv",
    #     fit_curve="g2calib_1",
    #     fit_parameters={
    #         "height": 40 / 0.1,
    #         # "gamma": 0.7,
    #         "sigma": 0.2,
    #         "offset": 0.01,
    #     },
    #     # t_center=4.4,
    #     t_center=24601.8,
    #     t_span=80,
    #     # time_bin=0.05,
    #     time_bin=0.5,
    #     sig_half=5.0,
    #     fit_window=50.0,
    #     adaptive_t_center=True,
    #     display_values=(
    #         "center",
    #         "fwhm",
    #         # "center_uncertainty",
    #         # "fwhm_uncertainty",
    #         "coincidences",
    #     ),
    #     freq=100e6,
    #     chA=1,
    #     chB=6,
    #     # show_fit=True,
    # )
    # save_time_dependent_results(time_dependent_results)

    # load_time_dependent_results()
