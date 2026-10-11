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
from qcs_analyzer.analysis import g2_process_sep, timetag_correction_sep
from qcs_analyzer.io import load_timetag_sep
from qcs_analyzer.results import save_time_dependent_results, save_dict_npz, load_dict_npz
from qcs_analyzer.qcs_kernal import qcs_kernal_sep, qcs_plot
from qcs_analyzer.allan_deviation import allan_deviation_estimator_sep, allan_deviation_estimator_tau_input

__all__ = _QCS_ANALYZER_PUBLIC_API


if __name__ == "__main__":
    timetag_to_h5_sep()
    # timetag_to_h5_TTXonly()
    # trancate_timetag(600)
    sys.exit()

    # results = load_g2_results()
    # fit = fit_g2(
    #     results,
    #     curve="expogaussconv",
    #     parameters={
    #         "center": 45.75,
    #         "height": 100,
    #         "gamma": 0.1,
    #         "sigma": 0.02,
    #         "offset": 0.1,
    #     },
    #     fit_window=8,
    #     use_rate=50e6,
    #     show=True,
    # )
    # print(f"sigma: {fit["parameters"]["sigma"]:.8g}", f"gamma: {fit["parameters"]["gamma"]:.8g}", f"center: {fit["parameters"]["center"]:.8g}")
    # sys.exit()
    # fit = fit_g2(
    #     results,
    #     curve="gaussian",
    #     parameters={
    #         "center": -0.02,
    #         "height": 30000,
    #         "sigma": 0.002,
    #         "offset": 0.0,
    #     },
    #     fit_window=8,
    #     use_rate=50e6,
    #     show=True,
    # )
    # print(f"sigma: {fit["parameters"]["sigma"]:.8g}")
    # sys.exit()

    # qcs_results = load_dict_npz()
    # tau_diff = qcs_results["tau_diff"]
    # tau_rt = qcs_results["tau_rt"]
    # t = qcs_results["center"]
    # # mask = (t < 480e12) & np.isfinite(tau_diff)
    # tau_diff = tau_diff - np.mean(tau_diff)
    # # tau_diff = tau_diff[mask]
    # tau_rt = tau_rt - np.mean(tau_rt)
    # # tau_rt = tau_rt[mask]
    # s1 = np.std(tau_diff, ddof=1)
    # s2 = np.std(tau_rt, ddof=1)
    # print("error_diff:", s1)
    # print("error_rt:", s2)
    # N = len(tau_diff)
    # se_s1 = s1 / np.sqrt(2 * (N - 1))
    # print(se_s1)
    # se_s2 = s2 / np.sqrt(2 * (N - 1))
    # print(se_s2)
    # CC = qcs_results["CC"]
    # print(np.mean(CC))
    # qcs_plot(qcs_results)
    # sys.exit()

    # t_ps, ch, filename = load_timetag(return_filename=True)
    t1_ps, ch1, t2_ps, ch2, filename = load_timetag_sep()
    print(r"Data import complete.")

    adev_result = allan_deviation_estimator_sep(t1_ps, ch1, t2_ps, ch2, number_of_tau_points=500, ch_clk_0=3, ch_clk_1=7)
    save_dict_npz(adev_result)

    # qcs_result = load_dict_npz()
    # adev_result = allan_deviation_estimator_tau_input(qcs_result["tau_ref"], qcs_result["center"], number_of_tau_points=200)
    # # save_dict_npz(adev_result)
    # adev_result = allan_deviation_estimator_tau_input(qcs_result["tau"], qcs_result["center"], number_of_tau_points=200)
    # # save_dict_npz(adev_result)
    sys.exit()

    t1_ps, ch1, t2_ps, ch2, details = timetag_correction_sep(
        t1_ps, ch1, t2_ps, ch2,
        ch_qutag=3,
        ch_hh=7,
        return_parameters=True,
        correction_mode="first",
        # correction_mode="global",
        # correction_mode="per_trigger",
    )

    # # Check reference jitter
    # ref0 = t_ps[ch==3]
    # ref1 = t_ps[ch==7]
    # ref_diff = ref0 - ref1
    # ref_jitter = np.std(ref_diff, ddof=1)
    # print(f"Reference jitter: {ref_jitter:.8g}")
    # sys.exit()

    # Ta_s = 1.0
    # qcs_result = qcs_kernal_sep(
    #     t1_ps,
    #     ch1,
    #     t2_ps,
    #     ch2,

    #     Ta_s_init=Ta_s,
    #     tau_ps_init=62200,
    #     tau_rt_ps_init=107400,

    #     fit_curve="g2calib_2",
    #     # time_bin_ps_init=100,
    #     time_bin_ps_init=20,
    #     # Ta_s_max=10,
    #     Ta_s_max=Ta_s,
    #     # tau_span_ps_init=120e3,

    #     # Ta_ref_s=0.1,
    #     Ta_ref_s=Ta_s,
    #     # show_fit=True,
    #     # second_order_fit=True,

    #     Ta_rt_s_max=Ta_s,

    #     Freq = 50e6,
    # )
    Ta_s = 3
    qcs_result = qcs_kernal_sep(
        t1_ps,
        ch1,
        t2_ps,
        ch2,

        ch_0 = 3,
        ch_1 = 7,
        ch_2 = 3,

        fit_parameters={"height": 4000,"sigma": 10,"offset": 0.01 },

        Ta_s_init=Ta_s,
        tau_ps_init=0.038,
        tau_rt_ps_init=0.0,
 
        fit_curve="gaussian",
        time_bin_ps_init=10,
        Ta_s_max=Ta_s,

        # Ta_ref_s=0.1,
        Ta_ref_s=Ta_s,
        # show_fit=True,

        Ta_rt_s_max=Ta_s,

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

    # results = g2_process_sep(
    #     t1_ps[ch1==3],
    #     t2_ps[ch2==7],
    #     # t_center=107.4,
    #     t_center=0,
    #     # t_center=45.75,
    #     # t_center=-0.025,
    #     # t_stop=600,
    #     t_span=10,
    #     time_bin=0.001,
    #     # time_bin=2,
    #     sig_half=5.0,
    #     freq=50e6,
    #     filename=filename,
    # )
    # g2_plot(results=results)
    # save_g2_results(results)

    # load_time_dependent_results()
