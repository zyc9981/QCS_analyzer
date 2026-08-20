from qcs_analyzer._implementation import *
import numpy as np
import matplotlib.pyplot as plt


def u0_estimator(t_ref_0, t_ref_1):
    """Return the fractional global drift of reference stream 1 relative to 0."""
    n = min(len(t_ref_0), len(t_ref_1))
    return (t_ref_1[n - 1] - t_ref_0[n - 1]) / (t_ref_0[n - 1] - t_ref_0[0])


def qcs_ref_estimator(
    tau,
    t,
    N_sample=None,
    t_eval_ps=None,
):
    """Return residual standard deviation and fitted tau at the time-span center."""
    if N_sample != None:
        N_sample = int(np.min([np.size(tau), N_sample]))
    else:
        N_sample = np.size(tau)
    if N_sample < 2:
        raise ValueError("tau and t must contain at least two samples for a linear fit.")

    tau = np.asarray(tau).ravel()[:N_sample]
    t = np.asarray(t).ravel()[:N_sample]
    fit = np.polyfit(t, tau, 1)
    # t_center = (t[0] + t[-1]) / 2
    # tau_center = np.polyval(fit, t_center)

    if t_eval_ps == None:
        t_eval_ps = (t[0] + t[-1]) / 2
    tau_at_point = np.polyval(fit, t_eval_ps)

    return tau_at_point, np.std(tau - np.polyval(fit, t)), N_sample


def qcs_kernal(
    t_ps,
    ch,
    ch_0=1,
    ch_1=6,
    ch_ref_0=2,
    ch_ref_1=7,

    tau_ps_init=24601.8e3,
    Ta_s_init=0.1,
    tau_span_ps_init=40e3,
    u0_init = 0.0,
    time_bin_ps_init=500,
    Ta_ref_s=0.1,

    Ta_s_max=10,
    fit_curve="g2calib_1",
    fit_parameters={
        "height": 40,
        # "gamma": 0.7,
        # "sigma": 0.1,
        "sigma": 100,
        "offset": 0.01,
    },
    # sour_freq=100e6,
    fwhm_calib_ps=1000,
    show_fit=False,
):
    """
    Chronologically scan paired-channel data in adaptive Ta intervals.

    Leave correction, fitting, and Ta-update logic in the marked sections.
    """
    t_ps = np.asarray(t_ps).ravel()
    ch = np.asarray(ch).ravel()

    if len(t_ps) == 0 or len(ch) == 0:
        raise ValueError("t_ps and ch must not be empty.")
    if len(t_ps) != len(ch):
        raise ValueError("t_ps and ch must have the same length.")

    t_end_ps = t_ps / Ta_s_init

    fit_curve = FIT_CURVE_ALIASES.get(str(fit_curve).strip().lower(), str(fit_curve).strip().lower())
    if fit_curve not in FIT_CURVES:
        available = ", ".join(sorted(FIT_CURVES))
        raise ValueError(f"Unknown fit_curve '{fit_curve}'. Choose one of: {available}.")
    try:
        fit_parameter_guess = {} if fit_parameters is None else dict(fit_parameters)
    except (TypeError, ValueError) as exc:
        raise TypeError("fit_parameters must be a dictionary of initial fit values.") from exc

    # Keep each channel chronological.
    t_0 = np.sort(t_ps[ch == ch_0])
    t_ref_0 = np.sort(t_ps[ch == ch_ref_0])
    t_1 = np.sort(t_ps[ch == ch_1])
    t_ref_1 = np.sort(t_ps[ch == ch_ref_1])

    if len(t_0) == 0 or len(t_1) == 0:
        raise ValueError("Both channels need at least one event.")

    # State carried from one adaptive interval to the next.
    state = {
        "Ta_s": float(Ta_s_init),
        "time_bin_ps": float(time_bin_ps_init),
        "tau_ps": float(tau_ps_init),
        "tau_span_ps": float(tau_span_ps_init),
        "u0": float(u0_init),
        "T_lastmid_ps": float(0.0),
        # "u1": 0.0,
    }

    # Temporary: calculate u0_init from classical recorded pulses
    state["u0"] = u0_estimator(t_ref_0, t_ref_1)
    # print(state["u0"])

    T_file_s = (np.max(t_ps) - np.min(t_ps)) * 1e-12
    buffer_size = int(np.floor(T_file_s / Ta_s_init))
    point_records = np.full(buffer_size, np.nan, dtype=float)
    tau_ps_records = np.full(buffer_size, np.nan, dtype=float)
    tau_ref_ps_records = np.full(buffer_size, np.nan, dtype=float)
    tau_ref_uncertainty_ps_records = np.full(buffer_size, np.nan, dtype=float)
    Ta_s_records = np.full(buffer_size, np.nan, dtype=float)
    fwhm_records = np.full(buffer_size, np.nan, dtype=float)
    sigma_records = np.full(buffer_size, np.nan, dtype=float)
    tau_s_uncertainty_records = np.full(buffer_size, np.nan, dtype=float)
    CC_count_records = np.full(buffer_size, np.nan, dtype=float)
    # S_peak_records = np.full(buffer_size, np.nan, dtype=float)
    # fwhm_uncertainty_records = np.full(buffer_size, np.nan, dtype=float)
    coincidence_records = np.full(buffer_size, np.nan, dtype=float)

    # fit_results = [None] * buffer_size
    # g2_results = [None] * buffer_size
    # errors = [None] * buffer_size

    acquisition_index = 0
    acquisition_results = []

    window_start_ps = min(t_0[0], t_1[0])
    recording_end_ps = max(t_0[-1], t_1[-1])

    index_0 = 0
    index_1 = 0
    index_ref = 0

    while window_start_ps < recording_end_ps:
        Ta_s = state["Ta_s"]

        remaining_s = (recording_end_ps - window_start_ps) * 1e-12
        if remaining_s < Ta_s:
            # Do not fit, correct, save, or update the state for this partial tail.
            break

        if not np.isfinite(Ta_s) or Ta_s <= 0:
            raise ValueError("State Ta_s must be positive.")

        # ------------------------------------------------------------
        # STEP 1:
        # Data acquisition and correction
        # ------------------------------------------------------------
        window_end_ps = window_start_ps + Ta_s * 1e12

        # Collect only events inside this chronological interval.
        end_index_0 = np.searchsorted(t_0, window_end_ps, side="left")
        end_index_1 = np.searchsorted(t_1, window_end_ps, side="left")

        acquisition_t_0 = t_0[index_0:end_index_0]
        acquisition_t_1 = t_1[index_1:end_index_1]
        acquisition_t_1 = acquisition_t_1 - (acquisition_t_1 - acquisition_t_1[0]) * state["u0"] + state["u0"] * state["Ta_s"] * 1e12 / 2

        acquisition = {
            "start_ps": window_start_ps,
            "end_ps": window_end_ps,
            "point_ps": (window_start_ps + window_end_ps) / 2,
            "Ta_s": Ta_s,
            "t_0": acquisition_t_0,
            "t_1": acquisition_t_1,
            # "state_before": state.copy(),
        }

        # Collect ref events inside this chronological interval.

        ref_width_ps = Ta_ref_s * 1e12
        available_start_ps = t_ref_0[0]
        available_end_ps = t_ref_0[-1]

        if ref_width_ps >= available_end_ps - available_start_ps:
            # Reference duration is shorter than the requested window.
            ref_start_ps = available_start_ps
            ref_end_ps = available_end_ps
        else:
            # Center when possible; shift inward near either boundary.
            ref_center_ps = np.clip(
                acquisition["point_ps"],
                available_start_ps + ref_width_ps / 2,
                available_end_ps - ref_width_ps / 2,
            )
            ref_start_ps = ref_center_ps - ref_width_ps / 2
            ref_end_ps = ref_center_ps + ref_width_ps / 2

        ref_start_index = np.searchsorted(t_ref_0, ref_start_ps, side="left")
        ref_end_index = np.searchsorted(t_ref_0, ref_end_ps, side="right")

        acquisition_ref_0 = t_ref_0[ref_start_index:ref_end_index]
        acquisition_ref_1 = t_ref_1[ref_start_index:ref_end_index]

        # ref_start_ps = acquisition["point_ps"] - Ta_ref_s * 1e12 / 2
        # ref_end_ps = acquisition["point_ps"] + Ta_ref_s * 1e12 / 2
        # ref_start_index = np.searchsorted(t_ref_0, ref_start_ps, side="left")
        # ref_end_index = np.searchsorted(t_ref_0, ref_end_ps, side="left")
        # acquisition_ref_0 = t_ref_0[ref_start_index:ref_end_index]
        # acquisition_ref_1 = t_ref_1[ref_start_index:ref_end_index]
        # # acquisition_ref_1 = acquisition_ref_1 - (acquisition_ref_1 - acquisition_ref_1[0]) * state["u0"]

        # ------------------------------------------------------------
        # STEP 2:
        # Create start-stop g2 processing results.
        # ------------------------------------------------------------
        tau_predicted_ps = state["tau_ps"] + (acquisition["point_ps"] - state["T_lastmid_ps"]) * state["u0"]
        t_lo = tau_predicted_ps - state["tau_span_ps"] * 0.5
        t_hi = tau_predicted_ps + state["tau_span_ps"] * 0.5
        time_bin_ps = state["time_bin_ps"]
        edges = np.arange(t_lo, t_hi + time_bin_ps, time_bin_ps)

        # print(t_lo, t_lo-t_hi, time_bin_ps, state["tau_span_ps"])
        if len(edges) < 2:
            raise ValueError("Window too small for given bin width. Increase span or decrease bin.")
    
        centers = edges[:-1] + time_bin_ps / 2
        start_stop_results = np.zeros(len(centers), dtype=np.int64)
    
        j_start = 0
        checkpoint_index = 0
        window_start_s = 0.0
        window_start_ns = 0.0

        for t_i in acquisition_t_0:
            while j_start < len(acquisition_t_1) and (acquisition_t_1[j_start] - t_i) < t_lo:
                j_start += 1
    
            j = j_start
            while j < len(acquisition_t_1):
                dt = acquisition_t_1[j] - t_i
    
                if dt > t_hi:
                    break
    
                kbin = int(np.floor((dt - t_lo) / time_bin_ps))
                if 0 <= kbin < len(start_stop_results):
                    start_stop_results[kbin] += 1
                    # numPairs += 1
    
                j += 1
        
        results = {
            "centers": centers.copy(),
            "histogram": start_stop_results.copy(),
            "T_acq_s": Ta_s,
            "chA": ch_0,
            "chB": ch_1,
        }

        # ------------------------------------------------------------
        # STEP 3:
        # Estimate time offset and frequency skew from g2 fit
        # ------------------------------------------------------------
        try:
            fit_parameters = fit_parameter_guess.copy()
            fit_parameters["center"] = float(
                results["centers"][np.argmax(results["histogram"])]
            )

            # if acquisition["point_ps"] > 3.36e12:
            #     show_fit = True,
            fit = fit_g2(
                results,
                curve=fit_curve,
                parameters=fit_parameters,
                fit_window=state["tau_span_ps"],
                show=show_fit,
            )

        except Exception as exc:
            raise

            print(f"Time-dependent g2 fit failed at t_stop={(acquisition["point_ps"]*1e-12):.6g} s: {exc}")
            fit = None
            exc_value = exc
        else:
            exc_value = None

        # ------------------------------------------------------------
        # STEP 4:
        # Update tau, u0, and records
        # ------------------------------------------------------------

        point_records[acquisition_index] = acquisition["point_ps"]
        tau_ps_records[acquisition_index] = fit["center"]
        Ta_s_records[acquisition_index] = state["Ta_s"]
        # fwhm_records[acquisition_index] = fwhm
        sigma_records[acquisition_index] = fit["parameters"]["sigma"]
        tau_s_uncertainty_records[acquisition_index] = fit["center_uncertainty"]
        # fwhm_uncertainty_records[acquisition_index] = fit["fwhm_uncertainty"]
        CC_count_records[acquisition_index] = np.sum(results["histogram"])
        # S_peak_records[acquisition_index] = np.sqrt(np.sum(results["histogram"]) ** 2 * np.size(results["centers"]) / np.size(acquisition_t_0) / np.size(acquisition_t_1))

        tau_ref_ps, tau_ref_uncertainty_ps, N_ref_samples = qcs_ref_estimator(
            acquisition_ref_1 - acquisition_ref_0,
            acquisition_ref_0,
            t_eval_ps=acquisition["point_ps"],
        )

        tau_ref_ps_records[acquisition_index] = tau_ref_ps
        tau_ref_uncertainty_ps_records[acquisition_index] = tau_ref_uncertainty_ps / np.sqrt(N_ref_samples)
        
        if acquisition_index > 0:
            udrift = (fit["center"] - tau_predicted_ps) / (acquisition["point_ps"] - state["T_lastmid_ps"])
            u0 = state["u0"] + udrift
            state["u0"] = u0

            # u0 = (fit["center"] - state["tau_ps"]) / (acquisition["point_ps"] - state["T_lastmid_ps"])
            # udrift = u0 -state["u0"]
            # state["u0"] = u0
            
            Ta_s = np.round((fwhm_calib_ps / 5 / np.abs(udrift) * 1e-12) / Ta_s_init) * Ta_s_init
            Ta_s = np.min([Ta_s, Ta_s_max])
            state["Ta_s"] = np.max([Ta_s_init, Ta_s])
            # state["time_bin_ps"] = round(time_bin_ps_init / state["Ta_s"] * Ta_s_init / 10.0) * 10.0

            # print(f"Ta_s: {state["Ta_s"]:.2f}")
        else:
            state["Ta_s"] = Ta_s

        state["tau_ps"] = fit["center"]
        state["T_lastmid_ps"] = acquisition["point_ps"]

        # acquisition_results.append(acquisition)

        # Advance both channel indices and move chronologically onward.
        acquisition_index += 1
        index_0 = end_index_0
        index_1 = end_index_1
        # index_ref = end_index_ref
        window_start_ps = window_end_ps

    # ------------------------------------------------------------
    # After scan: plot results
    # ------------------------------------------------------------

    start_index = 6
    record_slice = slice(min(start_index, acquisition_index), acquisition_index)
    point_records = point_records[record_slice]
    tau_ps_records = tau_ps_records[record_slice]
    tau_ref_ps_records = tau_ref_ps_records[record_slice]
    tau_ref_uncertainty_ps_records = tau_ref_uncertainty_ps_records[record_slice]
    Ta_s_records = Ta_s_records[record_slice]
    fwhm_records = fwhm_records[record_slice]
    sigma_records = sigma_records[record_slice]
    tau_s_uncertainty_records = tau_s_uncertainty_records[record_slice]
    CC_count_records = CC_count_records[record_slice]
    coincidence_records = coincidence_records[record_slice]
    acquisition_index = len(point_records)

    # Fig1: details
    fig1, axes = plt.subplots(5, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_ps_records - tau_ps_records[0], "o-")
    ax1.set_ylabel(r"$\tau$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, Ta_s_records, "o-")
    ax2.set_ylabel(r"$T_a$ (s)")
    ax2.grid(True)

    ax3 = axes[2]
    ax3.plot(point_records * 1e-12, sigma_records, "o-")
    ax3.set_ylabel(r"$\sigma$ (ps)")
    ax3.grid(True)

    ax4 = axes[3]
    ax4.plot(point_records * 1e-12, CC_count_records, "o-")
    ax4.set_ylabel(r"CC")
    ax4.grid(True)

    ax5 = axes[4]
    ax5.plot(point_records * 1e-12, tau_ref_ps_records - tau_ref_ps_records[0], "o-")
    ax5.set_ylabel(r"$\tau_{ref}$ (ps)")
    ax5.grid(True)

    # ax6 = axes[5]
    # ax6.plot(point_records[:acquisition_index] * 1e-12, tau_ref_uncertainty_ps_records[:acquisition_index], "o-")
    # ax6.set_ylabel(r"$\delta\tau_{ref}$ (ps)")
    # ax6.grid(True)
    # ax6.set_ylim(bottom=0)

    # ax6 = axes[5]
    # ax6.plot(point_records[:acquisition_index] * 1e-12, S_peak_records[:acquisition_index], "o-")
    # ax6.set_ylabel(r"S")
    # ax6.grid(True)
    # ax6.set_ylim(bottom=0)

    axes[-1].set_xlabel("Record Points (s)")
    fig1.tight_layout()

    # Fig2: summaries
    fig2, axes = plt.subplots(2, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    tau_diff = tau_ps_records - tau_ref_ps_records
    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_diff - np.mean(tau_diff), "o-")
    ax1.set_ylabel(r"$\tau$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, CC_count_records, "o-")
    ax2.set_ylabel(r"CC")
    ax2.grid(True)

    axes[-1].set_xlabel("Record Points (s)")
    fig2.tight_layout()

    print(np.std(tau_diff))
    print(np.mean(tau_diff))

    plt.show()

    return {
        "center": point_records,
        "tau": tau_ps_records,
        "tau_ref": tau_ref_ps_records,
        "tau_ref_uncertainty": tau_ref_uncertainty_ps_records,
        "tau_diff": tau_diff,
        "Ta": Ta_s_records,
        "sigma": sigma_records,
        "figure": fig1,
        "axes": axes,
    }
