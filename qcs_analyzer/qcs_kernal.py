from qcs_analyzer._implementation import *
import numpy as np
import matplotlib.pyplot as plt


def u0_estimator(t_ref_0, t_ref_1):
    """Return the fractional global drift of reference stream 1 relative to 0."""
    n = min(len(t_ref_0), len(t_ref_1))
    return (t_ref_1[n - 1] - t_ref_0[n - 1]) / (t_ref_0[n - 1] - t_ref_0[0])


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

    Ta_s_max=10,
    time_bin_ps=500,
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
        "tau_ps": float(tau_ps_init),
        "tau_span_ps": float(tau_span_ps_init),
        "u0": float(u0_init),
        "T_lastmid_ps": float(0.0),
        # "T_laststart_ps": float(0.0),
        # "u1": 0.0,
    }

    # Temporary: calculate u0_init from classical recorded pulses
    state["u0"] = u0_estimator(t_ref_0, t_ref_1)
    print(state["u0"])

    T_file_s = (np.max(t_ps) - np.min(t_ps)) * 1e-12
    buffer_size = int(np.floor(T_file_s / Ta_s_init))
    point_records = np.full(buffer_size, np.nan, dtype=float)
    tau_ps_records = np.full(buffer_size, np.nan, dtype=float)
    Ta_s_records = np.full(buffer_size, np.nan, dtype=float)
    fwhm_records = np.full(buffer_size, np.nan, dtype=float)
    sigma_records = np.full(buffer_size, np.nan, dtype=float)
    tau_s_uncertainty_records = np.full(buffer_size, np.nan, dtype=float)
    fwhm_uncertainty_records = np.full(buffer_size, np.nan, dtype=float)
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

        # # Collect ref events inside this chronological interval.
        # end_index_ref = np.searchsorted(t_ref_0, window_end_ps, side="left")
        # acquisition_ref_0 = t_ref_0[index_ref:end_index_ref]
        # acquisition_ref_1 = t_ref_1[index_ref:end_index_ref]
        # acquisition_ref_1 = acquisition_ref_1 - (acquisition_ref_1 - acquisition_ref_1[0]) * state["u0"]

        # plt.figure()
        # plt.plot(acquisition_ref_1 - acquisition_ref_0)
        # plt.show()

        acquisition = {
            "start_ps": window_start_ps,
            "end_ps": window_end_ps,
            "point_ps": (window_start_ps + window_end_ps) / 2,
            "Ta_s": Ta_s,
            "t_0": acquisition_t_0,
            "t_1": acquisition_t_1,
            # "state_before": state.copy(),
        }

        # ------------------------------------------------------------
        # STEP 2:
        # Create start-stop g2 processing results.
        # ------------------------------------------------------------
        tau_predicted_ps = state["tau_ps"] + (acquisition["point_ps"] - state["T_lastmid_ps"]) * state["u0"]
        t_lo = tau_predicted_ps - state["tau_span_ps"] * 0.5
        t_hi = tau_predicted_ps + state["tau_span_ps"] * 0.5
        edges = np.arange(t_lo, t_hi + time_bin_ps, time_bin_ps)
    
        if len(edges) < 2:
            raise ValueError("Window too small for given bin width. Increase span or decrease bin.")
    
        centers = edges[:-1] + time_bin_ps / 2
        start_stop_results = np.zeros(len(centers), dtype=np.int64)
    
        j_start = 0
        checkpoint_index = 0
        window_start_s = 0.0
        window_start_ns = 0.0

        for t_i in acquisition_t_0:
            # while checkpoint_index < len(t_stop_values) and t_i >= t_stop_values[checkpoint_index] * 1e9:
            #     fit_checkpoint(checkpoint_index)
            #     checkpoint_index += 1
    
            # if t_i >= t_stop_values[-1] * 1e9:
            #     break
    
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
            # "T_file_s": T_file_s,
            # "t_start": window_start_s,
            # "t_stop": float(t_stop),
            # "time_bin": time_bin,
            "chA": ch_0,
            "chB": ch_1,
            # "cntA_total": cntA_total,
            # "cntB_total": cntB_total,
            # "CC": CC,
            # "CCps": CCps,
            # "CCppulse": CCppulse,
            # "CC_raw": CC_raw,
            # "etaA": etaA,
            # "etaB": etaB,
            # "CAR": CAR,
            # "numA_used": numA_used,
            # "numPairs": numPairs,
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
            # fit_parameters["height"] = float(fit_parameters["height"] / Ta_s)
            # if "center" not in fit_parameters:
            #     fit_parameters["center"] = tau_ps_init
            # else:
            #     fit_parameters["center"] = float(
            #         results["centers"][np.argmax(results["histogram"])]
            #     )

            fit = fit_g2(
                results,
                curve=fit_curve,
                parameters=fit_parameters,
                fit_window=state["tau_span_ps"],
                show=show_fit,
            )

            # fwhm = 2.355 * np.sqrt(2*702.672**2 + fit["parameters"]["sigma"]**2)
            # print(f"FWHM: {fwhm:.2f}", f"gamma: {fit["parameters"]["sigma"]:.2f}")
            # if update_fit_parameters and fit["success"]:
            #     fit_parameter_guess.update(fit["parameters"])

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
        
        if acquisition_index > 0:
            udrift = (fit["center"] - tau_predicted_ps) / (acquisition["point_ps"] - state["T_lastmid_ps"])
            u0 = state["u0"] + udrift
            state["u0"] = u0

            # u0 = (fit["center"] - state["tau_ps"]) / (acquisition["point_ps"] - state["T_lastmid_ps"])
            # udrift = u0 -state["u0"]
            # state["u0"] = u0
            
            Ta_s = np.min([fwhm_calib_ps / 5 / np.abs(udrift) * 1e-12, Ta_s_max])
            state["Ta_s"] = np.max([Ta_s_init, Ta_s])

            print(state["u0"], udrift, fwhm_calib_ps / 5 / np.abs(udrift) * 1e-12, state["Ta_s"])
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

    # fig, axes = plt.subplots(len(display_values), 1, sharex=True, squeeze=False)
    fig, axes = plt.subplots(3, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    ax1 = axes[0]
    ax1.plot(point_records[:acquisition_index] * 1e-12, tau_ps_records[:acquisition_index], "o-")
    ax1.set_ylabel(r"$\tau$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records[:acquisition_index] * 1e-12, Ta_s_records[:acquisition_index], "o-")
    ax2.set_ylabel(r"$T_a$ (s)")
    ax2.grid(True)

    ax3 = axes[2]
    # ax3.plot(point_records[:acquisition_index] * 1e-12, fwhm_records[:acquisition_index], "o-")
    # ax3.set_ylabel(r"FWHM (ps)")
    ax3.plot(point_records[:acquisition_index] * 1e-12, sigma_records[:acquisition_index], "o-")
    ax3.set_ylabel(r"$\sigma$ (ps)")
    ax3.grid(True)

    axes[-1].set_xlabel("Record Points (s)")

    fig.tight_layout()
    plt.show()
