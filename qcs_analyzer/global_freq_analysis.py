"""Frequency estimation from sparse detections of a periodic pulse train."""

import numpy as np


def global_freq_alignment(
    t1_ps,
    t2_ps,
    *,
    sweep_range_ps_per_s=100_000.0,
    coarse_steps=31,
    refine_steps=21,
    min_bin_width_ps=100.0,
    max_histogram_bins=4194304,
    show_plot=True,
):
    """Estimate and compensate clock-frequency skew between two photon streams.

    This follows the coarse frequency-alignment step described in the
    supplied Phys. Rev. Applied article: repeatedly apply a trial linear time
    correction to the second stream, cross-correlate the two binned streams,
    and choose the correction with the strongest correlation peak. The search
    is performed once coarsely and then refined around its best candidate.

    Parameters
    ----------
    t1_ps, t2_ps : array_like
        Timestamp arrays in picoseconds, one for each detector/tagger. The
        arrays may be unsorted. They should cover overlapping portions of
        the same acquisition, with enough correlated photon pairs to produce
        a detectable cross-correlation peak.
    sweep_range_ps_per_s : float, default 100_000
        Search from ``-sweep_range_ps_per_s`` to ``+sweep_range_ps_per_s``.
        The default is 100,000 ps/s = 100 ns/s = 0.1 ppm fractional skew.
    coarse_steps : int, default 31
        Number of trial corrections in the initial search.
    refine_steps : int, default 21
        Number of trial corrections in a narrower search around the coarse
        optimum.
    min_bin_width_ps : float, default 100
        Do not use histogram bins narrower than this value.
    max_histogram_bins : int, default 4194304
        Cap the number of bins per stream to bound memory and FFT cost. If
        needed, the function widens the bins automatically.
    show_plot : bool, default True
        Show the coarse and refined score sweeps and the cross-correlation
        trace at the selected skew.

    Returns
    -------
    dict
        ``skew_s_per_s`` and ``skew_ps_per_s`` give the correction applied to
        stream 2 using
        ``t2_corrected = t2 + skew_s_per_s * (t2 - t2[0])`` after rebasing
        stream 2 to zero. ``skew_ppm`` is the same fractional correction in
        parts per million. ``offset_ps`` is the additional shift to apply to
        the corrected stream to place the correlation peak at zero delay.
        ``t2_aligned_ps`` contains the rebased, corrected, and offset-adjusted
        timestamps. Also returns the peak score, peak counts, and bin width.

    Notes
    -----
    This estimates relative clock skew, not an absolute frequency standard.
    The correction sign follows the article's convention of adjusting the
    second stream. A rough data-package/time overlap is still required; this
    function cannot identify which portions of two unrelated recordings
    correspond to each other. The histogram FFT is memory-bounded, so the
    effective time resolution may be wider than ``min_bin_width_ps`` for long
    acquisitions or broad skew searches.
    """
    first = np.asarray(t1_ps).ravel()
    second = np.asarray(t2_ps).ravel()
    if first.size < 2 or second.size < 2:
        raise ValueError("Both timestamp arrays must contain at least two events.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Timestamp arrays must contain only finite values.")
    sweep_range_ps_per_s = float(sweep_range_ps_per_s)
    if not np.isfinite(sweep_range_ps_per_s) or sweep_range_ps_per_s <= 0:
        raise ValueError("sweep_range_ps_per_s must be finite and positive.")
    skew_range_s_per_s = sweep_range_ps_per_s * 1e-12
    if int(coarse_steps) != coarse_steps or coarse_steps < 3:
        raise ValueError("coarse_steps must be an integer of at least 3.")
    if int(refine_steps) != refine_steps or refine_steps < 3:
        raise ValueError("refine_steps must be an integer of at least 3.")
    if not np.isfinite(min_bin_width_ps) or min_bin_width_ps <= 0:
        raise ValueError("min_bin_width_ps must be finite and positive.")
    if int(max_histogram_bins) != max_histogram_bins or max_histogram_bins < 16:
        raise ValueError("max_histogram_bins must be an integer of at least 16.")

    # Sort before subtracting the origins; preserve integer arithmetic during
    # subtraction when the input arrays contain integer picosecond tags.
    first = np.sort(first)
    second = np.sort(second)
    first_ps = (first - first[0]).astype(np.float64)
    second_ps = (second - second[0]).astype(np.float64)
    duration_ps = max(float(first_ps[-1]), float(second_ps[-1]))
    if duration_ps <= 0:
        raise ValueError("Each timestamp array must span a positive duration.")

    max_bins = int(max_histogram_bins)
    min_bin_width_ps = float(min_bin_width_ps)

    def search(candidates):
        if len(candidates) > 1:
            candidate_step = float(np.max(np.diff(candidates)))
        else:
            candidate_step = 0.0
        # Following the paper, the bin width is set by acquisition duration
        # times the spacing between adjacent skew candidates. Bound the FFT
        # array size for long recordings.
        bin_width_ps = max(
            min_bin_width_ps,
            duration_ps * candidate_step,
            duration_ps / max_bins,
        )
        n_bins = int(np.ceil(duration_ps / bin_width_ps)) + 1
        n_bins = min(n_bins, max_bins + 1)
        n_fft = 1 << (2 * n_bins - 2).bit_length()

        hist1 = np.histogram(
            first_ps, bins=n_bins, range=(0.0, n_bins * bin_width_ps)
        )[0].astype(np.float64)
        fft1 = np.fft.rfft(hist1, n=n_fft)
        best = None
        scores = []
        for skew in candidates:
            corrected2_ps = second_ps * (1.0 + float(skew))
            hist2 = np.histogram(
                corrected2_ps,
                bins=n_bins,
                range=(0.0, n_bins * bin_width_ps),
            )[0].astype(np.float64)
            # Linear cross-correlation: lag is t1 - corrected_t2.
            convolution = np.fft.irfft(
                fft1 * np.fft.rfft(hist2[::-1], n=n_fft), n=n_fft
            )[: 2 * n_bins - 1]
            peak_index = int(np.argmax(convolution))
            peak = float(convolution[peak_index])
            background = float(np.median(convolution))
            score = (peak - background) / np.sqrt(max(background, 1.0))
            scores.append(float(score))
            if best is None or score > best["score"]:
                lag_bins = peak_index - (n_bins - 1)
                best = {
                    "skew": float(skew),
                    "score": float(score),
                    "peak": peak,
                    "offset_ps": float(lag_bins * bin_width_ps),
                    "bin_width_ps": float(bin_width_ps),
                }
        return best, candidates, np.asarray(scores, dtype=float)

    coarse_candidates = np.linspace(
        -float(skew_range_s_per_s),
        float(skew_range_s_per_s),
        int(coarse_steps),
    )
    coarse_best, coarse_x, coarse_scores = search(coarse_candidates)
    coarse_step = coarse_candidates[1] - coarse_candidates[0]
    refine_half_range = abs(coarse_step)
    refine_candidates = np.linspace(
        max(-float(skew_range_s_per_s), coarse_best["skew"] - refine_half_range),
        min(float(skew_range_s_per_s), coarse_best["skew"] + refine_half_range),
        int(refine_steps),
    )
    best, refine_x, refine_scores = search(refine_candidates)

    t2_corrected_ps = second_ps * (1.0 + best["skew"])
    t2_aligned_ps = t2_corrected_ps + best["offset_ps"]
    if show_plot:
        import matplotlib.pyplot as plt

        # Recompute the selected cross-correlation so the search loop does
        # not need to retain a large correlation array for every trial skew.
        bin_width_ps = best["bin_width_ps"]
        n_bins = min(int(np.ceil(duration_ps / bin_width_ps)) + 1, max_bins + 1)
        n_fft = 1 << (2 * n_bins - 2).bit_length()
        hist1 = np.histogram(
            first_ps, bins=n_bins, range=(0.0, n_bins * bin_width_ps)
        )[0].astype(np.float64)
        hist2 = np.histogram(
            t2_corrected_ps, bins=n_bins, range=(0.0, n_bins * bin_width_ps)
        )[0].astype(np.float64)
        correlation = np.fft.irfft(
            np.fft.rfft(hist1, n=n_fft)
            * np.fft.rfft(hist2[::-1], n=n_fft),
            n=n_fft,
        )[: 2 * n_bins - 1]
        lag_ps = (np.arange(2 * n_bins - 1) - (n_bins - 1)) * bin_width_ps
        peak_index = int(np.argmax(correlation))
        peak_lag_ps = lag_ps[peak_index]
        view_half_width_ps = max(100 * bin_width_ps, 10_000.0)
        view = np.abs(lag_ps - peak_lag_ps) <= view_half_width_ps

        figure, axes = plt.subplots(3, 1, sharex=False, figsize=(8, 10))
        axes[0].plot(coarse_x * 1e12, coarse_scores, ".-")
        axes[0].axvline(coarse_best["skew"] * 1e12, color="tab:red", linestyle="--")
        axes[0].set_title("Coarse clock-skew search")
        axes[0].set_ylabel("Correlation peak score")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(refine_x * 1e12, refine_scores, ".-")
        axes[1].axvline(best["skew"] * 1e12, color="tab:red", linestyle="--", label="Selected")
        axes[1].set_title("Refined clock-skew search")
        axes[1].set_xlabel("Applied skew (ps/s)")
        axes[1].set_ylabel("Correlation peak score")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot((lag_ps[view] - peak_lag_ps) * 1e-3, correlation[view])
        axes[2].axvline(0.0, color="tab:red", linestyle="--", label="Correlation peak")
        axes[2].set_title("Cross-correlation at selected skew")
        axes[2].set_xlabel("Delay from peak (ns)")
        axes[2].set_ylabel("Coincidence count per bin")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        figure.tight_layout()
        plt.show()

    return {
        "skew_s_per_s": best["skew"],
        "skew_ps_per_s": best["skew"] * 1e12,
        "skew_ppm": best["skew"] * 1e6,
        "offset_ps": best["offset_ps"],
        "t2_aligned_ps": t2_aligned_ps,
        "peak_score": best["score"],
        "peak_counts": best["peak"],
        "bin_width_ps": best["bin_width_ps"],
        "duration_s": duration_ps * 1e-12,
    }


def estimate_pulse_train_frequency(
    t1_ps,
    nominal_frequency_hz=50e6,
    *,
    events_per_phase=100,
    min_phase_events=20,
    max_search_range_hz=1000.0,
):
    """Estimate pulse-train frequency from irregular photon arrival times.

    The method folds short groups of photon timestamps at the nominal period,
    measures each group's average pulse phase, unwraps phase drift over the
    full acquisition, and fits phase versus elapsed time. Missing detections
    are allowed: each timestamp contributes phase information without needing
    to know the number of pulses since the previous detection.

    Parameters
    ----------
    t1_ps : array_like
        Photon arrival timestamps in picoseconds. They need not be uniformly
        spaced and may be unsorted.
    nominal_frequency_hz : float, default 50e6
        Approximate pulse repetition frequency. It must be close enough that
        the phase drift between adjacent groups is less than half a cycle.
    events_per_phase : int, default 100
        Number of events per phase estimate. Larger groups reduce phase noise
        but make phase unwrapping more sensitive to nominal-frequency error.
    min_phase_events : int, default 20
        Discard a final group with fewer than this many events.
    max_search_range_hz : float, default 1000
        Maximum expected difference from ``nominal_frequency_hz``. Used to
        check that adjacent phase estimates are close enough to unwrap.

    Returns
    -------
    dict
        Contains ``frequency_hz``, ``period_ps``, ``frequency_offset_hz``,
        ``phase_coherence``, ``number_of_events``, ``number_of_phase_groups``,
        and ``duration_s``.

    Notes
    -----
    This estimates the source frequency relative to the time tagger's
    timestamp clock. Absolute accuracy therefore depends on that clock's
    calibration. Sparse data can make cycle unwrapping ambiguous; in that
    case use a closer nominal frequency, collect more photons, or provide an
    independent source-sync reference.
    """
    timestamps = np.asarray(t1_ps).ravel()
    if timestamps.size < 2:
        raise ValueError("t1_ps must contain at least two timestamps.")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("t1_ps must contain only finite timestamps.")
    if int(events_per_phase) != events_per_phase or events_per_phase < 2:
        raise ValueError("events_per_phase must be an integer of at least 2.")
    if int(min_phase_events) != min_phase_events or min_phase_events < 2:
        raise ValueError("min_phase_events must be an integer of at least 2.")

    nominal_frequency_hz = float(nominal_frequency_hz)
    max_search_range_hz = float(max_search_range_hz)
    if not np.isfinite(nominal_frequency_hz) or nominal_frequency_hz <= 0:
        raise ValueError("nominal_frequency_hz must be finite and positive.")
    if not np.isfinite(max_search_range_hz) or max_search_range_hz <= 0:
        raise ValueError("max_search_range_hz must be finite and positive.")

    # Sort and subtract while timestamps retain their original numeric type,
    # avoiding loss of picosecond precision for large absolute timestamps.
    timestamps = np.sort(timestamps)
    elapsed_ps = (timestamps - timestamps[0]).astype(np.float64)
    if np.any(np.diff(elapsed_ps) < 0):
        raise ValueError("timestamps must be sortable in chronological order.")
    duration_s = float(elapsed_ps[-1] * 1e-12)
    if duration_s <= 0:
        raise ValueError("t1_ps must span a positive duration.")

    # Use groups with a similar number of photon events. The complex mean
    # gives each group's circular phase while remaining insensitive to missed
    # pulses between detections.
    event_groups = [
        np.arange(start, min(start + int(events_per_phase), len(elapsed_ps)))
        for start in range(0, len(elapsed_ps), int(events_per_phase))
    ]
    if len(event_groups) > 1 and len(event_groups[-1]) < min_phase_events:
        event_groups.pop()
    if len(event_groups) < 3:
        raise ValueError(
            "Need at least three phase groups; provide more timestamps or "
            "reduce events_per_phase."
        )

    period_s = 1.0 / nominal_frequency_hz
    group_times_s = []
    group_phases = []
    group_coherence = []
    for indices in event_groups:
        times_s = elapsed_ps[indices] * 1e-12
        nominal_phase_cycles = np.remainder(times_s * nominal_frequency_hz, 1.0)
        phase_vectors = np.exp(2j * np.pi * nominal_phase_cycles)
        mean_vector = np.mean(phase_vectors)
        group_times_s.append(float(np.mean(times_s)))
        group_phases.append(float(np.angle(mean_vector)))
        group_coherence.append(float(np.abs(mean_vector)))

    group_times_s = np.asarray(group_times_s)
    group_phases = np.asarray(group_phases)
    group_coherence = np.asarray(group_coherence)
    time_gaps_s = np.diff(group_times_s)
    if np.any(time_gaps_s <= 0):
        raise ValueError("Could not form strictly increasing phase groups.")
    if np.any(max_search_range_hz * time_gaps_s >= 0.5):
        raise ValueError(
            "Phase unwrapping may be ambiguous for this event rate and search "
            "range. Increase events_per_phase only if the nominal frequency "
            "is more accurate, or provide a sync reference."
        )

    unwrapped_phase_cycles = np.unwrap(group_phases) / (2 * np.pi)
    fit_weights = np.maximum(group_coherence, 1e-6) ** 2
    fit_design = np.column_stack((group_times_s, np.ones_like(group_times_s)))
    weighted_design = fit_design * np.sqrt(fit_weights[:, None])
    weighted_phase = unwrapped_phase_cycles * np.sqrt(fit_weights)
    phase_slope_hz, _phase_intercept = np.linalg.lstsq(
        weighted_design, weighted_phase, rcond=None
    )[0]
    frequency_hz = nominal_frequency_hz + float(phase_slope_hz)
    if frequency_hz <= 0:
        raise ValueError("Estimated frequency is not positive.")
    if abs(phase_slope_hz) > max_search_range_hz:
        raise ValueError(
            "Estimated offset exceeds max_search_range_hz; increase the "
            "search range or check for phase-unwrapping ambiguity."
        )

    return {
        "frequency_hz": frequency_hz,
        "period_ps": 1e12 / frequency_hz,
        "frequency_offset_hz": float(phase_slope_hz),
        "phase_coherence": float(np.average(group_coherence, weights=fit_weights)),
        "number_of_events": int(len(elapsed_ps)),
        "number_of_phase_groups": int(len(group_times_s)),
        "duration_s": duration_s,
    }


def global_freq_alignment_g2(
    t1_ps,
    t2_ps,
    *,
    sweep_range_ps_per_s=20000.0,
    sweep_center_ps_per_s=0.0,
    init_time_offset_range_ms=5.0,
    init_bin_number=20_000,
    time_offset_range_ns=1000.0,
    max_bin_number=2_000,
    coarse_steps=21,
    show_plot=False,
):
    """Estimate relative clock skew by sweeping a g2 delay histogram.

    First apply ``sweep_center_ps_per_s`` to stream 2 and use FFT cross-correlation
    of coarse timestamp histograms over ``+/-init_time_offset_range_ms`` to find
    the initial delay. Alternatively, pass ``(min_ms, max_ms)`` to search an
    explicit interval such as ``(20, 80)``. Then sweep
    skew over a fixed window centered on that bin's center, extending
    ``+/-time_offset_range_ns``. Initialization uses ``init_bin_number``
    correlation-delay bins across the full initial delay span (default 10,000);
    the sweep uses ``max_bin_number`` bins.
    Each stage's bin width is its full delay span divided by the bin count.
    Trial skew corrections run from ``sweep_center_ps_per_s -
    sweep_range_ps_per_s`` to ``sweep_center_ps_per_s + sweep_range_ps_per_s``.
    Initialization and each trial apply their absolute correction to the
    original timestamps; the sweep does not apply the center correction twice.

    Initialization uses FFT cross-correlation of coarse binned event streams.
    Each sweep trial applies a linear time correction to stream 2 and calculates
    the pair-delay histogram with :func:`qcs_kernal.g2_histogram_numba`. The
    candidate with the strongest peak above the histogram background is kept.

    The returned ``g2_histogram`` contains raw pair counts, not normalized
    g2 values. ``peak_score`` is a heuristic ranking of the largest bin; it
    is not a calibrated test that correlated photons are present. A maximum
    exists even for uncorrelated data. If candidates tie, the returned skew
    is the smallest-magnitude tied candidate and ``unique_best_candidate``
    is False. A unique maximum alone does not establish statistical confidence.

    Timestamp arrays use picoseconds. Both inputs must use a common time
    origin (for example, their matched first triggers at zero). A single common origin is
    subtracted from both arrays internally, preserving their relative delay.

    The candidate ``c`` in ps/s is applied independently as
    ``t2_trial = t2 + (t2 - t2[0]) * c * 1e-12``. Positive corrections stretch
    stream 2's elapsed time; negative corrections compress it. This is the
    applied correction, whose sign opposes stream 2's measured clock error.
    The result's ``offset_ps`` is then added to align its g2 peak to zero.
    Returned timestamps are relative to ``time_origin_ps``.

    With ``show_plot=True``, show the initialization histogram before the
    sweep and a final figure after it completes. The final figure contains
    the selected g2 histogram and the completed sweep scores.

    The pair counter counts all event pairs inside the selected delay range.
    Consequently, runtime scales with the number of pairs in that range, not
    just the number of histogram bins; a broad window can be costly at high
    count rates.
    """
    first = np.asarray(t1_ps).ravel()
    second = np.asarray(t2_ps).ravel()
    if first.size < 2 or second.size < 2:
        raise ValueError("Both timestamp arrays must contain at least two events.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Timestamp arrays must contain only finite values.")

    sweep_range_ps_per_s = float(sweep_range_ps_per_s)
    sweep_center_ps_per_s = float(sweep_center_ps_per_s)
    time_offset_range_ns = float(time_offset_range_ns)
    if not np.isfinite(sweep_range_ps_per_s) or sweep_range_ps_per_s <= 0:
        raise ValueError("sweep_range_ps_per_s must be finite and positive.")
    if not np.isfinite(sweep_center_ps_per_s):
        raise ValueError("sweep_center_ps_per_s must be finite.")
    init_range_was_scalar = np.isscalar(init_time_offset_range_ms)
    if init_range_was_scalar:
        range_value = float(init_time_offset_range_ms)
        if not np.isfinite(range_value) or range_value <= 0:
            raise ValueError("init_time_offset_range_ms must be finite and positive.")
        init_min_ms, init_max_ms = -range_value, range_value
    else:
        try:
            init_min_ms, init_max_ms = map(float, init_time_offset_range_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "init_time_offset_range_ms must be a positive scalar or a "
                "(min_ms, max_ms) pair."
            ) from exc
        if (
            not np.isfinite(init_min_ms)
            or not np.isfinite(init_max_ms)
            or init_min_ms >= init_max_ms
        ):
            raise ValueError("Explicit range must have finite bounds with min_ms < max_ms.")
    init_time_offset_range_ms = (init_min_ms, init_max_ms)
    if not np.isfinite(time_offset_range_ns) or time_offset_range_ns <= 0:
        raise ValueError("time_offset_range_ns must be finite and positive.")
    if not np.isfinite(init_bin_number) or int(init_bin_number) != init_bin_number or init_bin_number < 2:
        raise ValueError("init_bin_number must be an integer of at least 2.")
    if int(max_bin_number) != max_bin_number or max_bin_number < 2:
        raise ValueError("max_bin_number must be an integer of at least 2.")
    if int(coarse_steps) != coarse_steps or coarse_steps < 3:
        raise ValueError("coarse_steps must be an integer of at least 3.")
    from .qcs_kernal import g2_histogram_numba

    first = np.sort(first)
    second = np.sort(second)
    time_origin_ps = min(first[0], second[0])
    first_ps = (first - time_origin_ps).astype(np.float64)
    second_ps = (second - time_origin_ps).astype(np.float64)
    correction_reference_ps = float(second_ps[0])
    second_elapsed_ps = (second - second[0]).astype(np.float64)
    duration_ps = max(float(first_ps[-1]), float(second_ps[-1]))
    if duration_ps <= 0:
        raise ValueError("Each timestamp array must span a positive duration.")

    acquisition_duration_s = duration_ps * 1e-12
    init_min_ps = init_min_ms * 1e9
    init_max_ps = init_max_ms * 1e9
    init_bin_width_ps = (init_max_ps - init_min_ps) / int(init_bin_number)
    init_second_ps = second_ps + second_elapsed_ps * (sweep_center_ps_per_s * 1e-12)
    # Coarsely bin both streams and compute their linear cross-correlation
    # using FFTs. Padding to at least twice the time-grid length prevents
    # circular wraparound; only delays within the requested initialization
    # range are retained.
    first_bin = np.floor(first_ps / init_bin_width_ps).astype(np.int64)
    second_bin = np.floor(init_second_ps / init_bin_width_ps).astype(np.int64)
    n_time_bins = max(int(first_bin.max()), int(second_bin.max())) + 1
    first_hist = np.bincount(first_bin, minlength=n_time_bins).astype(np.float64)
    second_hist = np.bincount(second_bin, minlength=n_time_bins).astype(np.float64)
    n_fft = 1 << (2 * n_time_bins - 2).bit_length()
    correlation_circular = np.fft.irfft(
        np.fft.rfft(second_hist, n=n_fft)
        * np.conj(np.fft.rfft(first_hist, n=n_fft)),
        n=n_fft,
    )
    min_lag_bin = max(int(np.ceil(init_min_ps / init_bin_width_ps)), -(n_time_bins - 1))
    max_lag_bin = min(int(np.floor(init_max_ps / init_bin_width_ps)), n_time_bins - 1)
    if min_lag_bin > max_lag_bin:
        raise ValueError(
            "The requested initialization interval does not overlap the "
            "available correlation lags."
        )
    init_lag_bins = np.arange(min_lag_bin, max_lag_bin + 1, dtype=np.int64)
    init_histogram = np.rint(
        correlation_circular[np.mod(init_lag_bins, n_fft)]
    ).astype(np.int64)
    init_centers_ps = init_lag_bins.astype(np.float64) * init_bin_width_ps
    init_peak_index = int(np.argmax(init_histogram))
    if init_histogram[init_peak_index] == 0:
        raise ValueError("No event pairs inside the initial delay range; cannot locate a peak.")
    init_peak_delay_ps = float(init_centers_ps[init_peak_index])
    half_range_ps = time_offset_range_ns * 1e3
    edges_ps = np.linspace(
        init_peak_delay_ps - half_range_ps,
        init_peak_delay_ps + half_range_ps,
        int(max_bin_number) + 1,
        dtype=np.float64,
    )
    print(
        f"g2 initialization complete: peak delay={init_peak_delay_ps * 1e-3:.6f} ns, "
        f"peak counts={int(init_histogram[init_peak_index]):,}; "
        f"sweep window=peak +/-{time_offset_range_ns:g} ns",
        flush=True,
    )
    bin_width_ps = float(edges_ps[1] - edges_ps[0])
    bin_centers_ns = (edges_ps[:-1] + edges_ps[1:]) * 0.5e-3
    skew_range_s_per_s = sweep_range_ps_per_s * 1e-12

    if show_plot:
        import matplotlib.pyplot as plt

        init_figure, init_axis = plt.subplots(figsize=(10, 5))
        init_centers_ms = init_centers_ps * 1e-9
        init_axis.plot(init_centers_ms, init_histogram, lw=1, label="Initial g2")
        init_axis.axvline(
            init_peak_delay_ps * 1e-9, color="tab:red", linestyle="--",
            label=f"Selected peak: {init_peak_delay_ps * 1e-3:.3f} ns",
        )
        init_axis.axvspan(
            edges_ps[0] * 1e-9, edges_ps[-1] * 1e-9,
            color="tab:orange", alpha=0.3,
            label=f"Sweep window: peak +/-{time_offset_range_ns:g} ns",
        )
        init_axis.set_title(
            f"Initialization g2: skew={sweep_center_ps_per_s:+.3f} ps/s, "
            f"bin width={init_bin_width_ps * 1e-3:g} ns"
        )
        init_axis.set_xlabel("Delay t2 - t1 (ms)")
        init_axis.set_ylabel("Pair count per bin")
        init_axis.set_ylim(bottom=0)
        init_axis.grid(True, alpha=0.3)
        init_axis.legend()
        init_figure.tight_layout()
        init_figure.canvas.draw()
        plt.show(block=False)
        plt.pause(0.05)

    def sweep(candidates):
        scores = np.empty(len(candidates), dtype=np.float64)
        peak_counts = np.empty(len(candidates), dtype=np.int64)
        background_counts = np.empty(len(candidates), dtype=np.float64)
        peak_delays_ps = np.empty(len(candidates), dtype=np.float64)
        changed_bins = np.zeros(len(candidates), dtype=np.int64)
        max_count_changes = np.zeros(len(candidates), dtype=np.int64)
        previous_histogram = None
        best = None
        for index, skew in enumerate(candidates):
            # Every trial starts with the original array, anchored at the
            # same reference event. No previous trial feeds into this one.
            corrected_second = second_ps + second_elapsed_ps * float(skew)
            result = g2_histogram_numba(
                first_ps,
                corrected_second,
                edges_ps,
                Ta_s=acquisition_duration_s,
                chA=None,
                chB=None,
                time_bin_ps_init=bin_width_ps,
            )
            histogram = result["histogram_raw"]
            peak_index = int(np.argmax(histogram))
            peak = int(histogram[peak_index])

            # Exclude a small neighborhood around the peak when estimating
            # the accidental-pair background.
            exclusion_bins = min(
                max(1, int(np.ceil(1_000.0 / bin_width_ps))),
                (histogram.size - 2) // 2,
            )
            background_mask = np.ones(histogram.size, dtype=bool)
            background_mask[
                max(0, peak_index - exclusion_bins) : min(
                    histogram.size, peak_index + exclusion_bins + 1
                )
            ] = False
            background = float(np.median(histogram[background_mask]))
            score = (peak - background) / np.sqrt(max(background, 1.0))
            scores[index] = score
            peak_counts[index] = peak
            background_counts[index] = background
            peak_delays_ps[index] = bin_centers_ns[peak_index] * 1e3
            if previous_histogram is not None:
                difference = histogram - previous_histogram
                changed_bins[index] = np.count_nonzero(difference)
                max_count_changes[index] = np.max(np.abs(difference))
            score_tied = best is not None and np.isclose(
                score, best["score"], rtol=1e-12, atol=1e-12
            )
            if (
                best is None
                or (score > best["score"] and not score_tied)
                or (score_tied and abs(skew) < abs(best["skew"]))
            ):
                best = {
                    "skew": float(skew),
                    "score": float(score),
                    "peak": peak,
                    "peak_index": peak_index,
                    "background": background,
                    "histogram": histogram,
                }
            previous_histogram = histogram.copy()
        return (
            best, np.asarray(candidates), scores, changed_bins, max_count_changes,
            peak_counts, background_counts, peak_delays_ps,
        )

    coarse_candidates = np.linspace(
        sweep_center_ps_per_s - sweep_range_ps_per_s,
        sweep_center_ps_per_s + sweep_range_ps_per_s,
        int(coarse_steps),
    ) * 1e-12
    (
        coarse_best, coarse_x, coarse_scores, changed_bins, max_count_changes,
        peak_counts, background_counts, peak_delays_ps,
    ) = sweep(coarse_candidates)

    best = coarse_best
    peak_delay_ps = float(edges_ps[best["peak_index"]]) + bin_width_ps / 2
    offset_ps = -peak_delay_ps
    t2_aligned_ps = second_ps + second_elapsed_ps * best["skew"] + offset_ps
    tied_best = np.isclose(coarse_scores, best["score"], rtol=1e-12, atol=1e-12)
    unique_best_candidate = np.count_nonzero(tied_best) == 1
    tied_message = (
        f"; {np.count_nonzero(tied_best)} candidates tie, so the result is not unique"
        if not unique_best_candidate
        else ""
    )
    print(
        f"g2 sweep complete: skew={best['skew'] * 1e12:+.3f} ps/s, "
        f"peak delay={peak_delay_ps * 1e-3:.6f} ns, "
        f"peak={best['peak']:,}, background={best['background']:.3f}, "
        f"score={best['score']:.3f}{tied_message}",
        flush=True,
    )

    if show_plot:
        import matplotlib.pyplot as plt

        final_figure, final_axes = plt.subplots(1, 2, figsize=(12, 4.5))
        final_hist_axis, final_score_axis = final_axes
        final_hist_axis.plot(bin_centers_ns, best["histogram"], lw=1)
        final_hist_axis.axvline(
            bin_centers_ns[best["peak_index"]],
            color="tab:red", linestyle="--", label="Selected peak",
        )
        final_hist_axis.axhline(
            best["background"], color="0.5", linestyle=":", label="Median background"
        )
        final_hist_axis.set_title("Final g2 histogram")
        final_hist_axis.set_xlabel("Delay t2 - t1 (ns)")
        final_hist_axis.set_ylabel("Pair count per bin")
        final_hist_axis.grid(True, alpha=0.3)
        final_hist_axis.legend()

        final_score_axis.plot(coarse_x * 1e12, coarse_scores, ".-")
        final_score_axis.axvline(
            best["skew"] * 1e12, color="tab:red", linestyle="--", label="Selected skew"
        )
        final_score_axis.set_title("Completed coarse sweep")
        final_score_axis.set_xlabel("Applied skew (ps/s)")
        final_score_axis.set_ylabel("Peak score")
        final_score_axis.grid(True, alpha=0.3)
        final_score_axis.legend()
        final_figure.tight_layout()
        plt.show()

    return {
        "skew_s_per_s": best["skew"],
        "skew_ps_per_s": best["skew"] * 1e12,
        "skew_ppm": best["skew"] * 1e6,
        "offset_ps": offset_ps,
        "t2_aligned_ps": t2_aligned_ps,
        "bin_centers_ns": bin_centers_ns,
        "g2_histogram": best["histogram"],
        "peak_score": best["score"],
        "peak_counts": best["peak"],
        "background_counts": best["background"],
        "bin_width_ps": bin_width_ps,
        "duration_s": acquisition_duration_s,
        "time_origin_ps": time_origin_ps.item(),
        "correction_reference_ps": correction_reference_ps,
        "init_time_offset_range_ms": (
            float(range_value)
            if init_range_was_scalar
            else (float(init_min_ms), float(init_max_ms))
        ),
        "sweep_center_ps_per_s": sweep_center_ps_per_s,
        "time_offset_range_ns": time_offset_range_ns,
        "init_peak_delay_ps": init_peak_delay_ps,
        "init_bin_width_ps": init_bin_width_ps,
        "init_bin_number": int(init_bin_number),
        "init_bin_centers_ns": init_centers_ps * 1e-3,
        "init_g2_histogram": init_histogram,
        "sweep_delay_range_ps": np.array([edges_ps[0], edges_ps[-1]]),
        "sweep_skew_ps_per_s": coarse_x * 1e12,
        "sweep_scores": coarse_scores,
        "sweep_changed_bins": changed_bins,
        "sweep_max_count_change": max_count_changes,
        "tied_best_skew_ps_per_s": coarse_x[tied_best] * 1e12,
        "unique_best_candidate": bool(unique_best_candidate),
        "sweep_peak_counts": peak_counts,
        "sweep_background_counts": background_counts,
        "sweep_peak_delays_ps": peak_delays_ps,
        "sweep_last_tag_corrections_ps": coarse_x * second_elapsed_ps[-1],
    }


__all__ = [
    "estimate_pulse_train_frequency",
    "global_freq_alignment",
    "global_freq_alignment_g2",
]
