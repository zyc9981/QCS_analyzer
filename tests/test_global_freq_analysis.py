"""Numerical checks for the g2 clock-skew sweep and its correction convention."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from qcs_analyzer.global_freq_analysis import global_freq_alignment_g2
from qcs_analyzer.qcs_kernal import g2_histogram_numba


class GlobalFrequencyG2Tests(unittest.TestCase):
    def run_quietly(self, first, second, **kwargs):
        kwargs.setdefault("max_bin_number", 10000)
        with contextlib.redirect_stdout(io.StringIO()):
            return global_freq_alignment_g2(first, second, show_plot=False, **kwargs)

    def test_nonzero_sweep_center_sets_absolute_trial_corrections(self):
        first = np.array([0, 1_000_000_000_000, 2_000_000_000_000])
        second = first + 5000
        for center in (-10000, 10000):
            with self.subTest(center=center):
                received = []

                def record_call(a, b, edges, **kwargs):
                    received.append(b.copy())
                    return g2_histogram_numba(a, b, edges, **kwargs)

                with patch("qcs_analyzer.qcs_kernal.g2_histogram_numba", new=record_call):
                    result = self.run_quietly(
                        first, second, sweep_center_ps_per_s=center,
                        sweep_range_ps_per_s=2000, coarse_steps=3,
                        init_time_offset_range_ms=0.0001, time_offset_range_ns=100,
                    )

                np.testing.assert_array_equal(
                    received[0], second + np.array([0, center, 2 * center])
                )
                np.testing.assert_array_equal(received[0], received[2])
                expected_skews = np.array([center - 2000, center, center + 2000])
                np.testing.assert_allclose(result["sweep_skew_ps_per_s"], expected_skews)
                for b, skew in zip(received[1:], expected_skews):
                    np.testing.assert_array_equal(b, second + np.array([0, skew, 2 * skew]))
                self.assertEqual(result["sweep_center_ps_per_s"], center)

    def test_nonfinite_sweep_center_is_rejected(self):
        for center in (np.nan, np.inf, -np.inf):
            with self.subTest(center=center):
                with self.assertRaisesRegex(ValueError, "sweep_center_ps_per_s"):
                    self.run_quietly([0, 1000], [0, 1000], sweep_center_ps_per_s=center)

    def test_wide_initialization_then_fixed_narrow_sweep(self):
        first = np.arange(200, dtype=np.int64) * 10_000_000_000
        delay_ps = 600_123_456
        second = first + delay_ps + np.rint(first * 500e-12).astype(np.int64)
        calls = []

        def record_call(a, b, edges, **kwargs):
            calls.append((b.copy(), edges.copy()))
            result = g2_histogram_numba(a, b, edges, **kwargs)
            delays = (b[:, None] - a[None, :]).ravel()
            expected, _ = np.histogram(
                delays[(delays >= edges[0]) & (delays < edges[-1])], edges
            )
            np.testing.assert_array_equal(result["histogram_raw"], expected)
            return result

        with patch("qcs_analyzer.qcs_kernal.g2_histogram_numba", new=record_call):
            result = self.run_quietly(first, second, sweep_range_ps_per_s=1000, coarse_steps=5)

        self.assertEqual(len(calls), 6)
        np.testing.assert_array_equal(calls[0][0], second)
        np.testing.assert_array_equal(calls[0][1][[0, -1]], [-5e9, 5e9])
        self.assertEqual(result["init_bin_width_ps"], 1e6)
        self.assertEqual(result["bin_width_ps"], 200)
        for _, edges in calls[1:]:
            self.assertEqual(len(edges), 10001)
            np.testing.assert_array_equal(edges, calls[1][1])
            np.testing.assert_array_equal(
                edges[[0, -1]], result["init_peak_delay_ps"] + np.array([-1e6, 1e6])
            )
        self.assertAlmostEqual(result["skew_ps_per_s"], -500)
        self.assertAlmostEqual(result["offset_ps"], -delay_ps, delta=200)
        self.assertEqual(len(result["init_g2_histogram"]), 10000)
        self.assertEqual(len(result["g2_histogram"]), 10000)

    def test_initial_range_without_pairs_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "No event pairs"):
            self.run_quietly(
                np.array([0, 10_000_000_000]),
                np.array([1_000_000_000, 11_000_000_000]),
                init_time_offset_range_ms=0.001,
            )

    def test_ps_per_second_correction_is_independent_and_preserves_offset(self):
        epoch = 9_000_000_000_000_000
        first = epoch + np.array([2_000_000_000_000, 0, 1_000_000_000_000])
        second = first + 5_000
        original_first, original_second = first.copy(), second.copy()
        received = []

        def check_counter_input(a, b, edges, **kwargs):
            received.append((a.copy(), b.copy()))
            result = g2_histogram_numba(a, b, edges, **kwargs)
            delays = (b[:, None] - a[None, :]).ravel()
            valid = (delays >= edges[0]) & (delays < edges[-1])
            expected, _ = np.histogram(delays[valid], bins=edges)
            np.testing.assert_array_equal(result["histogram_raw"], expected)
            return result

        with patch(
            "qcs_analyzer.qcs_kernal.g2_histogram_numba", new=check_counter_input
        ):
            result = self.run_quietly(
                first, second, sweep_range_ps_per_s=1000, coarse_steps=3,
                init_time_offset_range_ms=0.0001, time_offset_range_ns=100,
                max_bin_number=2000,
            )

        elapsed = np.array([0, 1_000_000_000_000, 2_000_000_000_000])
        self.assertEqual(len(received), 4)  # uncorrected initialization + 3 trials
        np.testing.assert_array_equal(received[0][1], elapsed + 5000)
        for (a, b), correction in zip(received[1:], [-1, 0, 1]):
            np.testing.assert_array_equal(a, elapsed)
            # At +/-1000 ps/s, the second and third events move by
            # +/-1000 ps and +/-2000 ps, while the first remains fixed.
            np.testing.assert_array_equal(
                b, elapsed + 5000 + correction * np.array([0, 1000, 2000])
            )
        self.assertEqual(result["skew_ps_per_s"], 0)
        self.assertAlmostEqual(result["offset_ps"], -5000, delta=100)
        self.assertEqual(result["time_origin_ps"], epoch)
        self.assertEqual(result["correction_reference_ps"], 5000)
        np.testing.assert_array_equal(first, original_first)
        np.testing.assert_array_equal(second, original_second)

    def test_recovers_known_skew_and_delay_from_missing_detections(self):
        epoch = 9_000_000_000_000_000
        elapsed = np.arange(200, dtype=np.int64) * 100_000_000_000
        first = epoch + elapsed
        second = epoch + elapsed + np.rint(elapsed * 1e-9).astype(np.int64) + 12_345
        # Missing detections do not affect the known clock scaling.
        second = np.delete(second, np.arange(5, len(second), 7))
        result = self.run_quietly(
            first, second, sweep_range_ps_per_s=2000, coarse_steps=9,
            init_time_offset_range_ms=0.0001, time_offset_range_ns=100,
            max_bin_number=2000,
        )
        self.assertAlmostEqual(result["skew_ps_per_s"], -1000)
        self.assertAlmostEqual(result["offset_ps"], -12_345, delta=100)
        self.assertEqual(result["peak_counts"], len(second))
        self.assertTrue(np.any(result["sweep_changed_bins"][1:] > 0))

    def test_unresolved_sweep_reports_all_tied_candidates(self):
        first = np.array([0, 10_000_000_000, 20_000_000_000])
        result = self.run_quietly(
            first, first + 5000, sweep_range_ps_per_s=1, coarse_steps=3,
            init_time_offset_range_ms=0.001,
        )
        np.testing.assert_array_equal(result["sweep_changed_bins"], 0)
        np.testing.assert_array_equal(result["tied_best_skew_ps_per_s"], [-1, 0, 1])
        self.assertEqual(result["skew_ps_per_s"], 0)
        self.assertFalse(result["unique_best_candidate"])

    def test_peak_response_depends_on_bin_width_and_duration(self):
        # 201 true pairs at 60 ns delay over two seconds. +/-10,000 ps/s
        # produces only +/-20 ns of drift, within one 200 ns histogram bin.
        first = np.arange(201, dtype=np.int64) * 10_000_000_000
        second = first + 60_000
        coarse = self.run_quietly(
            first, second, sweep_range_ps_per_s=10000, coarse_steps=3,
            init_time_offset_range_ms=1, time_offset_range_ns=1_000_000,
            max_bin_number=10000,
        )
        np.testing.assert_array_equal(coarse["sweep_peak_counts"], [201, 201, 201])
        np.testing.assert_array_equal(coarse["sweep_changed_bins"], 0)
        self.assertFalse(coarse["unique_best_candidate"])
        self.assertEqual(coarse["skew_ps_per_s"], 0)

        fine = self.run_quietly(
            first, second, sweep_range_ps_per_s=10000, coarse_steps=3,
            init_time_offset_range_ms=0.001, time_offset_range_ns=1000,
            max_bin_number=2000,
        )
        np.testing.assert_array_equal(fine["sweep_peak_counts"], [10, 201, 10])
        self.assertTrue(fine["unique_best_candidate"])
        self.assertEqual(fine["skew_ps_per_s"], 0)

        # With the original 200 ns bins but a 200-second span, wrong skew
        # spreads the signal over bins, reducing (not eliminating) the peak.
        long_first = first * 100
        long = self.run_quietly(
            long_first, long_first + 60_000,
            sweep_range_ps_per_s=10000, coarse_steps=3,
            init_time_offset_range_ms=1, time_offset_range_ns=1_000_000,
            max_bin_number=10000,
        )
        np.testing.assert_array_equal(long["sweep_peak_counts"], [20, 201, 20])

    def test_plot_does_not_rescale_a_weaker_peak_to_full_height(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        first = np.arange(201, dtype=np.int64) * 10_000_000_000
        heights, limits = [], []

        def inspect_frame(_interval):
            if len(plt.gcf().axes) != 4:
                return  # Separate initialization figure.
            axis = plt.gcf().axes[0]
            heights.append(np.max(axis.lines[0].get_ydata()))
            limits.append(axis.get_ylim())

        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(plt, "show"),
                patch.object(plt, "pause", side_effect=inspect_frame),
            ):
                global_freq_alignment_g2(
                    first, first + 60_000,
                    sweep_range_ps_per_s=10000, coarse_steps=3,
                    init_time_offset_range_ms=0.001, time_offset_range_ns=1000,
                    max_bin_number=2000,
                    show_plot=True,
                )
        finally:
            plt.close("all")
        np.testing.assert_array_equal(heights, [10, 201, 10])
        self.assertTrue(all(lower == 0 for lower, _ in limits))
        self.assertEqual(limits[1], limits[2])

    def test_initial_plot_contains_uncropped_histogram_and_peak(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        first = np.arange(20, dtype=np.int64) * 10_000_000_000
        frames = []

        def capture_frame(_interval):
            figure = plt.gcf()
            frames.append((
                id(figure), len(figure.axes),
                figure.axes[0].lines[0].get_ydata().copy(),
                figure.axes[0].lines[1].get_xdata(),
            ))

        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(plt, "show") as show,
                patch.object(plt, "pause", side_effect=capture_frame),
            ):
                result = global_freq_alignment_g2(
                    first, first + 60_000, coarse_steps=3, show_plot=True,
                    max_bin_number=10000,
                )
            self.assertEqual(len(frames), 4)
            self.assertEqual(frames[0][1], 1)
            self.assertTrue(all(frame[1] == 4 for frame in frames[1:]))
            self.assertNotEqual(frames[0][0], frames[1][0])
            np.testing.assert_array_equal(frames[0][2], result["init_g2_histogram"])
            np.testing.assert_allclose(frames[0][3], result["init_peak_delay_ps"] * 1e-9)
            self.assertEqual(show.call_args_list[0].kwargs, {"block": False})
        finally:
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
