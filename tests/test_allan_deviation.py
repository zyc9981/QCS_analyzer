"""Checks against analytic single-channel frequency stability examples."""

import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qcs_analyzer.allan_deviation import allan_deviation_estimator_single_channel


class SingleChannelAllanDeviationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_constant_frequency_offset_has_zero_adev(self):
        t = np.arange(11, dtype=np.int64) * 1_000_100
        result = allan_deviation_estimator_single_channel(
            t, np.full(t.size, 3), nominal_frequency_hz=1e6, show=False
        )
        np.testing.assert_allclose(result["allan_deviation"], 0, atol=1e-15)
        np.testing.assert_allclose(result["x_samples_ps"], np.arange(11) * 100)

    def test_quadratic_phase_matches_analytic_adev(self):
        # x[n] = a*n**2 gives sigma_y(m*T0) = sqrt(2)*a*m/T0.
        n = np.arange(51, dtype=np.int64)
        period_ps, a = 1_000_000, 2
        t = n * period_ps + a * n**2
        result = allan_deviation_estimator_single_channel(
            t, np.full(t.size, 3), nominal_frequency_hz=1e6, show=False
        )
        m = result["tau_s"] / (period_ps * 1e-12)
        np.testing.assert_allclose(
            result["allan_deviation"], np.sqrt(2) * a * m / period_ps
        )

    def test_large_integer_epoch_and_other_channels(self):
        n = np.arange(11, dtype=np.int64)
        t = 2**60 + n * 1_000_000 + n**2
        # Include another channel and deliberately reverse arrival order.
        mixed_t = np.concatenate((t, [t[0] + 50]))[::-1]
        mixed_ch = np.concatenate((np.full(t.size, 3), [1]))[::-1]
        result = allan_deviation_estimator_single_channel(
            mixed_t, mixed_ch, nominal_frequency_hz=1e6, show=False
        )
        np.testing.assert_array_equal(result["x_samples_ps"], n**2)

    def test_estimated_period_and_minimum_record(self):
        t = np.arange(6, dtype=np.int64) * 1_000_000
        with patch("matplotlib.pyplot.show") as show:
            result = allan_deviation_estimator_single_channel(
                t, np.full(t.size, 3), show=False
            )
        show.assert_not_called()
        self.assertEqual(result["nominal_frequency_hz"], 1e6)
        np.testing.assert_allclose(result["tau_s"], [1e-6])
        np.testing.assert_array_equal(result["allan_deviation"], [0])

    def test_invalid_records_raise(self):
        t = np.arange(11, dtype=np.int64) * 1_000_000
        ch = np.full(t.size, 3)
        cases = [
            (t, ch[:-1], {}, "same length"),
            (t, ch + 1, {}, "six pulses"),
            (t[:5], ch[:5], {}, "six pulses"),
            (np.delete(t, 5), np.delete(ch, 5), {}, "missing/extra"),
            (np.zeros(11), ch, {}, "strictly increasing"),
            (t.astype(float) * np.nan, ch, {}, "finite"),
            (t, ch, {"nominal_frequency_hz": 0}, "finite and positive"),
            (t, ch, {"nominal_frequency_hz": np.nan}, "finite and positive"),
            (t, ch, {"number_of_tau_points": 0}, "positive integer"),
            (t, ch, {"number_of_tau_points": 1.5}, "positive integer"),
        ]
        for times, channels, kwargs, message in cases:
            with self.subTest(message=message, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    allan_deviation_estimator_single_channel(
                        times, channels, show=False, **kwargs
                    )


if __name__ == "__main__":
    unittest.main()
