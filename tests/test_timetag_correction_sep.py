"""Parity checks for separate-stream clock correction."""

import unittest

import numpy as np

from qcs_analyzer import timetag_correction, timetag_correction_sep


class TimetagCorrectionSepTests(unittest.TestCase):
    def setUp(self):
        self.t1_ps = np.array([0, 100, 200, 400, 500, 600, 800, 900, 1100, 1250])
        self.ch1 = np.array([1, 3, 1, 1, 3, 1, 1, 3, 1, 1])
        self.t2_ps = np.array([0, 150, 300, 560, 700, 990, 1200])
        self.ch2 = np.array([5, 7, 5, 7, 5, 7, 5])

    def test_modes_match_existing_combined_correction(self):
        combined_times = np.concatenate((self.t1_ps, self.t2_ps))
        combined_channels = np.concatenate((self.ch1, self.ch2))
        for mode in ("first", "per_trigger", "linear_fit", "global"):
            with self.subTest(mode=mode):
                old_times, old_channels, old_details = timetag_correction(
                    combined_times,
                    combined_channels,
                    ch_qutag=3,
                    correction_mode=mode,
                    return_parameters=True,
                )
                t1, ch1, t2, ch2, details = timetag_correction_sep(
                    self.t1_ps,
                    self.ch1,
                    self.t2_ps,
                    self.ch2,
                    correction_mode=mode,
                    return_parameters=True,
                )
                first_stream = old_channels <= 4
                np.testing.assert_allclose(t1, old_times[first_stream])
                np.testing.assert_array_equal(ch1, old_channels[first_stream])
                np.testing.assert_allclose(t2, old_times[~first_stream])
                np.testing.assert_array_equal(ch2, old_channels[~first_stream])
                self.assertEqual(
                    details["events_removed_outside_shared_interval"],
                    old_details["events_removed_outside_shared_interval"],
                )
                self.assertEqual(details["shared_duration_ps"], old_details["shared_duration_ps"])

        np.testing.assert_array_equal(
            self.t1_ps, [0, 100, 200, 400, 500, 600, 800, 900, 1100, 1250]
        )
        np.testing.assert_array_equal(self.t2_ps, [0, 150, 300, 560, 700, 990, 1200])

    def test_mismatched_stream_lengths_raise(self):
        with self.assertRaisesRegex(ValueError, "match its channel array"):
            timetag_correction_sep(self.t1_ps, self.ch1[:-1], self.t2_ps, self.ch2)


if __name__ == "__main__":
    unittest.main()
