"""Checks for channel-separated g2 coincidence processing."""

import inspect
import unittest

import numpy as np

from qcs_analyzer._implementation import g2_process_sep


class G2ProcessSepTests(unittest.TestCase):
    def setUp(self):
        self.t1_ps = np.array([0, 1_000_000_000_000, 2_000_000_000_000])
        self.t2_ps = self.t1_ps + 6_000
        self.kwargs = {"t_center": 6, "t_span": 2, "time_bin": 1, "sig_half": 0.6}

    def test_histogram_uses_two_timestamp_arrays(self):
        result = g2_process_sep(self.t1_ps, self.t2_ps, **self.kwargs)

        self.assertNotIn("chA", inspect.signature(g2_process_sep).parameters)
        self.assertNotIn("chB", inspect.signature(g2_process_sep).parameters)
        self.assertNotIn("chA", result)
        self.assertNotIn("chB", result)
        np.testing.assert_array_equal(result["histogram"], [0, 0, 3, 0])
        np.testing.assert_allclose(result["centers"], [4.5, 5.5, 6.5, 7.5])
        self.assertEqual(result["cntA_total"], 3)
        self.assertEqual(result["cntB_total"], 3)
        self.assertEqual(result["numPairs"], 3)
        self.assertEqual(result["CC"], 3)
        self.assertEqual(result["single_photon_counts"], [1, 1])

    def test_stop_time_applies_to_both_channels(self):
        result = g2_process_sep(self.t1_ps, self.t2_ps, t_stop=1.5, **self.kwargs)

        np.testing.assert_array_equal(result["histogram"], [0, 0, 2, 0])
        self.assertEqual(result["cntA_total"], 2)
        self.assertEqual(result["cntB_total"], 2)
        self.assertEqual(result["T_acq_s"], 1.5)

    def test_unsorted_channels_are_sorted_independently(self):
        result = g2_process_sep(
            self.t1_ps[::-1], self.t2_ps[::-1], **self.kwargs
        )

        np.testing.assert_array_equal(result["histogram"], [0, 0, 3, 0])


if __name__ == "__main__":
    unittest.main()
