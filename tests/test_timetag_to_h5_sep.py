"""Small synthetic tests for the separate-stream timetag converter."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from qcs_analyzer import load_timetag_sep, timetag_to_h5_sep


class _FakeBuffer:
    def __init__(self, timestamps, channels):
        self.timestamps = timestamps
        self.channels = channels

    def getTimestamps(self):
        return self.timestamps

    def getChannels(self):
        return self.channels


class _FakeReader:
    def __init__(self, filename):
        self.timestamps = np.array([90, 100, 120, 130], dtype=np.int64)
        self.channels = np.array([1, -3, 1, 3], dtype=np.int64)
        self.index = 0

    def hasData(self):
        return self.index < len(self.timestamps)

    def getData(self, count):
        stop = min(self.index + count, len(self.timestamps))
        result = _FakeBuffer(
            self.timestamps[self.index:stop], self.channels[self.index:stop]
        )
        self.index = stop
        return result


class TimetagToH5SepTests(unittest.TestCase):
    def test_separate_rebased_streams_and_channels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source = Path(directory) / "recording.ttbin"
            source.touch()
            hh_source = Path(directory) / "recording_HH.h5"
            with h5py.File(hh_source, "w") as h5:
                h5["t_ps"] = np.array([50, 60, 80], dtype=np.int64)
                h5["ch"] = np.array([0, 2, 0], dtype=np.uint8)

            with patch(
                "qcs_analyzer._implementation.TimeTagger",
                SimpleNamespace(FileReader=_FakeReader),
            ):
                output = timetag_to_h5_sep(
                    filename=source,
                    chunk_size=2,
                    compression=None,
                    progress_every=0,
                )

            self.assertEqual(Path(output), Path(directory) / "recording_sep.h5")
            with h5py.File(output, "r") as h5:
                self.assertEqual(set(h5.keys()), {"t1_ps", "ch1", "t2_ps", "ch2"})
                np.testing.assert_array_equal(h5["t1_ps"][:], [0, 20, 30])
                np.testing.assert_array_equal(h5["ch1"][:], [3, 1, 3])
                np.testing.assert_array_equal(h5["t2_ps"][:], [0, 20])
                np.testing.assert_array_equal(h5["ch2"][:], [7, 5])
                self.assertEqual(h5.attrs["time_tagger_x_negative_timestamp_events_removed"], 1)
                self.assertEqual(h5.attrs["hydraharp_negative_timestamp_events_removed"], 1)

            t1_ps, ch1, t2_ps, ch2, loaded_filename = load_timetag_sep(output)
            np.testing.assert_array_equal(t1_ps, [0, 20, 30])
            np.testing.assert_array_equal(ch1, [3, 1, 3])
            np.testing.assert_array_equal(t2_ps, [0, 20])
            np.testing.assert_array_equal(ch2, [7, 5])
            self.assertEqual(loaded_filename, output)

    def test_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source = Path(directory) / "recording.ttbin"
            source.touch()
            hh_source = Path(directory) / "recording_HH.h5"
            with h5py.File(hh_source, "w") as h5:
                h5["t_ps"] = np.array([60], dtype=np.int64)
                h5["ch"] = np.array([2], dtype=np.uint8)
            output = Path(directory) / "recording_sep.h5"
            output.touch()

            with self.assertRaises(FileExistsError):
                timetag_to_h5_sep(filename=source)

    def test_loader_rejects_missing_or_mismatched_datasets(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            source = Path(directory) / "incomplete.h5"
            with h5py.File(source, "w") as h5:
                h5["t1_ps"] = np.array([0, 1], dtype=np.int64)
                h5["ch1"] = np.array([3], dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "missing dataset"):
                load_timetag_sep(source)

            with h5py.File(source, "a") as h5:
                h5["t2_ps"] = np.array([0], dtype=np.int64)
                h5["ch2"] = np.array([7], dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "t1_ps and ch1"):
                load_timetag_sep(source)


if __name__ == "__main__":
    unittest.main()
