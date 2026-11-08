"""Per-channel thinning across input formats and chunk boundaries."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from scipy.io import savemat

from qcs_analyzer import load_timetag, timetag_to_h5_divider


class TimetagDividerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.t = 2**60 + np.arange(19, dtype=np.int64) * 10
        self.ch = np.array([1, 3, 1, 3, 3, 2, 1, 2, 3, 1, 2, 3, 1, 3, 2, 3, 1, 2, 3], dtype=np.uint8)

    def write_h5(self, name="input.h5", times=None, channels=None):
        source = self.folder / name
        with h5py.File(source, "w") as h5:
            h5["t_ps"] = self.t if times is None else times
            h5["ch"] = self.ch if channels is None else channels
        return source

    def check_output(self, output, divider):
        keep = np.sort(np.concatenate([
            np.flatnonzero(self.ch == channel)[::divider]
            for channel in np.unique(self.ch)
        ]))
        t, ch = load_timetag(output)
        np.testing.assert_array_equal(t, self.t[keep])
        np.testing.assert_array_equal(ch, self.ch[keep])
        self.assertEqual(t.dtype, self.t.dtype)
        self.assertEqual(ch.dtype, self.ch.dtype)
        with h5py.File(output, "r") as h5:
            self.assertEqual(h5.attrs["channel_record_divider"], divider)
            self.assertEqual(h5.attrs["saved_records"], len(keep))
            self.assertEqual(h5.attrs["input_records"], len(self.t))

    def test_channel_counts_survive_chunk_boundaries(self):
        source = self.write_h5()
        for chunk_size in (1, 2, 5, 100):
            with self.subTest(chunk_size=chunk_size):
                output = timetag_to_h5_divider(
                    source, divider=3, chunk_size=chunk_size,
                    output_filename=self.folder / f"out_{chunk_size}.h5",
                    progress_every=0,
                )
                self.check_output(output, 3)
        # The source still contains every original tag.
        np.testing.assert_array_equal(load_timetag(source)[0], self.t)

    def test_npz_and_mat_input(self):
        for suffix in ("npz", "mat"):
            with self.subTest(suffix=suffix):
                source = self.folder / ("input." + suffix)
                arrays = {"t_ps": self.t, "ch": self.ch}
                if suffix == "npz":
                    np.savez(source, **arrays)
                else:
                    savemat(source, arrays)
                output = timetag_to_h5_divider(
                    source, divider=3, chunk_size=4, progress_every=0,
                    output_filename=self.folder / (suffix + ".h5"),
                )
                self.check_output(output, 3)

    def test_ttbin_uses_sequence_header_and_normalizes_edges(self):
        source = self.folder / "raw.ttbin"
        block = self.folder / "raw.1.ttbin"
        source.touch()
        block.touch()
        timestamps = self.t
        channels = self.ch.astype(np.int64)
        channels[channels == 3] = -3

        class Reader:
            def __init__(self, filename):
                self.index = 0
                self.filename = filename

            def hasData(self):
                return self.index < len(timestamps)

            def getData(self, count):
                start = self.index
                self.index = min(start + count, len(timestamps))
                return SimpleNamespace(
                    getTimestamps=lambda: timestamps[start:self.index],
                    getChannels=lambda: channels[start:self.index],
                )

        with patch("qcs_analyzer._implementation.TimeTagger", SimpleNamespace(FileReader=Reader)):
            output = timetag_to_h5_divider(block, divider=3, chunk_size=4, progress_every=0)
        self.assertEqual(Path(output).name, "raw_div3.h5")
        self.check_output(output, 3)

    def test_divider_one_and_divider_larger_than_record(self):
        source = self.write_h5()
        for divider in (1, 100):
            output = timetag_to_h5_divider(source, divider=divider, progress_every=0)
            self.check_output(output, divider)

    def test_internet_bin_divides_sync_and_inputs_across_overflows(self):
        def word(channel, time, special=False):
            return (int(special) << 31) | (channel << 25) | time

        source = self.folder / "hh.internet.bin"
        np.array([
            word(0, 10, True), word(0, 11), word(1, 12),
            word(2, 13, True),  # Marker: must not count as an input tag.
            word(0, 14, True), word(0, 15), word(63, 2, True),
            word(1, 1), word(0, 2, True), word(0, 3),
            word(63, 0, True),  # Legacy single-overflow encoding.
            word(1, 4), word(0, 5, True), word(0, 6), word(1, 7),
            word(0, 8, True),
        ], dtype="<u4").tofile(source)
        wrap = 1 << 25
        expected_times = np.array([
            10, 11, 12, 2 * wrap + 2, 2 * wrap + 3,
            3 * wrap + 4, 3 * wrap + 8,
        ], dtype=np.int64)
        for resolution in (1, 2):
            for chunk_size in (1, 2, 5, 100):
                with self.subTest(resolution=resolution, chunk_size=chunk_size):
                    output = timetag_to_h5_divider(
                        source, divider=2, chunk_size=chunk_size,
                        hydraharp_base_resolution_ps=resolution,
                        output_filename=self.folder / f"hh_{resolution}_{chunk_size}.h5",
                        progress_every=0,
                    )
                    times, channels = load_timetag(output)
                    np.testing.assert_array_equal(times, expected_times * resolution)
                    np.testing.assert_array_equal(channels, [0, 1, 2, 0, 1, 2, 0])
                    self.assertEqual(times.dtype, np.dtype("int64"))
                    self.assertEqual(channels.dtype, np.dtype("uint8"))
                    with h5py.File(output, "r") as h5:
                        self.assertEqual(h5.attrs["input_records"], 13)
                        self.assertEqual(h5.attrs["saved_records"], 7)
                        self.assertEqual(h5.attrs["hydraharp_base_resolution_ps"], resolution)

    def test_internet_bin_dialog_default_resolution_and_divider_one(self):
        source = self.folder / "hh.INTERNET.BIN"
        np.array([(1 << 31) | 10, 20, (1 << 25) | 30], dtype="<u4").tofile(source)
        with patch("qcs_analyzer._implementation._open_file_dialog", return_value=str(source)) as dialog:
            output = timetag_to_h5_divider(divider=1, progress_every=0)
        self.assertIn("*.internet.bin", dialog.call_args.args[1][0][1])
        times, channels = load_timetag(output)
        np.testing.assert_array_equal(times, [10, 20, 30])
        np.testing.assert_array_equal(channels, [0, 1, 2])
        self.assertEqual(Path(output).name, "hh.INTERNET_div1.h5")

    def test_internet_bin_rejects_invalid_resolution_and_truncated_records(self):
        source = self.folder / "hh.internet.bin"
        np.array([10], dtype="<u4").tofile(source)
        for resolution in (0, -1, 1.5, float("nan")):
            with self.subTest(resolution=resolution):
                with self.assertRaisesRegex(ValueError, "positive integer in ps"):
                    timetag_to_h5_divider(
                        source, hydraharp_base_resolution_ps=resolution,
                        output_filename=self.folder / f"invalid_{resolution}.h5",
                    )
        source.write_bytes(b"\0\0\0")
        with self.assertRaisesRegex(ValueError, "incomplete 32-bit"):
            timetag_to_h5_divider(source)

    def test_invalid_parameters_and_existing_outputs(self):
        for name in ("divider", "chunk_size"):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        timetag_to_h5_divider(**{name: value})
        source = self.write_h5()
        with self.assertRaises(FileExistsError):
            timetag_to_h5_divider(source, output_filename=source)
        output = self.folder / "existing.h5"
        output.write_bytes(b"unchanged")
        with self.assertRaises(FileExistsError):
            timetag_to_h5_divider(source, output_filename=output)
        self.assertEqual(output.read_bytes(), b"unchanged")

    def test_sorting_keeps_timestamp_channel_pairs(self):
        source = self.write_h5(times=self.t[::-1], channels=self.ch[::-1])
        output = timetag_to_h5_divider(source, divider=1, progress_every=0)
        t, ch = load_timetag(output)
        np.testing.assert_array_equal(t, self.t)
        np.testing.assert_array_equal(ch, self.ch)

    def test_dialog_and_public_import(self):
        from QCS_delay_estimator_v5 import timetag_to_h5_divider as exported
        self.assertIs(exported, timetag_to_h5_divider)
        source = self.write_h5()
        with patch("qcs_analyzer._implementation._open_file_dialog", return_value=str(source)):
            output = timetag_to_h5_divider(divider=3, progress_every=0)
        self.check_output(output, 3)


if __name__ == "__main__":
    unittest.main()
