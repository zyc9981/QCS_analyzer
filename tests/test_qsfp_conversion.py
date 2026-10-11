import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from qcs_analyzer import _implementation as impl


def word(channel, time, special=False):
    return (int(special) << 31) | (channel << 25) | time


class QsfpConversionTests(unittest.TestCase):
    def test_overflows_sync_and_channel_mapping_across_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hh.bin"
            np.array([
                word(0, 10), word(63, 2, True), word(0, 3, True),
                word(1, 4), word(63, 0, True), word(2, 5, True), word(0, 6),
            ], dtype="<u4").tofile(path)
            for chunk_size in (1, 2, 3, 100):
                with self.subTest(chunk_size=chunk_size):
                    chunks = list(impl._iter_hydraharp_qsfp_chunks(
                        path, chunk_size, channel_offset=5, base_resolution_ps=2,
                    ))
                    np.testing.assert_array_equal(
                        np.concatenate([t for t, _ in chunks]),
                        np.array([10, 2 * (1 << 25) + 3, 2 * (1 << 25) + 4,
                                  3 * (1 << 25) + 6]) * 2,
                    )
                    np.testing.assert_array_equal(
                        np.concatenate([ch for _, ch in chunks]), [6, 5, 7, 6],
                    )

    def test_separate_conversion_alignment_and_filename_spellings(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "recording.ttbin"
            source.touch()
            for suffix in ("_qsfp.bin", ".qsfp.bin"):
                with self.subTest(suffix=suffix):
                    hh_path = source.with_name(source.stem + suffix)
                    np.array([word(0, 90), word(1, 100), word(0, 110)], dtype="<u4").tofile(hh_path)
                    output = Path(directory) / (suffix + ".h5")

                    def tt_chunks(*args, **kwargs):
                        yield np.array([40, 50, 60]), np.array([1, 3, 1])

                    with patch.object(impl, "_iter_timetagger_ttbin_chunks", tt_chunks):
                        impl.timetag_to_h5_sep(source, output, chunk_size=1, progress_every=0)
                    with h5py.File(output) as h5:
                        self.assertEqual(set(h5), {"t1_ps", "ch1", "t2_ps", "ch2"})
                        np.testing.assert_array_equal(h5["t1_ps"][:], [0, 10])
                        np.testing.assert_array_equal(h5["ch1"][:], [3, 1])
                        np.testing.assert_array_equal(h5["t2_ps"][:], [0, 10])
                        np.testing.assert_array_equal(h5["ch2"][:], [7, 6])
                        self.assertEqual(h5["t2_ps"].dtype, np.dtype("int64"))
                        self.assertEqual(h5["ch2"].dtype, np.dtype("uint8"))
                        self.assertEqual(h5.attrs["hydraharp_source_filename"], str(hh_path))
                        self.assertEqual(h5.attrs["hydraharp_negative_timestamp_events_removed"], 1)
                    hh_path.unlink()

    def test_truncated_record_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hh.bin"
            path.write_bytes(b"\0\0\0")
            with self.assertRaisesRegex(ValueError, "incomplete 32-bit"):
                list(impl._iter_hydraharp_qsfp_chunks(path, 10))


if __name__ == "__main__":
    unittest.main()
