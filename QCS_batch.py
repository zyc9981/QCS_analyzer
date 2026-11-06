"""Batch conversion and count-rate reporting for Time Tagger recordings.

Run ``python QCS_batch.py`` and select multiple H5/HDF5/NPZ/MAT or .ttbin
files. Data files are loaded and their total count rates are printed.
Raw .ttbin recordings are converted to H5. Numbered blocks are read
automatically by FileReader, once per sequence. Existing H5 outputs are skipped.

Run ``python QCS_batch.py --rates`` to select multiple H5/NPZ/MAT files
and print each file's total count rate, using N / (last - first) * 1e12.
"""

import argparse
import sys
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np

from qcs_analyzer.config import get_default_file_dialog_directory
from qcs_analyzer.io import load_timetag, timetag_to_h5_TTXonly


def convert_files(filenames):
    """Convert each sequence once; return 1 if any recording fails, else 0."""
    recordings = []
    seen = set()
    for filename in filenames:
        source = Path(filename)
        parts = source.stem.rsplit(".", 1)
        if len(parts) == 2 and parts[1].isdigit():
            # measurement.1.ttbin belongs to measurement.ttbin.
            header = source.with_name(f"{parts[0]}{source.suffix}")
            if header.is_file():
                source = header
        key = source.resolve()
        if key not in seen:
            seen.add(key)
            recordings.append(source)

    converted = skipped = failed = 0
    for index, source in enumerate(recordings, start=1):
        output = source.with_suffix(".h5")
        print(f"\n[{index}/{len(recordings)}] {source.name}")
        if output.exists():
            print(f"Skipping existing output: {output.name}")
            skipped += 1
            continue
        try:
            timetag_to_h5_TTXonly(filename=source, output_filename=output)
        except Exception as exc:
            print(f"FAILED: {source.name}: {exc}")
            if output.exists():
                print(f"Incomplete output may remain at: {output}")
            failed += 1
        else:
            converted += 1

    print(f"\nFinished: {converted} converted, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


def print_file_rates(filenames):
    """Print total events per second across all channels in each selected file."""
    processed = failed = 0
    for filename in filenames:
        print(f"\nFile: {filename}")
        try:
            t1_ps, ch1, filename = load_timetag(filename, return_filename=True)
            if np.size(t1_ps) < 2:
                raise ValueError("Need at least two timestamps to calculate a rate.")
            # Convert scalars before subtraction to avoid integer overflow.
            duration_ps = t1_ps[-1].item() - t1_ps[0].item()
            if not np.isfinite(duration_ps) or duration_ps <= 0:
                raise ValueError("Last timestamp must be later than the first timestamp.")
            rate_hz = np.size(t1_ps) / duration_ps * 1e12
            print(f"Count rate: {rate_hz:.12g} counts/s")
        except Exception as exc:
            print(f"FAILED: {filename}: {exc}")
            failed += 1
        else:
            processed += 1

    print(f"\nFinished: {processed} processed, {failed} failed.")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rates",
        action="store_true",
        help="Select multiple H5/NPZ/MAT files and print their total count rates.",
    )
    args = parser.parse_args(argv)

    root = Tk()
    root.withdraw()
    try:
        if args.rates:
            filenames = filedialog.askopenfilenames(
                parent=root,
                title="Select timetag files to calculate count rates",
                initialdir=str(get_default_file_dialog_directory()),
                filetypes=[
                    ("Timetag arrays", "*.h5 *.hdf5 *.npz *.mat"),
                    ("HDF5 files", "*.h5 *.hdf5"),
                    ("NPZ files", "*.npz"),
                    ("MAT files", "*.mat"),
                    ("All files", "*.*"),
                ],
            )
        else:
            filenames = filedialog.askopenfilenames(
                parent=root,
                title="Select timetag files to process",
                initialdir=str(get_default_file_dialog_directory()),
                filetypes=[
                    ("Timetag files", "*.h5 *.hdf5 *.npz *.mat *.ttbin"),
                    ("HDF5 files", "*.h5 *.hdf5"),
                    ("NPZ files", "*.npz"),
                    ("MAT files", "*.mat"),
                    ("Time Tagger files", "*.ttbin"),
                    ("All files", "*.*"),
                ],
            )
    finally:
        root.destroy()

    if not filenames:
        print("No files selected. Exiting.")
        return 0
    if args.rates:
        return print_file_rates(filenames)
    recordings = [name for name in filenames if Path(name).suffix.lower() == ".ttbin"]
    data_files = [name for name in filenames if Path(name).suffix.lower() != ".ttbin"]
    conversion_status = convert_files(recordings) if recordings else 0
    rate_status = print_file_rates(data_files) if data_files else 0
    return max(conversion_status, rate_status)


if __name__ == "__main__":
    sys.exit(main())
