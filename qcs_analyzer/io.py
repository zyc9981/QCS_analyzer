"""Reading, converting, and truncating timetag files."""

from ._implementation import (
    load_timetag,
    timetag_to_h5,
    timetag_to_h5_qutag,
    timetag_to_h5_fast,
    trancate_timetag,
    truncate_timetag,
)

__all__ = [
    "load_timetag",
    "timetag_to_h5",
    "timetag_to_h5_qutag",
    "timetag_to_h5_fast",
    "trancate_timetag",
    "truncate_timetag",
]
