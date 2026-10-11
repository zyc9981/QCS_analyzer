# QCS Analyzer

The original script name is still supported:

```powershell
& 'C:\ProgramData\anaconda3\python.exe' .\QCS_delay_estimator_v4.py --help
```

Common command-line tasks:

```powershell
# Open a timetag file (or use the file dialog) and print its summary.
& 'C:\ProgramData\anaconda3\python.exe' .\QCS_delay_estimator_v4.py inspect

# Merge a QuTAG file with its matching HydraHarp H5 file.
& 'C:\ProgramData\anaconda3\python.exe' .\QCS_delay_estimator_v4.py convert

# Save only the first 300 seconds of a recording.
& 'C:\ProgramData\anaconda3\python.exe' .\QCS_delay_estimator_v4.py truncate 300
```

All file dialogs start in the configured acquisition folder.  To use another
folder for one command, add `--directory "C:\path\to\folder"` before the
command.

## Python imports

In v5, `timetag_to_h5_sep("recording.ttbin")` reads the matching
`recording_qsfp.bin` (also accepts `recording.qsfp.bin`) as raw HydraHarp V2
T2 data and writes `recording_sep.h5` with separate `t1_ps`/`ch1` and
`t2_ps`/`ch2` streams. It unfolds timestamp overflows and retains photon
and sync events, aligning each stream's first trigger to zero. The raw
HH base resolution defaults to 1 ps; set `hydraharp_base_resolution_ps`
if the recording uses a different integer picosecond resolution.

Use the focused modules in new notebooks and scripts:

```python
from qcs_analyzer.io import load_timetag
from qcs_analyzer.analysis import g2_process
from qcs_analyzer.plotting import g2_plot

t_ps, ch, filename = load_timetag(return_filename=True)
results = g2_process(t_ps, ch, chA=1, chB=6)
g2_plot(results)
```

## Peak fitting

Use one fitting function and select the curve with a parameter dictionary:

```python
from QCS_delay_estimator_v4 import fit_g2

fit = fit_g2(
    results,
    curve="lorentzian",
    parameters={"height": 15, "center": 4.4, "gamma": 0.7},
    fit_window=20,
)
```

Available curves and initial parameters are:

| Curve | Required parameters | Optional parameter |
| --- | --- | --- |
| `"gaussian"` | `height`, `center`, `sigma` | `offset` |
| `"lorentzian"` | `height`, `center`, `gamma` | `offset` |
| `"expo"` / `"exponential"` | `height`, `center`, `gamma` | `offset` |
| `"expogaussconv"` | `height`, `center`, `gamma`, `sigma` | `offset` |

`offset` defaults to the minimum y-value and is always fitted as a free background parameter.
If SciPy cannot converge, the input curve is plotted instead and the returned
dictionary has `success=False` and `used_initial_parameters=True`.

For time-dependent processing, use the same curve name and a dictionary that
omits `center` when each checkpoint should initialize its centre from the
histogram maximum:

```python
time_dependent_results = g2_process_time_dependent(
    t_ps,
    ch,
    t_stop_start=0.1,
    t_stop_stop=3600,
    t_stop_step=0.1,
    fit_curve="expogaussconv",
    fit_parameters={"height": 15, "gamma": 0.7, "sigma": 0.05},
)
```

| Module | Use it for |
| --- | --- |
| `qcs_analyzer.io` | Loading, merging, and truncating timetag files |
| `qcs_analyzer.analysis` | Time correction, g2 processing, and peak fitting |
| `qcs_analyzer.plotting` | Photon-count and g2 figures |
| `qcs_analyzer.results` | Saving and loading result files |
| `qcs_analyzer.config` | Changing the file-dialog default directory |
