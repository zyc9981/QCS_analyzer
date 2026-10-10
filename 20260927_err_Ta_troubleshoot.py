from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from qcs_analyzer.results import load_dict_npz

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 7))

def plot_result(filename):
    qcs_results = load_dict_npz(input_filename=filename)
    # The NPZ also contains the original analysis figures; only show this comparison.
    for value in qcs_results.values():
        if isinstance(value, plt.Figure):
            plt.close(value)
    tau_diff = qcs_results["tau_diff"]
    tau_diff = tau_diff - np.mean(tau_diff)
    tau_rt = qcs_results["tau_rt"]
    tau_rt = tau_rt - np.mean(tau_rt)
    tau_ref = qcs_results["tau_ref"]
    tau_ref = tau_ref - np.mean(tau_ref)
    t = qcs_results["center"] * 1e-12  # Saved timestamps are in ps.
    tau_ref = tau_ref - (tau_ref[-1] - tau_ref[0]) / (t[-1] - t[0]) * t
    tau_ref = tau_ref - np.mean(tau_ref)
    tau = qcs_results["tau"]
    tau = tau - np.mean(tau)
    label = Path(filename).stem
    s1 = np.std(tau_diff, ddof=1)
    s2 = np.std(tau_rt, ddof=1)
    axes[0].plot(t, tau_diff, label=f"{label}: std = {s1:.3f} ps")
    axes[0].set_ylabel(r"$\tau_{diff} - \langle\tau_{diff}\rangle$ (ps)")
    axes[0].grid(True)
    axes[1].plot(t, tau_rt, label=f"{label}: std = {s2:.3f} ps")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$\tau_{rt} - \langle\tau_{rt}\rangle$ (ps)")
    axes[1].grid(True)
    axes[2].plot(t, tau_ref, label=f"{label}: std = {np.std(tau_ref, ddof=1):.3f} ps")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel(r"$\tau_{ref} - \langle\tau_{ref}\rangle$ (ps)")
    axes[2].grid(True)
    axes[3].plot(t, tau, label=f"{label}: std = {np.std(tau, ddof=1):.3f} ps")
    axes[3].set_xlabel("Time (s)")
    # axes[3].set_ylabel(r"$\tau_{ref} - \langle\tau_{ref}\rangle$ (ps)")
    axes[3].grid(True)
    
# files = [
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\011_50MHz_1200s_truncated_560s\troubleshooting\1s_both.npz",
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\011_50MHz_1200s_truncated_560s\troubleshooting\2s_both.npz",
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\011_50MHz_1200s_truncated_560s\troubleshooting\3s_both.npz",
# ]
# files = [
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\011_50MHz_1200s_truncated_560s\troubleshooting\1s_both.npz",
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\011_50MHz_1200s_truncated_560s\troubleshooting\5s_both_binfixed.npz",
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\011_50MHz_1200s_truncated_560s\troubleshooting\10s_both_binfixed.npz",
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\011_50MHz_1200s_truncated_560s\troubleshooting\20s_both_binfixed.npz",
# ]
# files = [
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\011_50MHz_1200s_truncated_560s\troubleshooting\20s_binfixed_global.npz",
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\011_50MHz_1200s_truncated_560s\troubleshooting\20s_binfixed_pertrigger.npz",
# ]
files = [
    r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\023_50MHz_900s_1clock_sep\3p0.npz",
    r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\023_50MHz_900s_1clock_sep\5p0.npz",
    r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\023_50MHz_900s_1clock_sep\7p5.npz",
]
# files = [
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\027_50MHz_900s_2clock_sep\3p0.npz",
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\027_50MHz_900s_2clock_sep\5p0.npz",
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\027_50MHz_900s_2clock_sep\7p5.npz",
# ]
# files = [
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\027_50MHz_900s_2clock_sep\0p2.npz",
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\027_50MHz_900s_2clock_sep\0p5.npz",
#     r"C:\Yichi Zhang\QCS\20260924_Tx1_RB4_Characterization\027_50MHz_900s_2clock_sep\1p0.npz",
# ]

for file in files:
    plot_result(file)

for ax in axes:
    ax.legend()
fig.tight_layout()
plt.show()
