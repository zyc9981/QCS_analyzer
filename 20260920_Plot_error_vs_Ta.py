import numpy as np
import matplotlib.pyplot as plt

Ta_s = np.array([1.0, 2.0, 3.0,
                 5.0, 10.0, 15.0, 20.0, 25.0,
                 30.0, 35.0, 45.0])
precision_new = np.array([18.935109, 12.757044, 10.76794,
                          8.865574, 6.3068432, 5.036025, 4.5559711,
                          3.6048580, 3.3821567, 3.253177, 2.505751])
err_precision_new = np.array([0.387806, 0.3707440, 0.384569,
                              0.411573, 0.421393, 0.41966, 0.4467497, 0.3637162,
                              0.450607, 0.460253, 0.406486])

Ta_s2 = np.array([1.0, 2.0, 3.0,
                  5.0, 10.0, 15.0, 20.0, 25.0,
                  30.0, 35.0, 45.0])
precision_old = np.array([20.566837, 14.354077, 11.730545, 
                          9.870023, 14.61087, 29.00362, 40.819357, 53.622285, 
                          99.265476, 180.222718, 605.35455])
err_precision_old = np.array([0.37650, 0.3726133, 0.373956,
                              0.40842, 0.866995, 2.138175, 3.526255, 5.25809,
                              10.83075, 21.54073, 83.947572])

fig = plt.figure()
plt.errorbar(
    Ta_s,
    precision_new,
    yerr=err_precision_new,
    fmt="o-",
    capsize=3,
    color="tab:blue",     # marker edge and line
    # markerfacecolor="white",
    markeredgecolor="tab:blue",
    ecolor="tab:blue",     # error bars
    zorder=10,
    barsabove=True,  # draw error bars above the markers/line
)
plt.errorbar(
    Ta_s2,
    precision_old,
    yerr=err_precision_old,
    fmt="o-",
    capsize=3,
    color="tab:red",     # marker edge and line
    # markerfacecolor="white",
    markeredgecolor="tab:red",
    ecolor="tab:red",     # error bars
    zorder=10,
    barsabove=True,  # draw error bars above the markers/line
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Acquisition Time $T_a$ (s)", fontsize=14)
plt.ylabel("Precision (ps)", fontsize=14)
plt.legend(["New Rb Clocks", "Old OCXO Clocks"], fontsize=12)
plt.xticks([1, 10, 60], ["1", "10", "60"], fontsize=12)
plt.yticks([1, 10, 100], ["1", "10", "100"], fontsize=12)
plt.ylim(1, 100)
plt.grid(True, which="both")
plt.show()
