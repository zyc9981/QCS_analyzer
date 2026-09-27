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

Ta_s2 = np.array([0.01, 0.02, 0.05,
                  0.1, 0.2, 0.3, 0.4, 0.5,
                  0.6, 0.8, 1.0, 1.5, 2.0,
                  2.5, 3.0, 4.0, 5.0, 7.5,
                  10.0, 15.0, 20.0, 25.0, 30.0])
precision_old = np.array([12.194381, 8.759905, 5.777463,
                          4.269382, 3.238775, 2.763113, 2.490517, 2.282383,
                          2.174026, 2.003242, 1.831896, 1.643503, 1.465657,
                          1.422460, 1.389913, 1.367392, 1.341850, 1.299078,
                          1.211310, 1.457483, 0.818532, 1.110367, 1.449767])
err_precision_old = np.array([0.039360, 0.039989, 0.041710,
                              0.043606, 0.046816, 0.048953, 0.050986, 0.052279,
                              0.054590, 0.058169, 0.059560, 0.065687, 0.067895,
                              0.073950, 0.079456, 0.090958, 0.100576, 0.121670,
                              0.133767, 0.206119, 0.140377, 0.226653, 0.341713])

Ta_s3 = np.array([0.1, 0.2, 0.5, 1.0, 2.0,
                  5.0, 7.5, 10.0, 20.0, 30.0])
precision_old = np.array([0.459878, 0.534531, 0.477826, 0.420865, 0.453808,
                          0.482945, 0.422144, 0.512695, 0.564034, 0.422905])
err_precision_old = np.array([0.004697, 0.007727, 0.010945, 0.013683, 0.021022,
                              0.036198, 0.039537, 0.056618, 0.096731, 0.099680])

fig = plt.figure()
plt.fill_between(
    Ta_s,
    precision_new - err_precision_new,
    precision_new + err_precision_new,
    color="tab:blue",
    alpha=0.2,
    linewidth=0,
)
plt.plot(
    Ta_s,
    precision_new,
    "o-",
    color="tab:blue",
    label="Old Source",
    zorder=3,
)
plt.fill_between(
    Ta_s2,
    precision_old - err_precision_old,
    precision_old + err_precision_old,
    color="tab:red",
    alpha=0.2,
    linewidth=0,
)
plt.plot(
    Ta_s2,
    precision_old,
    "o-",
    color="tab:red",
    label="New Source",
    zorder=3,
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Acquisition Time $T_a$ (s)", fontsize=14)
plt.ylabel("Precision (ps)", fontsize=14)
plt.legend(fontsize=12)
plt.xticks([0.01, 0.1, 1, 10, 60], ["0.01", "0.1", "1", "10", "60"], fontsize=12)
plt.yticks([0.5, 1, 5, 10, 50], ["0.5", "1", "5", "10", "50"], fontsize=12)
plt.ylim(0.5, 50)
plt.grid(True, which="both")
plt.show()
