import numpy as np
import matplotlib.pyplot as plt

Ta_s = np.array([0.1 ,0.2, 0.4, 0.6, 0.8,
                 1.0, 1.5, 2.0, 2.5, 3.0,
                 4.0, 5.0, 6.0, 8.0, 10.0,
                 12.5, 15.0])
precision_2clk = np.array([69.128469, 48.096598, 33.965270, 27.659873, 23.623886, 
                          21.442541, 17.827232, 15.254179, 14.256569, 13.223261,
                          14.286570, 16.765430, 21.22995, 32.244583, 47.428873,
                          76.1646, 99.09732])
err_precision_2clk = np.array([0.399219, 0.392917, 0.392616, 0.391797, 0.386602, 
                              0.392537, 0.400233, 0.395979, 0.414323, 0.421542,
                              0.5273269, 0.693758, 0.964998, 1.6994388, 2.814386,
                              5.08896, 7.305552, ])
# Ta_s2 = np.array([0.1 ,0.2, 0.4, 0.6, 0.8,
#                  1.0, 1.5, 2.0, 2.5, 3.0,
#                  4.0, 5.0, 6.0, 8.0, 10.0,
#                  12.5, 15.0, 30.0, 60.0])
# precision_1clk = np.array([70.833306, 48.918244, 34.06119, 27.856931, 24.632258,
#                            21.746689, 17.300906, 15.473335, 13.785974, 13.092975,
#                            11.244644, 9.652487, 8.941176, 7.746954, 7.115275,
#                            6.689815, 6.260270, 4.3321858, 2.377687])
# err_precision_1clk = np.array([0.409065, 0.399629, 0.393725, 0.3945883, 0.4031040,
#                                0.398101, 0.388417, 0.4016678, 0.400647, 0.417389,
#                                0.415047, 0.399423, 0.406417, 0.408300, 0.422214,
#                                0.446982, 0.461513, 0.4726802, 0.40777])
Ta_s2 = np.array([0.1 ,0.2, 0.4, 0.6, 0.8,
                 1.0, 1.5, 2.0, 2.5, 3.0,
                 4.0, 5.0, 6.0, 8.0, 10.0,
                 12.5, 15.0])
precision_1clk = np.array([70.833306, 48.918244, 34.06119, 27.856931, 24.632258,
                           21.746689, 17.300906, 15.473335, 13.785974, 13.092975,
                           11.244644, 9.652487, 8.941176, 7.746954, 7.115275,
                           6.689815, 6.260270])
err_precision_1clk = np.array([0.409065, 0.399629, 0.393725, 0.3945883, 0.4031040,
                               0.398101, 0.388417, 0.4016678, 0.400647, 0.417389,
                               0.415047, 0.399423, 0.406417, 0.408300, 0.422214,
                               0.446982, 0.461513])
Ta_s3 = np.array([1.0, 1.5, 2.0, 2.5, 3.0,
                  4.0, 5.0])
precision_2clk2ch = np.array([19.282753, 16.1042799, 14.0163653, 12.252397, 11.698784,
                              11.858311, 12.875351])
err_precision_2clk2ch = np.array([0.3529959, 0.3615517, 0.3638467, 0.3560781, 0.3729436,
                                  0.4376982, 0.5327858])

fig = plt.figure()
plt.errorbar(
    Ta_s,
    precision_2clk,
    yerr=err_precision_2clk,
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
    precision_1clk,
    yerr=err_precision_1clk,
    fmt="o-",
    capsize=3,
    color="tab:red",     # marker edge and line
    # markerfacecolor="white",
    markeredgecolor="tab:red",
    ecolor="tab:red",     # error bars
    zorder=10,
    barsabove=True,  # draw error bars above the markers/line
)
plt.errorbar(
    Ta_s3,
    precision_2clk2ch,
    yerr=err_precision_2clk2ch,
    fmt="o-",
    capsize=3,
    color="tab:green",     # marker edge and line
    # markerfacecolor="white",
    markeredgecolor="tab:green",
    ecolor="tab:green",     # error bars
    zorder=10,
    barsabove=True,  # draw error bars above the markers/line
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Acquisition Time $T_a$ (s)", fontsize=14)
plt.ylabel("Precision (ps)", fontsize=14)
plt.legend(["2 CLK 1 WDM ch", "1 CLK 1 WDM ch", "2 CLK 2 WDM ch"], fontsize=12)
plt.xticks([0.1, 1, 10], ["0.1", "1", "10"], fontsize=12)
plt.yticks([10, 100], ["10", "100"], fontsize=12)
plt.grid(True, which="both")
plt.show()
