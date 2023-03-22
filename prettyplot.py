import pandas
import numpy as np
import matplotlib.pyplot as plt
import corner

run = "flatchain_2023-03-18_1503"

SM_truth = [1, 1, 1, 1, 1, 0, 0, 0]
var_names = ['k_v','k_b','k_t','k_mu','k_tau','k_gg','k_gamgam', 'BR_inv']

SM_points = np.array([SM_truth, SM_truth])

ndim = len(var_names)

flatchain = pandas.read_csv(f"flatchains/{run}.csv", sep=",")[var_names]
truths = np.loadtxt(f"truths/truth_2023-03-18_1503.csv", delimiter=",")

print(flatchain)

emcee_plot = corner.corner(flatchain, labels=var_names, levels = (0.69,0.95,), bins=50, smooth=True, truths=truths, verbose=True, plot_datapoints=False, quantiles=(0.16, 0.84))

corner.overplot_lines(emcee_plot, SM_truth, color = 'C1')
corner.overplot_points(emcee_plot, SM_points, marker = 's', color = 'orangered')

plt.show()