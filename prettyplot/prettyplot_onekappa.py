import pandas
import numpy as np
import matplotlib.pyplot as plt
import corner

run = "2023-04-14_1416"

# var_names = ['k_v','k_b','k_t','k_mu','k_tau','k_gg','k_gamgam', 'BR_inv']
# var_names = ['k_w','k_z','k_b','k_t','k_mu','k_tau','k_gg','k_gamgam']
var_names = ['k']

SM_truth = [1]
# SM_truth = [1, 1, 1, 1, 1, 1, 0, 0]
SM_points = np.array([SM_truth, SM_truth])
labels = [
    # r"$\kappa$",
    r"$\cos(\theta)$",
    # r"$1-BR_{inv}$"
    # r"$\kappa_w$",
    # r"$\kappa_z$",
    # r"$\kappa_f$",
]

flatchain = pandas.read_csv(f"../flatchains/flatchain_{run}.csv", sep=",")[var_names]
truths = np.loadtxt(f"../truths/truth_{run}.csv", delimiter=",")

# print(flatchain)

emcee_plot = corner.corner(flatchain, 
                           labels=labels, 
                           levels = (0.68,0.95,), 
                           bins=50, 
                           smooth=True, 
                           truths=[truths[0]], 
                           verbose=True, 
                           plot_datapoints=False, 
                           quantiles=(0.05,), 
                        #    title_quantiles=(0.025,0.5,0.975),
                        #    show_titles=True
                           )
ax = emcee_plot.get_axes()[0]
# corner.overplot_lines(emcee_plot, SM_truth, color = 'C1')
corner.overplot_points(emcee_plot, SM_points, marker = '*', color = 'orangered')

sec = ax.secondary_xaxis('top', functions=(np.square, np.sqrt))
sec.set_xlabel(r"$1-BR_{osynlig}$")

plt.show()
