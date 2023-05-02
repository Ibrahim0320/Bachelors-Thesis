import pandas
import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib

run = "2023-04-15_0055"

var_names = ['k_v','k_b','k_t','k_mu','k_tau','k_gg','k_gamgam', 'BR_inv']
# var_names = ['k_w','k_z','k_b','k_t','k_mu','k_tau','k_gg','k_gamgam']

# SM_truth = [1, 1, 1, 1, 1, 0, 0, 0]
SM_truth = [1, 1, 1, 1, 1, 1, 0, 0]
SM_points = np.array([SM_truth, SM_truth])
labels = [
    r"$\kappa_v$",
    # r"$\kappa_w$",
    # r"$\kappa_z$",
    r"$\kappa_b$",
    r"$\kappa_t$",
    r"$\kappa_{\mu}$",
    r"$\kappa_{\tau}$",
    r"$\kappa_{gg}$",
    r"$\kappa_{\gamma \gamma}$",
    r"$BR_{osynlig}$",
]

flatchain = pandas.read_csv(f"flatchains/flatchain_{run}.csv", sep=",")[var_names]
truths = np.loadtxt(f"truths/truth_{run}.csv", delimiter=",")#[:-1]

# print(flatchain)

matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13) 
matplotlib.rcParams.update({'font.size': 16})

emcee_plot = corner.corner(flatchain, 
                           labels=labels, 
                           levels = (0.68,0.95,), 
                           bins=50, 
                           smooth=True, 
                           truths=truths, 
                           verbose=True, 
                           plot_datapoints=False, 
                        #    quantiles=(0.16, 0.84), 
                        #    show_titles=True
                           )

# corner.overplot_lines(emcee_plot, SM_truth, color = 'C1')
corner.overplot_points(emcee_plot, SM_points, marker = '*', color = 'orangered')

plt.savefig("paramind_with_BRinv_2.svg")
plt.show()
