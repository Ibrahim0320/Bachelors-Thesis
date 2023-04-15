import pandas
import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib

run = "2023-04-15_0709"

var_names = ['tanB','cosBA','k_gamgam']

SM_truth = [1, 1, 1, 1, 1, 0, 0, 0]
SM_points = np.array([SM_truth, SM_truth])
labels = [
    r"$\log(\tan(\beta))$",
    r"$\cos(\beta-\alpha)$",
    r"$\kappa_{\gamma \gamma}$"
]

flatchain = pandas.read_csv(f"flatchains/flatchain_{run}.csv", sep=",")[var_names]
truths = np.loadtxt(f"truths/truth_{run}.csv", delimiter=",")[0:2]

matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13) 
matplotlib.rcParams.update({'font.size': 16})

emcee_plot = corner.corner(flatchain[flatchain.columns[::-1]], 
                           labels=labels[::-1], 
                           levels = (0.69,0.95,0.99), 
                           bins=30, 
                           smooth=True, 
                           verbose=True, 
                           plot_datapoints=True, 
                        #    range=[(-0.1,0.1), (-2,2)]
                        #    quantiles=(0.16, 0.84), 
                        #    show_titles=True
                           )

# corner.overplot_lines(emcee_plot, SM_truth, color = 'C1')
# corner.overplot_points(emcee_plot, SM_points, marker = '*', color = 'orangered')

plt.show()