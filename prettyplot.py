import pandas
import numpy as np
import matplotlib.pyplot as plt
import corner

run = "2023-03-08 17:56:35.578213"

var_names = ['k_w', 'k_z','k_b','k_t','k_mu','k_tau','k_gg','k_gamgam']
flatchain = pandas.read_csv(f"flatchains/flatchain_{run}.csv", sep=",", names=var_names)
truths = np.loadtxt(f"truths/truth_2023-03-08 17:56:36.130561.csv",delimiter=",")


emcee_plot = corner.corner(flatchain, labels=var_names, levels = (0.69,0.95,),bins=50,smooth=True,truths=truths[:-1],verbose=True, plot_datapoints=False,quantiles=(0.16, 0.84))



plt.show()