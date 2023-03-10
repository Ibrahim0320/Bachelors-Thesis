import pandas
import numpy as np
import matplotlib.pyplot as plt
import corner

run = "2023-03-08 17:56:35.578213"

var_names = ['k_w', 'k_z','k_b','k_t','k_mu','k_tau','k_gg','k_gamgam']
flatchain = pandas.read_csv(f"flatchains/flatchain_{run}.csv", sep=",", names=var_names)
truths = pandas.read_csv(f"truths/truth_2023-03-08 17:56:36.130561.csv", names=var_names)

emcee_plot = corner.corner(flatchain, labels=var_names, levels = (0.69,0.95),bins=50,smooth=True,truths=list(truths))

plt.show()