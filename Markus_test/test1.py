import numpy as np
import lmfit
import corner
import matplotlib.pyplot as plt

#BR_sm_gammagamma = 0.0022700
#BR_sm-ZZ = 0.0266700
BR_sm_bb = 0.5792000
BR_sm_WW = 0.2170000

#sönderfall till bb och WW

def residue(params, data, uncertainty):
    k_i = params['k_i']
    k_f = params['k_f']
    k_v = params['k_v']
    BR_inv = params['BR_inv']

    model =  (k_f *k_f * k_i * k_i * (1 - BR_inv)) / ( k_f * k_f * BR_sm_bb + k_v * k_v * BR_sm_WW )
    
    return (data - model)/uncertainty

#Skapa parametrar och initiella värden
par = lmfit.Parameters()
par.add('k_i',1, min = -5, max = 5)
par.add('k_f',1, min = -5, max = 5)
par.add('k_v',1, min = -5, max = 5,vary = False)
par.add('BR_inv', 0, min = 0, max = 1)

##ttH production, decay to bb and WW

data = [1.5, 0.7]
uncertainty = [1.1, 1.9]

#Gamma gamma data
#Produktionskanaler: ggH, VBF, VH, ttH

# data = [1.32 , 0.8 , 1 , 1.6]
# uncertainty = [0.38 , 0.7 , 1.6 , 2.7]

# data2 = [1.12, 1.58, -0.16, 2.69]
# uncertainty2 = [0.37, 0.77, 1.16, 2.51]

out = lmfit.minimize(residue, par, args=( data, uncertainty),method = 'leastsq')

# write error report
lmfit.report_fit(out)

#Corner plot
plot_grej = lmfit.minimize(residue, args=(data, uncertainty), method='emcee', nan_policy='omit', burn=0, steps=1000, thin=10, params=out.params, is_weighted=True, progress=True)
emcee_plot = corner.corner(plot_grej.flatchain, labels=plot_grej.var_names, levels = (0.69,), truths=list(plot_grej.params.valuesdict().values()))

plt.show()
