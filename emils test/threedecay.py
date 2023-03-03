import numpy as np
import lmfit
import corner
import matplotlib.pyplot as plt

BR_sm_ZZ = 0.0266700
BR_sm_bb = 0.5792000
BR_sm_WW = 0.2170000

mu_ww = [2.1, 1]
unc_ww = [1.4, 0.5]

mu_zz = [2.1,0.8]
unc_zz = [1.4, 0.46]

mu_bb = [1.5,0.7]
unc_bb = [1.1,1.9]

def residue(params):
    k_f = params['k_f']
    k_v = params['k_v']
    BR_inv = params['BR_inv']

    # BR_inv = 0

    mu_model_ww = (k_f**2 * k_v**2 * (1-BR_inv))/((BR_sm_WW + BR_sm_ZZ)*k_v**2 + BR_sm_bb*k_f**2)
    mu_model_zz = (k_f**2 * k_v**2 * (1-BR_inv))/((BR_sm_WW + BR_sm_ZZ)*k_v**2 + BR_sm_bb*k_f**2)
    mu_model_bb = (k_f**2 * k_f**2 * (1-BR_inv))/((BR_sm_WW + BR_sm_ZZ)*k_v**2 + BR_sm_bb*k_f**2)
    
    res_ww = (mu_model_ww - mu_ww)/unc_ww
    res_zz = (mu_model_zz - mu_zz)/unc_zz
    res_bb = (mu_model_bb - mu_bb)/unc_bb

    return np.hstack((res_ww, res_zz, res_bb))

# Skapa parametrar och initiala värden
par = lmfit.Parameters()
par.add('k_f',1, min = -5, max = 5)
par.add('k_v',1, min = -5, max = 5)
par.add('BR_inv', 0, min = 0, max = 1)

out = lmfit.minimize(residue, par,method = 'nelder')

# write error report
lmfit.report_fit(out)

# Corner plot
print("Sampling the posterior...")
bay = lmfit.minimize(residue, method='emcee',float_behavior = 'chi2', burn=300, steps=2000, thin=30, params=out.params, is_weighted=True, progress=True)
emcee_plot = corner.corner(bay.flatchain, labels=bay.var_names,truths=list(out.params.valuesdict().values()), levels = (0.69,))
plt.show()
