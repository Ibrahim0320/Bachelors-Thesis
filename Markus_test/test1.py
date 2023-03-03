import numpy as np
import lmfit
import corner
import matplotlib.pyplot as plt

BR_sm_gg = 0.0022700
BR_sm_ZZ = 0.0266700
BR_sm_bb = 0.5792000
BR_sm_WW = 0.2170000

#sönderfall till bb, ZZ och WW

mu_ww = [2.1, 1]
unc_ww = [1.4, 0.5]

mu_zz = [2.1,0.8]
unc_zz = [1.4, 0.46]

mu_bb = [1.5,0.7]
unc_bb = [1.1,1.9]

mu_gg = [1.6, 2.69]
unc_gg = [2.7, 2.51]

def residue(params):
    k_i = params['k_i']
    k_f = params['k_f'] #Fermioner
    k_b = params['k_b'] #Bosoner
    k_g = params['k_g'] #gamma gamma
    BR_inv = params['BR_inv']

    model_f = (k_f ** 2 * k_i ** 2 * (1 - BR_inv)) / ( k_f ** 2 * BR_sm_bb + k_b ** 2 * ( BR_sm_WW + BR_sm_ZZ) + k_g ** 2 *BR_sm_gg)
    model_b = (k_b ** 2 * k_i ** 2 * (1 - BR_inv)) / ( k_f ** 2 * BR_sm_bb + k_b ** 2 * ( BR_sm_WW + BR_sm_ZZ) + k_g ** 2 *BR_sm_gg)
    model_g = (k_g ** 2 * k_i ** 2 * (1 - BR_inv)) / ( k_f ** 2 * BR_sm_bb + k_b ** 2 * ( BR_sm_WW + BR_sm_ZZ) + k_g ** 2 *BR_sm_gg)

    res_ww = (mu_ww - model_b)/unc_ww
    res_zz = (mu_zz - model_b)/unc_zz
    res_bb = (mu_bb - model_f)/unc_bb
    res_gg = (mu_gg - model_g)/unc_gg

    return np.hstack((res_ww, res_zz, res_bb, res_gg))

#Skapa parametrar och initiella värden
par = lmfit.Parameters()
par.add('k_i', value = 1, min = -5, max = 5) #Produktionskappa
par.add('k_f', value = 1, min = -5, max = 5) #Ett av kappana, typ för fermioner
par.add('k_b', value = 1, min = -5, max = 5) #Ett av kappana, typ för bosoner
par.add('k_g', value = 1, expr = '1.59 * k_b ** 2 + 0.07 * k_i ** 2 - 0.66 * k_i * k_b') # Gamma gamma
par.add('BR_inv', 0, min = 0, max = 1) #Den ska vara med, har inte stelkoll på varför

##ttH production, decay to bb and WW

#Antaganden i datan
#kappa för alla bosoner samma, alla fermioner har samma kapppa också

#Minimizer objekt
out = lmfit.Minimizer(residue, par)
result = out.minimize( method = 'nelder')

# write error report
lmfit.report_fit(result)

# Covariansmatris
#print(result.covar)

#Confidence intervals

# ci = lmfit.conf_interval(out, result)
# lmfit.printfuncs.report_ci(ci)

#Corner plot
plot_grej = lmfit.minimize(residue, method='emcee', nan_policy='omit',float_behavior = 'chi2', burn=100, steps=2000, thin=50, params=result.params, is_weighted=True, progress=True)
emcee_plot = corner.corner(plot_grej.flatchain, labels=plot_grej.var_names,levels = (0.69,))#, truths=list(plot_grej.params.valuesdict().values()))

plt.show()
