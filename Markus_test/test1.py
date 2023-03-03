import numpy as np
import lmfit
import corner
import matplotlib.pyplot as plt

BR_sm_gg = 0.0022700
BR_sm_ZZ = 0.0266700
BR_sm_bb = 0.5792000
BR_sm_WW = 0.2170000
BR_sm_tt = 0.0624000

mu_ww = [2.1, 1] # WW
unc_ww = [1.4, 1]

mu_zz = [2.1,0.8] ## ZZ
unc_zz = [1.4, 0.46]

mu_bb = [1.5,0.7] # b b bar
unc_bb = [1.1,1.9]

mu_gg = [1.6, 2.69] # gamma gamma
unc_gg = [2.7, 2.51]

mu_tt = [2.1, -1.3] # tau tau
unc_tt = [1.4, -6.3]

def residue(params):
    k_i = params['k_i'] #produktion, tth
    k_f = params['k_f'] #Fermioner, bb
    k_b = params['k_b'] #Bosoner, zz, ww
    k_g = params['k_g'] #gamma gamma
    k_t = params['k_t'] #tau tau
    BR_inv = params['BR_inv']

    sum_over_f =  ( k_f ** 2 * BR_sm_bb + k_b ** 2 * ( BR_sm_WW + BR_sm_ZZ) + k_g ** 2 *BR_sm_gg + k_t ** 2 * BR_sm_tt)

    model_f = (k_f ** 2 * k_i ** 2 * (1 - BR_inv)) / sum_over_f
    model_b = (k_b ** 2 * k_i ** 2 * (1 - BR_inv)) / sum_over_f
    model_g = (k_g ** 2 * k_i ** 2 * (1 - BR_inv)) / sum_over_f
    model_t = (k_t ** 2 * k_i ** 2 * (1 - BR_inv)) / sum_over_f

    res_ww = (mu_ww - model_b)/unc_ww
    res_zz = (mu_zz - model_b)/unc_zz
    res_bb = (mu_bb - model_f)/unc_bb
    res_gg = (mu_gg - model_g)/unc_gg
    res_tt = (mu_tt - model_t)/unc_tt

    return np.hstack((res_ww, res_zz, res_bb, res_gg, res_tt))

#Skapa parametrar och initiella värden
par = lmfit.Parameters()
par.add('k_i', value = 1, min = -3, max = 3) #Produktionskappa
par.add('k_f', value = 1, min = -3, max = 3) #Ett av kappana, typ för fermioner
par.add('k_b', value = 1, min = -3, max = 3) #Ett av kappana, typ för bosoner
par.add('k_g', value = 1, expr = '1.59 * k_b ** 2 + 0.07 * k_i ** 2 - 0.66 * k_i * k_b') # Gamma gamma
par.add('k_t', value = 1, min = -3, max = 3) #tau tau
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

#print(result.params.values())

plot_grej = lmfit.minimize(residue, params=result.params, method='emcee', nan_policy='omit', burn=100, steps=10000, thin=50, float_behavior='chi2', is_weighted=True, progress=True)
emcee_plot = corner.corner(plot_grej.flatchain, labels=result.var_names, levels=(0.69,))#, truths=list(plot_grej.params.values()))

plt.show()


#print((result.params.valuesdict().values()))