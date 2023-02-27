import numpy as np
import lmfit
import corner
import matplotlib.pyplot as plt

#BR_sm_gammagamma = 0.0022700
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

def residue(params):
    k_i = params['k_i']
    k_f = params['k_f'] #Fermioner
    k_b = params['k_b'] #Bosoner
    BR_inv = params['BR_inv']

    model_f =  (k_f ** 2 * k_i ** 2 * (1 - BR_inv)) / ( k_f ** 2 * BR_sm_bb + k_b ** 2 * BR_sm_WW + k_b ** 2 * BR_sm_ZZ)
    model_b =  (k_b ** 2 * k_i ** 2 * (1 - BR_inv)) / ( k_f ** 2 * BR_sm_bb + k_b ** 2 * BR_sm_WW + k_b ** 2 * BR_sm_ZZ)
    
    # model = [model_f, model_f, model_b, model_b, model_b]
    # model = 2 * model_b + model_f
    # return (data - model)/uncertainty

    res_ww = (model_b - mu_ww)/unc_ww
    res_zz = (model_b - mu_zz)/unc_zz
    res_bb = (model_f - mu_bb)/unc_bb

    return np.hstack((res_ww, res_zz, res_bb))

#Skapa parametrar och initiella värden
par = lmfit.Parameters()
par.add('k_i',1, min = -5, max = 5) #Produktionskappa
par.add('k_f',1, min = -5, max = 5) #Ett av kappana, typ för fermioner
par.add('k_b',1, min = -5, max = 5) #Ett av kappana, typ för bosoner
par.add('BR_inv', 0, min = 0, max = 1) #Den ska vara med, har inte stelkoll på varför

##ttH production, decay to bb and WW

#Antaganden i datan
#kappa för alla bosoner samma, alla fermioner har samma kapppa också

#Nu test för datan för ttH -> bb, ttH -> WW, ttH -> ZZ
#Fermiondata
# data_F = [1.5, 0.7]
# uncertainty_F = [1.1, 1.9]

# #Bosondata
# data_B = [2.1, 2.1, 0.8]
# uncertainty_B = [1.4, 1.4, 0.46]

# data = data_F + data_B
# uncertainty = uncertainty_F + uncertainty_B

#Minimizer objekt
out = lmfit.Minimizer(residue, par)

result = out.minimize( method = 'leastsq')

# write error report
lmfit.report_fit(result)

# Covariansmatris
print(result.covar)

#Confidence intervals

# ci = lmfit.conf_interval(out, result)
# lmfit.printfuncs.report_ci(ci)

#Corner plot
plot_grej = lmfit.minimize(residue, method='emcee', nan_policy='omit', burn=100, steps=2000, thin=30, params=result.params, float_behavior='chi2', is_weighted=True, progress=True)
emcee_plot = corner.corner(plot_grej.flatchain, labels=plot_grej.var_names, truths=list(result.params.valuesdict().values()))#, levels=(0.68,))

plt.show()
