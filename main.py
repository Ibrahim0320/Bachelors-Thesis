import numpy as np
import lmfit
import corner
import matplotlib.pyplot as plt
import data
from data import hd,br
import os
import datetime
import pandas

def residue(params):
    res_ggF = residue_ggF(params)
    res_VBF = residue_VBF(params)
    res_ttH = residue_ttH(params)
    res_VH = residue_VH(params)

    return np.hstack((res_ggF, res_ttH, res_VBF, res_VH))

def residue_ggF(params):
    k_w = params['k_w']
    k_z = params['k_z']
    k_t = params['k_t']
    k_b = params['k_b']
    k_mu = params['k_mu']
    k_tau = params['k_tau']
    k_gg = params['k_gg']
    k_gamgam = params['k_gamgam']
    #k_zgam = params['k_zgam']
    BR_inv = params['BR_inv']
    sum_over_f = (k_w**2 * br['WW']
                  + k_z**2 * br['ZZ'] 
                  + k_b**2 *br['bb'] 
                  + k_t**2 * br['cc'] 
                  + k_mu**2 * br['mumu'] 
                  + k_tau**2 * br['tt'] 
                  + (1.58*k_w**2 - 0.67*k_t*k_w + 0.07*k_t**2 + 0.01*k_b*k_w+0.16*k_t*k_gamgam - 0.76*k_w*k_gamgam + 0.09*k_gamgam**2) * br['gamgam']
                  + (0.01 *k_b**2 - 0.16*k_b*k_gg + 1.93*k_gg**2 - 0.12*k_t*k_b + 2.93*k_gg*k_t + 1.11*k_t**2 ) * br['gg']
    )
    
    # k_ggF_7 = 1.06 * k_t**2 + 0.01 * k_b**2 - 0.07 * k_t * k_b
    # k_ggF_13 = 1.04*k_t**2 + 0.002*k_b**2 - 0.04*k_t*k_b
    k_gg = 0.01 *k_b**2 - 0.16*k_b*k_gg + 1.93*k_gg**2 - 0.12*k_t*k_b + 2.93*k_gg*k_t + 1.11*k_t**2
    k_ggF_13 = k_gg
    k_ggF_7 = k_gg

    mu_model_WW_78 = (k_z**2 * k_ggF_7 * (1 - BR_inv)) / sum_over_f
    mu_model_ZZ_78 = (k_w**2 * k_ggF_7 * (1 - BR_inv)) / sum_over_f
    mu_model_bb_78 = (k_b**2 * k_ggF_7 * (1 - BR_inv)) / sum_over_f 
    mu_model_mumu_78 = (k_mu**2 * k_ggF_7 * (1 - BR_inv)) / sum_over_f
    mu_model_tau_78 = (k_tau**2 * k_ggF_7 * (1 - BR_inv)) / sum_over_f
    mu_model_gamgam_78 = ((1.58*k_w**2 - 0.67*k_t*k_w + 0.07*k_t**2 + 0.01*k_b*k_w+0.16*k_t*k_gamgam - 0.76*k_w*k_gamgam + 0.09*k_gamgam**2) * k_ggF_7 * (1 - BR_inv)) / sum_over_f
    mu_model_gg_78 = (k_gg * k_ggF_7 * (1 - BR_inv)) / sum_over_f

    mu_model_WW_13 = (k_z**2 * k_ggF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_ZZ_13 = (k_w**2 * k_ggF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_bb_13 = (k_b**2 * k_ggF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_mumu_13 = (k_mu**2 * k_ggF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_tau_13 = (k_tau**2 * k_ggF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_gamgam_13 = ((1.58*k_w**2 - 0.67*k_t*k_w + 0.07*k_t**2 + 0.01*k_b*k_w+0.16*k_t*k_gamgam - 0.76*k_w*k_gamgam + 0.09*k_gamgam**2) * k_ggF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_gg_13 = (k_gg * k_ggF_13 * (1 - BR_inv)) / sum_over_f

    res_WW_78 = (hd('mu','ggF','WW','78') - mu_model_WW_78)/hd('unc','ggF','WW','78')
    res_ZZ_78 = (hd('mu','ggF','ZZ','78') - mu_model_ZZ_78)/hd('unc','ggF','ZZ','78')
    res_bb_78 = (hd('mu','ggF','bb','78') - mu_model_bb_78)/hd('unc','ggF','bb','78')
    res_mumu_78 = (hd('mu','ggF','mumu','78') - mu_model_mumu_78)/hd('unc','ggF','mumu','78')
    res_tau_78 = (hd('mu','ggF','tt','78') - mu_model_tau_78)/hd('unc','ggF','tt','78')
    res_gamgam_78 = (hd('mu','ggF','gamgam','78') - mu_model_gamgam_78)/hd('unc','ggF','gamgam','78')
    res_gg_78 = (hd('mu','ggF','gg','78') - mu_model_gg_78)/hd('unc','ggF','gg','78')

    res_WW_13 = (hd('mu','ggF','WW','13') - mu_model_WW_13)/hd('unc','ggF','WW','13')
    res_ZZ_13 = (hd('mu','ggF','ZZ','13') - mu_model_ZZ_13)/hd('unc','ggF','ZZ','13')
    res_bb_13 = (hd('mu','ggF','bb','13') - mu_model_bb_13)/hd('unc','ggF','bb','13')
    res_mumu_13 = (hd('mu','ggF','mumu','13') - mu_model_mumu_13)/hd('unc','ggF','mumu','13')
    res_tau_13 = (hd('mu','ggF','tt','13') - mu_model_tau_13)/hd('unc','ggF','tt','13')
    res_gamgam_13 = (hd('mu','ggF','gamgam','13') - mu_model_gamgam_13)/hd('unc','ggF','gamgam','13')
    res_gg_13 = (hd('mu','ggF','gg','13') - mu_model_gg_13)/hd('unc','ggF','gg','13')
    # res_gg = (hd('mu','ggF','gg') - mu_model_gg)/hd('unc','ggF','gg')
    
    return np.hstack((res_WW_78, res_ZZ_78, res_bb_78, res_mumu_78, res_tau_78, res_gamgam_78,res_WW_13, res_ZZ_13, res_bb_13, res_mumu_13, res_tau_13, res_gamgam_13, res_gg_13, res_gg_78))

def residue_VBF(params):
    k_w = params['k_w']
    k_z = params['k_z']
    k_t = params['k_t']
    k_b = params['k_b']
    k_mu = params['k_mu']
    k_tau = params['k_tau']
    k_gg = params['k_gg']
    k_gamgam = params['k_gamgam']
    #k_zgam = params['k_zgam']
    BR_inv = params['BR_inv']

    k_VBF_78 = 0.74*k_w**2 + 0.74*k_z**2
    k_VBF_13 = 0.73*k_w**2 + 0.27*k_z**2
    k_gg = 0.01 *k_b**2 - 0.16*k_b*k_gg + 1.93*k_gg**2 - 0.12*k_t*k_b + 2.93*k_gg*k_t + 1.11*k_t**2

    sum_over_f = (k_w**2 * br['WW']
                  + k_z**2 * br['ZZ'] 
                  + k_b**2 *br['bb'] 
                  + k_t**2 * br['cc'] 
                  + k_mu**2 * br['mumu'] 
                  + k_tau**2 * br['tt'] 
                  + (1.58*k_w**2 - 0.67*k_t*k_w + 0.07*k_t**2 + 0.01*k_b*k_w+0.16*k_t*k_gamgam - 0.76*k_w*k_gamgam + 0.09*k_gamgam**2) * br['gamgam']
                  + k_gg * br['gg']
    )
    
    mu_model_WW_78 = (k_z**2 * k_VBF_78 * (1 - BR_inv)) / sum_over_f
    mu_model_ZZ_78 = (k_w**2 * k_VBF_78 * (1 - BR_inv)) / sum_over_f
    mu_model_bb_78 = (k_b**2 * k_VBF_78 * (1 - BR_inv)) / sum_over_f 
    mu_model_mumu_78 = (k_mu**2 * k_VBF_78 * (1 - BR_inv)) / sum_over_f
    mu_model_tau_78 = (k_tau**2 * k_VBF_78 * (1 - BR_inv)) / sum_over_f
    mu_model_gamgam_78 = ((1.58*k_w**2 - 0.67*k_t*k_w + 0.07*k_t**2 + 0.01*k_b*k_w+0.16*k_t*k_gamgam - 0.76*k_w*k_gamgam + 0.09*k_gamgam**2) * k_VBF_78 * (1 - BR_inv)) / sum_over_f
    mu_model_gg_78 = (k_gg * k_VBF_78 * (1 - BR_inv)) / sum_over_f

    mu_model_WW_13 = (k_z**2 * k_VBF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_ZZ_13 = (k_w**2 * k_VBF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_bb_13 = (k_b**2 * k_VBF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_mumu_13 = (k_mu**2 * k_VBF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_tau_13 = (k_tau**2 * k_VBF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_gamgam_13 = ((1.58*k_w**2 - 0.67*k_t*k_w + 0.07*k_t**2 + 0.01*k_b*k_w+0.16*k_t*k_gamgam - 0.76*k_w*k_gamgam + 0.09*k_gamgam**2) * k_VBF_13 * (1 - BR_inv)) / sum_over_f
    mu_model_gg_13 = (k_gg * k_VBF_13 * (1 - BR_inv)) / sum_over_f

    res_WW_78 = (hd('mu','VBF','WW','78') - mu_model_WW_78)/hd('unc','VBF','WW','78')
    res_ZZ_78 = (hd('mu','VBF','ZZ','78') - mu_model_ZZ_78)/hd('unc','VBF','ZZ','78')
    res_bb_78 = (hd('mu','VBF','bb','78') - mu_model_bb_78)/hd('unc','VBF','bb','78')  
    res_mumu_78 = (hd('mu','VBF','mumu','78') - mu_model_mumu_78)/hd('unc','VBF','mumu','78')
    res_tau_78 = (hd('mu','VBF','tt','78') - mu_model_tau_78)/hd('unc','VBF','tt','78')
    res_gamgam_78 = (hd('mu','VBF','gamgam','78') - mu_model_gamgam_78)/hd('unc','VBF','gamgam','78')
    res_gg_78 = (hd('mu','VBF','gg','78') - mu_model_gg_78)/hd('unc','VBF','gg','78')
    
    res_WW_13 = (hd('mu','VBF','WW','13') - mu_model_WW_13)/hd('unc','VBF','WW','13')
    res_ZZ_13 = (hd('mu','VBF','ZZ','13') - mu_model_ZZ_13)/hd('unc','VBF','ZZ','13')
    res_bb_13 = (hd('mu','VBF','bb','13') - mu_model_bb_13)/hd('unc','VBF','bb','13')
    res_mumu_13 = (hd('mu','VBF','mumu','13') - mu_model_mumu_13)/hd('unc','VBF','mumu','13')
    res_tau_13 = (hd('mu','VBF','tt','13') - mu_model_tau_13)/hd('unc','VBF','tt','13')
    res_gamgam_13 = (hd('mu','VBF','gamgam','13') - mu_model_gamgam_13)/hd('unc','VBF','gamgam','13')
    res_gg_13 = (hd('mu','VBF','gg','13') - mu_model_gg_13)/hd('unc','VBF','gg','13')

    return np.hstack((res_WW_78, res_ZZ_78, res_bb_78, res_mumu_78, res_tau_78, res_gamgam_78, res_WW_13, res_ZZ_13, res_bb_13, res_mumu_13, res_tau_13, res_gamgam_13,res_gg_78, res_gg_13))

def residue_ttH(params):
    k_w = params['k_w']
    k_z = params['k_z']
    k_t = params['k_t']
    k_b = params['k_b']
    k_mu = params['k_mu']
    k_tau = params['k_tau']
    k_gg = params['k_gg']
    k_gamgam = params['k_gamgam']
    #k_zgam = params['k_zgam']
    BR_inv = params['BR_inv']

    k_gg = 0.01 *k_b**2 - 0.16*k_b*k_gg + 1.93*k_gg**2 - 0.12*k_t*k_b + 2.93*k_gg*k_t + 1.11*k_t**2

    sum_over_f = (k_w**2 * br['WW']
                  + k_z**2 * br['ZZ'] 
                  + k_b**2 *br['bb'] 
                  + k_t**2 * br['cc'] 
                  + k_mu**2 * br['mumu'] 
                  + k_tau**2 * br['tt'] 
                  + (1.58*k_w**2 - 0.67*k_t*k_w + 0.07*k_t**2 + 0.01*k_b*k_w+0.16*k_t*k_gamgam - 0.76*k_w*k_gamgam + 0.09*k_gamgam**2) * br['gamgam']
                  + k_gg * br['gg']
    )
    
    mu_model_WW = (k_z**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_ZZ = (k_w**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_bb = (k_b**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_mumu = (k_mu**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_tau = (k_tau**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_gamgam = ((1.58*k_w**2 - 0.67*k_t*k_w + 0.07*k_t**2 + 0.01*k_b*k_w+0.16*k_t*k_gamgam - 0.76*k_w*k_gamgam + 0.09*k_gamgam**2) * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_gg = (k_gg * k_t**2 * (1 - BR_inv)) / sum_over_f
    

    res_WW = (hd('mu','ttH','WW') - mu_model_WW)/hd('unc','ttH','WW')
    res_ZZ = (hd('mu','ttH','ZZ') - mu_model_ZZ)/hd('unc','ttH','ZZ')
    res_bb = (hd('mu','ttH','bb') - mu_model_bb)/hd('unc','ttH','bb')
    res_mumu = (hd('mu','ttH','mumu') - mu_model_mumu)/hd('unc','ttH','mumu')
    res_tau = (hd('mu','ttH','tt') - mu_model_tau)/hd('unc','ttH','tt')
    res_gamgam = (hd('mu','ttH','gamgam') - mu_model_gamgam)/hd('unc','ttH','gamgam')
    res_gg = (hd('mu','ttH','gg') - mu_model_gg)/hd('unc','ttH','gg')
    

    return np.hstack((res_WW, res_ZZ, res_bb, res_mumu, res_tau, res_gamgam, res_gg))

def residue_VH(params):
    k_w = params['k_w']
    k_z = params['k_z']
    k_t = params['k_t']
    k_b = params['k_b']
    k_mu = params['k_mu']
    k_tau = params['k_tau']
    k_gg = params['k_gg']
    k_gamgam = params['k_gamgam']
    #k_zgam = params['k_zgam']
    BR_inv = params['BR_inv']

    k_VH = 0.5*k_z**2 + 0.5*k_w**2
    k_gg = 0.01 *k_b**2 - 0.16*k_b*k_gg + 1.93*k_gg**2 - 0.12*k_t*k_b + 2.93*k_gg*k_t + 1.11*k_t**2

    sum_over_f = (k_w**2 * br['WW']
                  + k_z**2 * br['ZZ'] 
                  + k_b**2 *br['bb'] 
                  + k_t**2 * br['cc'] 
                  + k_mu**2 * br['mumu'] 
                  + k_tau**2 * br['tt'] 
                  + (1.58*k_w**2 - 0.67*k_t*k_w + 0.07*k_t**2 + 0.01*k_b*k_w+0.16*k_t*k_gamgam - 0.76*k_w*k_gamgam + 0.09*k_gamgam**2) * br['gamgam']
                  + k_gg * br['gg']
    )
    
    mu_model_WW = (k_z**2 * k_VH * (1 - BR_inv)) / sum_over_f
    mu_model_ZZ = (k_w**2 * k_VH * (1 - BR_inv)) / sum_over_f
    mu_model_bb = (k_b**2 * k_VH * (1 - BR_inv)) / sum_over_f
    mu_model_mumu = (k_mu**2 *k_VH * (1 - BR_inv)) / sum_over_f
    mu_model_tau = (k_tau**2 * k_VH * (1 - BR_inv)) / sum_over_f
    mu_model_gamgam = ((1.58*k_w**2 - 0.67*k_t*k_w + 0.07*k_t**2 + 0.01*k_b*k_w+0.16*k_t*k_gamgam - 0.76*k_w*k_gamgam + 0.09*k_gamgam**2) * k_VH * (1 - BR_inv)) / sum_over_f
    mu_model_gg = (k_gg * k_VH * (1 - BR_inv)) / sum_over_f
    # mu_model_gamgam = (k_gamgam**2 * k_VH * (1 - BR_inv)) / sum_over_f

    res_WW = (hd('mu','VH','WW') - mu_model_WW)/hd('unc','VH','WW')
    res_ZZ = (hd('mu','VH','ZZ') - mu_model_ZZ)/hd('unc','VH','ZZ')
    res_bb = (hd('mu','VH','bb') - mu_model_bb)/hd('unc','VH','bb')
    res_mumu = (hd('mu','VH','mumu') - mu_model_mumu)/hd('unc','VH','mumu')
    res_tau = (hd('mu','VH','tt') - mu_model_tau)/hd('unc','VH','tt')
    res_gamgam = (hd('mu','VH','gamgam') - mu_model_gamgam)/hd('unc','VH','gamgam')
    res_gg = (hd('mu','VH','gg') - mu_model_gg)/hd('unc','VH','gg')

    return np.hstack((res_WW, res_ZZ, res_bb, res_mumu, res_tau, res_gamgam,res_gg))

# Skapa parametrar
par = lmfit.Parameters()
par.add('k_w', value = 1, min = -5, max = 5)
par.add('k_z', value = 1, min = -5, max = 5)
par.add('k_b', value = 1, min = -5, max = 5)
par.add('k_t', value = 1, min = -5, max = 5)
# par.add('k_c', value = 1, min = -5, max = 5) # är lika med k_t
par.add('k_mu', value = 1, min = -5, max = 5)
par.add('k_tau', value = 1, min = -5, max = 5)
par.add('k_gg', value = 0, min = -5, max = 5)
par.add('k_gamgam', value = 0, min = -5, max = 5)
#par.add('k_zgam', value = 1, min = -5, max = 5)
par.add('BR_inv', value = 0, min = 0, max = 0.5, vary=True)

print('Finding best fit parameters...')
# MinimizerResult objekt
out = lmfit.minimize(residue, par, method = 'nelder', nan_policy= 'omit')

# write error report
lmfit.report_fit(out)

print(f"PID: {os.getpid()}")
print("Sampling the posterior...")
bay = lmfit.minimize(residue, method='emcee',float_behavior = 'chi2', burn=300, steps=1000, thin=30, params=out.params, is_weighted=True, progress=True)
print("Sampling done. Saving...")
bay.flatchain.to_csv(f'flatchains/flatchain_{datetime.datetime.now()}.csv', sep=',')
np.savetxt(f'truths/truth_{datetime.datetime.now()}.csv', list(out.params.valuesdict().values()),delimiter=',')

emcee_plot = corner.corner(bay.flatchain, labels=bay.var_names, levels = (0.69,),truths=list(out.params.valuesdict().values())) # med br_inv
# emcee_plot = corner.corner(bay.flatchain, labels=bay.var_names, levels = (0.69,),truths=list(out.params.valuesdict().values())[:-1]) # utan br_inv
plt.savefig(f"plots/corner_{datetime.datetime.now()}.svg")
print("I'm done here. Goodbye!")
# plt.show()