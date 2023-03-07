import numpy as np
import lmfit
import corner
import matplotlib.pyplot as plt
import data
from data import hd,br
import os

def residue(params):
    res_ggF = residue_ggF(params)
    res_VBF = residue_VBF(params)
    res_ttH = residue_ttH(params)
    res_VH = residue_VH(params)

    return np.hstack((res_ggF, res_ttH, res_VBF, res_VH))
    # return res_ttH #np.hstack(res_ttH)

def residue_ggF(params):
    k_w = params['k_w']
    k_z = params['k_z']
    k_t = params['k_t']
    k_b = params['k_b']
    k_mu = params['k_mu']
    k_tau = params['k_tau']
    # k_gg = params['k_gg']
    # k_gamgam = params['k_gamgam']
    #k_zgam = params['k_zgam']
    BR_inv = params['BR_inv']

    sum_over_f = (k_w**2 * br['WW']
                  + k_z**2 * br['ZZ'] 
                  + k_b**2 *br['bb'] 
                  + k_t**2 * br['cc'] 
                  + k_mu**2 * br['mumu'] 
                  + k_tau**2 * br['tt'] 
                #   + k_gg**2 * br['gg'] 
                #   + k_gamgam**2 * br['gamgam']
    )
    
    mu_model_WW = (k_z**2 * (1.06*k_t**2 + 0.01*k_t**2 - 0.07*k_t*k_b) * (1 - BR_inv)) / sum_over_f
    mu_model_ZZ = (k_w**2 * (1.06*k_t**2 + 0.01*k_t**2 - 0.07*k_t*k_b) * (1 - BR_inv)) / sum_over_f
    mu_model_bb = (k_b**2 * (1.06*k_t**2 + 0.01*k_t**2 - 0.07*k_t*k_b) * (1 - BR_inv)) / sum_over_f
    mu_model_cc = (k_t**2 * (1.06*k_t**2 + 0.01*k_t**2 - 0.07*k_t*k_b) * (1 - BR_inv)) / sum_over_f
    mu_model_mumu = (k_mu**2 * (1.06*k_t**2 + 0.01*k_t**2 - 0.07*k_t*k_b) * (1 - BR_inv)) / sum_over_f
    mu_model_tau = (k_tau**2 * (1.06*k_t**2 + 0.01*k_t**2 - 0.07*k_t*k_b) * (1 - BR_inv)) / sum_over_f
    mu_model_gamgam = ((1.59*k_w**2 + 0.07*k_t**2 - 0.66*k_t*k_w) * (1.06*k_t**2 + 0.01*k_t**2 - 0.07*k_t*k_b) * (1 - BR_inv)) / sum_over_f
    # mu_model_gg = (k_gg**2 * (1.06*k_t**2 + 0.01*k_t**2 - 0.07*k_t*k_b) * (1 - BR_inv)) / sum_over_f
    # mu_model_gamgam = (k_gamgam**2 * (1.06*k_t**2 + 0.01*k_t**2 - 0.07*k_t*k_b) * (1 - BR_inv)) / sum_over_f

    res_WW = (hd('mu','ggF','WW') - mu_model_WW)/hd('unc','ggF','WW')
    res_ZZ = (hd('mu','ggF','ZZ') - mu_model_ZZ)/hd('unc','ggF','ZZ')
    res_bb = (hd('mu','ggF','bb') - mu_model_bb)/hd('unc','ggF','bb')
    res_cc = (hd('mu','ggF','cc') - mu_model_cc)/hd('unc','ggF','cc')
    res_mumu = (hd('mu','ggF','mumu') - mu_model_mumu)/hd('unc','ggF','mumu')
    res_tau = (hd('mu','ggF','tt') - mu_model_tau)/hd('unc','ggF','tt')
    res_gamgam = (hd('mu','ggF','gamgam') - mu_model_gamgam)/hd('unc','ggF','gamgam')
    # res_gg = (hd('mu','ggF','gg') - mu_model_gg)/hd('unc','ggF','gg')
    

    return np.hstack((res_WW, res_ZZ, res_bb, res_cc, res_mumu, res_tau, res_gamgam))

def residue_VBF(params):
    k_w = params['k_w']
    k_z = params['k_z']
    k_t = params['k_t']
    k_b = params['k_b']
    k_mu = params['k_mu']
    k_tau = params['k_tau']
    # k_gg = params['k_gg']
    # k_gamgam = params['k_gamgam']
    #k_zgam = params['k_zgam']
    BR_inv = params['BR_inv']

    sum_over_f = (k_w**2 * br['WW']
                  + k_z**2 * br['ZZ'] 
                  + k_b**2 *br['bb'] 
                  + k_t**2 * br['cc'] 
                  + k_mu**2 * br['mumu'] 
                  + k_tau**2 * br['tt'] 
                #   + k_gg**2 * br['gg'] 
                #   + k_gamgam**2 * br['gamgam']
    )
    
    mu_model_WW = (k_z**2 * (0.74*k_w**2 + 0.74*k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_ZZ = (k_w**2 * (0.74*k_w**2 + 0.74*k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_bb = (k_b**2 * (0.74*k_w**2 + 0.74*k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_cc = (k_t**2 * (0.74*k_w**2 + 0.74*k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_mumu = (k_mu**2 * (0.74*k_w**2 + 0.74*k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_tau = (k_tau**2 * (0.74*k_w**2 + 0.74*k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_gamgam = ((1.59*k_w**2 + 0.07*k_t**2 - 0.66*k_t*k_w) * (0.74*k_w**2 + 0.74*k_z**2) * (1 - BR_inv)) / sum_over_f
    # mu_model_gg = (k_gg**2 * (0.74*k_w**2 + 0.74*k_z**2) * (1 - BR_inv)) / sum_over_f
    # mu_model_gamgam = (k_gamgam**2 * (0.74*k_w**2 + 0.74*k_z**2) * (1 - BR_inv)) / sum_over_f

    res_WW = (hd('mu','VBF','WW') - mu_model_WW)/hd('unc','VBF','WW')
    res_ZZ = (hd('mu','VBF','ZZ') - mu_model_ZZ)/hd('unc','VBF','ZZ')
    res_bb = (hd('mu','VBF','bb') - mu_model_bb)/hd('unc','VBF','bb')
    res_cc = (hd('mu','VBF','cc') - mu_model_cc)/hd('unc','VBF','cc')
    res_mumu = (hd('mu','VBF','mumu') - mu_model_mumu)/hd('unc','VBF','mumu')
    res_tau = (hd('mu','VBF','tt') - mu_model_tau)/hd('unc','VBF','tt')
    res_gamgam = (hd('mu','VBF','gamgam') - mu_model_gamgam)/hd('unc','VBF','gamgam')
    # res_gg = (hd('mu','VBF','gg') - mu_model_gg)/hd('unc','VBF','gg')
    

    return np.hstack((res_WW, res_ZZ, res_bb, res_cc, res_mumu, res_tau, res_gamgam))

def residue_ttH(params):
    k_w = params['k_w']
    k_z = params['k_z']
    k_t = params['k_t']
    k_b = params['k_b']
    k_mu = params['k_mu']
    k_tau = params['k_tau']
    # k_gg = params['k_gg']
    # k_gamgam = params['k_gamgam']
    #k_zgam = params['k_zgam']
    BR_inv = params['BR_inv']

    sum_over_f = (k_w**2 * br['WW']
                  + k_z**2 * br['ZZ'] 
                  + k_b**2 *br['bb'] 
                  + k_t**2 * br['cc'] 
                  + k_mu**2 * br['mumu'] 
                  + k_tau**2 * br['tt'] 
                #   + k_gg**2 * br['gg'] 
                #   + k_gamgam**2 * br['gamgam']
    )
    
    mu_model_WW = (k_z**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_ZZ = (k_w**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_bb = (k_b**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_cc = (k_t**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_mumu = (k_mu**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_tau = (k_tau**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    mu_model_gamgam = ((1.59*k_w**2 + 0.07*k_t**2 - 0.66*k_t*k_w) * k_t**2 * (1 - BR_inv)) / sum_over_f
    # mu_model_gg = (k_gg**2 * k_t**2 * (1 - BR_inv)) / sum_over_f
    # mu_model_gamgam = (k_gamgam**2 * k_t**2 * (1 - BR_inv)) / sum_over_f

    res_WW = (hd('mu','ttH','WW') - mu_model_WW)/hd('unc','ttH','WW')
    res_ZZ = (hd('mu','ttH','ZZ') - mu_model_ZZ)/hd('unc','ttH','ZZ')
    res_bb = (hd('mu','ttH','bb') - mu_model_bb)/hd('unc','ttH','bb')
    res_cc = (hd('mu','ttH','cc') - mu_model_cc)/hd('unc','ttH','cc')
    res_mumu = (hd('mu','ttH','mumu') - mu_model_mumu)/hd('unc','ttH','mumu')
    res_tau = (hd('mu','ttH','tt') - mu_model_tau)/hd('unc','ttH','tt')
    res_gamgam = (hd('mu','ttH','gamgam') - mu_model_gamgam)/hd('unc','ttH','gamgam')
    # res_gg = (hd('mu','ttH','gg') - mu_model_gg)/hd('unc','ttH','gg')
    

    return np.hstack((res_WW, res_ZZ, res_bb, res_cc, res_mumu, res_tau, res_gamgam))

def residue_VH(params):
    k_w = params['k_w']
    k_z = params['k_z']
    k_t = params['k_t']
    k_b = params['k_b']
    k_mu = params['k_mu']
    k_tau = params['k_tau']
    # k_gg = params['k_gg']
    # k_gamgam = params['k_gamgam']
    #k_zgam = params['k_zgam']
    BR_inv = params['BR_inv']

    sum_over_f = (k_w**2 * br['WW']
                  + k_z**2 * br['ZZ'] 
                  + k_b**2 *br['bb'] 
                  + k_t**2 * br['cc'] 
                  + k_mu**2 * br['mumu'] 
                  + k_tau**2 * br['tt'] 
                #   + k_gg**2 * br['gg'] 
                #   + k_gamgam**2 * br['gamgam']
    )
    
    mu_model_WW = (k_z**2 * (k_w**2 + k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_ZZ = (k_w**2 * (k_w**2 + k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_bb = (k_b**2 * (k_w**2 + k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_cc = (k_t**2 * (k_w**2 + k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_mumu = (k_mu**2 *(k_w**2 + k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_tau = (k_tau**2 * (k_w**2 + k_z**2) * (1 - BR_inv)) / sum_over_f
    mu_model_gamgam = ((1.59*k_w**2 + 0.07*k_t**2 - 0.66*k_t*k_w) * (k_w**2 + k_z**2) * (1 - BR_inv)) / sum_over_f
    # mu_model_gg = (k_gg**2 * (k_w**2 + k_z**2) * (1 - BR_inv)) / sum_over_f
    # mu_model_gamgam = (k_gamgam**2 * (k_w**2 + k_z**2) * (1 - BR_inv)) / sum_over_f

    res_WW = (hd('mu','VH','WW') - mu_model_WW)/hd('unc','VH','WW')
    res_ZZ = (hd('mu','VH','ZZ') - mu_model_ZZ)/hd('unc','VH','ZZ')
    res_bb = (hd('mu','VH','bb') - mu_model_bb)/hd('unc','VH','bb')
    res_cc = (hd('mu','VH','cc') - mu_model_cc)/hd('unc','VH','cc')
    res_mumu = (hd('mu','VH','mumu') - mu_model_mumu)/hd('unc','VH','mumu')
    res_tau = (hd('mu','VH','tt') - mu_model_tau)/hd('unc','VH','tt')
    res_gamgam = (hd('mu','VH','gamgam') - mu_model_gamgam)/hd('unc','VH','gamgam')
    # res_gg = (hd('mu','VH','gg') - mu_model_gg)/hd('unc','VH','gg')
    

    return np.hstack((res_WW, res_ZZ, res_bb, res_cc, res_mumu, res_tau, res_gamgam))

# Skapa parametrar
par = lmfit.Parameters()
par.add('k_w', value = 1, min = -5, max = 5)
par.add('k_z', value = 1, min = -5, max = 5)
par.add('k_b', value = 1, min = -5, max = 5)
par.add('k_t', value = 1, min = -5, max = 5)
# par.add('k_c', value = 1, min = -5, max = 5) # är lika med k_t
par.add('k_mu', value = 1, min = -5, max = 5)
par.add('k_tau', value = 1, min = -5, max = 5)
# par.add('k_gg', value = 1, min = -5, max = 5)
# par.add('k_gamgam', value = 1, min = -5, max = 5)
#par.add('k_zgam', value = 1, min = -5, max = 5)
par.add('BR_inv', value = 0, min = 0, max = 1)

# # Produktionskappan (k_i) för 7-8 TeV
# par.add('k_t_7', value = 1, min = -5, max = 5)
# par.add('k_ggH_7', expr = "1.06 * k_t_7**2 + 1.01 * k_b**2 - 0.07 * k_t_7 * k_b")
# par.add('k_VB_7', expr = '0.74 * k_v**2 + 0.26 * k_v**2')
# par.add('k_VH_7', expr = '0.5 * k_v**2 + 0.5 * k_v **2')

# # Produktionskappan (k_i) för 13 TeV
# par.add('k_t_13', value = 1, min = -5, max = 5)
# par.add('k_ggH_13', expr = "1.04 * k_t_13**2 - 0.002 * k_b**2 - 0.04 * k_t_13 * k_b")
# par.add('k_VB_13', expr = '0.73 * k_v**2 + 0.27 * k_v**2')
# par.add('k_VH_13', expr = '0.5 * k_v**2 + 0.5 * k_v **2')

#Interferenskappan
# par.add('')
# par.add('')
# par.add('')

print('Finding best fit parameters...')
# MinimizerResult objekt
out = lmfit.minimize(residue, par, method = 'nelder', nan_policy= 'omit')

# write error report
lmfit.report_fit(out)

print(f"PID: {os.getpid()}")
print("Sampling the posterior...")
bay = lmfit.minimize(residue, method='emcee',float_behavior = 'chi2', burn=300, steps=5000, thin=30, params=out.params, is_weighted=True, progress=True)
emcee_plot = corner.corner(bay.flatchain, labels=bay.var_names, levels = (0.69,))#truths=list(out.params.valuesdict().values())
plt.show()