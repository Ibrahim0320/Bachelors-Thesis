import lmfit
import numpy as np
import matplotlib.pyplot as plt
import corner

data = [0.98, 1.03, 1.01]
uncertainty = [0.03, 0.04, 0.03]

def res(params):
    a = params['a']
    b = params['b']

    model = a**2+b**2

    return (data-model)/uncertainty

par = lmfit.Parameters()
par.add_many(('a',1), ('b',1))

out = lmfit.minimize(res, par, method='leastsq', nan_policy='omit')
print(lmfit.fit_report(out))

bay = lmfit.minimize(res, method='emcee', nan_policy='omit', burn=300, steps=1000, thin=20,params=out.params, is_weighted=True, progress=True)

a = out.params['a']
b = out.params['b']

# x = [0,1,2]
# plt.errorbar(x,data,uncertainty, capsize=10, ecolor='red')
# plt.plot(x, np.ones(3,)*(0.5*a**2+0.5*b**2))
# plt.grid(True)
# plt.show()


emcee_plot = corner.corner(bay.flatchain, labels=bay.var_names,truths=list(bay.params.valuesdict().values()))
plt.show()