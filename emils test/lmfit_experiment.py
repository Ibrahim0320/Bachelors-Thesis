import lmfit
import numpy as np
import matplotlib.pyplot as plt
import corner

def res(params, x, data, uncertainty):
    a = params['a']
    b = params['b']

    model = a*x+b

    return (data-model)/uncertainty


a = 0.1
b = 1

x = np.linspace(0,100)
noise = np.random.normal(size=x.size, scale=0.1)

y = a*x+b
data = y+noise

uncertainty = abs(0.05+np.random.normal(size=x.size, scale=0.1))

par = lmfit.Parameters()
par.add_many(('a',0.2), ('b',0))

out = lmfit.minimize( res, par, args=(x,data, uncertainty) )
abest = out.params['a']
bbest = out.params['b']

print(lmfit.fit_report(out))

# plt.plot(x,y, '-')
# plt.plot(x,data,'+')
# plt.plot(x, abest*x+bbest)
# plt.legend(['Data', f'Chisq: {out.redchi}'])
# plt.grid(True)
# plt.show()

bay = lmfit.minimize(res, args=(x,data, uncertainty), method='emcee', nan_policy='omit', burn=300, steps=1000, thin=20,params=out.params, is_weighted=True, progress=True)
emcee_plot = corner.corner(bay.flatchain, labels=bay.var_names,levels = (0.68,), truths=list(bay.params.valuesdict().values()))

plt.show()