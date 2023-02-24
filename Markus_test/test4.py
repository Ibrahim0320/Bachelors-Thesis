import lmfit
import numpy as np
#import corner
import matplotlib.pyplot as plt


def residue(params, x, data, uncertainty):
    k_i = params['k_i']
    k_f = params['k_f']
    BR_inv = params['BR_inv']
    BR_sm = 1

    mu = (k_f ** 2 * k_i ** 2 * (1- BR_inv)) / (k_f ** 2 * BR_sm)

    model = mu * x 

    #model = k_i ** 2 * k_f ** 2 *(1-BR_inv)

    return (data-model)/uncertainty


k_i = 1
k_f = 1
BR_inv = 0
BR_sm = 1

mu_model = (k_f ** 2 * k_i ** 2 * (1- BR_inv)) / (k_f ** 2 * BR_sm)

x = np.linspace(0,100)
noise = np.random.normal(size=x.size, scale=0.1)

mu = mu_model * x
data = mu + noise

uncertainty = abs(0.05+np.random.normal(size=x.size, scale=0.1))

par = lmfit.Parameters()
par.add_many(('k_i',1), ('k_f',1),('BR_inv',0))

out = lmfit.minimize( residue, par, args=(x,data, uncertainty) )
k_ibest = out.params['k_i']
k_fbest = out.params['k_i']

print(out.params)

# plt.plot(x,y, '-')
plt.plot(x,data,'+')
plt.plot(x, k_ibest*x+k_fbest)
plt.legend(['Data', f'Chisq: {out.redchi}'])
plt.grid(True)
plt.show()