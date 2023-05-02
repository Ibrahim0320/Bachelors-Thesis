import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
v = 246
mh = 125
gammatot = 4.1e-3
brinv = 0.08

def gamma_ss(m,a):
    return (v*10**a)**2 / (32 * np.pi * mh) * np.sqrt(1- 4*m**2/mh**2)

def gamma_DD(m,a):
    return (10**a)**2 * mh / (32 * np.pi) * np.power(1- 4*m**2/mh**2, 3/2)

def gamma_MM(m,a):
    return (10**a)**2 * mh / (16 * np.pi) * np.power(1- 4*m**2/mh**2, 3/2)

def gamma_VV(m,a):
    return (v*10**a)**2 * mh**3 / (128 * np.pi * m**4) * np.sqrt(1- 4*m**2/mh**2) * (1 - 4*m**2/mh**2 + 12*m**4/mh**4)

a = np.linspace(-4,1,100)
m = np.linspace(-1,62.5,100)
mforbidden = np.linspace(62.5,125,100)

A,M = np.meshgrid(a,m)
A,Mforbidden = np.meshgrid(a,mforbidden)

F = gamma_ss(M,A)/(gammatot + gamma_ss(M,A))
G = gamma_DD(M,A)/(gammatot + gamma_DD(M,A))
H = gamma_VV(M,A)/(gammatot + gamma_VV(M,A))
vectorcolor = 'lightsteelblue'
scalarcolor = 'lightcoral'
fermioncolor = 'lightgreen'

plt.contourf(Mforbidden,A,np.zeros((100,100)),[0,1],colors='orangered',alpha=0.5, hatches='X')
plt.contourf(M,A,H,[brinv,np.infty],colors=vectorcolor, hatches='o')
plt.contourf(M,A,F,[brinv,np.infty],colors=scalarcolor, hatches='*')
plt.contourf(M,A,G,[brinv,np.infty],colors=fermioncolor, hatches='s')
plt.xlim(0,125)
plt.xlabel(r"$m_{DM}$ [GeV]")
plt.ylabel(r"$\log(\lambda)$")
vpat = pat.Patch(color=vectorcolor,label='Vektor')
spat = pat.Patch(color=scalarcolor,label='Skalär')
fpat = pat.Patch(color=fermioncolor,label='Fermion')
plt.legend(handles=[vpat,spat,fpat])

# plt.legend(['A','B','C'])
# plt.legend(['Skalär','Diracfermion','Vektor'])
plt.show()