# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:07:32 2024

@author: xavier
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

hbar = 6.62607015e-34/2/np.pi
Eps0=8.85418782e-12
e=1.602176565e-19
me = 9.1093837015e-31
mp = 1.672649e-27
m=4*mp
#c= 299792458
#ma = 4.0026*931.5e6*e/c**2
#m=ma
r0=1.4e-15

plt.clf()
Ealpha1=4.9e6*e
R1=7.3e-15
Rc1=50e-15
Zd1=86
A1=226

def V(x,Zd):
    """ Potentiel coulombien"""
    y=2*Zd*e**2/(4*np.pi*Eps0)/x
    return y

def integrand(x, Zd, Ealpha):
    return np.sqrt(V(x,Zd)-Ealpha)

def lnT(Zd, Ealpha, A):
    R=r0*A**(1/3)
    Rc=2*Zd*e**2/(4*np.pi*Eps0*Ealpha)
    I=integrate.quad(integrand, R, Rc, args=(Zd,Ealpha))
    y=-np.sqrt(8*m)/hbar*I[0]
    return y

np.exp(lnT(Zd1,Ealpha1,A1))

def lntdemi(Zd, Ealpha, A):
    R=r0*A**(1/3)
    y=np.log(np.sqrt(2*m/Ealpha)*R*np.log(2))-lnT(Zd,Ealpha,A)
    return y


[np.exp(lnT(Zd1,Ealpha1,A1)),np.exp(lntdemi(Zd1,Ealpha1,A1))]

Ealpha=np.linspace(3, 10, 100)*1e6*e
y=[]

for elmt in Ealpha:
    y.append(lntdemi(Zd1,elmt,A1))
    

tdemi = [4e4, 3e-7, 1e-4, 3.3e5, 5.4e10, 4.4e17, 7.2e14, 1.4e7]  # s
E = [6.2, 9.0,  8.1, 5.6, 4.9, 4.0, 4.4, 6.2]   # MeV
Zd = [81, 82, 83, 84, 86, 88, 90, 94]
A = [212, 212, 215, 222, 226, 232, 236, 242]
lntdemiGamow=[]

for i in range(np.size(E)):
    lntdemiGamow.append(lntdemi(Zd[i],E[i]*1e6*e,A[i]))

fitGeigerNuttal=np.polyfit(1/np.sqrt(E),np.log(tdemi),1)
GeigerNuttal_fn = np.poly1d(fitGeigerNuttal)


#plt.rc('font', size=20)          # controls default text sizes
#fig = plt.figure(figsize=(15,9))
##plt.plot(Ealpha/(e*1e6), y,'.--')
#plt.plot(E, np.log(tdemi),'.',markersize=20,label='data')
#plt.plot(E, lntdemiGamow,'+',markersize=20,label='Gamow,Gurney,Condon')
#plt.xlabel('E_alpha (MeV)')
#plt.ylabel('ln(T)')
#plt.title('geiger et Nuttal')
#plt.legend()


plt.rc('font', size=26)          # controls default text sizes
fig = plt.figure(1,figsize=(15,9))
#plt.plot(Ealpha/(e*1e6), y,'.--')
plt.plot(1/np.sqrt(E), np.log(tdemi),'.',markersize=20,label='data')
plt.plot([0.325, 0.5], GeigerNuttal_fn([0.325, 0.5]),'-g',markersize=20,label='Geiger et Nuttal')
plt.plot(1/np.sqrt(E), lntdemiGamow,'+r',markersize=20,label='Gamow,Gurney,Condon')
plt.xlabel(r'1/$\sqrt{E_\alpha (MeV)}$')
plt.ylabel(r'ln(t$_{1/2}$)')
plt.title('Test du modèle de Gamow, Gurney et Condon')
#plt.title('Loi de Geiger et Nuttal')
plt.legend()
plt.tight_layout()
plt.savefig('gamow')
plt.show()








