# -*- coding: utf-8 -*-




import numpy as np
from numpy.lib.scimath import sqrt
import matplotlib.pyplot as plt


titre = " Effet tunnel"

description = r"""
Ce programme permet de calculer la transmission d'une barrière de potentiel pour 
une onde de matière incidente d'énergie $E$ variable. Il permet en particulier de 
mettre en évidence l'effet tunnel. 

La transmission est tracée en fonction de l'énergie de la particule incidente. 
Sont également représentés, l'équivalent classique de la transmission et 
l'approximation de barrière large habituelle en mécanique quantique dans sa limite de validité.

$T = \frac{4K^2k^2}{(K^2+k^2)^2\mathrm{sh}^2(Kd)+4K^2k^2}$

$q = \sqrt{2m(V_0-E)}/\hbar$

$k = \sqrt{2mE}/\hbar$
"""

#===========================================================
# --- Variables globales et paramètres ---------------------
#===========================================================
plt.clf()
hbar = 6.62607015e-34/2/np.pi
e=1.602176565e-19
me = 9.1093837015e-31
mp = 1.672649e-27

m = me
V0 = 4*e # Hauteur de la barriere, c'est l'unite d'energie
approx = 0.5 # Valeur minimale acceptable pour q*d afin que l'approximation de barriere large soit verifiée

E_max=6*V0
d=3e-10

#m = 4*mp
#V0 = 10e6*e # Hauteur de la barriere, c'est l'unite d'energie
#approx = 0.5 # Valeur minimale acceptable pour q*d afin que l'approximation de barriere large soit verifiée
#
#E_max=2*V0
#d=(50-7.3)*1e-15

#===========================================================
# --- Modèle physique --------------------------------------
#===========================================================

def transmission(E, V, d):
    """ Tranmission par effet tunnel: formule exacte"""
    k = sqrt(2*m*E)/hbar # Vecteur d'onde a l'exterieur de la barriere
    q = sqrt(2*m*(V-E))/hbar # Vecteur d'onde a l'interieur de la barriere
    #t = 2*i*k*q*np.exp(-i*k*d)*1/ ( (q**2+k**2)*np.sinh(q*d) + 2*i*q*k*np.cosh(q*d) ) # coefficient de transmission en amplitude
    T = np.real(4*q**2*k**2 / ( (q**2+k**2)**2*np.sinh(q*d)**2 + 4*q**2*k**2 ) ) # coefficient de transmission en probabilite
    return T

def transmission_classique(E, V, d):
    """Tranmission par effet tunnel: cas classique"""
    T = (E-V) > 0 # Si E>V, la particule passe, sinon elle est reflechie
    return T

def limite_large_barriere(E, V, d): 
    """Tranmission par effet tunnel: Cas d'une barriere épaisse, ou la formule se simplifie """
    k = sqrt(2*m*E)/hbar # Vecteur d'onde a l'exterieur de la barriere
    q = sqrt(2*m*(V-E))/hbar # Vecteur d'onde a l'interieur de la barriere
    T=np.real(16*q**2*k**2 / (q**2+k**2)**2 *np.exp(-2*q*d))
    validite = (E<V) & (q*d>approx)
    return T, validite

#===========================================================
# --- Réalisation du plot ----------------------------------
#===========================================================


E_abscisse = np.linspace(0, E_max, 1000)
T_exact = transmission(E_abscisse, V0, d)
T_classique = transmission_classique(E_abscisse, V0, d)
T_large_barriere, validite_large_barriere = limite_large_barriere(E_abscisse, V0, d)



plt.rc('font', size=20)          # controls default text sizes
fig = plt.figure(1,figsize=(15,9))
plt.plot(E_abscisse/V0, T_exact, label='Quantique',lw=2, color='red')
plt.plot(E_abscisse/V0, T_classique, label='Classique', ls='--', color='blue')
plt.plot(E_abscisse[validite_large_barriere]/V0, T_large_barriere[validite_large_barriere], label='Barriere epaisse', lw=3, ls='--', color='green')
plt.xlabel('Energie E/V0')
plt.ylabel('Transmission')
plt.title(titre)
leg=plt.legend(loc='lower right')
plt.savefig('effet_tunnel')
#ax=plt.gca()
#ax.set_ylim(-1, 2)

fig = plt.figure(2,figsize=(15,8))
plt.plot(E_abscisse/V0, T_exact, label='Quantique',lw=2, color='red')
plt.plot(E_abscisse/V0, T_classique, label='Classique', ls='--', color='blue')
plt.plot(E_abscisse[validite_large_barriere]/V0, T_large_barriere[validite_large_barriere], label='Barriere epaisse', lw=3, ls='--', color='green')
plt.xlabel('Energie E/V0')
plt.ylabel('Transmission')
plt.title(titre)
leg=plt.legend(loc='upper left')
ax=plt.gca()
ax.set_ylim(0, 0.3)
ax.set_xlim(0, 1)
plt.savefig('effet_tunnel_2')


q = sqrt(2*m*(V0-V0/2))/hbar
[transmission(V0/2, V0, d),
limite_large_barriere(V0/2, V0, d)[0],
1/q]

plt.show()

