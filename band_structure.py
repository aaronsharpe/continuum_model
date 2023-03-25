from __future__ import division
import numpy as np
from numpy import linalg as lg
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'STIXGeneral'

from funcs import *
from class_tBLG import *




"""
    band structure variables
"""
theta = 0.5
valley = 1
w0 = 117 # meV, interlayer tunneling strength
T_ratio = 0.7 # w_AA/w_AB
N_shell = 3 # num of shells, N_shell >= 2, N_shell <= 4 for this compare_neighbors
compare_neighbors = np.array(([0, 1, -2],\
                              [0, 1, 2, -2, -3],\
                              [0, 1, 2, 3, -2, -3, -4],\
                              [0, 1, 2, 3, 4, -2, -3, -4, -5],\
                              [0, 1, 2, 3, 4, 5, -2, -3, -4, -5, -6]), dtype=object)


"""
    set up instances
"""
masses = np.array([0, 0])
TG_case = tBLG(theta, valley, masses, w0, T_ratio, N_shell, compare_neighbors)
V_sites  = 0*np.array([1/2, -1/2]) # onsite energy


"""
    some useful parameters
"""
theta_rad = theta*np.pi/180
k_BZ_sl = 4*np.pi/3/TG_case.a_graphene # graphene BZ side length
k_MBZ_sl = abs(2*k_BZ_sl*np.sin( theta/2*np.pi/180 ))#abs( 8*np.pi/3/TG_case.a_graphene*np.sin(theta/2*np.pi/180) )
a_M = TG_case.a_graphene/(theta_rad) # lattice constant of Moire superlattice
area_M = a_M**2*np.sin(60*np.pi/180) # area of Moire superlattice
b_M = k_MBZ_sl*np.sqrt(3)

q_3, layer1_q, layer2_q = basis(theta, valley, TG_case.a_graphene, N_shell, compare_neighbors)
N_bands = TG_case.N_layers * (np.shape(layer1_q)[0] + np.shape(layer2_q)[0])
k, k_prime, gamma, gamma2 = high_sym_points(theta, valley, TG_case.a_graphene)

"""
    set initial energy shift to make sure that energy at kappa is 0
"""
V_sites0 = 0*np.array([1/2, -1/2])
val0, vec0 = lg.eigh( TG_case.hamil(0, 0, V_sites0, 0) )
shift0 = - (val0[N_bands//2-1] + val0[N_bands//2])/2


"""
    sampling along high-symmetry lines
"""
Nk = 80 # num of k points on segment with length k_MBZ_sl
Nk_D = np.int(np.sqrt(3)*Nk)

### k' -> k -> gamma -> m -> k'
kx1 = list( np.linspace(k_prime[0], k[0], Nk) )
ky1 = list( np.linspace(k_prime[1], k[1], Nk) )
kx2 = list( np.linspace(k[0]+(gamma[0]-k[0])/(Nk-1), gamma[0], Nk-1) )
ky2 = list( np.linspace(k[1]+(gamma[1]-k[1])/(Nk-1), gamma[1], Nk-1) )
kx3 = list( np.linspace(gamma[0]+(gamma2[0]-gamma[0])/(Nk_D-1), gamma2[0], Nk_D-1) )
ky3 = list( np.linspace(gamma[1]+(gamma2[1]-gamma[1])/(Nk_D-1), gamma2[1], Nk_D-1) )
kx4 = list( np.linspace(gamma2[0]+(k_prime[0]-gamma2[0])/(Nk-1), k_prime[0], Nk-1) )
ky4 = list( np.linspace(gamma2[1]+(k_prime[1]-gamma2[1])/(Nk-1), k_prime[1], Nk-1) )

kxx = np.array( kx1 + kx2 + kx3 + kx4 )
kyy = np.array( ky1 + ky2 + ky3 + ky4 )
Nk_tot = np.size(kxx)


"""
    band structure
"""
eigen_val = np.zeros((Nk_tot, N_bands))
for ii in range(Nk_tot):
    kx, ky = kxx[ii], kyy[ii]
    val, vec = lg.eigh( TG_case.hamil(kx, ky, V_sites, shift0) )
    eigen_val[ii] = val

plt.figure()
for i in range(0, N_bands):
    plt.plot(np.linspace(0, Nk_tot-1, Nk_tot), eigen_val[:, i], color='black',linewidth=1.5)
plt.axvline(Nk-1,linestyle='--',linewidth=0.5, color='grey')
plt.axvline(2*Nk-2,linestyle='--',linewidth=0.5, color='grey')
plt.axvline(2*Nk+Nk_D-3,linestyle='--',linewidth=0.5, color='grey')
plt.ylabel('E (meV)',fontsize=18)
plt.xlim(0, Nk_tot-1)
plt.ylim([-50, 50])
plt.xticks([0, Nk-1, 2*Nk-2, 2*Nk+Nk_D-3, Nk_tot-1], (['$\kappa^\prime$','$\kappa$','$\gamma$','$\gamma$','$\kappa^\prime$']),fontsize=18)
plt.yticks(fontsize=16)
plt.title(r'$\theta=$'+str(theta), fontsize=18)
#plt.gca().set_aspect(Nk_tot/400)
plt.savefig(str(theta)+'.pdf', bbox_inches='tight')