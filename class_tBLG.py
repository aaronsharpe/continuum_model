from __future__ import division
import numpy as np
from numpy import linalg as lg
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from bisect import bisect, bisect_left, bisect_right

from funcs import *


class tBLG(object):
    """
        twisted bilayer graphene
    """  
    a_graphene = 0.246 # nm
    t0 = -3089
    hv = np.sqrt(3)/2 * a_graphene * abs(t0) #=658.2 # meV*nm, hbar*v_F = 6.582e-13 * 1e15
    N_layers = 2 # num of graphene layers
      
    def __init__(self, theta, valley, masses, w0, T_ratio, N_shell, compare_neighbors):
        self.theta = theta
        self.valley = valley
        self.masses = masses
        self.w0 = w0
        self.T_ratio = T_ratio
        self.N_shell = N_shell
        self.compare_neighbors = compare_neighbors
        
        self.half_theta_rad = theta/2*np.pi/180
        self.k_BZ_sl = 4*np.pi/3/tBLG.a_graphene # graphene BZ side length
        self.k_MBZ_sl = abs( 8*np.pi/3/tBLG.a_graphene*np.sin(theta/2*np.pi/180) )  # MBZ side length
        self.a_M = tBLG.a_graphene/(self.half_theta_rad*2) # lattice constant of Moire superlattice
        self.area_M = self.a_M**2*np.sin(60*np.pi/180) # area of Moire superlattice
        self.q_3, self.layer1_q, self.layer2_q = basis(theta, valley, tBLG.a_graphene, N_shell, compare_neighbors)
        self.T1, self.T2, self.T3 = interlayer_T_moire(valley, T_ratio)
        self.tau_A = np.array([0,0])
        self.tau_B = np.dot( rotation(self.half_theta_rad), np.array([0, 1])*tBLG.a_graphene/np.sqrt(3) )
        self.N_bands = tBLG.N_layers * (np.shape(self.layer1_q)[0] + np.shape(self.layer2_q)[0])
    
        ## rhombus MBZ
        self.b_M = self.k_MBZ_sl*np.sqrt(3)
        self.b_M1 = valley * np.array([1/2, -np.sqrt(3)/2]) * self.b_M
        self.b_M2 = valley * np.array([1/2,  np.sqrt(3)/2]) * self.b_M        
        self.o_point  = valley*np.array([1, 0])*self.k_BZ_sl + valley*np.array([0, 1])*self.k_BZ_sl*np.sin(self.half_theta_rad) \
                      + valley*np.array([-np.sqrt(3)/2, -1/2])*self.k_MBZ_sl
        self.K_layer1 = valley*np.array([1, 0])*self.k_BZ_sl + valley*np.array([0, 1])*self.k_BZ_sl*np.sin(self.half_theta_rad) # k point of layer1    
    
    
    
    def hamil(self, qx, qy, V_sites, shift):
        """
            Hamiltotian H(k).
            (qx, qy) is wrt the K point of layer1.
        """        
        V_1, V_2 = V_sites[0], V_sites[1]    
        q1, q2, q3 = self.q_3[0], self.q_3[1], self.q_3[2]  
        mass1, mass2 = self.masses[0], self.masses[1]                
        mat = np.eye(2)-rotation(-2*self.half_theta_rad)
        mat_inv = lg.inv(mat)
        size_l1_q, size_l2_q = np.shape(self.layer1_q)[0], np.shape(self.layer2_q)[0]  
        
        ## block11
        for i in range(size_l1_q):
            if i==0:
                block11 = h_k_low_energy(self.valley, qx+self.layer1_q[i][0], qy+self.layer1_q[i][1], \
                                         self.half_theta_rad, V_1, mass1, tBLG.hv, shift)
            else:
                block11 = block_diag( block11, h_k_low_energy(self.valley, qx+self.layer1_q[i][0], \
                                      qy+self.layer1_q[i][1], self.half_theta_rad, V_1, mass1, tBLG.hv, shift) )
                    
        ## block22
        for i in range(size_l2_q):
            if i==0:
                block22 = h_k_low_energy(self.valley, qx+self.layer2_q[i][0], qy+self.layer2_q[i][1], \
                                         -self.half_theta_rad, V_2, mass2, tBLG.hv, shift)
            else:
                block22 = block_diag( block22, h_k_low_energy(self.valley, qx+self.layer2_q[i][0], \
                                      qy+self.layer2_q[i][1], -self.half_theta_rad, V_2, mass2, tBLG.hv, shift) )
            
        ## block12
        block12 = np.zeros((2*size_l1_q, 2*size_l2_q), dtype=complex)
        for i in range(size_l1_q):
            for j in ( i+np.array(self.compare_neighbors[np.int(self.N_shell-2)]) ):#range(np.shape(layer2_q)[0]):#[i,i+1,i+2,i-2,i-3]:
                if j<size_l2_q and j>=0:
                    diff = self.layer2_q[j] - self.layer1_q[i]
                    critiria1 = abs( diff-q1 )
                    critiria2 = abs( diff-q2 )
                    critiria3 = abs( diff-q3 )
                    if np.all( critiria1 < 1e-10 ):
                        block12[2*i : 2*i+2, 2*j : 2*j+2] = self.w0 * self.T1
                    if np.all( critiria2 < 1e-10 ):
                        block12[2*i : 2*i+2, 2*j : 2*j+2] = self.w0 * self.T2
                    if np.all( critiria3 < 1e-10 ):
                        block12[2*i : 2*i+2, 2*j : 2*j+2] = self.w0 * self.T3
#                        
#                    if np.all( critiria1 < 1e-10 ) or np.all( critiria2 < 1e-10 ) or np.all( critiria3 < 1e-10 ):
##                        q = np.array([self.layer1_q[i][0], self.layer1_q[i][1]])
#                        G1 = np.dot( mat_inv, diff+self.valley*np.sign(self.theta)*np.array([0,-1])*self.k_MBZ_sl )                
#                        block12[2*i:2*i+2, 2*j:2*j+2] = self.w0 * phi_alpha_beta(self.T_ratio, self.tau_A, self.tau_B, G1)
#                        #t_q(k_wrt_Gamma + q + G1)/A_uc * phi_alpha_beta(T_ratio, tau_A, tau_B, G1) 
                                                        
        ## block21
        block21 = block12.transpose().conj()
        
        H_1row = np.concatenate((block11, block12),axis=1)
        H_2row = np.concatenate((block21, block22),axis=1)
        return np.concatenate((H_1row, H_2row),axis=0)
    
        

    def val_kspace(self, Nk_side, active_bands_each_side_val, V_sites, shift):
        """
            On k-space mesh, store all eigenvals to eig_val[ #_n, #_m, #_active_bands ]     
        """
        delta_k = 1/Nk_side
        active_bands = active_bands_each_side_val*2
        eig_val = np.zeros((Nk_side+1, Nk_side+1, active_bands))
        
        for n in range(Nk_side+1):
            for m in range(Nk_side+1):
                k_vector = self.o_point + m*self.b_M1*delta_k + n*self.b_M2*delta_k
                q_vector = k_vector - self.K_layer1        
                val, vec = lg.eigh( self.hamil(q_vector[0], q_vector[1], V_sites, shift) )
                eig_val[n,m] = val[ (self.N_bands//2-active_bands_each_side_val) : (self.N_bands//2+active_bands_each_side_val) ]
        return eig_val