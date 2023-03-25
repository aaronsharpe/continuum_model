from __future__ import division
import numpy as np
from numpy import linalg as lg
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from bisect import bisect, bisect_left, bisect_right



"""
    Functions related to Hamiltonian construction
"""
def rm_duplicates(list_2d):
    """
        remove duplicates in a 2D list
    """
    rounded_list_2d = np.round(list_2d,10)
    p = np.lexsort(( rounded_list_2d[:,1], rounded_list_2d[:,0] ))
    sorted_list_2d = rounded_list_2d[p]
    size = np.shape(sorted_list_2d)[0]
    
    diff = []
    for i in range(size-1):
        diff.append( list(np.array(sorted_list_2d[i+1]) - np.array(sorted_list_2d[i])) )
    
    new_list = []
    new_list.append( sorted_list_2d[0] )
    for row in range(1, size):
        if abs(diff[row-1][0]) > 1e-10 or abs(diff[row-1][1]) > 1e-10:
            new_list.append(sorted_list_2d[row])
    return new_list


def rotation(theta): 
    """
        2D rotation matrix
    """
    return np.array(( [np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)] ))


def t_q(q_vec): 
    """
        q-dependent hopping strength (Rafi)
    """
    q = lg.norm(q_vec)
    t0 = 20 # meV * nm^2
    alpha = 0.13
    d_vertical = 0.334 # nm
    gamma = 1.25  
    t = t0*np.exp( -alpha*(q*d_vertical)**gamma )
    return t


def h_k_low_energy(valley, qx, qy, half_theta_rad, V_site, mass, hv, shift): 
    """
       Dirac Hamiltonian 
       hv = hbar*v_F = sqrt(3)*a*|t0|/2 
    """
    q_plus  = valley*qx + 1j*qy
    q_minus = valley*qx - 1j*qy
    
    hamil = np.zeros((2,2),dtype=complex)
    hamil[0,0] = V_site + mass + shift
    hamil[1,1] = V_site - mass + shift
    hamil[0,1] = hv * q_minus * np.exp( valley*1j*half_theta_rad)
    hamil[1,0] = hv * q_plus  * np.exp(-valley*1j*half_theta_rad)    
    return hamil


def phi_alpha_beta(T_ratio, tau_A, tau_B, G1): 
    """
       Interlayer tunneling matrix T1, T2, T3,
       AB stacking
    """
    phi_AA = np.dot( G1,tau_B )
    phi_AB = np.dot( G1,tau_A )
    phi_BA = np.dot( G1,2*tau_B-tau_A )
    phi_BB = np.dot( G1,tau_B )
    result = np.array(( [T_ratio * np.exp(1j*phi_AA), np.exp(1j*phi_AB)],\
                        [np.exp(1j*phi_BA),           T_ratio * np.exp(1j*phi_BB)] ))
    return result

#def phi_alpha_beta(tau_A, tau_B, G1, T_ratio): 
#    """
#       Interlayer tunneling matrix T1, T2, T3,
#       BA stacking
#    """
#    phi_AA = np.dot( G1,-tau_B )
#    phi_AB = np.dot( G1,tau_A-2*tau_B )
#    phi_BA = np.dot( G1,-tau_A )
#    phi_BB = np.dot( G1,-tau_B )
#    result = np.array(( [T_ratio * np.exp(1j*phi_AA), np.exp(1j*phi_AB)],\
#                        [np.exp(1j*phi_BA),           T_ratio * np.exp(1j*phi_BB)] ))
#    return result

#def phi_alpha_beta(tau_A, tau_B, G1, T_ratio): 
#    """
#       Interlayer tunneling matrix T1, T2, T3,
#       AA stacking
#    """
#    phi_AA = np.dot( G1,np.array([0,0]) )
#    phi_AB = np.dot( G1,tau_A-tau_B )
#    phi_BA = np.dot( G1,tau_B-tau_A )
#    phi_BB = np.dot( G1,np.array([0,0]) )
#    result = np.array(( [T_ratio * np.exp(1j*phi_AA), np.exp(1j*phi_AB)],\
#                        [np.exp(1j*phi_BA),           T_ratio * np.exp(1j*phi_BB)] ))
#    return result
    

def interlayer_T_moire(valley, T_ratio):
    """
        Interlayer tunneling matrices for moire
    """
    phi = 2*np.pi/3
    T1 = np.array(([T_ratio, 1.0], \
                   [1.0,     T_ratio]))
    T2 = np.array(([T_ratio * np.exp(1j*phi*valley),  1.0],\
                   [np.exp(-1j*phi*valley),           T_ratio * np.exp(1j*phi*valley)]))
    T3 = np.array(([T_ratio * np.exp(-1j*phi*valley), 1.0],\
                   [np.exp(1j*phi*valley),            T_ratio * np.exp(-1j*phi*valley)]))
    return T1, T2, T3


def AB_bilayer_T(valley, qx, qy, half_theta_rad, hv, t0, t1, t3, t4): 
    """
        AB-stacking bilayer graphene interlayer tunneling matrix
    """
    q_plus  = valley*qx + 1j*qy
    q_minus = valley*qx - 1j*qy
    
    hamil = np.zeros((2,2),dtype=complex)
    hamil[0,0] = hv * q_minus * np.exp( valley*1j*half_theta_rad) * t4/t0
    hamil[1,1] = hv * q_minus * np.exp( valley*1j*half_theta_rad) * t4/t0
    hamil[0,1] = hv * q_plus  * np.exp(-valley*1j*half_theta_rad) * t3/t0
    hamil[1,0] = t1    
    return hamil


def dAB_bilayerT_dkx(valley, half_theta_rad, hv, t0, t3, t4): 
    """
        derivative of AB-stacking bilayer graphene interlayer tunneling matrix wrt kx
    """
    hamil = np.zeros((2,2),dtype=complex)
    hamil[0,0] = hv * valley * np.exp( valley*1j*half_theta_rad) * t4/t0
    hamil[1,1] = hv * valley * np.exp( valley*1j*half_theta_rad) * t4/t0
    hamil[0,1] = hv * valley * np.exp(-valley*1j*half_theta_rad) * t3/t0
    hamil[1,0] = 0    
    return hamil


def dAB_bilayerT_dky(valley, half_theta_rad, hv, t0, t3, t4): 
    """
        derivative of AB-stacking bilayer graphene interlayer tunneling matrix wrt kx
    """
    hamil = np.zeros((2,2),dtype=complex)
    hamil[0,0] = -hv * 1j * np.exp( valley*1j*half_theta_rad) * t4/t0
    hamil[1,1] = -hv * 1j * np.exp( valley*1j*half_theta_rad) * t4/t0
    hamil[0,1] =  hv * 1j * np.exp(-valley*1j*half_theta_rad) * t3/t0
    hamil[1,0] = 0    
    return hamil


def basis(theta, valley, a_graphene, N_shell, compare_neighbors):
    """
        moire reciprocal lattice vectors till N_shell
    """
    half_theta_rad = theta/2*np.pi/180
    k_BZ_sl = 4*np.pi/3/a_graphene # graphene BZ side length
    k_MBZ_sl = abs(2*k_BZ_sl*np.sin( half_theta_rad )) 

    ## three transfer momenta
    q1 = valley * np.sign(theta) * np.array([0,1])                * k_MBZ_sl
    q2 = valley * np.sign(theta) * np.array([-np.sqrt(3)/2,-1/2]) * k_MBZ_sl
    q3 = valley * np.sign(theta) * np.array([np.sqrt(3)/2,-1/2])  * k_MBZ_sl
    q_3 = np.array([q1, q2, q3])
    
    ## six moire reciprocal lattice vector
    globals()[ 'g'+str(1) ] = np.array([ 1/2,  np.sqrt(3)/2]) * np.sqrt(3) * k_MBZ_sl
    globals()[ 'g'+str(2) ] = np.array([-1/2,  np.sqrt(3)/2]) * np.sqrt(3) * k_MBZ_sl
    globals()[ 'g'+str(3) ] = np.array([-1,    0])            * np.sqrt(3) * k_MBZ_sl
    globals()[ 'g'+str(4) ] = np.array([-1/2, -np.sqrt(3)/2]) * np.sqrt(3) * k_MBZ_sl
    globals()[ 'g'+str(5) ] = np.array([ 1/2, -np.sqrt(3)/2]) * np.sqrt(3) * k_MBZ_sl
    globals()[ 'g'+str(6) ] = np.array([ 1,    0])            * np.sqrt(3) * k_MBZ_sl
    
    ## generate basis
    globals()[ 'layer1_shell'+str(1) ] = [ [0,0] ]
    for n in range(2, N_shell+1):
        globals()[ 'layer1_shell'+str(n) ] = []
        for j_layer1 in range( np.shape( globals()['layer1_shell'+str(n-1)] )[0] ):
            for i in range(1, 7):
                globals()[ 'layer1_shell'+str(n) ].append( \
                       list(globals()['layer1_shell'+str(n-1)][j_layer1] + globals()['g'+str(i)]) )
    
    ## combine all shells to set the basis
    for n in range(1, N_shell+1):
        if n==1:
            layer1_shelltot = globals()[ 'layer1_shell'+str(n) ]
        else:
            layer1_shelltot += globals()[ 'layer1_shell'+str(n) ]
    
    if valley==1:
        layer1_shelltot_uniq = np.array( rm_duplicates(layer1_shelltot) )
        layer2_shelltot_uniq = layer1_shelltot_uniq + np.array([0, k_MBZ_sl]) * np.sign(theta)
    else:
        layer1_shelltot_uniq = np.array( rm_duplicates(layer1_shelltot) )[::-1]
        layer2_shelltot_uniq = layer1_shelltot_uniq - np.array([0, k_MBZ_sl]) * np.sign(theta)
    
    return q_3, layer1_shelltot_uniq, layer2_shelltot_uniq


def high_sym_points(theta, valley, a_graphene):
    """
        high symmetric points in MBZ
    """
    half_theta_rad = theta/2*np.pi/180
    k_BZ_sl = 4*np.pi/3/a_graphene # graphene BZ side length
    k_MBZ_sl = abs(2*k_BZ_sl*np.sin( half_theta_rad )) 
    
    k =       np.array([0, 0])  * k_MBZ_sl * valley # Dirac point of layer1
    k_prime = np.array([0, -1]) * k_MBZ_sl * valley * np.sign(theta) # Dirac point of layer2
    gamma =   np.array([-np.sqrt(3)/2, -1/2])  * k_MBZ_sl * valley * np.sign(theta)
    gamma2 =  np.array([np.sqrt(3)/2, -1/2]) * k_MBZ_sl * valley * np.sign(theta)        
    return k, k_prime, gamma, gamma2



"""
    Functions related to DOS calculation
"""
def heaviside_10(E, mu):
    if E<mu:
        result = 1
    elif E>mu:
        result = 0
    else:
        result = 1/2
    return result


def heaviside_01(E, mu):
    if E<mu:
        result = 0
    elif E>mu:
        result = 1
    else:
        result = 1/2
    return result


def sorted_val(Nk_side, active_bands_each_side_val, eig_val):
    """
        Sort eigenvals and store them for down and up triangulars in sort_E_down and sort_E_up respectively.
        sort_E_down[ #_active_bands, #_n, #_m, 3 ] 
        Nk_side: # of k points on one side of k-space unit cell
        eig_val[ #_n, #_m, #_active_bands ], is the output of val_k_space()
        active_bands_each_side_val: # of bands to keep on two sides of CNP
    """
    active_bands = active_bands_each_side_val*2
    sort_E_down = np.zeros((active_bands, Nk_side, Nk_side, 3))
    sort_E_up   = np.zeros((active_bands, Nk_side, Nk_side, 3))

    E_down = np.zeros((active_bands, 3))
    E_up   = np.zeros((active_bands, 3))
    for n in range(Nk_side):
        for m in range(Nk_side):
            E_down[:,0] = eig_val[n,  m]
            E_down[:,1] = eig_val[n+1,m+1]
            E_down[:,2] = eig_val[n,  m+1]
            p1 = np.argsort(E_down, axis=1) # sort each row
            for band in range(active_bands):
                sort_E_down[band,n,m] = E_down[band][p1[band]]
                
            E_up[:,0] = eig_val[n,  m]
            E_up[:,1] = eig_val[n+1,m+1]
            E_up[:,2] = eig_val[n+1,m]
            p2 = np.argsort(E_up,axis=1) # sort each row
            for band in range(active_bands):
                sort_E_up[band,n,m] = E_up[band][p2[band]]                    
    return sort_E_down, sort_E_up


def d_o_s(mu, E1, E2, E3, A_MZ): 
    """
        DOS in a triangular.
        A_MZ: area of micro-zone (one triangular)
        mu: chemical potential
        E1, E2, E3: eigenvalues at three vertices of the triangular
        Note that E1 <= E2 <= E3
    """
    if len(set([E1,E2,E3])) == 3: # if E1 < E2 < E3
        if mu>E1 and mu<=E2:
            den_of_state = A_MZ*(mu-E1)/(E3-E1)/(E2-E1)                    
        elif mu>E2 and mu<E3:
            den_of_state = A_MZ*(E3-mu)/(E2-E3)/(E1-E3)                
        else:
            den_of_state = 0
            
    if len(set([E1,E2,E3])) == 2 and E1==E2: # if E1 = E2 < E3
        if mu>E1 and mu<E3:
            den_of_state = A_MZ*(E3-mu)/(E2-E3)/(E1-E3) 
        else:
            den_of_state = 0
    
    if len(set([E1,E2,E3])) == 2 and E2==E3: # if E1 < E2 = E3
        if mu>E1 and mu<E3:
            den_of_state = A_MZ*(mu-E1)/(E3-E1)/(E2-E1)     
        else:
            den_of_state = 0
            
    if len(set([E1,E2,E3])) == 1:  # if E1 = E2 = E3
        den_of_state = 0    
    return den_of_state



def dos_of_oneband(mu, band, sort_E_down, sort_E_up, Nk_side, A_MZ):
    """
        DOS of a specific band, band=0~active_bands.        
    """
    #*****  For sort_E_down *****#
    for i in range(np.shape(sort_E_down)[1]): # np.shape(sort_E_down)[1] = #_n
        if i==0:
            comb_nm_Edown_list =  list(sort_E_down[band,i])
        else:
            comb_nm_Edown_list += list(sort_E_down[band,i])
    comb_nm_Edown_array = np.array(comb_nm_Edown_list) # n = row//Nk_side, m = row%Nk_side
    p_Edown1 = np.lexsort(( comb_nm_Edown_array[:,2], comb_nm_Edown_array[:,0] )) 
               # sort comb_n_m_array first by column 0 then by column 2
    comb_nm_Edown_sort = comb_nm_Edown_array[p_Edown1]
    # mu should satisfy: for each row of comb_nm_sort, comb_nm_sort[row,0] < mu < comb_nm_sort[row,2]
    
    ## For up_limit_row
    row_start = 0
    row_end = Nk_side**2
    row_middle = np.int( (row_end+row_start)/2 )
    while (row_end-row_start) > 1:
        if mu >= comb_nm_Edown_sort[row_middle][0]:
            row_start  = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
        else:
            row_end = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
    up_limit_row_Edown = 1*row_end # need to include bands with row# <= up_limit_row
    comb_nm_Edown_sort2 = comb_nm_Edown_sort[0 : up_limit_row_Edown+1]
    
    ## For lower_limit_row
    p_Edown2 = np.lexsort(( comb_nm_Edown_sort2[:,0], comb_nm_Edown_sort2[:,2] ))
    comb_nm_Edown_sort3 = comb_nm_Edown_sort2[p_Edown2] 
    row_start = 0
    row_end = np.shape(comb_nm_Edown_sort2)[0]
    row_middle = np.int( (row_end+row_start)/2 )
    while (row_end-row_start) > 1:
        if mu <= comb_nm_Edown_sort3[row_middle][2]:
            row_end = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
        else:
            row_start = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
    lower_limit_row_Edown = 1*row_start # need to include bands with row# >= lower_limit_row
    active_triangles_Edown = comb_nm_Edown_sort3[lower_limit_row_Edown:]
    
    dos_Edown = 0
    for i in range(np.shape(active_triangles_Edown)[0]):
        dos_Edown += d_o_s(mu, active_triangles_Edown[i][0], active_triangles_Edown[i][1], \
                           active_triangles_Edown[i][2], A_MZ)
        
    
    #*****  For sort_E_up *****#
    for i in range(np.shape(sort_E_up)[1]): # np.shape(sort_E_up)[1] = #_n
        if i==0:
            comb_nm_Eup_list = list(sort_E_up[band,i])
        else:
            comb_nm_Eup_list += list(sort_E_up[band,i])
    comb_nm_Eup_array = np.array(comb_nm_Eup_list) # n = row//Nk_side, m = row%Nk_side
    p_Eup1 = np.lexsort(( comb_nm_Eup_array[:,2], comb_nm_Eup_array[:,0] )) 
             # sort comb_n_m_array first by column 0 then by column 2
    comb_nm_Eup_sort = comb_nm_Eup_array[p_Eup1]
    # mu should satisfy: for each row of comb_n_m_sort, comb_n_m_sort[row,0] < mu < comb_n_m_sort[row,2]
    
    ## For up_limit_row
    row_start = 0
    row_end = Nk_side**2
    row_middle = np.int( (row_end+row_start)/2 )
    while (row_end-row_start) > 1:
        if mu >= comb_nm_Eup_sort[row_middle][0]:
            row_start = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
        else:
            row_end = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
    up_limit_row_Eup = 1*row_end # need to include bands with row# <= up_limit_row
    comb_nm_Eup_sort2 = comb_nm_Eup_sort[0:up_limit_row_Eup+1]
    
    ## For lower_limit_row
    p_Eup2 = np.lexsort(( comb_nm_Eup_sort2[:,0], comb_nm_Eup_sort2[:,2] ))
    comb_nm_Eup_sort3 = comb_nm_Eup_sort2[p_Eup2] 
    row_start = 0
    row_end = np.shape(comb_nm_Eup_sort2)[0]
    row_middle = np.int( (row_end+row_start)/2 )
    while (row_end-row_start) > 1:
        if mu <= comb_nm_Eup_sort3[row_middle][2]:
            row_end = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
        else:
            row_start = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
    lower_limit_row_Eup = 1*row_start # need to include bands with row# >= lower_limit_row
    active_triangles_Eup = comb_nm_Eup_sort3[lower_limit_row_Eup:]
    
    dos_Eup = 0
    for i in range(np.shape(active_triangles_Eup)[0]):
        dos_Eup += d_o_s(mu, active_triangles_Eup[i][0], active_triangles_Eup[i][1], \
                           active_triangles_Eup[i][2], A_MZ)
        
    dos_tot = dos_Edown + dos_Eup    
    return dos_tot



"""
    Functions related to orbital magnetization calculation
"""
def sorted_val_Mz(Nk_side, num_bands, eig_val, Mz1_k, Mz2_k):
    """
        store the sorted eigenvals for down and up triangulars in sort_E_down and sort_E_up respectively,
        sort_E_down[ #num_bands, #_n, #_m, 3 ].
        num_bands: # of bands that we are interested in.
    """
    sort_E_down   = np.zeros((num_bands, Nk_side, Nk_side, 3))
    sort_E_up     = np.zeros((num_bands, Nk_side, Nk_side, 3))
    sort_Mz1_down = np.zeros((num_bands, Nk_side, Nk_side, 3))
    sort_Mz1_up   = np.zeros((num_bands, Nk_side, Nk_side, 3))
    sort_Mz2_down = np.zeros((num_bands, Nk_side, Nk_side, 3))
    sort_Mz2_up   = np.zeros((num_bands, Nk_side, Nk_side, 3))

    E_down,   E_up   = np.zeros((num_bands,3)), np.zeros((num_bands,3))
    Mz1_down, Mz1_up = np.zeros((num_bands,3)), np.zeros((num_bands,3))
    Mz2_down, Mz2_up = np.zeros((num_bands,3)), np.zeros((num_bands,3))
    for n in range(Nk_side):
        for m in range(Nk_side):
            E_down[:,0] = eig_val[n,  m]
            E_down[:,1] = eig_val[n+1,m+1]
            E_down[:,2] = eig_val[n,  m+1]
            p1 = np.argsort(E_down, axis=1) # sort each row
            
            Mz1_down[:,0] = Mz1_k[n,  m]
            Mz1_down[:,1] = Mz1_k[n+1,m+1]
            Mz1_down[:,2] = Mz1_k[n,  m+1]
            Mz2_down[:,0] = Mz2_k[n,  m]
            Mz2_down[:,1] = Mz2_k[n+1,m+1]
            Mz2_down[:,2] = Mz2_k[n,  m+1]
            for band in range(num_bands):
                sort_E_down[band,n,m]   = E_down[band][p1[band]]
                sort_Mz1_down[band,n,m] = Mz1_down[band][p1[band]]
                sort_Mz2_down[band,n,m] = Mz2_down[band][p1[band]]
                
            E_up[:,0] = eig_val[n,  m]
            E_up[:,1] = eig_val[n+1,m+1]
            E_up[:,2] = eig_val[n+1,m]
            p2 = np.argsort(E_up, axis=1) # sort each row
            
            Mz1_up[:,0] = Mz1_k[n,  m]
            Mz1_up[:,1] = Mz1_k[n+1,m+1]
            Mz1_up[:,2] = Mz1_k[n+1,m]
            Mz2_up[:,0] = Mz2_k[n,  m]
            Mz2_up[:,1] = Mz2_k[n+1,m+1]
            Mz2_up[:,2] = Mz2_k[n+1,m]
            for band in range(num_bands):
                sort_E_up[band,n,m]   = E_up[band][p2[band]]
                sort_Mz1_up[band,n,m] = Mz1_up[band][p2[band]]
                sort_Mz2_up[band,n,m] = Mz2_up[band][p2[band]]
                
    return sort_E_down, sort_E_up, sort_Mz1_down, sort_Mz1_up, sort_Mz2_down, sort_Mz2_up


def Mz_one_MZ(mu, E1, E2, E3, Mz1, Mz2, Mz3, A_MZ):
    """
        orbital magnetization Mz in a triangular micro-zone.
        A_MZ: area of micro-zone (one triangular)
        mu: chemical potential
        E1, E2, E3: eigenvalues at three vertices of the triangular
        Note that E1 <= E2 <= E3
    """
    if len(set([E1,E2,E3])) == 3: # E1<E2<E3
        if mu>E1 and mu<=E2:
            f_31 = (mu-E1)/(E3-E1)
            f_13 = 1-f_31
            f_21 = (mu-E1)/(E2-E1)
            f_12 = 1-f_21
            n_portion = f_31*f_21
            magnetization = A_MZ * n_portion * (1/3) * ( (1+f_12+f_13)*Mz1 + f_21*Mz2 + f_31*Mz3 )
        elif mu>E2 and mu<E3:
            f_31 = (mu-E1)/(E3-E1)
            f_13 = 1-f_31
            f_23 = (mu-E3)/(E2-E3)
            f_32 = 1-f_23
            magnetization = A_MZ * (1/3) * ( (f_31*(1+f_13)+f_32*f_13**2)*Mz1 + \
                                             (f_31+f_32*f_13*(1+f_23))*Mz2 + \
                                             (f_31**2+f_32*f_13*(f_31+f_32))*Mz3 )
        elif mu>=E3:
            magnetization = A_MZ * (1/3) * ( Mz1 + Mz2 + Mz3 )
        else:
            magnetization = 0
    
    if len(set([E1,E2,E3])) == 2 and E1==E2:
        if mu>E2 and mu<E3:
            f_31 = (mu-E1)/(E3-E1)
            f_13 = 1-f_31
            f_23 = (mu-E3)/(E2-E3)
            f_32 = 1-f_23
            magnetization = A_MZ * (1/3) * ( (f_31*(1+f_13)+f_32*f_13**2)*Mz1 + \
                                             (f_31+f_32*f_13*(1+f_23))*Mz2 + \
                                             (f_31**2+f_32*f_13*(f_31+f_32))*Mz3 )            
        elif mu>=E3:
            magnetization = A_MZ * (1/3) * ( Mz1 + Mz2 + Mz3 )
        else:
            magnetization = 0
            
    if len(set([E1,E2,E3])) == 2 and E2==E3:
        if mu>E1 and mu<E2:
            f_31 = (mu-E1)/(E3-E1)
            f_13 = 1-f_31
            f_21 = (mu-E1)/(E2-E1)
            f_12 = 1-f_21
            n_portion = f_31*f_21
            magnetization = A_MZ * n_portion * (1/3) * ( (1+f_12+f_13)*Mz1 + f_21*Mz2 + f_31*Mz3 )
        elif mu>=E3:
            magnetization = A_MZ * (1/3) * ( Mz1 + Mz2 + Mz3 )
        else:
            magnetization = 0
    
    if len(set([E1,E2,E3])) == 1:  
        if mu>=E3:
            magnetization = A_MZ * (1/3) * ( Mz1 + Mz2 + Mz3 )
        else:
            magnetization = 0
    
    return magnetization


def Mz_oneband(mu, band, sort_E_down, sort_E_up, sort_Mz1_down, sort_Mz1_up, \
               sort_Mz2_down, sort_Mz2_up, Nk_side, A_MZ):
    """
        orbital magnetization for a specific band, band=0~interest_bands 
    """
    #***** For sort_E_down *****#
    for i in range(np.shape(sort_E_down)[1]): # np.shape(sort_E_down)[1] = #_n
        if i==0:
            comb_nm_Edown_list = list( sort_E_down[band,i] )
            comb_nm_Mz1_down_list = list( sort_Mz1_down[band,i] )
            comb_nm_Mz2_down_list = list( sort_Mz2_down[band,i] )
        else:
            comb_nm_Edown_list += list( sort_E_down[band,i] )
            comb_nm_Mz1_down_list += list( sort_Mz1_down[band,i] )
            comb_nm_Mz2_down_list += list( sort_Mz2_down[band,i] )
    comb_nm_Edown_array = np.array(comb_nm_Edown_list) # n = row//Nk_side, m = row%Nk_side
    comb_nm_Mz1_down_array = np.array(comb_nm_Mz1_down_list)
    comb_nm_Mz2_down_array = np.array(comb_nm_Mz2_down_list)
    p_Edown1 = np.lexsort(( comb_nm_Edown_array[:,2], comb_nm_Edown_array[:,0] )) 
               # sort comb_nm_array first by column 0 then by column 2
    comb_nm_Edown_sort = comb_nm_Edown_array[p_Edown1]
    comb_nm_Mz1_down_sort = comb_nm_Mz1_down_array[p_Edown1]
    comb_nm_Mz2_down_sort = comb_nm_Mz2_down_array[p_Edown1]
    # mu should satisfy: for each row of comb_nm_sort, comb_nm_sort[row,0] < mu
    
    ## For up_limit_row
    row_start = 0
    row_end = Nk_side**2
    row_middle = np.int( (row_end+row_start)/2 )
    while row_end-row_start>1:
        if mu >= comb_nm_Edown_sort[row_middle][0]:
            row_start = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
        else:
            row_end = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
    up_limit_row_Edown = 1*row_end # need to include bands with row# <= up_limit_row
    active_Edown = comb_nm_Edown_sort[0 : up_limit_row_Edown+1]
    active_Mz1_down = comb_nm_Mz1_down_sort[0:up_limit_row_Edown+1]
    active_Mz2_down = comb_nm_Mz2_down_sort[0:up_limit_row_Edown+1]
    
    Mz1_Edown = 0 # refer to Mz1 contribution
    Mz2_Edown = 0 # refer to Mz2 contribution
    for i in range(np.shape(active_Edown)[0]):
        E1 = active_Edown[i][0]
        E2 = active_Edown[i][1]
        E3 = active_Edown[i][2]
        Mz_EnprimeEn_1 = active_Mz1_down[i][0]
        Mz_EnprimeEn_2 = active_Mz1_down[i][1]
        Mz_EnprimeEn_3 = active_Mz1_down[i][2]
        Mz_sum_1 = active_Mz2_down[i][0]
        Mz_sum_2 = active_Mz2_down[i][1]
        Mz_sum_3 = active_Mz2_down[i][2]
        Mz1_Edown += Mz_one_MZ(mu, E1, E2, E3, Mz_EnprimeEn_1, Mz_EnprimeEn_2, Mz_EnprimeEn_3, A_MZ)
        Mz2_Edown += Mz_one_MZ(mu, E1, E2, E3, Mz_sum_1, Mz_sum_2, Mz_sum_3, A_MZ)
        
    #***** For sort_E_up *****#
    for i in range(np.shape(sort_E_up)[1]): # np.shape(sort_E_up)[1] = #_n
        if i==0:
            comb_nm_Eup_list = list(sort_E_up[band,i]  )
            comb_nm_Mz1_up_list = list( sort_Mz1_up[band,i] )
            comb_nm_Mz2_up_list = list( sort_Mz2_up[band,i] )
        else:
            comb_nm_Eup_list += list(sort_E_up[band,i]  )
            comb_nm_Mz1_up_list += list( sort_Mz1_up[band,i] )
            comb_nm_Mz2_up_list += list( sort_Mz2_up[band,i] )
    comb_nm_Eup_array = np.array(comb_nm_Eup_list) # n = row//Nk_side, m = row%Nk_side
    comb_nm_Mz1_up_array = np.array(comb_nm_Mz1_up_list)
    comb_nm_Mz2_up_array = np.array(comb_nm_Mz2_up_list)
    p_Eup1 = np.lexsort(( comb_nm_Eup_array[:,2], comb_nm_Eup_array[:,0] )) 
             # sort comb_n_m_array first by column 0 then by column 2
    comb_nm_Eup_sort = comb_nm_Eup_array[p_Eup1]
    comb_nm_Mz1_up_sort = comb_nm_Mz1_up_array[p_Eup1]
    comb_nm_Mz2_up_sort = comb_nm_Mz2_up_array[p_Eup1]
    # mu should satisfy: for each row of comb_nm_sort, comb_nm_sort[row,0] < mu
    
    ## For up_limit_row
    row_start = 0
    row_end = Nk_side**2
    row_middle = np.int( (row_end+row_start)/2 )
    while row_end-row_start>1:
        if mu >= comb_nm_Eup_sort[row_middle][0]:
            row_start = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
        else:
            row_end = 1*row_middle
            row_middle = np.int( (row_end+row_start)/2 )
    up_limit_row_Eup = 1*row_end # need to include bands with row# <= up_limit_row
    active_Eup = comb_nm_Eup_sort[0:up_limit_row_Eup+1]
    active_Mz1_up = comb_nm_Mz1_up_sort[0:up_limit_row_Eup+1]
    active_Mz2_up = comb_nm_Mz2_up_sort[0:up_limit_row_Eup+1]     
    
    Mz1_Eup = 0
    Mz2_Eup = 0
    for i in range(np.shape(active_Eup)[0]):
        E1 = active_Eup[i][0]
        E2 = active_Eup[i][1]
        E3 = active_Eup[i][2]
        Mz_EnprimeEn_1 = active_Mz1_up[i][0]
        Mz_EnprimeEn_2 = active_Mz1_up[i][1]
        Mz_EnprimeEn_3 = active_Mz1_up[i][2]
        Mz_sum_1 = active_Mz2_up[i][0]
        Mz_sum_2 = active_Mz2_up[i][1]
        Mz_sum_3 = active_Mz2_up[i][2]
        Mz1_Eup += Mz_one_MZ(mu, E1, E2, E3, Mz_EnprimeEn_1, Mz_EnprimeEn_2, Mz_EnprimeEn_3, A_MZ)
        Mz2_Eup += Mz_one_MZ(mu, E1, E2, E3, Mz_sum_1, Mz_sum_2, Mz_sum_3, A_MZ)
            
    Mz_EnprimeEn = Mz1_Edown + Mz1_Eup
    Mz_twomu = (Mz2_Edown + Mz2_Eup)*(-2*mu)
    return Mz_EnprimeEn, Mz_twomu

