# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:52:15 2024

@author: S.H.Reeder

! THESE FUNCTIONS ARE YET TO BE VALIDATED - USE WITH CAUTION !
"""
from .Stark_Matrix import dip_mtrx_elem, lookup_eigval
from .Calculate_He_Transitions import W_tot, defect
from .constants import e, a0, eps0, c, h
import numpy as np
from tqdm import tqdm

##-------------- Calculate dipole spectral intensities ------------------------
def dip_spec_int(basis, QD_arr, state1, state2, q=0, r_exp=1, step=0.0065, rcore=0.65):
    """
    Inputs
    -------
    basis  = 2D Array. Basis set of atomic states [n,l,ml] - index as [state][quantum number].
    QD_arr = 1D array of quantum defects for each state in basis.
    state  = 1D Array. Eigenvector of the state. 
    q      = Electric field polarisation vector, q = 0 [linear polarised], q= +/- 1 [Circularly polarised]
    r_exp  = Exponent of r in matrix element for radial integral. (Default = 1).
    step   = Radial integration step size. (Default = 0.005).
    rcore  = Minimum r value in numerov. (The default is 0.65 -> dipole (polarizability)^(1/3) of He core).

    Returns
    -------
    S_kq  = Dipole spectral intensity of one transition (SI).
    """
    # Initialise value
    size = len(basis)
    S_kq = 0
    
    # Sum over all states
    for i in range(size):
        basisi = basis[i]  # unpack state i
        QDi    = QD_arr[i]
        c_ik   = state1[i] # get eigenvector coefficient for ith state
        
        for j in range(size):
            basisj = basis[j]  # unpack state j
            QDj    = QD_arr[j]
            c_jq   = state2[j] # get eigenvector coefficient for ith state
            
            # Add component to spectral intensity
            S_kq += (c_ik**2) * (c_jq**2) * (dip_mtrx_elem(basisi,basisj,QDi,QDj,q,r_exp,step,rcore)**2)
            
    return S_kq * e * a0 # converted to SI units

##------------------- Calculate einstein A coeffs -----------------------------
# With no pre-calculated values
def ein_A(basis1,basis2,S=1,dJ=1,q=0,r_exp=1,step=0.0065,rcore=0.65):
    """Calcualte field free Einstein A coefficient between two states. This function assumes no previous calculations.
    
    Inputs
    -------
    basis = 1D Array. Basis set of atomic states [n,l,ml].
    S     = Spin quantum number. [Default = 1 (triplets)].
    dJ    = Difference from l i.e. J = l+dJ - used in quantum defect calculation.
            (Default is 1. Can take values dJ = -1,0,+1 for the triplet state [S=1], must be 0 for S=0)
    q     = Electric field polarisation vector, q = 0 [linear polarised], q= +/- 1 [Circularly polarised]
    r_exp = Exponent of r in matrix element for radial integral. (Default = 1).
    step  = Radial integration step size. (Default = 0.0065).
    rcore = Minimum r value in numerov. (The default is 0.65 -> dipole (polarizability)^(1/3) of He core).
            [For hydrogen rcore=0.05]
    
    
    Returns
    -------
    A = Einsiein-A coefficient (S.I.).
    """
    
    # Take out constants
    C = (2. * e**2)/(3. * eps0 * h * c**3)
    
    ## Calculate omega (field free)
    n1, l1 = basis1[0], basis1[1]
    n2, l2 = basis2[0], basis2[1]
    
    # Quantum defect
    if S==1: 
        J1, J2 = l1+dJ, l2+dJ
    else:
        J1=J2=0
        
    QD1 = defect(n1,l1,J1,S)
    QD2 = defect(n2,l2,J2,S)
    
    # Energies
    W1 = W_tot(n1,QD1,wn=False) 
    W2 = W_tot(n2,QD2,wn=False) 
    W  = abs(W1-W2)
    
    # angular frequency
    omega = 2*np.pi*W
    
    # Calculate dipole matrix element in S.I. units (function is a.u.)
    D = dip_mtrx_elem(basis1,basis2,QD1,QD2,q,r_exp,step,rcore) * a0
    
    # Einstein-A coeff
    A = C * omega**3 * D**2
    
    return A

##------------------- Calculate flourescence lifetimes ------------------------
def fl_life(basis,state1,S=0,dJ=1,q=0,r_exp=1,step=0.005,rcore=0.05):
    """Calcualte field free Einstein A coefficient between two states. This function assumes no previous calculations.
    
    Inputs
    -------
    basis  = 1D Array. Basis set of atomic states [n,l,ml].
    state1 = Iniitial state (that you want the lifetime of). [n1,l1,ml1].
    S      = Spin quantum number. [Default = 1 (triplets)].
    dJ     = Difference from l i.e. J = l+dJ - used in quantum defect calculation.
            (Default is 1. Can take values dJ = -1,0,+1 for the triplet state [S=1], must be 0 for S=0)
    q      = Electric field polarisation vector, q = 0 [linear polarised], q= +/- 1 [Circularly polarised]
    r_exp  = Exponent of r in matrix element for radial integral. (Default = 1).
    step   = Radial integration step size. (Default = 0.0065).
    rcore  = Minimum r value in numerov. (The default is 0.65 -> dipole (polarizability)^(1/3) of He core).
            [For hydrogen rcore=0.05]
    
    
    Returns
    -------
    Fluorescence lifetime (s).
    """
    # Get initial state
    n1, l1, = state1[0], state1[1]
    ind     = lookup_eigval(basis,n1,l1)
    
    # Calculate Einstein-A coefficients for all states of lower energy
    A_s = 0.0
    for i in tqdm(range(len(basis[:ind-1]))):
        state_i = basis[i]
        l_i     = basis[i][1]

        if abs(l_i-l1)==1:
            A    = ein_A(state1,state_i,S,dJ,q,r_exp,step,rcore)
            A_s += A # Sum coefficients
            
    return A_s**(-1)
