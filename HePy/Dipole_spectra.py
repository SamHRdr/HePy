# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:52:15 2024

@author: S.H.Reeder

"""
from .Stark_Matrix import dip_mtrx_elem
from .constants import e, a0, eps0, c, h, hbar
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
    rcore  = Minimum r value in numerov. (The default is core radius: 0.65 -> dipole (polarizability)^(1/3) of He core).

    Returns
    -------
    S_kq  = Dipole spectral intensity of one transition (SI). - 𝑆𝑘𝑞=|∑𝑞∑𝑘𝑐𝑖𝑞𝑐𝑗𝑘⟨𝜓𝑖|𝜇|𝜓𝑗⟩|2
    """
    # Initialise value
    size = len(basis)
    S_kq = 0
    
    # Sum over all states
    for i in range(size):
        basisi = basis[i]  # unpack state i
        li     = basisi[1]
        QDi    = QD_arr[i]
        c_ik   = state1[i] # get eigenvector coefficient for ith state
        
        for j in range(size):
            basisj = basis[j]  # unpack state j
            lj     = basisj[1]
            QDj    = QD_arr[j]
            c_jq   = state2[j] # get eigenvector coefficient for jth state
            
            # ensure dl must be 1
            if abs(li-lj)==1: 
                
                # Add component to spectral intensity
                S_kq += (c_ik) * (c_jq) * (dip_mtrx_elem(basisi,basisj,QDi,QDj,q,r_exp,step,rcore))
            
    return S_kq * e * a0 # converted to SI units - Take modulus and square for spectral intensities

##------------------ Field-free Einstein-A coefficients -----------------------
def dip_matrix(basis,QD_arr,q=0,r_exp=1,step=0.0065,rcore=0.65):
    """Generates matrix of dipole matrix elements for polarisation q. 
    !!N.B. Calculates one half of the matrix to save time as matrix would be symmetrical!!
    
    Inputs
    -------
    basis  = 2D Array. Basis set of atomic states [n,l,ml].
    QD_arr = Qauntum defects of basis states.
    q      = Electric field polarisation vector, q = 0 [linearly polarised], q= +/- 1 [Circularly polarised]
    r_exp  = Exponent of r in matrix element for radial integral. (Default = 1).
    step   = Radial integration step size. (Default = 0.0065).
    rcore  = Minimum r value in numerov. (The default is core radius: 0.65 -> dipole (polarizability)^(1/3) of He core).
            [For hydrogen rcore=0.05]
    
    Returns
    -------
    Mml = Matrix of transition dipole moments (SI units).    
    """
    # Initialise matrix
    size = len(basis)
    Mml  = np.zeros((size,size)) # Square matrix

    # Loop over all states and calculate the dipole matrix element for allowed transitions
    for i in tqdm(range(size)): # rows
        # state 1
        ni, li, mli = basis[i][0], basis[i][1], basis[i][2]
        QDi         = QD_arr[i]

        for j in range(size): # columns
            # state 2
            nj, lj, mlj = basis[j][0], basis[j][1], basis[j][2]
            QDj         = QD_arr[j]

            if j<i: # off diagonal elements, only one half of matrix (state has no dip. mom. with itself)
                if abs(li-lj)==1: # dl must be 1

                    ## Define states
                    statei = [ni,li,mli]
                    statej = [nj,lj,mlj]

                    ## Populate [i,j] element of matrix 
                    Mml[i,j] += dip_mtrx_elem(statei,statej,QDi,QDj,q,r_exp,step,rcore) * a0 * e

            else:break # kills inner loop if element is off diagonal or in top half of matrix
    return Mml

def EinA_mtrx(Mdip,H0):
    """Generate matrix of Einstein-A coefficients from dipole moments and field free energies.
    
    Inputs
    -------
    Mdip = 2D array of transition dipole moments (SI).
    H0   = Feild free energies (Hz).
    
    Returns
    -------
    MEA = Matrix of Einstein-A coefficients.
    """
    # Find indices of non-zero matrix elements
    NZ = np.nonzero(Mdip)

    # Initialise new matrix
    MD = np.zeros(np.shape(Mdip))

    for k in range(len(NZ[0])): # Assumes 2D square matrix
        # Get row and column index for each element
        i, j = NZ[0][k], NZ[1][k]

        # Get dipole matrix element
        D = Mdip[i,j]

        # Get the relevant energies for that dipole moment and subtract them
        Ei, Ej = H0[i,i], H0[j,j]
        T = abs(Ej-Ei)

        # populate the new matrix with the product of the angular transition frequency and the dipole moment
        MD[i,j] = (2*np.pi*T)**3 * abs(D)**2
        
    # define constant
    C  = (2)/(3*eps0*h*c**3)
    
    # create final array of Einstein-A coefficients
    MEA = C * MD
    return MEA

##------------------ Stark-mixed Einstein-A coefficients ----------------------
def EinA_F(Aeinarr, state):
    """Calculate the Stark mixed fluorescence decay rate of a state.
    
    Inputs
    -------
    Aeinarr = Array of field free fluorescence decay rates. 
    state   = Eigenvector of the state of interest.
    
    Returns
    -------
    rate = Field mixed Einstein-A coefficient. 
    """
    # Initialise values
    size = len(state)
    rate = 0
    
    # For each of the coeffcients
    for j in range(size):
        c_j   = state[j]
        Aeinj = Aeinarr[j]
        
        # Sum the components of the mixed rates to the total mixed rate
        rate += abs(c_j)**2 * Aeinj
        
    return rate

##--------------------------- Rabi frequency ----------------------------------
def rabi_freq(basis1,basis2,QD1,QD2,I0,q=0,r_exp=1,step=0.0065,rcore=0.65):
    """Calculate Rabi frequency of transition in Hz. N.B. Only between n1,l1 -> n2,l2 (J not included).
    
    Inputs
    -------
    basis = 2D Array. Basis set of atomic states [n,l,ml].
    QD    = Qauntum defect of basis state.
    I0    = Electric field intensisty of radiation. (W/m^2)
    q     = Electric field polarisation vector, q = 0 [linear polarised], q= +/- 1 [Circularly polarised]
    r_exp = Exponent of r in matrix element for radial integral. (Default = 1).
    step  = Radial integration step size. (Default = 0.0065).
    rcore = Minimum r value in numerov. (The default is 0.65 -> dipole (polarizability)^(1/3) of He core).
            [For hydrogen rcore=0.05]
    
    Returns
    -------
    Rabi_frq = Rabi frequency in Hz
    """
    # Photon field amplitude - (V/m)
    F0 = np.sqrt((2*I0)/(c*eps0))
    
    # Dipole transition moment - S.I.
    dip_trans = dip_mtrx_elem(basis1,basis2,QD1,QD2,q,r_exp,step,rcore) * e * a0
    
    # Rabi frequency
    Rabi_frq = (abs(dip_trans) * F0)/hbar
    
    return Rabi_frq
