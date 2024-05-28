# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:00:44 2024

@author: S.H.Reeder

Field free and Stark hamiltonians for Rydberg He.
"""

from .numerov import radial_integral
from .Calculate_He_Transitions import W_tot, defect
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
from .constants import e,a0,h
from sympy.physics.wigner import wigner_3j

def Basis(ns,ml_arr):
    """Generate a basis of atomic states.
    Inputs
    -------
    ns     = array of principal quantum numbers.
    ml_arr = desired values of ml (projection of the orbital angular quantum number).
             e.g. ml_arr = [-1,0,1]

    Returns
    -------
    basis = 2D list of states 
            format [[n,l,ml],...]
    """
    basis = [] # Initialise list
    for n in ns:
        for l in range(n): # all allowed ls
            mls = np.arange(-l,l+1,1) # all allowed mls
            
            for ml in mls: # check if ml is desired
                if ml in ml_arr:
                    state = [n,l,ml] # build basis
                    basis.append(state)
    return basis

def QD_Array(basis,S=1,dJ=1):
    """Function to calculate the quantum defects of a Rydberg atom in a particular state.
    
    Inputs
    -------
    basis = 2D Array. Basis set of atomic states [n,l,ml].
    S     = Spin quantum number (Defult is S=1 for triplet states)
    dJ    = Difference from l i.e. J = l+dJ - used in quantum defect calculation.
            (Default is 1. Can take values dJ = -1,0,+1 for the triplet state [S=1], must be 0 for S=0)
    
    Returns
    --------
    QD_arr = array of quantum defects for the input basis.
     """
    # Iniitlise square matrix the size of the basis
    size = len(basis)
    QD_arr = np.zeros(size)

    for state in range(size):
        # Extract quantum numbers
        ni = basis[state][0]
        li = basis[state][1]
        
        if (S==1) & (li==0): # Ensure J for triplet s = 1
            Ji = 1
        elif (S==1) & (li>0): # Can choose J for all other l
            Ji = li+dJ
        elif (S==0): # Singlet states, J=l
            Ji = li
        else:
            raise ValueError("https://giphy.com/gifs/wetv-no-facepalm-cant-3xz2BLBOt13X9AgjEA/fullscreen" )

        # Calculate field free energies
        D = defect(ni,li,Ji,S)
        QD_arr[state] = D
        
    return QD_arr

def order(eigenvalues,eigenvectors):
    """Sort eigenvalues and eigenvectors in ascending order of eigenvalue."""
    
    idx = eigenvalues.argsort()[::1] # square brackets included for completeness - change to -1 for descending order 
    eigenvalues  = eigenvalues[idx] # sort eigenvalues
    eigenvectors = eigenvectors[:,idx] # sort eigenvectors
    
    return eigenvalues,eigenvectors

def lookup_eigval(basis,n,l):
    """Locate eigenvalues of specific state.
    
    Inputs:
    -------
    basis = [n,l] basis of states.
    n     = Principal quantum number.
    l     = Orbital angular momentum quantum number.
    
    Returns
    -------
    inds_nl = Indices of desired state
    """
    basis_arr = np.array(basis)
    inds_n  = np.where(basis_arr == n)[0]
    inds_l  = np.where(basis_arr == l)[0]
    inds_nl = np.intersect1d(inds_n,inds_l)[0]
    
    return inds_nl

def dip_mtrx_elem(basis1,basis2,QD1,QD2,q=0,r_exp=1,step=0.0065,rcore=0.65):
    """Calculate dipole matrix element <n'l'm'|r|nlm> per unit field.
    
    Inputs
    -------
    basis = 2D Array. Basis set of atomic states [n,l,ml].
    QD    = Qauntum defect of basis state.
    q     = Electric field polarisation vector, q = 0 [linear polarised], q= +/- 1 [Circularly polarised]
    r_exp = Exponent of r in matrix element for radial integral. (Default = 1).
    step  = Radial integration step size. (Default = 0.0065).
    rcore = Minimum r value in numerov. (The default is 0.65 -> dipole (polarizability)^(1/3) of He core).
            [For hydrogen rcore=0.05]
    
    Returns
    -------
    dipole_matrix_element - In atomic units.
    """
    # state 1
    n1, l1, ml1 = basis1[0], basis1[1], basis1[2]
    N1 = n1-QD1

    # state 2
    n2, l2, ml2 = basis2[0], basis2[1], basis2[2]
    N2 = n2-QD2

    # Use numerov method to calculate radial integral
    radint = radial_integral(N1, l1, N2, l2, r_exp, step, rcore)
    
    # Calculate angular initegral
    angint = (-1.)**ml2 * np.sqrt((2*l2+1)*(2*l1+1)) * float(wigner_3j(l2,1,l1,-ml2,q,ml1)) * float(wigner_3j(l2,1,l1,0,0,0))

    dipole_matrix_element = angint*radint

    return dipole_matrix_element

def H_0(basis,QD_arr):
    """Function to calculate the field free hamiltonian for a Rydberg atom.
    
    Inputs
    -------
    basis  = 2D Array. Basis set of atomic states [n,l,ml].
    QD_arr = 1D array of quantum defects for each state in basis.
    
    Returns
    --------
    H0 = Square diagonal matrix of field free energies.
     """
    # Iniitlise square matrix the size of the basis
    size = len(basis)
    H0 = np.zeros((size,size))
    En = []

    for state in range(size):
        # Extract quantum numbers
        ni = basis[state][0]

        # Calculate field free energies
        D = QD_arr[state]
        W = W_tot(ni,D)
        En.append(W)

    # Populate diagonal of matrix    
    np.fill_diagonal(H0,En)
    return H0

def H_s(basis,QD_arr,Fz=1.0,q=0,r_exp=1,step=0.0065,rcore=0.65):
    """Calculate the off-diagonal matrix elements that make up the field hamiltonian Hs
    
    Inputs
    -------
    basis  = 2D Array. Basis set of atomic states [n,l,ml].
    QD_arr = 1D array of quantum defects for each state in basis.
    Fz     = Electric field magnitude (V/m) [Default = 1V/m]
    q      = Electric field polarisation vector, q = 0 [linear polarised], q= +/- 1 [Circularly polarised]
            (Default = 0, linearly polarised)
    r_exp  = Exponent of r in matrix element for radial integral. (Default = 1).
    step   = Radial integration step size. (Default = 0.0065 a.u.).
    rcore  = Minimum r value in numerov. (The default is 0.65 -> dipoole polarisability of He core).
    
    Returns
    -------
    Hs = Stark matrix in units of Hz [multiply by h for SI].
    """
    
    # Initialise matrix
    size = len(basis)
    Hs = np.zeros((size,size))
    
    # Run over all possible states
    for i in tqdm(range(size)):
        ni  = basis[i][0]
        li  = basis[i][1]
        mli = basis[i][2]
        QDi = QD_arr[i]
        
        for j in range(size):
            nj  = basis[j][0]        
            lj  = basis[j][1]
            mlj = basis[j][2]
            QDj = QD_arr[j]
            
            ## Conditions for calculation
            if i!=j: # off diagonal
                if mli==mlj: # Projection not changing
                    if abs(li-lj)==1: # dl must be 1

                        ## Define states
                        state1 = [ni,li,mli]
                        state2 = [nj,lj,mlj]

                        ## Populate [i,j] element of matrix
                        Hs[i,j] = dip_mtrx_elem(state1,state2,QDi,QDj,q,r_exp,step,rcore)
                    
    return (Hs * Fz * e * a0 / h)

def E_Stark(H0,Hs,Fz_arr):
    """Calculate the Stark energies.
    
    Inputs
    -------
    H0 = [MxM] Square diagonal matrix of field free energies.
    Hs = [MxM] Off-diagonal field pertubation matrix.
    Fz = Array of field magnitudes (V/m).
    
    Returns
    -------
    Eigvals = Correspond to the stark energies - index as [state, field].
    Eigvecs = Correspond to the wavefunctions - index as [field, basis, state]. 
    """
    size_H0 = np.shape(H0)[0]
    size_Hs = np.shape(Hs)[0]
    size_Fz = len(Fz_arr)
    
    if size_H0 != size_Hs:
        raise ValueError("Matrices H0 and Hs are not the same shape. Check basis sets.")
        
    # Initalise arrays of eigenvalues and eigenvectors against field values.
    Eigvals = np.zeros((size_H0,size_Fz)) # [MxF] array - index as [state, field].
    Eigvecs = np.zeros((size_Fz,size_H0,size_H0)) # [FxMxM] array - index as [field, basis, state].
        
    # Run over all field values.
    for i in tqdm(range(size_Fz)):
        Fz = Fz_arr[i]
        
        # Calculate total hamiltonian matrix H.
        H = H0 + (Hs * Fz)
        
        # Compute eigenvalues and eigenvectors of H.
        eigvals, eigvecs = LA.eig(H)
        eigval_sort, eigvec_sort = order(eigvals, eigvecs) # sort into ascending order.
        
        # Records eigenvalues and eigenvectors for each field.
        Eigvals[:,i] = eigval_sort 
        Eigvecs[i,:,:] = eigvec_sort
        
    return Eigvals, Eigvecs