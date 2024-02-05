# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:00:44 2024

@author: sam_r

Field free and Stark hamiltonians for Rydberg He.
"""

from .numerov import radial_integral
from Calculate_He_Transitions import W_tot, defect
import numpy as np
from tqdm import tqdm
from numpy import linalg as LA
from .constants import e,a0,h

def Basis(ns):
    """Generate a basis of atomic states.
    Inputs
    -------
    ns = array of principal quantum numbers.

    Returns
    -------
    basis = 2D list of states 
            format [[n,l],...]
    """
    basis = [] # Initialise list
    for n in ns:
        for l in range(n): # all allowed ls
            state = [n,l]
            basis.append(state)
    return basis


def ang_int(li,lj,m=0):
    """Calculate the angular integral component of the Stark hamiltonian.
    Assumes dm = 0 and dl = +/- 1.
    
    Inputs
    -------
    li = Orbital angular momentum quantum number of first state.
    lj = Orbital angular momentum quantum number of first state.
    m  = Projection of the orbital angular momentum quantum number. (Default = 0)
    
    Returns
    -------
    angint = Evaluation of the angular integral.
    """
    if abs(li-lj)!=1:
        raise ValueError("lj must be li+1 or li-1")
    
    if   lj == li+1:
        angint = np.sqrt(((li+2)**2 - m**2) / ((2*li+3)*(2*li+1)))
    elif lj == li-1:
        angint = np.sqrt((li**2 - m**2) / ((2*li+1)*(2*li-1)))
        
    return angint

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

def H_0(basis,dJ=1):
    """Function to calculate the field free hamiltonian for a Rydberg atom.
    
    Inputs
    -------
    basis = 2D Array. Basis set of atomic states [n,l].
    dJ    = Difference from l i.e. J = l+dJ - used in quantum defect calculation.
            (Default is 1. Can take values dJ = -1,0,+1 for the triplet state [S=1])
    
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
        li = basis[state][1]
        Ji = li+dJ

        # Calculate field free energies
        D = defect(ni,li,Ji)
        W = W_tot(ni,D)
        En.append(W)

    # Populate diagonal of matrix    
    np.fill_diagonal(H0,En)
    return H0

def H_s(basis,Fz=1,m=0,r_exp=1,step=0.005,rcore=0.65):
    """Calculate the off-diagonal matrix elements that make up the field hamiltonian Hs
    
    Inputs
    -------
    basis = 2D Array. Basis set of atomic states [n,l].
    Fz    = Electric field magnitude (V/m) [Default = 1V/m]
    m     = Projection of the orbital angular momentum quantum number. (Default = 0)
    r_exp = Exponent of r in matrix element for radial integral. (Default = 1).
    step  = Radial integration step size. (Default = 0.005).
    rcore = Minimum r value in numerov. (The default is 0.65 -> dipoole polarizability of He core).
    
    Returns
    -------
    Hs = Stark matrix.
    """
    
    # Initialise matrix
    size = len(basis)
    Hs = np.zeros((size,size))
    
    # Run over all possible states
    for i in tqdm(range(size)):
        ni = basis[i][0]
        li = basis[i][1]
        
        for j in range(size):
            nj = basis[j][0]        
            lj = basis[j][1]
            
            ## Conditions for calculation
            if i!=j: # off diagonal
                if abs(li-lj)==1: # dl must be 1
                    
                    ## Calculate angular integral
                    angint = ang_int(li,lj,m)
                    ## Calculate radial integral
                    radint = radial_integral(ni,li,nj,lj,r_exp,step,rcore)
                    ## Populate [i,j] element of matrix
                    Hs[i,j] = angint*radint
                    
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
    Eigvals = Correspond to the stark energies.
    Eigvecs = Correspond to the wavefunctions. 
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
