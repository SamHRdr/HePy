# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12th 2024

@author: S.H.Reeder

Functions required to calculate He energy levels etc.
"""
from .constants import R_He,c,alpha,h,e,a0,eps0,hbar
from numpy import pi, sqrt

## Binding energy and Quantum defects.
# Less correct function
def W(n,defect): 
    """Binding energy of the eigenstates of the Hamiltonian H0 associated 
    with a single Rydberg electron in the He atom in units of Hz.
    
    Inputs
    -------
    n      = principal quantum number.
    defect = quantum defect calcualted using Ritz expansion.
    """
    return -(R_He*c)/((n-defect)**2)

# More correct function
def W_tot(n,defect,wn=False):
    """Include relativistic and finite mass corrections to standard Rydberg formuala above.
    From Drake 1999 - For He atom only (units of Hz).
    
    Inputs
    -------
    n      = principal quantum number.
    defect = quantum defect calcualted using Ritz expansion.
    wn     = Return energy in wavenumber (1/cm)? [Default = False].
    """
    N = n-defect
    Term1     = 1/N**2
    Term2     = (3*alpha**2)/(4*n**4)
    Term3_num = 1+((5/6)*(alpha*2)**2)
    Term3     = (1.370745620e-4)**2 * (Term3_num/n**2)
    
    if wn == False:
        W_tot = R_He * c * (Term1-Term2+Term3)
    else:
        W_tot = 0.01 * (R_He  * (Term1-Term2+Term3))
    return -W_tot

def defect(n,L,J,S=1):
    """Recursive Ritz expasion giving the value of quantum defects, 
    where c0,c2,c4,c6 are calculated contants from Drake 1999.
    
    Inputs
    -------
    n = principal quantum number.
    L = orbital angualr momentum quantum number.
    J = Total angular momentum quantum number.
        Can take values |L-S| ≤ J ≤ L+S
    S = Spin quantum number. [Default = 1 (triplets)]
    """
    
    if (S!= 1 and S!=0):
        raise ValueError("S must be equal to 1 or 0.")
    if (J>(L+S) or J<abs(L-S)):
        raise ValueError("J can take values |L-S| ≤ J ≤ L+S.")
        
    ## Constants all taken from Drake Table VII.
    
    if S==1:
        ## Triplet states (S=1)
        if J==L+1:
            # In order [S, P, D, F, G, H, I]
            c0 = [ 0.29665648771, 0.06836028379, 0.002891328825, 0.00044737927, 0.00012714167, 0.000048729846, 0.000023047609]
            c2 = [ 0.038296666,  -0.018629228,  -0.0063577040,  -0.001739217 , -0.000796484 , -0.000433281,   -0.0002610672]
            c4 = [ 0.0075131,    -0.01233275,    0.00033670,     0.00010478,   -0.00000985,   -0.00000810,    -0.00000404]
            c6 = [-0.0045476,    -0.0079527,     0.0008395,      0.0000331,    -0.000019,      0,              0] 
        elif J==L:
            c0 = [0,  0.06835785765, 0.002890941493, 0.00044859483, 0.00012871316, 0.000049757614, 0.000023768483]
            c2 = [0, -0.018630462,  -0.0063571836,  -0.001727232,  -0.000796246,  -0.0004332274,  -0.0002610662]
            c4 = [0, -0.01233040,    0.00033777,     0.0001524,    -0.00001189,   -0.00000813,    -0.000004076]
            c6 = [0, -0.0079512,     0.0008392,     -0.0002486,    -0.0000141,     0,              0]
        elif J==L-1:
            c0 = [0,  0.06832800251, 0.002885580281, 0.00044486989, 0.00012570743, 0.000047797067, 0.000022390759]
            c2 = [0, -0.018641975,  -0.0063576012,  -0.001739275,  -0.000796498,  -0.0004332322,  -0.0002610680]
            c4 = [0, -0.01233165,    0.00033667,     0.00010476,   -0.00000980,   -0.00000807,    -0.000004042]
            c6 = [0, -0.0079515,     0.0008394,      0.0000337,    -0.000019,      0,              0]
    
    else:
        ## Singlet states (S=0)
        # In order [S, P, D, F, G, H, I]
        c0 = [ 0.13971806486, -0.012141803603, 0.002113378464, 0.00044029426, 0.000124734490, 0.000047100899, 0.000021868881]
        c2 = [ 0.027835737,    0.0075190804,  -0.0030900510,  -0.001689446,  -0.000796230,   -0.0004332277,  -0.0002610673]
        c4 = [ 0.01679229,     0.01397780,     0.00000827,    -0.0001183,    -0.00001205,    -0.00000814,    -0.000004048]
        c6 = [-0.0014590,      0.0048373,     -0.0003094,      0.000326,     -0.0000136,      0,              0]
        
    ## Recursive Ritz function
    if L <= 6:
        m = n - c0[L]
        defect = c0[L] + c2[L]*m**(-2.) + c4[L]*m**(-4.) + c6[L]*m**(-6.);
    else:
        defect = 0
    
    return defect

def trans(n1,l1,J1,n2,l2,J2,S=1):
    """Calcuate difference between two energies.
    Inputs
    -------
    n1(2) = principal quantum number of first(second) state.
    l1(2) = orbital angualr momentum quantum number of first(second) state.
    J1(2) = Total angular momentum quantum number of first(second) state.
    S     = Total spin quantum number (Default S=1 [triplets]).
    
    N.B. This is only really effective for high n. For transitions to or from
         n<10 I reccommend using the values from https://physics.nist.gov/PhysRefData/ASD/levels_form.html
    """
    En1 = W_tot(n1, defect(n1,l1,J1,S))
    En2 = W_tot(n2, defect(n2,l2,J2,S))
    
    return abs(En2-En1)

## Classical field ionisation for Helium.
def Fion_He_adibatic(n):
    """Calculates the electric field required to classically ionise 
    Rydberg atom of state n (From Gallagher), constants from NIST."""
    
    F0    = (2*R_He*h*c)/(e*a0) # adaptation from atomic units
    F_ion = (F0)/(16*n**4) #V/m
    return F_ion

def Inglis_Teller(n):
    """Calculate the inglis teller limit for a given n.
    
    Inputs
    -------
    n = principal quantum number
    
    Returns
    -------
    F_IT = Inglis teller limit (V/m).
    """
    F_IT = e / (12.* pi * eps0 * a0**2. * n**5.)
    
    return F_IT
    
