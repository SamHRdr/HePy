# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:32:28 2024
@author: sam_r

Constants for use in He calculations. 
Values taken from NIST database. 
CODATA 2018 DOI:https://doi.org/10.1103/RevModPhys.93.025010
"""
c     = 299792458 # Speed of light in a vacuum (m/s)
h     = 6.62607015e-34 # Planck's constant (J/Hz)
e     = 1.602176634e-19 # elementry charge (C)
alpha = 7.2973525693e-3 # Fine-structure constant 
a0    = 5.29177210903e-11 # Bohr radius (m)
R_inf = 109737.31568160*100 # Rydberg constant (m-1)
amu   = 1.66053906660e-27 # atomic mass (kg)
m_He  = 4.00260*amu # Helium atom mass (kg)
m_e   = 9.1093837015e-31 # Mass of the electron (kg)
k     = 1.380649e-23 # Boltzmann constant (J/k)
eps0  = 8.8541878128e-12 # Vacuum electric permittivity (F/m)

# Derived constants
mu_He = (m_He-m_e)*m_e/m_He # reduced mass He
R_He  = R_inf*mu_He/m_e # Rydberg constant for He

