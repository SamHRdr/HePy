# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:38 2024

@author: S.H.Reeder

Based on Appendix A of Zimmerman et. al (1979). 

Adapted from S.D.Hogan Matlab implementation.
"""
import numpy as np
from numba import jit

# Define g_x_1 and g_x_2 functions with JIT compilation
@jit(nopython=True)
def g_x_1(ri, ns, ls):
    W0 = -1. / (2. * ns[0]**2.) # Energy
    return 2. * np.exp(2. * np.log(ri)) * (-1. / ri - W0) + (ls[0] + 0.5)**2.

@jit(nopython=True)
def g_x_2(ri, ns, ls):
    W1 = -1. / (2. * ns[1]**2.) # Energy
    return 2. * np.exp(2. * np.log(ri)) * (-1. / ri - W1) + (ls[1] + 0.5)**2.

# Define the radial_integral function with JIT compilation
# For comparing two states
@jit(nopython=True)
def radial_integral(n1, l1, n2, l2, r_exp=1, step=0.0065, rcore=0.65):
    """Function to calculate radial integral using the Numerov method. Based on 
    Appendix A of Zimmerman et. al (1979). 
    
    Inputs
    ----------
    n     = Principal quantum number.
    l     = Azimuthal quantum number.
    r_exp = Exponent of r in matrix element. (Default = 1).
    step  = Integration step size. (Default = 0.0065).
    rcore = minimum r value. (The default is 0.65 -> dipole (polarizability)^(1/3) of He core).
            [For hydrogen rcore=0.05]

    Returns
    -------
    MM_elem = Radial matrix elements.

    """
    # Ensure correct order of states
    if l1 >= n1 or l2 > n2:
        raise ValueError("Error: Azimuthal quantum number should be less than principal quantum number!")

    # Ensure n1 < n2
    ns, ls = np.array([n1,n2]), np.array([l1,l2])
    arg = ns.argsort()[::-1]
    ns, ls = ns[arg], ls[arg]

    # Starting points
    rs_1 = 2. * ns[0] * (ns[0] + 15.)
    rs_2 = 2. * ns[1] * (ns[1] + 15.)
    r1_0 = r2_0 = rs_1
    Y1_0 = Y2_0 = 1e-10

    # Starting values
    r1_1 = r2_1 = rs_1 * np.exp(-step)
    Y1_1 = Y1_0 * (1. + step * np.sqrt(g_x_1(rs_1, ns, ls)))
    Y2_1 = Y2_0 * (1. + step * np.sqrt(g_x_2(rs_2, ns, ls)))
    
    # End points
    # rcore
    r_in_1 = ns[0]**2. - ns[0] * np.sqrt(ns[0]**2. - ls[0]*(ls[0]+1.))
    r_in_2 = ns[1]**2. - ns[1] * np.sqrt(ns[1]**2. - ls[1]*(ls[1]+1.))

    # Begin Numerov approach
    rvals1 = [r1_0, r1_1]
    Yvals1 = [Y1_0, Y1_1]
    rvals2 = [r2_0, r2_1]
    Yvals2 = [Y2_0, Y2_1]

    hsq = step**2.

    i = 2 # Starts at second integration (third point)
    j = 2

    # Integration for n1 ([0])
    while r1_1 > rcore:
        r1_1 = rs_1 * np.exp(-i * step)
        # Calculate g for i-1, i and i+1 (remember python indexes from 0)
        g1 = [g_x_1(rvals1[i-2], ns, ls), g_x_1(rvals1[i-1], ns, ls), g_x_1(r1_1, ns, ls)]
        Y1 = ((Yvals1[i-2] * (g1[0] - (12. / hsq)) + Yvals1[i-1] * (10. * g1[1] + (24. / hsq))) /
              ((12. / hsq) - g1[2]))
        
        ## Check for divergence
        if r1_1 < r_in_1:
            dY = abs((Y1-Yvals1[i-1]) / Yvals1[i-1])
            dr = (r1_1**(-ls[0]-1) - rvals1[i-1]**(-ls[0]-1)) / (rvals1[i-1]**(-ls[0]-1))
            if dY > dr:
                break

        # Store values
        rvals1.append(r1_1)
        Yvals1.append(Y1)

        i += 1

    idx = int(np.where(np.abs(np.array(rvals1) - rs_2) == min(np.abs(np.array(rvals1) - rs_2)))[0][0])

    # Update values for n2 ([1])
    r2_0 = rvals1[idx]
    r2_1 = rs_1 * np.exp(-idx * step)
    Y2_0 = 1e-10
    Y2_1 = Y2_0 * (1. + step * np.sqrt(g_x_2(r2_0, ns, ls)))

    rvals2 = [r2_0, r2_1]
    Yvals2 = [Y2_0, Y2_1]

    # Integration for n2
    while r2_1 > rcore:
        r2_1 = rs_1 * np.exp(-(idx + j - 1) * step)
        g2 = [g_x_2(rvals2[j-2], ns, ls), g_x_2(rvals2[j-1], ns, ls), g_x_2(r2_1, ns, ls)]
        Y2 = ((Yvals2[j-2] * (g2[0] - 12. / hsq) + Yvals2[j-1] * (10. * g2[1] + 24. / hsq)) /
              (12. / hsq - g2[2]))
        
        ## Check for divergence
        if r2_1 < r_in_2:
            dY = abs((Y2-Yvals2[j-1])/Yvals2[j-1])
            dr = (r2_1**(-ls[1]-1)-rvals2[j-1]**(-ls[1]-1))/(rvals2[j-1]**(-ls[1]-1))
            if dY > dr:
                break

        # Store values
        rvals2.append(r2_1)
        Yvals2.append(Y2)

        j += 1

    # Line up lists of values
    I1, I2 = [], []
    for s in range(len(rvals1)):
        for p in range(len(rvals2)):
            if rvals1[s] == rvals2[p]:
                I1.append(s)
                I2.append(p)
                break

    # Convert to arrays for final sum
    Y1_arr = np.array(Yvals1)
    Y2_arr = np.array(Yvals2)
    r1_arr = np.array(rvals1)
    r2_arr = np.array(rvals2)

    YY1 = Y1_arr[np.array(I1)]
    YY2 = Y2_arr[np.array(I2)]
    rr2 = r2_arr[np.array(I2)]

    # Calculate radial matrix element
    MM_elem = np.sum((YY1 * YY2 * rr2**(2. + r_exp)) /
                     np.sqrt(np.sum((Y1_arr**2.) * (r1_arr)**2.) * np.sum((Y2_arr**2.) * (r2_arr)**2.)))

    return MM_elem
