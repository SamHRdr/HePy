# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:22:51 2023

@author: S.H.Reeder

Based on Appendix A of Zimmerman et. al (1979). 

Thanks to S.D.Hogan for showing me his Matlab implementation.
"""
import numpy as np
from math import exp, log, sqrt


# For comparing two states - this is how the Stark states are calculated
def radial_integral(n1,l1,n2,l2,r_exp=1,h=0.005,rcore=0.65):
    """Function to calculate radial integral using the Numerov method. Based on 
    Appendix A of Zimmerman et. al (1979). 
    
    Inputs
    ----------
    n     = Principal quantum number.
    l     = Azimuthal quantum number.
    r_exp = Exponent of r in matrix element. (Default = 1).
    h     = Integration step size. (Default = 0.005).
    rcore = minimum r value. (The default is 0.65 -> dipoole polarizability of He core).

    Returns
    -------
    MM_elem = Radial matrix elements.

    """
    ## Make sure states are the right way around
    if l1>=n1 or l2>n2:
        raise ValueError("Error: Azimuthal quantum number should be "
                         "less than principal quantum number!")
        
        
    # If I use arrays I can do perform operations on lots of number at once
    ns, ls = np.array([n1,n2]), np.array([l1,l2])
    if n1 < n2:
        ns, ls = np.array([ns[1],ns[0]]) , np.array([ls[1],ls[0]])
    
    
    # Energies
    W0 = -1./(2.*ns[0]**2.)
    W1 = -1./(2.*ns[1]**2.)
    
    # Functions to calculate g(x)
    g_x_1 = lambda ri: 2.*exp(2.*log(ri)) * (-1./ri - W0) + (ls[0]+0.5)**2.
    g_x_2 = lambda ri: 2.*exp(2.*log(ri)) * (-1./ri - W1) + (ls[1]+0.5)**2.
    
    ## Starting points
    # Y_{-1}
    r1_0 = rs_1 = 2.*ns[0]*(ns[0]+15.) # first r, far from nucleus
    r2_0 = rs_2 = 2.*ns[1]*(ns[1]+15.)
    Y1_0 = 1e-10
    Y2_0 = 1e-10
    
    # Y_0
    r1 = r1_1 = rs_1 * exp(-h)
    r2 = r2_1 = rs_2 * exp(-h)
    Y1_1 = Y1_0 * (1.+ h * sqrt(g_x_1(rs_1)))
    Y2_1 = Y2_0 * (1.+ h * sqrt(g_x_2(rs_2)))
    
    ## End points
    #rcore
    r_in_1 = ns[0]**2. - ns[0] * sqrt(ns[0]**2. - ls[0]*(ls[0]+1.))
    r_in_2 = ns[1]**2. - ns[1] * sqrt(ns[1]**2. - ls[1]*(ls[1]+1.))
    
    ## Begin Numerov approach
    rvals1 = [r1_0,r1_1]
    Yvals1 = [Y1_0,Y1_1]
    
    rvals2 = [r2_0,r2_1]
    Yvals2 = [Y2_0,Y2_1]
    
    hsq = h**2.
    
    i=2 # starts at second integration (third point)
    j=2
    
    # While loop for each n 
    # n1 ([0])
    while r1 > rcore:
        # Next step
        r1 = rs_1*exp(-i*h)
        g1 = [g_x_1(rvals1[i-2]),g_x_1(rvals1[i-1]),g_x_1(r1)] # calculate g for i-1, i and i+1 (remember python indexes from 0)
        Y1 = ((Yvals1[i-2] * (g1[0]-(12./hsq)) + Yvals1[i-1] * (10.*g1[1]+(24./hsq)))/
             ((12./hsq)-g1[2]))
    
        ## Check for divergence
        if r1 < r_in_1:
            dY = abs((Y1-Yvals1[i-1]) / Yvals1[i-1])
            dr = (r1**(-ls[0]-1) - rvals1[i-1]**(-ls[0]-1)) / (rvals1[i-1]**(-ls[0]-1))
            if dY > dr:
                break
    
        # Store values
        rvals1.append(r1)
        Yvals1.append(Y1)
    
        i+=1
    
    idx = np.where(abs(rvals1-rs_2)==min(abs(rvals1-rs_2)))[0][0]
    
    # update values for n2
    r2_0   = rvals1[idx]
    r2     = r2_1 = rs_1*exp(-idx*h)
    Y2_0   = 1e-10
    Y2_1   = Y2_0 * (1.+ h * sqrt(g_x_2(r2_0)))
    
    rvals2 = [r2_0,r2_1]
    Yvals2 = [Y2_0,Y2_1]
    
    # n2 ([1])
    while r2 > rcore:
        # Next step
        r2 = rs_1*exp(-(idx+j-1)*h)
        g2 = [g_x_2(rvals2[j-2]),g_x_2(rvals2[j-1]),g_x_2(r2)] # calculate g for i-1, i and i+1 (remember python starts at 0)
        Y2 = ((Yvals2[j-2] * (g2[0]-12./hsq) + Yvals2[j-1] * (10.*g2[1]+24./hsq))/
             (12./hsq-g2[2]))
    
        ## Check for divergence
        if r2 < r_in_2:
            dY = abs((Y2-Yvals2[j-1])/Yvals2[j-1])
            dr = (r2**(-ls[1]-1)-rvals2[j-1]**(-ls[1]-1))/(rvals2[j-1]**(-ls[1]-1))
            if dY > dr:
                break
    
        # Store values
        rvals2.append(r2)
        Yvals2.append(Y2)
    
        j+=1  
    
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
    
    YY1 = Y1_arr[I1] 
    YY2 = Y2_arr[I2]
    rr2 = r2_arr[I2]
    
    # Calculate radial matrix element
    MM_elem = np.sum(
       (YY1 * YY2 * rr2**(2.+r_exp))/sqrt(np.sum((Y1_arr**2.)*(r1_arr)**2.) * np.sum((Y2_arr**2.)*(r2_arr**2.)))
        )
    
    return (MM_elem)
