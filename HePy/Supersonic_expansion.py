# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 20:27:37 2023

@author: sam_r
"""
import numpy as np
import matplotlib.pyplot as plt
from .constants import k

#---------------------------------------- Effusive Beams --------------------------------------------------------
def v_w(T1,m):
    '''Calculate most probable velocity from inital temperature and mass.
    Inputs:
    -------
    T1 = Initial temperature (K).
    m  = atomic mass (kg).
    
    Returns:
    -------
    Most probable velocity (m/s).
    '''
    vw = np.sqrt((2*k*T1)/m)
    return vw

def f_eff(m,v,T):
    '''Effusive beam speed probability distribution.
    Inputs:
    -------
    m          = atomic mass (kg).
    v (array)  = speeds (m/s).
    T1         = Initial temperature (K).
    
    Returns:
    -------
    Speed probability distribution (array).
    '''
    a = 4/np.sqrt(np.pi)
    b = (m/2*k*T)**(3/2)
    c = (m*v**2)/(2*k*T)
    f = a*b * v**2 * np.exp(-c)
    return f

#---------------------------------------- Supersonic Beams --------------------------------------------------
def T_2(P1,P2,T1,gamma):
    '''Calculate gas temperature after the nozzle.
    Inputs:
    -------
    P1    = reservoir pressure (Pa).
    P2    = pressure after nozzle (Pa).
    T1    = Initial temperature (K).
    gamma = Contstant volume/pressure specific heat ratio.
    
    Returns:
    -------
    Temperature of the supersonic expansion after the nozzle.
    '''
    T2 = np.power((P2/P1),((gamma-1)/gamma))*T1
    return T2

def vmax_est(T1,m,gamma):
    '''Estimate maxium longitudinal speed if T2=P2=0.
    Inputs:
    -------
    T1    = Initial temperature (K).
    m     = atomic mass (kg).
    gamma = Contstant volume/pressure specific heat ratio.
        
    Returns:
    -------
    Maximum possible longitudinal speed (m/s).
    '''
    vw = np.sqrt(2*k*T1/m)
    vmax = vw*np.sqrt(gamma/(gamma-1))
    return vmax

def v_2(P1,P2,gamma,vw):
    '''Average longitudinal velocity.
    Inputs:
    -------
    P1    = reservoir pressure (Pa).
    P2    = pressure after nozzle (Pa).
    gamma = Contstant volume/pressure specific heat ratio.
    vw    = Most probable velocity (m/s).
    
    Returns:
    Mean lognitudinal vlocity of the beam (m/s)
    -------
    
    '''
    a = gamma/(gamma-1)
    b = P2/P1
    c = (gamma-1)/gamma
    v2 = vw*np.sqrt(a*(1-b**c))
    return v2

def f(m,v,v2,T,C=1):
    '''Probability distribution of longitudinal velocities for supersonic beam.
    Inputs:
    -------
    m         = atomic mass (kg).
    v (array) = speeds (m/s).
    v2        = Mean lognitudinal vlocity of the beam (m/s)
    T         = Initial temperature (K).
    C         = Normalisation constant (default=1).
    
    Returns:
    -------
    Velocity probability distribution (array).
    '''
    vw = np.sqrt(2*k*T/m) # define vw again for ease
    a = (v/vw)**3
    b = (m*(v-v2)**2)/(2*k*T)
    f = C * a * np.exp(-b)
    return f

def M_F(P1,P2,gamma):
    '''Calculate mach number after the nozzle.
    Inputs:
    -------
    P1    = reservoir pressure (Pa).
    P2    = pressure after nozzle (Pa).
    gamma = Contstant volume/pressure specific heat ratio.
    
    Returns:
    -------
    Mach number.
    '''
    a = (2/(gamma-1))
    b = np.power((P1/P2),((gamma-1)/gamma))
    M_F = np.sqrt(a*(b-1))
    return M_F

def v_2_mach(T1,m,gamma,MF):
    '''Average velocity in the longitudinal direction as a function of Mach number.
    Inputs:
    -------
    T1    = Initial temperature (K).
    m     = atomic mass (kg).
    gamma = Contstant volume/pressure specific heat ratio.
    M_F   = Mach number.
    
    Returns:
    -------
    Mean lognitudinal vlocity of the beam (m/s)
    '''
    a = (k*T1)/m
    b = (gamma-1)/2
    v2 = MF * np.sqrt((gamma*a)/(1+b*MF**2))
    return v2

#---------------------------------------- z dependent functions --------------------------------------------------

def M_z(A,B,C,D,z,gamma):
    '''Mach number at distance z from valve with diammeter D.
    Inputs:
    -------
    A,B,C = Constants taken from Hogan Thesis (2014).
    D     = Dimmeter of exit aperture (m).
    z     = distance from aperture (m).
    gamma = Contstant volume/pressure specific heat ratio.
    
    Returns:
    -------
    Mz = z dependent Mach number.
    '''
    Mz = A*((z/D)-B)**(gamma-1) - C*((z/D)-B)**(1-gamma)
    return Mz

def T_z(T0,gamma,Mz):
    '''Beam temperature at distance z from the valve.
    Inputs:
    -------
    T0    = Initial temperature (K).
    gamma = Contstant volume/pressure specific heat ratio.
    Mz    = z dependent Mach number.
    
    Returns:
    -------
    Tz = z dependent beam temperature (K).
    '''
    a = 1 + 0.5*(gamma-1)*Mz**2
    Tz = T0/a
    return Tz
 
def v_z(Mz,gamma,m,Tz):
    '''Mean longitudinal velocity.
    Inputs:
    -------
    gamma = Contstant volume/pressure specific heat ratio.
    Mz    = z dependent Mach number.
    Tz    = z dependent beam temperature (K).
    m     = atomic mass (kg).
 
    Returns:
    -------
    vz = z dependent longitudinal velocity (m/s).
    '''
    from scipy.constants import k
    vz = Mz*np.sqrt((gamma*k*Tz)/m)
    return vz

def f_hog(m,v,vz,Tz):
    '''Beam velocity distribution from Hogan (2012).
    Inputs:
    -------
    Tz    = z dependent beam temperature (K).
    m     = atomic mass (kg).
    vz    = z dependent longitudinal velocity (m/s).
    v     = 1D array of velocities (m/s). 
    
    Returns:
    -------
    f = probability distribution of velocities.
    '''
    from scipy.constants import k
    a = m*(v-vz)**2
    b = 2*k*Tz
    f = v**2 * np.exp(-a/b)
    return f

#------------------------------------ Composite supersonic mega function -----------------------------------------------
def f_mSS(m,v,P1,P2,T1,gamma):
    '''Probability distribution of longitudinal velocities for supersonic beam.
    Inputs:
    -------
    m     = atomic mass (kg).
    v     = 1D array of velocities (m/s). 
    P1    = reservoir pressure (Pa).
    P2    = pressure after nozzle (Pa).
    T1    = Initial temperature (K).
    gamma = Contstant volume/pressure specific heat ratio.
    
    Returns:
    -------
    f = probability distribution of velocities.
    '''
    T2 = np.power((P2/P1),((gamma-1)/gamma))*T1 # temp after nozzle
    vw = np.sqrt(2*k*T1/m) # KE = 2kT
    v2 = vw*np.sqrt((gamma/(gamma-1))*(1-(P2/P1)**((gamma-1)/gamma))) # Av. velocity after nozzle
    
    a = (v/vw)**3
    b = (m*(v-v2)**2)/(2*k*T2)
    f = a * np.exp(-b)
    return f


#---------------- Test Function with this--------------------------
# m_He = 6.6464731e-27 #Helium mass (kg)
# T1 = 372 # Valve temperature (K)
# P1 = 3e5 # Stagnation (reservoir) pressure (Pa) - 3bar
# P2 = 1e-2 # Chamber pressure (Pa)
# gamma = 5/3 # for monoatomic gas

# vs = np.arange(0,3001,1) # range of speeds to feed into the equations
# fm = f_mSS(m_He,vs,P1,P2,T1,gamma)

# plt.plot(vs,fm/max(fm),label='Supersonic vel. dist')
# plt.xlim(1800,2150)
# plt.xlabel('Speed (m/s)',fontsize=13)
# plt.ylabel('Normalised Probability',fontsize=13)
# plt.title(r'$T$='+str(T1)+'K, $P_{stag}$='+str('%.0f'%(P1*1e-5))+'bar',
#           fontsize=13)
# plt.legend()
