# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:52:06 2020

@author: Yuri
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, tan, sin

from Ting_numerical import ting_numerical
from relaxation_functions import relaxation_function

def tingFC_constructor(Pars, indentationfull):
    modelprobe = Pars['probe_shape']
    probe_dim = Pars['probe_dimension']
    Poisson = Pars['Poisson']          # Poisson's ratio of the sample
    dT = Pars['dT']                    # Sampling time
    modelting = Pars['viscomodel']
    Height = Pars['height']
    pars = Pars['Vpars']
    
    try:
        indpars = Pars['indpars']
    except:
        indpars = 0
        

    PointN = indentationfull.size
    time = np.linspace(dT,dT*PointN, PointN)
    time = time-dT
    MaxInd = np.argmax(indentationfull)
    Depth = indentationfull[MaxInd]
    Speed = Depth/time[MaxInd]

    if indpars[0]==1: # [yes/no; depth; speed; numpoimts; ramp/sin];
        Depth=indpars[1]  # [nm]
        Speed=indpars[2]  # [nm/s]
        MaxInd=indpars[3]  # num points in approach-indentation
        dT2=Depth/Speed/MaxInd
        dT = dT2  # suppress original dT
        PointN = MaxInd*2
        time = np.linspace(0,dT*(PointN-1), PointN)
        timeL = np.linspace(dT,dT*(PointN*2-1), PointN*2)
        ind_ramp = np.zeros([MaxInd*2])
        ind_rampL = np.zeros([MaxInd*4])
        if indpars[4]==0: # ramp or sin
            for ii in range(MaxInd):
                ind_ramp[ii] = Speed*time[ii]

            for ii in range(MaxInd, MaxInd*2):
                ind_ramp[ii] = Speed*time[-1]-Speed*time[ii]
                
            for ii in range(2*MaxInd):
                ind_rampL[ii] = Speed*timeL[ii]-Speed*time[MaxInd]-Speed*dT

            for ii in range(2*MaxInd, MaxInd*4):
                ind_rampL[ii] = Speed*timeL[-1]-Speed*time[MaxInd]-Speed*timeL[ii]-Speed*dT;
                
        elif indpars[4]==1:
            Freq=Speed/Depth/4
            ind_ramp=Depth*sin(2*pi*Freq*time) # -1/4period
            ind_rampL=Depth*sin(2*pi*Freq*timeL-pi/2)  # -1/2period
        
        indentationfull=ind_ramp
        indentationfullL=ind_rampL
        
    [force, cradius, contact_time, t1_ndx2, Et] = ting_numerical(pars, Poisson, probe_dim, dT, MaxInd, Height, modelting, modelprobe, indentationfull)[0:5]

    forceL = np.pad(force, MaxInd, 'constant', constant_values=(0, 0))
    
    return time, indentationfull, force, cradius, indentationfullL, forceL
    
    
    
    
    
if __name__ == '__main__':
    Pars={}
    Pars['probe_shape'] = 'sphere'
    Pars['probe_dimension'] = 5000
    Pars['Poisson'] = 0.5         # Poisson's ratio of the sample
    Pars['dT'] = 1e-3                 # Sampling time (time step)

    Pars['height'] = 0
    Pars['indpars'] = np.array([1, 50, 50, 1000, 1])
    
    Pars['viscomodel'] = 'sPLR'
    Pars['Vpars'] = np.array([1000, 0.8, 0, 20])
    indentationfull = np.array([0, 1])
    time, indentationfull, force, cradius, indentationfullL, forceL = tingFC_constructor(Pars, indentationfull)
    plt.figure(num='Force vs Indentation')
    plt.plot(indentationfull, force)
    plt.xlabel('Indentation')
    plt.ylabel('Force')
    plt.figure(2)
    plt.plot(time, cradius)
    plt.figure(3)
    plt.plot(indentationfullL, forceL)