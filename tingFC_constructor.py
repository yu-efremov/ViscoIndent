# -*- coding: utf-8 -*-
"""
@author: Yuri Efremov
function tingFC_constructor takes parameters from ViscoIndent GUI and
construct force vs distance curve via ting_numerical function
most of the used parameters described in Ting_numerical.py
additional parameters:
    Pars['indpars'] - constructor for indentation history
    indpars[0]:  [yes/no] 0 - use constructor, 1 - use provided indentation history
    indpars[1]:  maximum indentation depth [nm]
    indpars[2]:  indentation speed [nm/s]
    indpars[3]:  number of points in approach phase
    indpars[4]:  [ramp/sin] - 0 - ramp (triangular indentation profile), 1 - sinusoidal indentation profile
    indpars[5]:  add dwell phase [s]
    Pars['noise'] noise level relative to median force [%] (random.normal)
    Pars['hydrodrag'] coefficient of viscous drag of the probe [nN*s/nm] 
"""
import numpy as np
from numpy import pi, tan, sin
import matplotlib.pyplot as plt

from Ting_numerical import ting_numerical


def tingFC_constructor(Pars, indentationfull):
    probe_geom = Pars['probe_shape']  # probe shape
    probe_size = Pars['probe_dimension']  # probe size
    Poisson = Pars['Poisson']          # Poisson's ratio of the sample
    dT = Pars['dT']                    # Sampling time
    modelting = Pars['viscomodel']  # relaxation function
    Height = Pars['height']  # sample thickness
    pars = Pars['Vpars']  # relaxation function parameters
    noise = Pars['noise']  # % noise level from median force
    hydrodrag = Pars['hydrodrag']  # [nN*s/nm] coefficient of viscous drag

    try:
        indpars = Pars['indpars']
    except:
        indpars = 0  # if no indentation parameters were provided, the "indentationfull" is used

    PointN = indentationfull.size
    time = np.linspace(dT, dT*PointN, PointN)
    time = time-dT
    MaxInd = np.argmax(indentationfull)
    Depth = indentationfull[MaxInd]
    Speed = Depth/time[MaxInd]

    # indpars: constructor for the indentation history
    if indpars[0] == 1:  # [yes/no; depth; speed; numpoints; ramp/sin; dwell points];
        Depth = indpars[1]  # [nm]
        Speed = indpars[2]  # [nm/s]
        MaxInd = int(indpars[3])  # number of points in approach-indentation
        dT2 = Depth/Speed/MaxInd  # calculate time step based on number of points
        dT = dT2  # suppress original dT
        PointN = MaxInd*2
        time = np.linspace(0, dT*(PointN-1), PointN)
        timeL = np.linspace(dT, dT*(PointN*2-1), PointN*2)
        ind_ramp = np.zeros([MaxInd*2])
        ind_rampL = np.zeros([MaxInd*4])  # extended indentation with noncontact region
        if indpars[4] == 0:  # ramp or sin
            for ii in range(MaxInd):
                ind_ramp[ii] = Speed*time[ii]

            for ii in range(MaxInd, MaxInd*2):
                ind_ramp[ii] = Speed*time[-1]-Speed*time[ii]

            for ii in range(2*MaxInd):
                ind_rampL[ii] = Speed*timeL[ii]-Speed*time[MaxInd]-Speed*dT

            for ii in range(2*MaxInd, MaxInd*4):
                ind_rampL[ii] = Speed*timeL[-1]-Speed*time[MaxInd]-Speed*timeL[ii]-Speed*dT

        elif indpars[4] == 1:  # sinuspidal indentation
            Freq = Speed/Depth/4
            ind_ramp = Depth*sin(2*pi*Freq*time)  # -1/4period
            ind_rampL = Depth*sin(2*pi*Freq*timeL-pi/2)  # -1/2period

        indentationfull = ind_ramp
        indentationfullL = ind_rampL

        if len(indpars) > 5:  # add dwell region
            Dwell_time = indpars[5]  # dwell time in seconds
            dwell_points = int(Dwell_time/dT)
            PointN = MaxInd*2 + dwell_points
            time = np.linspace(0, dT*(PointN-1), PointN)
            timeL = np.linspace(dT, dT*(PointN*2-1), PointN*2)
            inddwell = np.ones(dwell_points)*ind_ramp[MaxInd]
            indentationfull = np.concatenate((ind_ramp[0:MaxInd+1], inddwell, ind_ramp[MaxInd:-1]))
            indentationfullL = np.concatenate((ind_rampL[0:2*MaxInd+1], inddwell, ind_rampL[2*MaxInd:-1]))
            MaxInd = MaxInd + dwell_points
        else:
            dwell_points = 0

    # launch ting_numerical to construct force
    [force, cradius, contact_time, t1_ndx2, Et] = ting_numerical(pars, Pars['adhesion'],
        Poisson, probe_size, dT, MaxInd, Height, modelting, probe_geom,
        indentationfull)[0:5]

    # add noise
    force = force + np.random.normal(0, abs(0.01*noise*np.median(force)), len(force))

    # force for the extended indentation with non-contact region
    # forceL = np.pad(force, MaxInd - dwell_points, 'constant', constant_values=(0, 0))
    forcePadL = np.zeros(MaxInd - dwell_points) + np.random.normal(0, abs(0.01*noise*np.median(force)), MaxInd - dwell_points)
    forcePadR = np.zeros(MaxInd - dwell_points) + np.random.normal(0, abs(0.01*noise*np.median(force)), MaxInd - dwell_points)
    forceL = np.concatenate((forcePadL, force, forcePadR), axis=0)

    # add hydrodynamic drag
    speedfull = np.diff(indentationfullL)/dT
    speedfull = np.append(speedfull, speedfull[-1])
    drag = speedfull*hydrodrag
    forceL = forceL + drag

    return time, indentationfull, force, cradius, indentationfullL, forceL


if __name__ == '__main__':
    import time as mtime
    Pars = {}
    Pars['probe_shape'] = 'sphere'
    Pars['probe_dimension'] = 5000
    Pars['Poisson'] = 0.5         # Poisson's ratio of the sample
    Pars['dT'] = 1e-3                 # Sampling time (time step)

    Pars['height'] = 0
    Pars['indpars'] = np.array([1, 50, 500, 500, 0, 2])  # [yes/no; depth; speed; numpoimts; ramp/sin; dwell_time];

    Pars['viscomodel'] = 'sPLR'
    Pars['Vpars'] = np.array([1000, 0.1, 0, 0])
    Pars['noise'] = 3  # % noise level from median force
    Pars['hydrodrag'] = 1e-5  # [nN*s/nm] coefficient of viscous drag
    # Pars['viscomodel'] = 'dSLS'
    # Pars['Vpars'] = np.array([1000, 0.4, 1, 0.01, 0])

    indentationfull = np.array([0, 1])  # arbitrary indentation profile
    t1 = mtime.time()
    time, indentationfull, force, cradius, indentationfullL, forceL = tingFC_constructor(Pars, indentationfull)
    t2 = mtime.time()
    print(t2-t1)
    # curvedata = [time, indentationfull, force, cradius, indentationfullL, forceL]
    # arr = np.vstack([curvedata[0], curvedata[1], curvedata[2]])
    # arr.transpose()
    plt.figure(num='Force vs Indentation')
    plt.plot(indentationfull, force)
    plt.xlabel('Indentation')
    plt.ylabel('Force')
    plt.figure(num='Contact Radius vs Time')
    plt.plot(time, cradius)
    plt.xlabel('Time')
    plt.ylabel('Contact radius')
    plt.figure(num='Simulated Indentation')
    plt.plot(indentationfullL, forceL)
    plt.xlabel('Indentation')
    plt.ylabel('Force')
    plt.figure(num='Simulated Force vs Time')
    timeL = np.arange(0, len(forceL))*Pars['dT']
    plt.plot(timeL, forceL)
    plt.xlabel('Time')
    plt.ylabel('Force')
