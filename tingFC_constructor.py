# -*- coding: utf-8 -*-
"""
@author: Yuri Efremov
"""
import numpy as np
from numpy import pi, tan, sin
import matplotlib.pyplot as plt

from Ting_numerical import ting_numerical


def tingFC_constructor(Pars, indentationfull):
    modelprobe = Pars['probe_shape']
    probe_dim = Pars['probe_dimension']
    Poisson = Pars['Poisson']          # Poisson's ratio of the sample
    dT = Pars['dT']                    # Sampling time
    modelting = Pars['viscomodel']
    Height = Pars['height']
    pars = Pars['Vpars']
    noise = Pars['noise']  # % noise level from median force
    hydrodrag = Pars['hydrodrag']  # [nN*s/nm] coefficient of viscous drag

    try:
        indpars = Pars['indpars']
    except:
        indpars = 0

    PointN = indentationfull.size
    time = np.linspace(dT, dT*PointN, PointN)
    time = time-dT
    MaxInd = np.argmax(indentationfull)
    Depth = indentationfull[MaxInd]
    Speed = Depth/time[MaxInd]

    if indpars[0] == 1:  # [yes/no; depth; speed; numpoimts; ramp/sin];
        Depth = indpars[1]  # [nm]
        Speed = indpars[2]  # [nm/s]
        MaxInd = int(indpars[3])  # num points in approach-indentation
        dT2 = Depth/Speed/MaxInd
        dT = dT2  # suppress original dT
        PointN = MaxInd*2
        time = np.linspace(0, dT*(PointN-1), PointN)
        timeL = np.linspace(dT, dT*(PointN*2-1), PointN*2)
        ind_ramp = np.zeros([MaxInd*2])
        ind_rampL = np.zeros([MaxInd*4])
        if indpars[4] == 0:  # ramp or sin
            for ii in range(MaxInd):
                ind_ramp[ii] = Speed*time[ii]

            for ii in range(MaxInd, MaxInd*2):
                ind_ramp[ii] = Speed*time[-1]-Speed*time[ii]

            for ii in range(2*MaxInd):
                ind_rampL[ii] = Speed*timeL[ii]-Speed*time[MaxInd]-Speed*dT

            for ii in range(2*MaxInd, MaxInd*4):
                ind_rampL[ii] = Speed*timeL[-1]-Speed*time[MaxInd]-Speed*timeL[ii]-Speed*dT

        elif indpars[4] == 1:
            Freq = Speed/Depth/4
            ind_ramp = Depth*sin(2*pi*Freq*time)  # -1/4period
            ind_rampL = Depth*sin(2*pi*Freq*timeL-pi/2)  # -1/2period

        indentationfull = ind_ramp
        indentationfullL = ind_rampL

        if len(indpars) > 5:  # add dwell region
            Dwell_time = indpars[5]  # in seconds
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

    [force, cradius, contact_time, t1_ndx2, Et] = ting_numerical(pars,
        Poisson, probe_dim, dT, MaxInd, Height, modelting, modelprobe,
        indentationfull)[0:5]

    force = force + np.random.normal(0, 0.01*noise*np.median(force), len(force))

    # forceL = np.pad(force, MaxInd - dwell_points, 'constant', constant_values=(0, 0))
    forcePadL = np.zeros(MaxInd - dwell_points) + np.random.normal(0, 0.01*noise*np.median(force), MaxInd - dwell_points)
    forcePadR = np.zeros(MaxInd - dwell_points) + np.random.normal(0, 0.01*noise*np.median(force), MaxInd - dwell_points)
    forceL = np.concatenate((forcePadL, force, forcePadR), axis=0)

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
    plt.plot(time, force)
    plt.xlabel('Time')
    plt.ylabel('Force')
