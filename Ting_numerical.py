# -*- coding: utf-8 -*-
"""
numerical calculation of the force using the Ting's equations
based on the provided indentation history and parameters
of the probe and material
Probe parameters:
    probe_geom = sphere, cone, cylinder, pyramid, cylinder  (geomerty)
    probe_size = probe dimension, probe_size of sphere/cylinder, angle of the cone/pyramid
Material parameters:
    Possion = Poisson's ratio
    viscomodel = relaxation function
    par = parameters of the relaxation function (viscoelastic parameters)
    par[0] = EHertz in elastic, E0 in SLS, E1 in PLR model
    par[1] = tau in SLS, alpha in PLR model
    par[2] = Einf in SLS
    ... see full description in the relaxation_functions.py
    Height = thickness of the sample (0=inf)
Indentation history parameters:
    indentationfull = pointwise indentation history
    dT = sampling time
    MaxInd = point of maximum indentation (maximum contact area)
    
"""

import numpy as np
from numpy import pi, tan
import matplotlib.pyplot as plt
# import scipy
# from scipy.integrate import simps # slower, without substantial improvement
from bottom_effect_correction import bottom_effect_correction
from relaxation_functions import relaxation_function
from scipy.optimize import fsolve

def ting_numerical(par, adhesion_pars, Poisson, probe_size, dT, MaxInd, Height, viscomodel, probe_geom, indentationfull):
    PointN = len(indentationfull)
    time = np.linspace(0, dT*(PointN-1), PointN)

    # Et construction
    [Et, eta] = relaxation_function(par, viscomodel, time)[0:2]
    # plt.plot(time, Et)
    # remove zero-time singularity if needed
    if np.isinf(Et[0]):
        Et[0] = 2*Et[1]
    if np.isnan(Et[0]):
        Et[0] = Et[1]+(Et[1]-Et[2])
        
    adhesion_model = adhesion_pars[0]
    adhesion_region = adhesion_pars[1]
    adhesion_force = adhesion_pars[2]

    if Height > 0:  # bottom effect correction
        [BEC, BECspeed] = bottom_effect_correction(Poisson, probe_size, Height, probe_geom, indentationfull)
    else:
        BEC = np.ones(len(indentationfull))
        BECspeed = np.ones(len(indentationfull))

    if probe_geom == 'sphere':
        power = 1.5
        K1 = 4*probe_size**0.5/3
        K12 = probe_size**0.5
    elif probe_geom == 'cone' or probe_geom == 'pyramid':
        power = 2
        if probe_geom == 'cone':
            K1 = 2/pi*tan(probe_size*pi/180)
        elif probe_geom == 'pyramid':
            K1 = 1.406/2*tan(probe_size*pi/180)
        K12 = K1
    elif probe_geom == 'cylinder': # cylinder
        power = 1
        K1 = 2*probe_size
        K12 = probe_size
    K1 = K1/(1-Poisson**2)*1e-9  # compensate nm

    # indentation history
    ind2speed = np.diff(indentationfull**power)/dT
    ind2speed = np.append(ind2speed, ind2speed[-1])
    ind2speed = smoothM(ind2speed, 5)
    indspeed = np.diff(indentationfull)/dT
    indspeed = np.append(indspeed, indspeed[-1])
    indspeed = smoothM(indspeed, 5)
    # plt.plot(indspeed)

    ForceR = np.zeros(len(indentationfull))  # Lee_Radok force prediction
    ForceT = np.zeros(len(indentationfull))  # Ting's force prediction

    # for i in range (1, PointN-1):  # MaxInd or PointN-1
    #     ndx = np.asarray(range (0, i+1))  # integration limits, dummy variable
    #     ForceR[i]= K1 * (BEC[i]*np.trapz(Et[i-ndx]*ind2speed[ndx], time[ndx]) + 
    #                      power*eta*indentationfull[i]**(power-1)*indspeed[i]*BECspeed[i])

    for i in range(1, PointN-1):  # MaxInd or PointN-1
        ForceR[i] = K1 * (BEC[i]*np.trapz(Et[i::-1]*ind2speed[:i+1], dx=dT) + power*eta*indentationfull[i]**(power-1)*indspeed[i]*BECspeed[i])

    ForceT[:MaxInd]=ForceR[:MaxInd]
    # plt.plot(indentationfull,ForceR)

    cntrad = K12*indentationfull**(power-1)  # contact probe_size
    cntrad[MaxInd:] = 0

    # retraction part calculation=============================================
    t1_ndx = np.zeros(len(time), dtype=int)
    endofalgorithm2 = len(time)
    b = MaxInd-1  # upper limit for max search

    for i in range(MaxInd, endofalgorithm2-2):  # force for retraction part
        res2 = np.zeros(len(time))
        localend = 0

        for j in range(b, localend-1, -1):
            if localend == 0:
                # ndx = np.asarray(range (j, i+1))
                # res2[j] = np.trapz(Et[i-ndx]*indspeed[ndx], time[ndx]) + eta*indspeed[i]
                res2[j] = np.trapz(Et[i-j::-1]*indspeed[j:i+1], dx=dT) + eta*indspeed[i]
                if res2[j] > 0:
                    localend = j

        if abs(res2[localend]) <= abs(res2[localend+1]):
            Imin = localend
        else:
            Imin = localend+1

        if Imin > MaxInd+1:
            t1_ndx[i] = Imin-1
            # print("position1")  # check of trigger position
        elif (Imin <= 1):
            t1_ndx[i] = Imin
            t1_ndx[i+1] = 1
            endofalgorithm2 = i
            cntrad[i] = cntrad[t1_ndx[i]]
            cntrad[i+1] = cntrad[t1_ndx[i+1]]
            # print("position2")
            # print(i)
            break
        else:
            b = Imin
            t1_ndx[i] = Imin
            endofalgorithm2 = PointN-1
            # print("position3")

        cntrad[i] = cntrad[t1_ndx[i]]  # a is recalculated
        # indentationfull2[i] = indentationfull[t1_ndx[i]]  # effective indentation

        # ndx = np.asarray(range (0, t1_ndx[i]+1))
        ijk = t1_ndx[i]
        if ijk == i:
            ijk = i-1
        # ForceT[i] = K1 * (BEC[ijk]*np.trapz(Et[i-ndx]*ind2speed[ndx], time[ndx]))
        # ForceT[i] = K1 * (BEC[ijk]*np.trapz(Et[i:i-ijk-1:-1]*ind2speed[:ijk+1], dx=dT))
        ForceT[i] = K1 * (BEC[ijk]*np.trapz(Et[i:i-ijk-1:-1]*ind2speed[:ijk+1], dx=dT))

    if adhesion_model in ['JKR', 'JKR_transition']:
        endofalgorithm2 = len(time)
    if adhesion_model == 'DMT':
        if adhesion_region  == 'approach' or adhesion_region  == 'both':
            ForceT[:MaxInd] += adhesion_force
        if adhesion_region  == 'retraction' or adhesion_region  == 'both':
            ForceT[MaxInd:endofalgorithm2] += adhesion_force
    if adhesion_model == 'JKR' and viscomodel == 'elastic' and probe_geom == 'sphere':
        if adhesion_region  == 'approach' or adhesion_region  == 'both':
            ForceT[:MaxInd], cntrad[:MaxInd] = JKR(Et[0], adhesion_force, Poisson, probe_size, indentationfull[:MaxInd])
        if adhesion_region  == 'retraction' or adhesion_region  == 'both':
            ForceT[MaxInd:endofalgorithm2], cntrad[MaxInd:endofalgorithm2] = JKR(Et[0], adhesion_force, Poisson, probe_size, indentationfull[MaxInd:endofalgorithm2])
    if adhesion_model == 'JKR_transition' and viscomodel == 'elastic' and probe_geom == 'sphere':
         if adhesion_region  == 'approach':
             pass
         if adhesion_region  == 'retraction' or adhesion_region  == 'both':
             ForceT[MaxInd:endofalgorithm2], cntrad[MaxInd:endofalgorithm2] = JKR_transition(Et[0], adhesion_force, Poisson, probe_size, indentationfull[MaxInd:endofalgorithm2])
    if adhesion_model == 'JKR' and viscomodel != 'elastic':
        print('Sorry, no model combining JKR and viscoelasticity yet')
    if adhesion_model == 'JKR' and probe_geom != 'sphere':
        print('Sorry, JKR model only for sphere/parabolic indenter')        
    cntrad = cntrad[0:len(indentationfull)]
    contact_time = endofalgorithm2*dT
    # plt.plot(indentationfull,ForceT)  # linestyle=':'
    return ForceT, cntrad, contact_time, t1_ndx, Et, ForceR

def JKR(E_Hertz, adhesion_force, Poisson, probe_size, ind):
    K1 = 4/3/(1-Poisson**2)*1e-9
    K = K1*E_Hertz
    Radius = probe_size
    Fadh = abs(adhesion_force) # should work with negative and positive imput
    cradius_JKR = np.zeros(len(ind))
    MincradJKR = (Radius*Fadh/K)**(1/3)
    prevsolution1 = MincradJKR
    IndJKR = ind
    for k in range(0, len(ind)):
        def IndJKRfun(ax):
            return (ax**2/Radius)-4/3*(ax*Fadh/Radius/K)**0.5-IndJKR[k]
        ax0 = prevsolution1
        ax = fsolve(IndJKRfun, ax0)
        cradius_JKR[k] = ax
        prevsolution1 = cradius_JKR[k]
    ForceJKR = ((K/Radius)**0.5*cradius_JKR**1.5-Fadh**0.5)**2-Fadh
    # ForceJKR = (K/Radius)*cradius_JKR**3-2*(Fadh*(K/Radius)*cradius_JKR**3)**0.5 # not working
    # plt.plot(ind,ForceHertz,IndJKR,ForceJKR)
    return ForceJKR, cradius_JKR

def JKR_transition(E_Hertz, adhesion_force, Poisson, probe_size, ind):
    K1 = 4/3/(1-Poisson**2)*1e-9
    K = K1*E_Hertz
    Radius = probe_size
    Fadh = abs(adhesion_force) # should work with negative and positive imput
    cradMax=(np.max(ind)*probe_size)**0.5
    ForceJKRcrit =  (K/Radius)*cradMax**3-2*(Fadh*(K/Radius)*cradMax**3)**0.5
    IndJKRcrit = (cradMax**2/Radius)-4/3*(cradMax*Fadh/Radius/K)**0.5
    cradius_JKR=np.zeros(len(ind))
    MincradJKR = (Radius*Fadh/K)**(1/3)
    prevsolution1 = MincradJKR
    IndJKR = ind
    kk=0
    deltaPunch=(2-4/3)*E_Hertz/(1-Poisson**2)/1e9*np.max(ind)**1.5*Radius**0.5
    ForceJKR = np.zeros(len(ind))

    while ind[kk]>IndJKRcrit and kk<len(ind)-1:
        cradius_JKR[kk]=cradMax
        ForceJKR[kk]= E_Hertz/(1-Poisson**2)/1e9*2*cradMax*ind[kk]-deltaPunch
        kk += 1

    for k in range(kk, len(ind)):
        def IndJKRfun(ax):
            return (ax**2/Radius)-4/3*(ax*Fadh/Radius/K)**0.5-IndJKR[k]
        ax0 = prevsolution1
        ax = fsolve(IndJKRfun, ax0)
        cradius_JKR[k] = ax
        if ax == ax0:
            break
        prevsolution1 = cradius_JKR[k]

    ForceJKR[kk:] = ((K/Radius)**0.5*cradius_JKR[kk:]**1.5-Fadh**0.5)**2-Fadh
    # ForceJKR = (K1/Radius)*cradius_JKR**3-2*(Fadh*(K1/Radius)*cradius_JKR**3)**0.5 #fine
    # plt.plot(ind,ForceHertz,IndJKR,ForceJKR)
    return ForceJKR, cradius_JKR

def smoothM(d, parS):
    # auxillary function for smoothing the data by moving average
    y = d
    DL = len(d)-1
    for ij in range(len(d)-1):
        if np.isnan(y[ij]):
            k = 0
            while np.isnan(y[ij]) and ij+k < DL:
                k = k+1
                y[ij] = y[ij+k]
    if parS > 1:
        y[1] = (d[1] + d[2] + d[3])/3
        y[-2] = (d[DL-2] + d[DL-1] + d[DL])/3
    if parS == 2 or parS == 3: #for 2 and 3
        for ij in range(2, DL-2):
            y[ij] = (d[ij-1] + d[ij] + d[ij+1])/3
    if parS >= 4:  # :for 4 and 5 and any more
        for n in range(2, DL-2):
            y[n] = (d[n-2] + d[n-1] + d[n] + d[n+1] + d[n+2])/5
    return y


if __name__ == '__main__':
    import time
    Poisson = 0.5
    probe_size = 1000
    MaxInd = 501
    dT = 1/MaxInd
    Height = 0
    viscomodel = 'springpot-dashpot-parallel'
    viscomodel = 'sPLR'
    viscomodel = 'elastic'
    probe_geom = 'sphere'
    adhesion_pars = ['JKR_transition', 'retraction', 0.03]  # JKR_transition
    t = np.linspace(0, 2, MaxInd*2+1)
    t = np.linspace(0, 3, MaxInd*3)
    indentationfull = np.piecewise(t, [t <= 1, t >= 1], [lambda t: t, lambda t: 2-t])
    ind_magnitude = 50
    indentationfull = ind_magnitude *indentationfull
    # plt.plot(t, indentationfull)
    par = [10000, 0.1, 10, 1, 10]  # par[0] = EHertz in elastic
    t1 = time.time()
    ForceT, cntrad, contact_time, t1_ndx, Et, ForceR = ting_numerical(par, adhesion_pars,
    Poisson, probe_size, dT, MaxInd, Height, viscomodel, probe_geom, indentationfull)
    t2 = time.time()
    print(t2-t1)
    f = plt.figure(1)
    plt.plot(indentationfull, ForceT)
    plt.plot(indentationfull, ForceR)
    g = plt.figure(2)
    PointN = len(indentationfull)
    time = np.linspace(0, dT*(PointN-1), PointN)
    plt.plot(time, ForceT) # t1_ndx
    plt.figure(num='Contact radius')
    plt.plot(time, cntrad)
    # plt.plot(indentationfull)
