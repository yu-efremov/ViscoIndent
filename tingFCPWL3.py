# -*- coding: utf-8 -*-
"""
@author: Yuri

"""


import numpy as np
from math import pi, tan
import matplotlib.pyplot as plt

from bottom_effect_correction import bottom_effect_correction
from relaxation_functions import relaxation_function


def smoothM(d, parS):
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
    if parS == 2 or parS == 3:  # for 2 and 3
        for ij in range(2, DL-2):
            y[ij] = (d[ij-1] + d[ij] + d[ij+1])/3
    if parS >= 4:  # :for 4 and 5 and any more
        for n in range(2, DL-2):
            y[n] = (d[n-2] + d[n-1] + d[n] + d[n+1] + d[n+2])/5
    # box = np.ones(parS)/parS
    # y_smooth = np.convolve(y, box, mode='same')
    # return y_smooth
    return y


def tingFCPWL3uni(par, Poisson, Radius, dT, MaxInd, Height, modelting, modelprobe, indentationfull):
    PointN = len(indentationfull)
    time = np.linspace(dT, dT*PointN, PointN)
    time = time-dT
    # fixedpars assignment - removed from the function
    # fixedp=Fixedpars[0,:]
    # fixedv=Fixedpars[1,:]
    # #par_initial=par
    # par=np.asarray(par)
    # par[fixedp==1]=fixedv[fixedp==1]

    # Et construction
    [Et, eta] = relaxation_function(par, modelting, time)

    # Et[0]=Et[1]*2

    if Height > 0:  # BEC for the purely elastic case
        [BEC, BECspeed] = bottom_effect_correction(Poisson, Radius, Height, modelprobe, indentationfull)
    else:
        BEC = np.ones(len(indentationfull))
        BECspeed = np.ones(len(indentationfull))

    if modelprobe == 'sphere':
        power = 1.5
        K1 = 4*Radius**0.5/3
        K12 = Radius**0.5
    elif modelprobe == 'cone' or modelprobe == 'pyramid':
        power = 2
        if modelprobe == 'cone':
            K1 = 2/pi()*tan(Radius*pi/180)
        elif modelprobe == 'pyramid':
            K1 = 1.406/2*tan(Radius*pi/180)
        K12 = K1
    elif modelprobe == 'cylinder':  # cylinder
        power = 1
        K1 = 2*Radius  # before integral no speed
        K12 = Radius
    K1 = K1/(1-Poisson**2)*1e-9

    # indentation history
    ind2speed = np.diff(indentationfull**power)/dT
    ind2speed = np.append(ind2speed, ind2speed[-1])
    ind2speed = smoothM(ind2speed, 5)
    indspeed = np.diff(indentationfull)/dT
    indspeed = np.append(indspeed, indspeed[-1])
    indspeed = smoothM(indspeed, 5)
    # plt.plot(indspeed)

    Force2 = np.zeros(len(indentationfull))
    for i in range(1, MaxInd):  # MaxInd PointN-1
        ndx = np.asarray(range(0, i))  # lower and upper limits for dummy var
        Force2[i] = K1 * (BEC[i]*np.trapz(Et[i+1-ndx]*ind2speed[ndx], time[ndx]) + power*eta*indentationfull[i]**(power-1)*indspeed[i]*BECspeed[i])
    # ForceR=Force2
    # plt.plot(indentationfull,ForceR)

    carea2 = K12*indentationfull**(power-1)
    carea2[MaxInd:] = 0

    # retraction part calculation=============================================
    t1_ndx2 = np.zeros(len(time), dtype=int)
    endofalgorithm2 = len(time)
    b = MaxInd-1  # upper limit for max search

    for i in range(MaxInd, endofalgorithm2-2):  # force for retraction part
        res2 = np.zeros(len(time))
        localend = 0

        for j in range(b, localend-1, -1):
            if localend == 0:
                ndx = np.asarray(range(j, i))
                res2[j] = np.trapz(Et[i+1-ndx]*indspeed[ndx], time[ndx]) + eta*indspeed[i]
                if res2[j] > 0:
                    localend = j

        if abs(res2[localend]) <= abs(res2[localend+1]):
            Imin = localend
        else:
            Imin = localend+1

        if (Imin > MaxInd+1):
            t1_ndx2[i] = Imin-1
            # print("position1")  # check of trigger position
        elif (Imin <= 1):
            t1_ndx2[i] = Imin
            t1_ndx2[i+1] = 1
            endofalgorithm2 = i
            carea2[i] = carea2[t1_ndx2[i]]
            carea2[i+1] = carea2[t1_ndx2[i+1]]
            # print("position2")
            # print(i)
            break
        else:
            b = Imin
            t1_ndx2[i] = Imin
            endofalgorithm2 = PointN-1
            # print("position3")

        carea2[i] = carea2[t1_ndx2[i]]  # a is recalculated
        # indentationfull2[i] = indentationfull[t1_ndx2[i]] # effective indentation

        ndx = np.asarray(range(1, t1_ndx2[i]))
        ijk = t1_ndx2[i]
        Force2[i] = K1 * (BEC[ijk]*np.trapz(Et[i+1-ndx]*ind2speed[ndx], time[ndx]))  # checked - no eta for retraction

    carea2 = carea2[0:len(indentationfull)]
    contact_time = endofalgorithm2*dT
    # plt.plot(indentationfull,Force2)  # linestyle=':'
    return Force2, carea2, contact_time, t1_ndx2, Et


if __name__ == '__main__':
    Fixedpars = np.array([[0, 0, 0], [0, 0, 0]])
    Poisson = 0.5
    Radius = 1
    MaxInd = 1000
    dT = 1/MaxInd
    Height = 0
    modelting = 'springpot-spring-serial'
    modelprobe = 'sphere'
    t = np.linspace(0, 2, MaxInd*2+1)
    indentationfull = np.piecewise(t, [t <= 1, t >= 1], [lambda t: t, lambda t: 2-t])
    # plt.plot(t, indentationfull)
    par = [10, 0.2, 100]
    [Force2, carea2] = tingFCPWL3uni(par, Poisson, Radius, dT, MaxInd, Height, modelting, modelprobe, indentationfull)[0:2]
    f = plt.figure(1)
    plt.plot(indentationfull, Force2)
    g = plt.figure(2)
    PointN = len(indentationfull)
    time = np.linspace(dT, dT*PointN, PointN)
    time = time-dT

    plt.plot(time, carea2)
