# -*- coding: utf-8 -*-
"""
numerical calculation of the force using the Ting's equations


"""

import numpy as np
from numpy import pi, tan
import matplotlib.pyplot as plt
# import scipy
# from scipy.integrate import simps # slower, without substantial improvement
from bottom_effect_correction import bottom_effect_correction
from relaxation_functions import relaxation_function


def ting_numerical(par, Poisson, Radius, dT, MaxInd, Height, modelting, modelprobe, indentationfull):
    PointN = len(indentationfull)
    time = np.linspace(0,dT*(PointN-1), PointN)

    
    # Et construction
    [Et, eta] = relaxation_function(par, modelting, time)[0:2]
    # remove zero-time singularity if needed
    if np.isinf(Et[0]):
        Et[0] = 2*Et[1]
        
    if Height > 0:  # bottom effect correction
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
            K1=2/pi*tan(Radius*pi/180)
        elif modelprobe== 'pyramid':
            K1 = 1.406/2*tan(Radius*pi/180)
        K12=K1
    elif modelprobe== 'cylinder': # cylinder
        power = 1
        K1 = 2*Radius
        K12 = Radius
    K1 = K1/(1-Poisson**2)*1e-9   
    
    # indentation history
    ind2speed = np.diff(indentationfull**power)/dT
    ind2speed = np.append(ind2speed, ind2speed[-1])
    # ind2speed = smoothM(ind2speed, 5)
    indspeed = np.diff(indentationfull)/dT
    indspeed = np.append(indspeed, indspeed[-1])
    # indspeed = smoothM(indspeed, 5)
    # plt.plot(indspeed)
    
    ForceR = np.zeros(len(indentationfull)) # Lee_Radok force prediction
    ForceT = np.zeros(len(indentationfull)) # Ting's force prediction
    
    for i in range (1, PointN-1):  # MaxInd or PointN-1
        ndx = np.asarray(range (0, i+1))  # integration limits, dummy variable
        ForceR[i]= K1 * (BEC[i]*np.trapz(Et[i-ndx]*ind2speed[ndx], time[ndx]) + 
                         power*eta*indentationfull[i]**(power-1)*indspeed[i]*BECspeed[i])
    ForceT[:MaxInd]=ForceR[:MaxInd]
    # plt.plot(indentationfull,ForceR)
    
    cntrad = K12*indentationfull**(power-1) # contact radius
    cntrad[MaxInd:]=0

    # retraction part calculation=============================================
    t1_ndx = np.zeros(len(time), dtype=int)  
    endofalgorithm2=len(time)
    b=MaxInd-1  # upper limit for max search

    for i in range(MaxInd,endofalgorithm2-2):  # force for retraction part
        res2 = np.zeros(len(time)) 
        localend = 0

        for j in range (b, localend-1, -1): 
            if localend == 0:
                ndx = np.asarray(range (j, i+1))
                res2[j] = np.trapz(Et[i-ndx]*indspeed[ndx], time[ndx]) + eta*indspeed[i]
                if res2[j]>0:
                    localend=j
                    
        if abs(res2[localend]) <= abs(res2[localend+1]):
            Imin = localend
        else:
            Imin = localend+1
 
        if (Imin>MaxInd+1):
            t1_ndx[i] = Imin-1
            #print("position1")  # check of trigger position
        elif (Imin<=1):
            t1_ndx[i] = Imin
            t1_ndx[i+1] = 1
            endofalgorithm2 = i
            cntrad[i] = cntrad[t1_ndx[i]]
            cntrad[i+1] = cntrad[t1_ndx[i+1]]
            #print("position2")
            #print(i)
            break 
        else:
            b=Imin
            t1_ndx[i] = Imin 
            endofalgorithm2 = PointN-1
            #print("position3")
        
        cntrad[i] = cntrad[t1_ndx[i]]  # a is recalculated
        #indentationfull2[i] = indentationfull[t1_ndx[i]]  # effective indentation
        
        ndx = np.asarray(range (1, t1_ndx[i]+1))
        ijk=t1_ndx[i]
        ForceT[i] = K1 * (BEC[ijk]*np.trapz(Et[i-ndx]*ind2speed[ndx], time[ndx]))
    
    cntrad=cntrad[0:len(indentationfull)]
    contact_time = endofalgorithm2*dT
    #plt.plot(indentationfull,Force2)  # linestyle=':'
    return ForceT, cntrad, contact_time, t1_ndx, Et, ForceR

def smoothM(d, parS):
    y=d
    DL = len(d)-1;
    for ij in range(len(d)-1):
        if np.isnan(y[ij]):
            k=0
            while np.isnan(y[ij]) and ij+k<DL:
                k=k+1
                y[ij]=y[ij+k]
    if parS>1:
        y[1]=(d[1] + d[2] + d[3])/3
        y[-2]=(d[DL-2] + d[DL-1] + d[DL])/3
    if parS==2 or parS==3: #for 2 and 3
        for ij in range(2, DL-2):
            y[ij] = (d[ij-1] + d[ij] + d[ij+1])/3;
    if parS>=4:  # :for 4 and 5 and any more
        for n in range(2, DL-2):
            y[n] = (d[n-2] + d[n-1] + d[n] + d[n+1] + d[n+2])/5;
    return y

if __name__ == '__main__':
    import time
    t1 = time.time()
    Poisson = 0.5
    Radius = 1
    MaxInd = 1000
    dT = 1/MaxInd   
    Height = 0
    modelting = 'springpot-dashpot-parallel'
    modelprobe = 'sphere'
    t = np.linspace(0, 2, MaxInd*2+1)
    indentationfull = np.piecewise(t, [t <= 1, t >= 1], [lambda t: t, lambda t: 2-t])
    indentationfull = 50*indentationfull
    # plt.plot(t, indentationfull)
    par = [10000, 0.4, 0]
    ForceT, cntrad, contact_time, t1_ndx, Et, ForceR = ting_numerical(par, Poisson, Radius, dT, MaxInd, Height, modelting, modelprobe, indentationfull)
    t2 = time.time()
    print(t2-t1)
    f = plt.figure(1)
    plt.plot(indentationfull, ForceT)
    plt.plot(indentationfull, ForceR)    
    g = plt.figure(2)
    PointN = len(indentationfull)
    time = np.linspace(0,dT*(PointN-1), PointN)
    plt.plot(time, cntrad)