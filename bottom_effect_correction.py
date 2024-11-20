# -*- coding: utf-8 -*-
"""
@author: Yuri Efremov
bottom effect correction coefficients
"""
import numpy as np
from numpy import pi


def bottom_effect_correction(Poisson, Probe_dimension, Height, modelprobe,
                             indentationfull):
    BEC = np.ones(len(indentationfull))
    indentationfull[indentationfull < 0] = np.nan  # to remove warnings in sqrt
    if 0.45 <= Poisson < 0.5:  # coefficients from doi.org/10.1016/j.bpj.2018.05.012
        if modelprobe == 'sphere':  # sphere
            Dpar1 = 1.133
            Dpar2 = 1.497
            Dpar3 = 1.469
            Dpar4 = 0.755
            R = Probe_dimension
            h = Height
            b = (R*indentationfull)**(0.5)/h
            BEC = 1 + (Dpar1*b) +\
                      (Dpar2*b**2) +\
                      (Dpar3*b**3) +\
                      (Dpar4*b**4)

            BECspeed = 1 + 4/3*(Dpar1*b) +\
                           5/3*(Dpar2*b**2) +\
                             2*(Dpar3*b**3) +\
                           7/3*(Dpar4*b**4)

        elif modelprobe in ['pyramid', 'cone']:
        # coefficients from doi.org/10.1016/j.bpj.2018.05.012
            Dpar1 = 0.721
            Dpar2 = 0.650
            Dpar3 = 0.491
            Dpar4 = 0.225
            h = Height;
            b = np.tan(Probe_dimension*pi/180)*indentationfull/h
            BEC = 1 + Dpar1*b + Dpar2*b**2 + Dpar3*b**3 + Dpar4*b**4
            BECspeed = 1 + 1.5*Dpar1*b + 2*Dpar2*b**2 + 2.5*Dpar3*b**3 + 3*Dpar4*b**4
            # BEC = np.ones(len(indentationfull))
            # BECspeed = np.ones(len(indentationfull))
            print('BEC anal. coeffs. used')

        elif modelprobe == 'cylinder':
        # coefficients from doi.org/10.1016/j.bpj.2018.05.012
            Dpar1 = 1.133
            Dpar2 = 1.283
            Dpar3 = 0.598
            Dpar4 = -0.291
            R = Probe_dimension
            h = Height;
            b = R/h
            BEC = 1 + Dpar1*b + Dpar2*b**2 + Dpar3*b**3 + Dpar4*b**4
            BEC = BEC*indentationfull/indentationfull
            BECspeed = BEC
            # BEC = np.ones(len(indentationfull))
            # BECspeed = np.ones(len(indentationfull))
            print('BECspeed not ready yet')

    else:
        if modelprobe == 'sphere':
            # coefficients are from doi.org/10.1016/S0006-3495(02)75620-8
            alpha = -(1.2876-1.4678*Poisson+1.3442*Poisson**2)/(1-Poisson)  # bonded case
            bettaD = (0.6387-1.0277*Poisson+1.5164*Poisson**2)/(1-Poisson)  # bonded case
            # alpha = -0.347*(3-2*Poisson)/(1-Poisson)  # nonbonded case
            # bettaD = 0.056*(5-2*Poisson)/(1-Poisson)  # nonbonded case
            # print('nonbonded_case!!') manual switch between bonded/ non-bonded

            Dpar1 = -2*alpha/pi
            Dpar2 = 4*(alpha/pi)**2
            #Dpar2 = 4*(alpha/pi)**2 + alpha/pi*2/3*alpha/pi # Garcia et al. very close
            Dpar3 = -(8/pi**3)*(alpha**3+(4*(pi**2)*bettaD/15))
            #Dpar3 = -(8/pi**3)*(alpha**3+(4*(pi**2)*bettaD/15)) - Dpar2*2/3*alpha/pi
            Dpar4 = (16*alpha/pi**4)*(alpha**3+(3*(pi**2)*bettaD/5))
            #Dpar4 = (16*alpha/pi**4)*(alpha**3+(3*(pi**2)*bettaD/5)) - Dpar3*2/3*alpha/pi
            R = Probe_dimension
            h = Height
            b = (R*indentationfull)**(0.5)/h
            BEC = 1 + (Dpar1*b) +\
                      (Dpar2*b**2) +\
                      (Dpar3*b**3) +\
                      (Dpar4*b**4)

            BECspeed = 1 + 4/3*(Dpar1*b) +\
                           5/3*(Dpar2*b**2) +\
                             2*(Dpar3*b**3) +\
                           7/3*(Dpar4*b**4)

        elif modelprobe in ['pyramid', 'çone', 'cylinder']:
            # in the curent version of the script the correction is not yet implemented
            BEC = np.ones(len(indentationfull))
            BECspeed = np.ones(len(indentationfull))
            print('BEC is not yet avilable')

    return BEC, BECspeed


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Poisson = 0.49
    Probe_dimension = 1000  # nm or degrees
    Height = 1000
    modelprobe = 'cylinder'  # 'sphere' 'pyramid' 'cylinder'
    indentationfull = np.linspace(0, 400, 400)
    BEC = bottom_effect_correction(Poisson, Probe_dimension, Height, modelprobe, indentationfull)[0]
    plt.plot(indentationfull, BEC)
