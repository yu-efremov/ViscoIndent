# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 22:02:27 2020

@author: Yuri
Frequcnsies in [radians/s]

"""

import numpy as np
import math
from math import pi
from math import tan
from math import sin


def timetofreqspectrumFun(Wpars, Freqs, modelting, Pars):
    # y = np.zeros([len(Freqs),2])
    y = np.full([len(Freqs), 2], np.nan)

    if modelting == 'sPLR' or modelting == 'sPLReta' or modelting == 'mPLR' or modelting == 'sPLRdouble':
        # plr_par = [E1f; alpha; nu; Einf]
        E1t = Wpars[0]
        alpha = Wpars[1]
        Einf = Wpars[2]
        nu = 0  # newtonian viscosity

        if modelting == 'mPLR':
            powerlaw2 = [Wpars[1], Wpars[0], 1, Pars['dTmPLR'], Wpars[2]]
            plr_fun = lambda time, parplr: parplr[0] + (parplr[1]-parplr[0])/(parplr[2]+time/parplr[3])**parplr[4]
            # Et = plr_fun(powerlaw2,Timefull)
            E1t = plr_fun(powerlaw2, 1)
            alpha = Wpars[2]

        if modelting == 'sPLReta':
            nu = Wpars[2]
            Einf = 0
        E1w = (E1t-Einf)*pi/2/sin(alpha*pi/2)/math.gamma(alpha)  # checked
        y[:, 0] = E1w*(Freqs**alpha)+Einf
        y[:, 1] = E1w*tan(pi*alpha/2)*(Freqs**alpha)+nu*Freqs

    elif modelting == 'SLS':
        # y = Einf + (E0-Einf).*exp(-x./(tau);
        E0 = Wpars[0]
        Einf = Wpars[1]
        Ed = E0 - Einf
        tau = Wpars[2]

        # SLS_Estor= SLSPar(1)+(SLSPar(2)-SLSPar(1)).*(Freq.*SLSPar(3)).^2./(1+(Freq.*SLSPar(3)).^2);
        # SLS_Eloss=(SLSPar(2)-SLSPar(1)).*(Freq.*SLSPar(3))./(1+(Freq.*SLSPar(3)).^2);
        y[:, 0] = Einf+Ed*(Freqs*tau)**2/(1+(Freqs*tau)**2)
        y[:, 1] = Ed*(Freqs*tau)/(1+(Freqs*tau)**2)

    return y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Pars_class import Pars_gen
    # Wpars = [1000, 500, 0.2]
    Pars = Pars_gen()
    Freqs = np.array([100, 200])
    Freqs = np.logspace(np.log10(0.1), np.log10(100), num=10)
    Wpars = [1000, 0.1, 10]
    contact_time = 0.1
    Freq = np.array([1/contact_time])
    modelting = 'sPLReta'
    y = timetofreqspectrumFun(Wpars, Freqs, modelting, Pars)
    plt.loglog(Freqs, y[:, 0])+plt.loglog(Freqs, y[:, 1])