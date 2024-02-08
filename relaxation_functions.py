# -*- coding: utf-8 -*-
"""
set of relaxation functions
file contains equations for different relaxation functions (viscoelastic models)
some functions have several names (e.g. 'springpot-spring-parallel2' = 'sPLR')
list can be extended
parameters of the functions are stored in "par" list
names of the parameters are stored in "parnames" list
currently functions with up to 5 parameters are included
modellist:  list of all relaxation functions available
"""

import numpy as np
from numpy import exp as exp
import matplotlib.pyplot as plt
import warnings


from MLF import mlf


def modellist():  # list of all relaxation functions currently available
    modellist = ['elastic', 'DMT', 'KV', 'MW', 'SLS', 'SLS2', 'dSLS', 'springpot',
                 'springpot-spring-parallel', 'springpot-spring-parallel2',
                 'sPLR', 'springpot-dashpot-parallel', 'sPLReta',
                 'springpot-spring-serial', 'springpot-spring-serial2',
                 'springpot-dashpot-serial', 'springpot-dashpot-serial2',
                 'springpot-springpot-parallel', 'springpot-springpot-serial',
                 'springpot-springpot-serial2', 'springpot-springpot-serial3',
                 'fractionalSLS', 'fractionalSLS2', 'fractionalMWe',
                 'fractionalMWe2', 'mPLR', 'sPLRetatest'
                 ]
    return modellist


def relaxation_function(par, model, Time):
    eta0 = 0
    Einf0 = 0
    parnames = ['par1', 'par2', 'par3', 'par4', 'par5']
    warnings.filterwarnings("ignore", message="divide by zero encountered in power")
    warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
    warnings.filterwarnings("ignore", message="divide by zero encountered in reciprocal")
    warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
    warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

    if model == 'elastic' or model == 'DMT':
        parnames = ['E']
        E = par[0]
        Et = E*np.ones(Time.shape)
    elif model == 'KV':
        parnames = ['Einf', 'eta']
        Einf = par[0]
        eta0 = par[1]
        Et = Einf*np.ones(Time.shape)
    elif model == 'MW':  # Maxwell
        parnames = ['E0', 'tau']
        E0 = par[0]
        tau = par[1]
        relfun = lambda E0, tau, t: E0*exp(-t/tau)
        Et = relfun(E0, tau, Time)
    elif model == 'SLS':
        parnames = ['E0', 'tau', 'Einf']
        E0 = par[0]
        tau = par[1]
        Einf = par[2]
        Es1 = E0 - Einf
        Es2 = Einf
        relfun1 = lambda E0, Einf, tau, t: (E0 - Einf) * exp(-t/tau) + Einf
        # relfun2 = lambda E0, Einf, tau, t: E0 - (E0 - Einf) * (1 - exp(-t / tau))
        Et = relfun1(E0, Einf, tau, Time)
    elif model == 'SLS2':
        parnames = ['Es1', 'tau', 'Es2']
        Es1 = par[0]
        tau = par[1]
        Es2 = par[2]
        E0 = Es1 + Es2
        Einf = Es2
        relfun = lambda Es1, Es2, tau, t: Es1 * exp(-t / tau) + Es2
        Et = relfun(Es1, Es2, tau, Time)
    elif model == 'dSLS':  # double SLS, 5pars
        parnames = ['Es1', 'tau1', 'Es2', 'tau2', 'Einf']
        Es1 = par[0]
        tau1 = par[1]
        Es2 = par[2]
        tau2 = par[3]
        Einf = par[4]
        relfun = lambda Es1, tau1, Es2, tau2, Einf, t:\
            Es1 * exp(-t / tau1) + Es2 * exp(-t / tau2) + Einf
        Et = relfun(Es1, tau1, Es2, tau2, Einf, Time)
    elif model == 'springpot':   #1 -single spring-pot 'PLR'
        parnames = ['Ea1', 'alpha']
        Ea1 = par[0]
        alpha = par[1]
        relfun = lambda Ea1, alpha, t: Ea1 * t ** (-alpha)
        Et = relfun(Ea1, alpha, Time)
    elif model == 'springpot-spring-parallel':   # 'PLRe'
        parnames = ['Ea1', 'alpha', 'Einf']
        Ea1 = par[0]
        alpha = par[1]
        Einf = par[2]
        relfun = lambda Ea1, alpha, Einf, t: Ea1 * t ** (-alpha) + Einf
        Et = relfun(Ea1, alpha, Einf, Time)
    elif model == 'springpot-spring-parallel2' or model == 'sPLR':   #1 -single spring-pot E1 global
        parnames = ['E1', 'alpha', 'Einf']
        E1 = par[0]
        alpha = par[1]
        Einf = par[2]
        relfun = lambda E1, alpha, Einf, t: (E1 - Einf) * t ** (-alpha) + Einf
        Et = relfun(E1, alpha, Einf, Time)
    elif model == 'springpot-dashpot-parallel' or model == 'sPLReta' or model == 'sPLRetatest':   #1 'sPLReta'
        parnames = ['Ea1', 'alpha', 'eta0']
        Ea1 = par[0]
        alpha = par[1]
        eta0 = par[2]
        relfun = lambda Ea1, alpha, t: Ea1 * t ** (-alpha)
        Et = relfun(Ea1, alpha, Time)
    elif model == 'springpot-spring-serial':   # 'fractMW' fractional Maxwell model
        parnames = ['Ea1', 'alpha', 'Es0']
        Ea1 = par[0]
        alpha = par[1]
        Es0 = par[2]
        tau = (Ea1 / Es0) ** (1 / alpha)
        relfun = lambda Ea1, alpha, Es0, t: Es0 * mlf(alpha, 1, -(Es0 / Ea1) * t ** alpha)
        Et = relfun(Ea1, alpha, Es0, Time)
    elif model == 'springpot-spring-serial2':   # 'fractMW' fractional Maxwell model with tau
        parnames = ['Ea1', 'alpha', 'tau']
        Ea1 = par[0]
        alpha = par[1]
        tau = par[2]
        Es0 = Ea1 * tau ** (-alpha)
        relfun = lambda Es, alpha, tau, t: Ea1 * tau ** (-alpha) * mlf(alpha, 1, -(t / tau) ** alpha)
        Et = relfun(Ea1, alpha, tau, Time)
    elif model == 'springpot-dashpot-serial':   # 'fractMW' fractional Maxwell model
        parnames = ['Ea1', 'alpha', 'eta']
        Ea1 = par[0]
        alpha = par[1]
        eta = par[2]
        tau = (eta / Ea1) ** (1 / (1 - alpha))
        relfun = lambda Ea1, alpha, eta, t:\
            eta ** (alpha / (alpha - 1)) * Ea1 ** (1. / (1 - alpha)) * t ** (-alpha) * mlf(1 - alpha, 1 - alpha, -(Ea1 / eta) * t ** (1 - alpha))
        Et = relfun(Ea1, alpha, eta, Time)
    elif model == 'springpot-dashpot-serial2':   # 'fractMW' fractional Maxwell model
        parnames = ['Ea1', 'alpha', 'tau']
        Ea1 = par[0]
        alpha = par[1]
        tau = par[2]
        eta = Ea1 * tau ** (1 - alpha)
        relfun = lambda Ea1, alpha, tau, t:\
            Ea1 * (t * tau) ** (-alpha) * mlf(1 - alpha, 1 - alpha, -(t / tau) ** (1 - alpha))
        Et = relfun(Ea1, alpha, tau, Time)
    elif model == 'springpot-springpot-parallel':
        parnames = ['Ea1', 'alpha', 'Eb1', 'betta']
        Ea1 = par[0]
        alpha = par[1]
        Eb1 = par[2]
        betta = par[3]
        relfun = lambda Ea1, alpha, Eb1, betta, t: Ea1 * t ** (-alpha) + Eb1 * t ** (-betta)
        Et = relfun(Ea1, alpha, Eb1, betta, Time)
    elif model == 'springpot-springpot-serial':
        parnames = ['Ea1', 'alpha', 'Eb1', 'betta']
        Ea1 = par[0]
        alpha = par[1]
        Eb1 = par[2]
        betta = par[3]
        tau = (Ea1 / Eb1) ** (1 / (alpha - betta))
        # assignin('base', 'tau', tau)
        relfun = lambda Ea1, alpha, Eb1, betta, t:\
            Ea1 ** (betta / (betta - alpha)) * Eb1 ** (alpha / (alpha - betta)) * t ** (-betta) * mlf(alpha - betta, 1 - betta, -(Eb1 / Ea1) * t ** (alpha - betta))
        Et = relfun(Ea1, alpha, Eb1, betta, Time)
    elif model == 'springpot-springpot-serial2':   #with tau
        parnames = ['Ea1', 'alpha', 'betta', 'tau']
        Ea1 = par[0]
        alpha = par[1]
        betta = par[2]
        tau = par[3]
        Eb1 = Ea1 / tau ** (alpha - betta)
        relfun = lambda Ea1, alpha, betta, tau, t:\
            Ea1 * tau ** (-alpha) * t ** (-betta) * mlf(alpha - betta, 1 - betta, -(t / tau) ** (alpha - betta))
        Et = relfun(Ea1, alpha, betta, tau, Time)
    elif model == 'springpot-springpot-serial3':   #with tau
        parnames = ['Eb1', 'alpha', 'betta', 'tau']
        Eb1 = par[0]
        alpha = par[1]
        betta = par[2]
        tau = par[3]
        Ea1 = Eb1 * tau ** (alpha - betta)
        relfun = lambda Eb1, alpha, betta, tau, t:\
            Eb1 * (t * tau) ** (-betta) * mlf(alpha - betta, 1 - betta, -(t / tau) ** (alpha - betta))
        # relfun = @(Eb1, alpha, betta, tau, t) Eb1.*tau.^betta.*t.^(-betta).*mlf(alpha-betta,1-betta,-(t./tau).^(alpha-betta))
        Et = relfun(Eb1, alpha, betta, tau, Time)
    elif model == 'fractionalSLS':   #fractional SLS (Zener) model
        parnames = ['Ea1', 'alpha', 'Es1', 'Es2']
        Ea1 = par[0]
        alpha = par[1]
        Es1 = par[2]
        Es2 = par[3]
        tau = (Ea1 / Es1) ** (1 / alpha)
        E0 = Es1 + Es2
        relfun = lambda Ea1, alpha, Es1, Es2, t:\
            Es1 * mlf(alpha, 1, -(Es1 / Ea1) * t ** alpha) + Es2
        Et = relfun(Ea1, alpha, Es1, Es2, Time)
    elif model == 'fractionalSLS2':  # fractional SLS (Zener) model with tau, E0, Einf
        parnames = ['E0', 'alpha', 'tau', 'Einf']
        E0 = par[0]
        alpha = par[1]
        tau = par[2]
        Einf = par[3]
        Es1 = E0 - Einf
        Ea1 = (E0 - Einf) * tau ** (alpha)
        relfun = lambda E0, Einf, alpha, tau, t:\
        (E0 - Einf) * mlf(alpha, 1, -(t / tau) ** alpha) + Einf
        Et = relfun(E0, Einf, alpha, tau, Time)
    elif model == 'fractionalMWe':    # fractional model, springpot-dashpot in parallel with spring
        parnames = ['Ea1', 'alpha', 'eta', 'Es1']
        Ea1 = par[0]
        alpha = par[1]
        eta = par[2]
        Es1 = par[3]
        tau = (eta / Ea1) ** (1 / (1 - alpha))
        relfun = lambda Ea1, alpha, eta, Es1, t:\
        eta ** (alpha / (alpha - 1)) * Ea1 ** (1. / (1 - alpha)) * t ** (-alpha) * mlf(1 - alpha, 1 - alpha, -(Ea1 / eta) * t ** (1 - alpha)) + Es1
        Et = relfun(Ea1, alpha, eta, Es1, Time)
    elif model == 'fractionalMWe2':    # fractional model, springpot-dashpot in parallel with spring
        parnames = ['Ea1', 'alpha', 'tau', 'Es1']
        Ea1 = par[0]
        alpha = par[1]
        tau = par[2]
        Es1 = par[3]
        eta = Ea1 * tau ** (1 - alpha)
        relfun = lambda Ea1, alpha, tau, Es1, t:\
        Ea1 * (t * tau) ** (-alpha) * mlf(1 - alpha, 1 - alpha, -(t / tau) ** (1 - alpha)) + Es1
        Et = relfun(Ea1, alpha, tau, Es1, Time)
    elif model == 'mPLR':    # powerlaw function Einf+(E0-Einf)/(1+time/dT)^alpha
        parnames = ['E0', 'alpha', 'Einf', 'dTp']
        E0 = par[0]
        alpha = par[1]
        Einf = par[2]
        dTp = par[3]
        relfun = lambda E0, Einf, dTp, alpha, t: Einf + (E0 - Einf) / (1 + t / dTp) ** alpha
        Et = relfun(E0, Einf, dTp, alpha, Time)
    else:
        print('relaxation function is not recognized')

    return Et, eta0, parnames


if __name__ == '__main__':
    # view some relaxation function E(t) on a log-log scale
    # plt.figure
    Time = np.logspace(-5, 5, 100)
    Time = np.linspace(0, 1, 1000)
    # Et=relaxation_function([1,0.1,0.1,0.1,0.1], 'SLS2', Time)
    [Et, eta, parnames] = relaxation_function([1, 10, 1, 0.01, 0.1], 'dSLS', Time)
    # [Et, eta, parnames] = relaxation_function([2, 0.1, 1024], 'springpot-dashpot-serial', Time)
    # [Et, eta, parnames] = relaxation_function([100, 0.5, 300, 20], 'fractionalSLS', Time)[0:2]
    plt.loglog(Time, Et)
