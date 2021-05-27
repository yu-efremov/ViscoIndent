# -*- coding: utf-8 -*-
"""
@author: Yuri
# set of relaxation functions as class
# another approach to read lamba function by inspect
# also can add a string with latex string for the equation
"""

import numpy as np
from numpy import exp as exp
import matplotlib.pyplot as plt
import warnings
import inspect

from MLF import mlf


class relaxation_function:

    def __init__(self, model, *par_time):
        nargin = len(par_time)
        if nargin == 2:
            par = par_time[0]
            Time = par_time[1]
        else:
            par = [0.4, 0.3, 0.2, 0.1, 0]
            Time = np.nan

        self.eta0 = 0
        self.Einf0 = 0
        self.parnames = ['par1', 'par2', 'par3', 'par4', 'par5']
        self.model_name = model
        # self.Et = self.model(par, Time)
        # rel_fun_selected = getattr(self, model)
        self.Et = getattr(self, model)(par, Time)  # execute model by name
        # return Et

        self.modellist = ['elastic', 'KV', 'MW', 'SLS', 'SLS2', 'dSLS', 'springpot',
                 'springpot-spring-parallel', 'springpot-spring-parallel2',
                 'sPLR', 'springpot-dashpot-parallel', 'sPLReta',
                 'springpot-spring-serial', 'springpot-spring-serial2',
                 'springpot-dashpot-serial', 'springpot-dashpot-serial2',
                 'springpot-springpot-parallel', 'springpot-springpot-serial',
                 'springpot-springpot-serial2', 'springpot-springpot-serial3',
                 'fractionalSLS', 'fractionalSLS2', 'fractionalMWe',
                 'fractionalMWe2', 'mPLR'
                 ]
        if model in self.modellist:
            self.Et = getattr(self, model)(par, Time)  # execute model by name
            func_string = str(inspect.getsourcelines(self.relfun1)[0])
            self.rel_equation = func_string.strip("['\\n']").split(": ")[1]

        else:
            print(f'model {model} is not supported.\n')

    def SLS(self, par, Time):
        self.parnames = ['E0', 'tau', 'Einf']
        E0 = par[0]
        tau = par[1]
        Einf = par[2]
        Es1 = E0 - Einf
        Es2 = Einf
        self.relfun1 = lambda E0, Einf, tau, t: (E0 - Einf) * exp(-t/tau) + Einf
        # relfun2 = lambda E0, Einf, tau, t: E0 - (E0 - Einf) * (1 - exp(-t / tau))
        Et = self.relfun1(E0, Einf, tau, Time)
        return Et

    # def SLSsym(self, par, Time):
    #     self.parnames = ['E0', 'tau', 'Einf']
    #     [t, E0, tau, Einf] = sympy.symbols('t E_0 tau E_inf')
    #     E0 = par[0]
    #     tau = par[1]
    #     Einf = par[2]
    #     Es1 = E0 - Einf
    #     Es2 = Einf
    #     self.relfun1 = (E0 - Einf) * sympy.exp(-t/tau) + Einf
    #     [t, Es0, alpha, Ea1] = sympy.symbols('t Es0 alpha Ea1')
    #     # relfun1 = Es0 * mlf(alpha, 1, -(Es0 / Ea1) * t ** alpha)
    #     # relfun2 = lambda E0, Einf, tau, t: E0 - (E0 - Einf) * (1 - exp(-t / tau))
    #     Et = self.relfun1(E0, Einf, tau, Time)
    #     return Et

    def elastic(self, par, Time):
        self.parnames = ['E']
        E = par[0]
        self.relfun1 = lambda E, t: E*t**0
        Et = self.relfun1(E, Time)
        return Et

    def KV(self, par, Time):
        self.parnames = ['Einf', 'eta']
        Einf = par[0]
        eta0 = par[1]
        self.relfun1 = lambda Einf, t: Einf*t**0
        Et = self.relfun1(Einf, Time)
        return Et


if __name__ == '__main__':

    # plt.figure
    Time = np.logspace(-5, 5, 100)
    # Time = np.nan
    # Time = np.linspace(0, 1, 1000)
    # Et=relaxation_function([1,0.1,0.1,0.1,0.1], 'SLS2', Time)
    Et = relaxation_function('SLS', [1, 10, 0.1, 0.01, 0.1], Time)
    # [Et, eta, parnames] = relaxation_function([2, 0.1, 1024], 'springpot-dashpot-serial', Time)
    # [Et, eta, parnames] = relaxation_function([100, 0.5, 300, 20], 'fractionalSLS', Time)[0:2]
    plt.loglog(Time, Et.Et)
    print(relaxation_function('KV').rel_equation)