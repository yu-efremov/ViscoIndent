# -*- coding: utf-8 -*-
"""
@author: Yuri Efremov
fixed fit Ting
"""
import numpy as np
from scipy.optimize import curve_fit
import sys
if not sys.warnoptions:
    import warnings


def interlace(a, x, fix):
    a[~fix] = x
    return a


def fixedfit(ModelFun, par0, bounds, Fixedpars, x, y):
    fixedp = Fixedpars[0, :]
    fixedv = Fixedpars[1, :]
    par0 = np.asarray(par0)
    par2 = par0
    par2 = np.asarray(par2)
    parboundsnp = np.asarray(bounds)
    FixedparsNF = np.array([[0, 0, 1], [0, 0.5, 0]])
    #   if np.isnan(SurfZ)==0 and HeightfromZ==1:
    #       Height=-Displ[indstart]-SurfZ

    # fixed variables without lmfit
    numfitpars = 3-sum(fixedp)
    warnings.simplefilter("ignore")
    parbounds2 = parboundsnp[:, fixedp == 0]
    parbounds = tuple(map(tuple, parboundsnp))
    parbounds2 = tuple(map(tuple, parbounds2))
    par2 = par0[fixedp == 0]
    if numfitpars == 3:
        funTing = lambda x, a, b, c: ModelFun([a, b, c], x)
    elif numfitpars == 2:
        if fixedp[0] == 1:
            funTing = lambda x, a, b: ModelFun([fixedv[0], a, b], x)
        elif fixedp[1] == 1:
            funTing = lambda x, a, b: ModelFun([a, fixedv[1], b], x)
        elif fixedp[2] == 1:
            funTing = lambda x, a, b: ModelFun([a, b, fixedv[2]], x)
    elif numfitpars == 1:
        if fixedp[0] == 0:
            funTing = lambda x, a: ModelFun([a, fixedv[1], fixedv[2]], x)
        elif fixedp[1] == 0:
            funTing = lambda x, a: ModelFun([fixedv[0], a, fixedv[2]], x)
        elif fixedp[2] == 0:
            funTing = lambda x, a: ModelFun([fixedv[0], fixedv[1], a], x)

    fitTingpar, fitTingcov = curve_fit(funTing, x, y, par2, bounds=parbounds2)
    Force_fitn = funTing(x, *fitTingpar)
    return Force_fitn, fitTingpar, fitTingcov
