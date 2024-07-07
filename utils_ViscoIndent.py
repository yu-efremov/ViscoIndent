# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:29:54 2020

@author: Yuri
"""

import os.path  # for check existance of the spm file
import numpy as np
import shelve
import pickle
from Pars_class import Pars_gen
from make_Results import make_Results
from import_AFM_data import Bruker_import
from tingprocessing_class4 import locate_position, smoothM, HertzBEC, npmax
from Ting_numerical import ting_numerical as tingFCPWL3uni
import sys
if not sys.warnoptions:
    import warnings


def save_AFM_data_pickle(filename, Pars, Data, Results):
    # saving complete dataset
    # global Pars, Data, Results
    # filename = 'D:/MEGAsync/My materials/python/Ting_code/temp_spy_data/test_pickle.dat'
    Pars_dict = Pars.class2dict()
    with open(filename, 'wb') as f:
        dictAFM = {'Pars': Pars_dict, 'Data': Data, 'Results': Results}
        pickle.dump(dictAFM, f)


def save_AFM_data_pickle_short(filename, Pars, Data, Results):
    # saving only minimal set of Data
    # global Pars, Data, Results
    Data2 = np.copy(Data)
    Data2[:, 1] = np.zeros([Data2.shape[0]])  # remove curves data
    print('saving data')
    print(filename)
    Pars_dict = Pars.class2dict()
    with open(filename, 'wb') as f:
        dictAFM = {'Pars': Pars_dict, 'Data': Data2, 'Results': Results}
        pickle.dump(dictAFM, f)
    print('saved')


def load_AFM_data_pickle(filename):
    # load from .dat file
    with open(filename, 'rb') as f:
        data2 = pickle.load(f)
    Pars_dict = data2['Pars']
    if isinstance(Pars_dict, dict):
        Pars = Pars_gen()
        Pars.dict2class(Pars_dict)
    else:
        Pars = Pars_dict
    Data = data2['Data']
    Results = data2['Results']
    return Pars, Data, Results


def load_AFM_data_pickle_short(filename):
    # load from .dat file (short)
    spmfilename = filename[:-3] + 'spm'
    print(spmfilename)
    if not os.path.isfile(spmfilename):  # search for truncated filename
        spmfilename = filename[:-5] + '.spm'
    if not os.path.isfile(spmfilename):
        print('No corresponding .spm file was found in the same folder!')
        Data = [0]
        Data = np.asarray(Data, dtype=object)
        Results = make_Results(np.shape(Data)[0])
        Pars = Pars_gen()
    else:
        with open(filename, 'rb') as f:
            data2 = pickle.load(f)
        Pars_dict = data2['Pars']
        if isinstance(Pars_dict, dict):
            Pars = Pars_gen()
            Pars.dict2class(Pars_dict)
        else:
            Pars = Pars_dict
        Pars.filedir[0] = spmfilename
        Data = data2['Data']
        Results = data2['Results']
        Bruker_data = Bruker_import(Pars)
        Data2 = Bruker_data.Data
        print('data loaded from file')
        kk = 0
        for ii in range(np.shape(Data2)[0]):
            if Data2[ii][0] == Data[kk][0]:
                Data[kk][1] = Data2[ii][1]
                if kk < np.shape(Data)[0]-1:
                    kk = kk+1
                else:
                    break

    return Pars, Data, Results

def AFM_data_pickle_to_short(filename):
    Pars, Data, Results = load_AFM_data_pickle(filename)
    save_AFM_data_pickle_short(filename, Pars, Data, Results)

def curve_from_saved_pars(Pars, cData, cResults):
    currentcurve3 = cData[1]
    DFL_corrs = cData[2]
    cpHertz = cResults['cpHertz']
    Height = cResults['Height']
    EHertz = cResults['EHertz']
    EHertzBEC = cResults['EHertzBEC']

    cpTing = cResults['cpTing']
    E0 = cResults['E0']
    alpha_tau = cResults['alpha_tau']
    Einf = cResults['Einf']
    E0BEC = cResults['E0BEC']
    alpha_tauBEC = cResults['alpha_tauBEC']
    EinfBEC = cResults['EinfBEC']

    # Sens = Pars.InvOLS                  # sensetivity, nm/V
    # Stiffness = Pars.k                  # cantilever spring constant, nN/nm
    modelprobe = Pars.probe_shape
    Radius = Pars.probe_dimension       # radius (nm) or angle (degrees)
    Poisson = Pars.Poisson              # Poisson's ratio of the sample
    dT = Pars.dT                        # Sampling time
    modelting = Pars.viscomodel
    # hydrodragcorr = Pars.hydro.corr # hydrodinamic drag correction
    # try:
    #     Fixedpars = Pars.fixed_values
    # except:
    #     Fixedpars = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)

    if modelprobe == 'sphere':
        K1 = 4*Radius**0.5/3
        power = 1.5
    elif modelprobe == 'cone' or modelprobe == 'pyramid':
        power = 2
        if modelprobe == 'cone':
            K1 = 2/np.pi*np.tan(Radius*np.pi/180)
        elif modelprobe == 'pyramid':
            K1 = 1.406/2*np.tan(Radius*np.pi/180)
    elif modelprobe == 'cylinder':
        K1 = 2*Radius  # before integral no speed
        power = 1
    K1 = K1/(1-Poisson**2)*1e-9

    # try:
    #     HeightfromZ = Pars.HeightfromZ
    # except:
    #     HeightfromZ = 0

    # Height = Pars.height

    rawZ = currentcurve3[:, 0]
    rawDFL = currentcurve3[:, 1]

    if rawZ[0] > rawZ[30]:
        CurveDir = -1  # decreasing curve
    else:  # default case
        CurveDir = 1  # increasing curve
    Displ = CurveDir*(rawZ-rawZ[0])
    DisplSpeed = np.diff(Displ)/dT
    DisplSpeed = np.append(DisplSpeed, DisplSpeed[-1])
    DisplSpeed = smoothM(DisplSpeed, 5)
    DFLc = rawDFL - DFL_corrs[0]*Displ - DFL_corrs[1] - DFL_corrs[2]*DisplSpeed/Pars.InvOLS/Pars.k
    Forcec = DFLc*Pars.k
    indentationfull = Displ - np.min(Displ) - cpHertz - DFLc*Pars.InvOLS

    funHertzfit = lambda ind, a, b: HertzBEC([a, b], K1, Poisson,
                                             Radius, power, 0, np.nan,
                                             modelprobe, ind)

    warnings.simplefilter("ignore")

    if EHertz > 0:
        FitAppr = funHertzfit(indentationfull, EHertz, 0)

        if EHertzBEC > 0:
            funHertzfitBEC = lambda ind, a, b: HertzBEC([a, b], K1, Poisson,
                                                        Radius, power, Height,
                                                        np.nan, modelprobe, ind)
            FitApprBEC = funHertzfitBEC(indentationfull, EHertzBEC, 0)
            # plt.plot(Displ, rawDFL)
            # #plt.plot(indentationfull)
            # plt.plot(indentationfull, Forcec)
            # plt.plot(indentationfull, FitApprBEC)
            # plt.plot(currentcurve3[:, 2], currentcurve3[:, 4])
        elif EHertzBEC == 0 and EHertz > 0:
            FitApprBEC = FitAppr
        else:
            FitApprBEC = np.nan*indentationfull
        currentcurve3 = np.c_[currentcurve3, indentationfull, Forcec, FitApprBEC]



    if E0 > 0:
        locationcpTing = locate_position(cpTing, Displ)
        [ApprLength, maxZ] = npmax(Displ)
        MaxInd = int(ApprLength-locationcpTing)  # points in Approach indentation
        parf = [E0, alpha_tau, Einf]
        indentationfullTing = Displ - cpTing - DFLc*Pars.InvOLS
        FullLength = len(indentationfullTing)-1
        indend = np.min([locationcpTing+int(MaxInd*2.15), len(indentationfullTing)-1])
        for ala in range(ApprLength, FullLength):  # alternative search for indend based on Z value
            if Displ[ala] < Displ[locationcpTing]:
                indend = ala
                break
        indentationTing = indentationfullTing[locationcpTing+1:indend]
        indentationTing[indentationTing<0] = 0
        Force_fitT = tingFCPWL3uni(parf, 0, Poisson, Radius, dT, MaxInd, 0, modelting, modelprobe, indentationTing)[0]
        # plt.plot(indentationTing, Force_fitT)
        # plt.plot(indentationfullTing, Forcec)

        if E0BEC > 0:
            locationcpTing = locate_position(cpTing, Displ)
            [ApprLength, maxZ] = npmax(Displ)
            MaxInd = int(ApprLength-locationcpTing) # points in Approach indentation
            parf = [E0BEC, alpha_tauBEC, EinfBEC]
            indentationfullTing = Displ - cpTing - DFLc*Pars.InvOLS
            FullLength = len(indentationfullTing)-1
            indend = np.min([locationcpTing+int(MaxInd*2.15), len(indentationfullTing)-1])
            for ala in range(ApprLength, FullLength):  # alternative search for indend based on Z value
                if Displ[ala] < Displ[locationcpTing]:
                    indend = ala
                    break
            indentationTing = indentationfullTing[locationcpTing+1:indend]
            indentationTing[indentationTing<0] = 0
            Force_fitTBEC = tingFCPWL3uni(parf, 0, Poisson, Radius, dT, MaxInd, Height, modelting, modelprobe, indentationTing)[0]
            # plt.plot(indentationTing, Force_fitT)
            # plt.plot(currentcurve3[:, 2]-(cpTing-cpHertz), currentcurve3[:, 5])
            # plt.plot(indentationfullTing, Forcec)
        elif E0BEC == 0 and E0 > 0:
            Force_fitTBEC = Force_fitT
        Force_fitTBEC2 = np.full(np.shape(Forcec), np.nan)
        Force_fitTBEC2[locationcpTing+1:locationcpTing+1+len(Force_fitTBEC)] = Force_fitTBEC
        currentcurve3 = np.c_[currentcurve3, Force_fitTBEC2]
        # plt.plot(Force_fitTBEC2)
        # plt.plot(currentcurve3[:, 2], currentcurve3[:, 5])

    # currentcurve3 = np.c_[currentcurve3, indentationfull, Forcec, FitApprBEC, Force_fitTBEC, FitAppr, Force_fitT]

    warnings.simplefilter("default")
    return currentcurve3


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    # filename = file_import_dialog_qt5_simplest(import_opt='single', window_title='select files')
    #  filename = 'D:/MEGAsync/My materials/python/Ting_code/examples/Bruker_forcevolume_cells.dat'
    # Pars, Data, Results = load_AFM_data_pickle(filename)
    # Pars, Data, Results = load_AFM_data_pickle_short(filename)
    # Pars, Data, Results = save_AFM_data_pickle_short(filename, Pars, Data, Results)
    # filename= 'D:/MailCloud/AFM_data/BrukerResolve/cytotoxicity/20211118_Ref52_ACR+NaOCL/control.0_000062.dat'
    # Pars, Data, Results = load_AFM_data_pickle_short(filename)
    currentcurve3 = curve_from_saved_pars(Pars, Data[0], Results.loc[0, :])
    plt.plot(currentcurve3[:, 2], currentcurve3[:, 3])
    plt.plot(currentcurve3[:, 2], currentcurve3[:, 4])