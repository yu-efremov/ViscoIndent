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
from import_Bruker_spm import Bruker_import
from force_curve_fit import locate_position, smoothM, HertzBEC, DMTBEC, JKRnoBEC, npmax
from Ting_numerical import ting_numerical
import sys
if not sys.warnoptions:
    import warnings

def check_Pars(Pars):
    if not hasattr(Pars, 'adhesion_model'):
        Pars.adhesion_model = 'none'
    if not hasattr(Pars, 'adhesion_region'):
        Pars.adhesion_region = 'retraction'
    return Pars        

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
    currentcurve3 = cData[1][:, 0:2]
    DFL_corrs = cData[2]
    cpHertz = cResults['cpHertz']
    Height = cResults['Height']
    EHertz = cResults['EHertz']
    EHertzBEC = cResults['EHertzBEC']
    E_adhesion = cResults['E_adhesion']
    adhesion = cResults['adhesion']
    try:
        cp_adhesion = cResults['cp_adhesion']
    except:
        cp_adhesion = cpHertz
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
    try:
        dT = cData[3]  # dT in Data array position
        if dT == 0:
            dT = Pars.dT
    except:
        dT = Pars.dT

    modelting = Pars.viscomodel
    
    adhesion_model = Pars.adhesion_model  # none, DMT, JKR
    adhesion_region = Pars.adhesion_region # 'approach' 'retraction' 'both'
    adhesion_pars = [adhesion_model, adhesion_region, adhesion]
    # hydrodragcorr = Pars.hydro.corr # hydrodinamic drag correction
    # try:
    #     Fixedpars = Pars.fixed_values
    # except:
    #     Fixedpars = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)

    if modelprobe in {'sphere', 'sphere_correction1'}:
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
    elif modelprobe == 'spheroid': # cylinder
        power = 1.5
        K1 = 4*Radius**0.5/3/2**1.5  # doubled deformation, radius should be same for all
        # K1 = K1/(Radius/Height)**0.5  # base diameter on measured height
        # K12 = Radius**0.5    
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
    Forcec = DFLc*Pars.InvOLS*Pars.k
    # plt.plot(DFLc) plt.plot(rawDFL)
    # indentationfull = Displ - np.min(Displ) - cpHertz - DFLc*Pars.InvOLS
    indentationfull_noCP = Displ - Displ[0] - DFLc*Pars.InvOLS  #  - cpHertz for hertzian

    funHertzfit = lambda ind, a, b: HertzBEC([a, b], K1, Poisson,
                                             Radius, power, 0, np.nan,
                                             modelprobe, ind)
    if adhesion_model == 'DMT':
        funAdhfit = lambda ind, a, b, c: DMTBEC([a, b, c], K1, Poisson, 
                                                Radius, power, 0, np.nan, 
                                                modelprobe, ind)
    elif adhesion_model == 'JKR':
        funAdhfit = lambda ind, a, b, c: JKRnoBEC([a, b, c], K1, Poisson, 
                                                Radius, power, 0, np.nan, 
                                                modelprobe, ind)        
    

    warnings.simplefilter("ignore")

    if EHertz > 0:
        [ApprLength, maxZ] = npmax(Displ)
        locationcpHertz = locate_position(cpHertz, Displ)
        cp_Selected = cpHertz
        MaxInd = int(ApprLength-locationcpHertz)  # points in Approach indentation
        FitAppr = np.full(len(Displ), np.nan)
        FitAppr[locationcpHertz:ApprLength] = funHertzfit(indentationfull_noCP[locationcpHertz:ApprLength] - cpHertz, EHertz, 0)
        FitAdh = np.full(len(Displ), np.nan)
        if E_adhesion>0:
            cp_Selected = cp_adhesion
            locationcp_adhesion = locate_position(cp_adhesion, Displ)
            FitAdh[locationcp_adhesion:ApprLength] = funAdhfit(indentationfull_noCP[locationcp_adhesion:ApprLength] - cp_adhesion, E_adhesion, 0, adhesion)
        else:
            FitAdh = np.full(len(Displ), np.nan)
        # FitAdh[:locationcpHertz-1] = np.nan
        # FitAdh[apprlength:locationcpHertz-1] = np.nan

        # plt.plot(indentationfull_noCP, DFLc)
        # plt.plot(indentationfull_noCP, FitAppr)
        # plt.plot(indentationfull_noCP, FitAdhBEC)

        if EHertzBEC > 0:
            funHertzfitBEC = lambda ind, a, b: HertzBEC([a, b], K1, Poisson,
                                                        Radius, power, Height,
                                                        np.nan, modelprobe, ind)
            funAdhfit = lambda ind, a, b, c: DMTBEC([a, b, c], K1, Poisson, 
                                                    Radius, power, 0, np.nan, 
                                                    modelprobe, ind)
            FitApprBEC = np.full(len(Displ), np.nan)
            FitApprBEC[locationcpHertz:ApprLength] = funHertzfitBEC(indentationfull_noCP[locationcpHertz:ApprLength] - cpHertz, EHertzBEC, 0)
            if E_adhesion>0:
                FitAdhBEC = np.full(len(Displ), np.nan)
                FitAdhBEC[locationcp_adhesion:ApprLength] = funAdhfit(indentationfull_noCP[locationcp_adhesion:ApprLength] - cp_adhesion, E_adhesion, 0, adhesion)
            # plt.plot(Displ, rawDFL)
            # plt.plot(indentationfull)
            # plt.plot(indentationfull, Forcec)
            # plt.plot(indentationfull, FitApprBEC)
            # plt.plot(indentationfull, FitAdhBEC)
            # plt.plot(currentcurve3[:, 2], currentcurve3[:, 4])
            else:
                FitAdhBEC = np.full(len(Displ), np.nan)
        elif (EHertzBEC == 0 or np.isnan(EHertzBEC)) and EHertz > 0:
            FitApprBEC = FitAppr
            FitAdhBEC = FitAdh
        else:
            FitApprBEC = np.nan*indentationfull_noCP
        # plt.plot(FitApprBEC)
        currentcurve3 = np.c_[currentcurve3, indentationfull_noCP - cp_Selected, Forcec, FitApprBEC, FitAdhBEC]



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
        Force_fitT = ting_numerical(parf, adhesion_pars, Poisson, Radius, dT, MaxInd, 0, modelting, modelprobe, indentationTing)[0]
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
            Force_fitTBEC = ting_numerical(parf, adhesion_pars, Poisson, Radius, dT, MaxInd, Height, modelting, modelprobe, indentationTing)[0]
            # plt.plot(indentationTing, Force_fitT)
            # plt.plot(currentcurve3[:, 2]-(cpTing-cpHertz), currentcurve3[:, 5])
            # plt.plot(indentationfullTing, Forcec)
        elif (E0BEC == 0 or np.isnan(E0BEC)) and E0 > 0:
            Force_fitTBEC = Force_fitT
        Force_fitTBEC2 = np.full(np.shape(Forcec), np.nan)
        Force_fitTBEC2[locationcpTing+1:locationcpTing+1+len(Force_fitTBEC)] = Force_fitTBEC
        currentcurve3 = np.c_[currentcurve3, Force_fitTBEC2]
        # plt.plot(Force_fitTBEC2)
        # plt.plot(currentcurve3[:, 2], currentcurve3[:, 5])

    # currentcurve3 = np.c_[currentcurve3, indentationfull, Forcec, FitApprBEC, Force_fitTBEC, FitAppr, Force_fitT]

    warnings.simplefilter("default")
    # print(currentcurve3)
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
    global Pars, Results, Data
    kk=0
    currentcurve3 = curve_from_saved_pars(Pars, Data[kk], Results2.loc[kk, :])
    plt.plot(currentcurve3[:, 2], currentcurve3[:, 3])
    plt.plot(currentcurve3[:, 2], currentcurve3[:, 4])
    plt.plot(currentcurve3[:, 2], currentcurve3[:, 5])
