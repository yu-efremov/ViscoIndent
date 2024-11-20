
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 23:11:51 2022

@author: Yuri

# multicore processing test
"""

import multiprocessing
from itertools import product
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import shelve
import os.path  # for check existance of the spm file
import pickle

from make_Results import make_Results
from tingprocessing_class4 import tingsprocessingd1
from utils_ViscoIndent import save_AFM_data_pickle, load_AFM_data_pickle, \
    load_AFM_data_pickle_short, save_AFM_data_pickle, \
    save_AFM_data_pickle_short, curve_from_saved_pars
from file_import_qt5simple import file_import_dialog_qt5
from Pars_class import Pars_gen
from import_AFM_data import Bruker_import
from selection_windows_common_gui import selection_win1


def load_AFM_data_pickle_short2(filename):
    spmfilename = filename[:-5] + '.spm'  # modified
    print(spmfilename)
    if not os.path.isfile(spmfilename):
        print('No corresponding .spm file was found in the same folder!')
        Data = [0]
        Data = np.asarray(Data)
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


def open_AFM_data(filename):

    with shelve.open(filename) as my_shelf:
        for key in my_shelf:
            print(key)
            Pars = my_shelf['Pars']
            Data = my_shelf['Data']
            Results = my_shelf['Results']
    my_shelf.close()
    return Pars, Data, Results


def save_test(Pars, Data, Results):
    # global Pars, Data, Results
    filename = 'D:/MEGAsync/My materials/python/Ting_code/temp_spy_data/test_multi'
    with shelve.open(filename, 'n') as my_shelf:
        my_shelf['Pars'] = Pars
        my_shelf['Data'] = Data
        my_shelf['Results'] = Results
    print('Saved')


def tingformulti(kk, Pars, Data, Results):
    Datam = []
    Datam[0] = np.copy(Data[kk])
    Resultsm = Results.iloc[kk:kk+1].copy()
    Results2, Data2 = tingsprocessingd1(0, Pars, Datam, Resultsm)
    return Results2, Data2


def tingformulti2(kk, Pars, Data, Results):
    try:
        Results, Data = tingsprocessingd1(kk, Pars, Data, Results)
    except:
        True
    return Results, Data


if __name__ == '__main__':
    print("Number of cpu : ", multiprocessing.cpu_count())
    num_cores = multiprocessing.cpu_count()
    import_opt = 'all_in_selected_folders'
    import_opt = 'multi'
    filenames = file_import_dialog_qt5(import_opt=import_opt,
                                       file_types='*.dat',
                                       window_title='select dat file')
    # fileName = filenames[0]
    for fileName in filenames:
        Pars, Data, Results = load_AFM_data_pickle_short(fileName)
        Pars.viscomodel = 'sPLRetatest'
        Results3 = make_Results(1)
        t1 = time.time()
        
        #out = Parallel(n_jobs=num_cores-2)(delayed(tingsprocessingd1)(Pars, Data[kk]) for kk in range(0, 25))  # from class4
        out = Parallel(n_jobs=num_cores-2)(delayed(tingsprocessingd1)(Pars, Data[kk]) for kk in range(0, Data.shape[0]-1))  # from class4
        t2 = time.time()
        print(t2-t1)
        if np.shape(Data)[1]<3:
            # Data.append(np.zeroes(np.shape(Data)[0]))
            Data = np.append(Data, np.zeros((np.shape(Data)[0],1)), axis=1)
        for kk in range(0, len(out)):
            Data[kk][2] = out[kk][2]  # TODO 2 is dT, move to 3?
            Results.loc[kk, :] = out[kk][0].loc[0, :]
            
        fileNameSav = fileName # rewrite file
        save_AFM_data_pickle_short(fileNameSav, Pars, Data, Results)
