# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 22:16:26 2022
select several folders for import microtester data
@author: Yuri
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('D:/MEGAsync/My materials/python')  # /Ting_code
sys.path.append('D:/MEGAsync/My materials/python/Ting_code')
sys.path.append('D:/Yuri Efremov/MEGAsync/My materials/python')  # /Ting_code
sys.path.append('D:/Yuri Efremov/MEGAsync/My materials/python/Ting_code')

from Pars_class import Pars_gen
from make_Results import make_Results



def import_Microtester(filename):
    DataTable = pd.read_csv(filename, header=0, sep=',', decimal='.', encoding = 'ISO-8859-1')
    DataTable.columns = DataTable.columns.str.replace('[(,), ,&,°]', '', regex='True')
    return DataTable

def import_Microtester_toVI(fileName, Data):
    DataTable = import_Microtester(fileName)
   
    # Z = (DataTable.BaseDisplacementum-DataTable.TipDisplacementum)*1000 # to nm
    dT = np.mean(np.diff(DataTable.Timems))/1000
    Z = (DataTable.TipDisplacementum - DataTable.CurrentSizeum[0])*1000  # to nm
    F = DataTable.ForceuN*1000  # to nN 
    curve = np.column_stack([Z, F])
    # plt.plot(Z, F)
    #ind = Data.shape[0]
    ind = len(Data)
    Data.append([ind, curve, 0, dT])
    # Data = np.append(Data, np.array([ind, curve, 0, dT]), axis=0)

    return Data

def import_Microtester_toVI_multi(fileNames):
    Pars = Pars_gen()
    Pars.filedir.append(fileNames[0])
    Pars.TipType = 'Macro'
    Pars.InvOLS = 1e-9
    Pars.k = 1e9
    Pars.probe_shape = 'sphere'
    Pars.probe_dimension = 0.25*1e6
    Pars.Poisson = 0.5         # Poisson's ratio of the sample
    Pars.dT = 1                 # Sampling time
    Pars.height = 0
    Pars.HeightfromZ = 1
    Pars.viscomodel = 'elastic'
    Pars.hydro.corr = 1  # -1: no line subtraction too, 1 - subtract
    Pars.hydro.corr_type = 2
    Pars.hydro.speedcoef = 0
    Pars.cp_adjust = 1
    fnameshort = fileNames[0].split("/")
    fnameshort = fnameshort[-1]
    Pars.fnameshort = []
    Pars.fnameshort.append(fnameshort)
    Pars.right_border_cp1 = 0.99
    
    Data = []
    # Data = np.asarray(Data, dtype=object)
    for fileName in fileNames:
        Data = import_Microtester_toVI(fileName, Data)
    Data = np.asarray(Data, dtype=object)
    Results = make_Results(np.shape(Data)[0])
    Pars.PL = np.zeros(np.shape(Data)[0])
    
    return Pars, Data, Results

    
if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    import sys
    import chardet  # decode file encoding
    import glob
    sys.path.append("../Ting_code")
    from file_import_qt5simple import file_import_dialog_qt5 as file_import_qt5
    filename = "D:/MailCloud/Microsquisher/20200215-MSC7d-2k1k/Test068/Test068Data.csv"
    fileNames = ["D:/MailCloud/Microsquisher/20220804_lenticules/Test007_4Voroncov_OD_d1/Test007Data.csv", \
                 "D:/MailCloud/Microsquisher/20220804_lenticules/Test008_4Voroncov_OD_d2/Test008Data.csv"]

    select_manually = 1    
    if select_manually == 1:
        import_opt = 'multi_folders'  # 
        start_folder = 'D:/MailCloud/Microsquisher'
        # start_folder = 'D:/MailCloud/AFM_data/BrukerResolve'
        pathnames = file_import_qt5(import_opt=import_opt, start_folder=start_folder)
        fileNames = []
        for pathnamec in pathnames:
            tfiles = glob.glob(pathnamec+"/"+"*.csv")
            if len(tfiles)>0:
                fileNames.append(tfiles[0])
        # fileNames = list(filter(None, fileNames))
        
    with open(filename, 'rb') as f:
        enc = chardet.detect(f.readline())  # or readline if the file is large
    
    DataTable = pd.read_csv(filename, header=0, sep=',', decimal='.', encoding = 'ISO-8859-1')
    DataTable.columns = DataTable.columns.str.replace('[(,), ,&,°]', '', regex='True')
    
    plt.plot(DataTable.Timems, DataTable.ForceuN)
    
    Pars, Data, Results = import_Microtester_toVI_multi(fileNames)