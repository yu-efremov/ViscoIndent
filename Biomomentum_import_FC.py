# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:08:03 2020

@author: Yuri
"""

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import sys
sys.path.append('D:/MEGAsync/My materials/python')  # /Ting_code
sys.path.append('D:/MEGAsync/My materials/python/Ting_code')
from Biomomentum_Import_common_fun import Biomomentum_Import_common_fun
from file_import_qt5simple import file_import_dialog_qt5 as file_import_qt5
from Pars_class import Pars_gen
from make_Results import make_Results


def import_specific_BM(fileName, num_regs):
        # fileName = Pars.filedir[0]
        fnameshort = fileName.split("/")
        fnameshort = fnameshort[-1]
        [fields_array, lines_array, Headers, data_cell] = Biomomentum_Import_common_fun('common', fileName)
        fields_names = list(lines_array.iloc[:,0].values)
        dT = data_cell[0].iloc[1,0] - data_cell[0].iloc[0,0]
        Data = []
        if num_regs==0:
            num_regs = range(0, len(data_cell))
        for ind, i in enumerate(num_regs):
           idf = data_cell[i]
           dT = idf.iloc[1,0] - idf.iloc[0,0]
           Z = idf.iloc[:, 1].values*1000000  # to nm
           F = idf.filter(regex='Fz').values*1e9  # to nN (from gF - at import)
           # F = idf.iloc[:, -1].values*1e9  # to nN (from gF - at import)
           # curve = np.asarray([Z, F])
           curve = np.column_stack([Z, F])
           # plt.plot(Z, F)
           Data.append([ind, curve, 0, dT, fnameshort])

        Data = np.asarray(Data)

        return Data
           
def import_multifiles_BM(fileNames, num_regs):
    Pars = Pars_gen()
    Pars.filedir.append(fileNames[0])
    Pars.TipType = 'Macro'
    Pars.InvOLS = 1e-9
    Pars.k = 1e9
    Pars.probe_shape = 'sphere'
    Pars.probe_dimension = 6.35*1e6/2  # radius in nm
    Pars.Poisson = 0.5         # Poisson's ratio of the sample
    Pars.dT = 1                 # Sampling time
    Pars.height = 0
    Pars.HeightfromZ = 1
    Pars.viscomodel = 'elastic'
    Pars.hydro.corr = 1  # -1: no line subtraction too, 1 - subtract
    Pars.hydro.corr_type = 3  # for Biomomentum
    Pars.hydro.speedcoef = 0
    Pars.cp_adjust = 1
    fnameshort = fileNames[0].split("/")
    fnameshort = fnameshort[-1]
    Pars.fnameshort = []
    Pars.fnameshort.append(fnameshort)
    Pars.right_border_cp1 = 0.99
    
    Data = np.empty([0, 5])
    for fileName in fileNames:
        DataC = import_specific_BM(fileName, num_regs)
        Data = np.vstack((Data,DataC))
        # Data.append(DataC)
    # Data = np.asarray(Data, dtype=object)
    Results = make_Results(np.shape(Data)[0])
    Pars.PL = np.zeros(np.shape(Data)[0])
    
    return Pars, Data, Results           
        
        
if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    import_opt='single'
    import_opt='multi_from_folders'
    fileNames = file_import_qt5(import_opt, file_types='*.txt', window_title='select files', start_folder='D:/MailCloud/BioMomentum')
    # fileName = 'D:/MailCloud/BioMomentum/20201016_ecoflex_indentation/long_sample1_map2.txt'
    num_regs = list(range(0, 9))
    num_regs = [1, 3, 5, 7, 9]
    num_regs = 0
    # Pars = Pars_gen()
    # vPars.filedir[0] = fileName
    # Pars.filedir.append(fileName)
    # Pars, Data, Results = import_specific_BM(Pars, num_regs)
    Pars, Data, Results = import_multifiles_BM(fileNames, num_regs)
    # runfile('D:/MEGAsync/My materials/python/Ting_code/Viscoindent_dataGUI.py', wdir='D:/MEGAsync/My materials/python/Ting_code', current_namespace=True)    # try:
    #     del app
    # except:
    #     print('noapp') 
    # app = QApplication(sys.argv)
    # ex = App()
    # app.exec()

