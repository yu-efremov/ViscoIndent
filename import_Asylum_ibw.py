# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:39:29 2023

@author: Yuri
# import ibw files
"""
# from data_import_functions import binarywave as ibw
import igor2.binarywave as ibw           # https://pypi.org/project/igor/
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import sys
sys.path.append('D:/MEGAsync/My materials/python')  # /Ting_code
sys.path.append('D:/MEGAsync/My materials/python/Ting_code')

from file_import_qt5simple import file_import_dialog_qt5 as file_import_qt5
from Pars_class import Pars_gen
from make_Results import make_Results





def import_ibw_curve(fileName):
    # fileName = Pars.filedir[0]
    fnameshort = fileName.split("/")
    fnameshort = fnameshort[-1]
    indata = ibw.load(fileName) # Loads data from ibw file into indata

    data = {}    # dictionary for ibw force curve data
    for key in enumerate(indata['wave']['labels'][1][1:]):     
        data[str(key[1])[2:-1]] = indata['wave']['wData'][:,key[0]]  # Places Z and DFL in data

    notedic = {}                                                                           
    for item in str(indata['wave']['note']).split('\\r'):
        try: 
            notedic[item.split(':')[0]] = item.split(':')[1]
        except: 
            # print("Warrning cannot parse note entry: " + item)
            pass

    Parsdict = {}
    Parsdict['InvOLS'] = float(notedic['InvOLS'])*1e9
    Parsdict['k'] = float(notedic['SpringConstant'])
    Parsdict['dT'] = 1/float(notedic['NumPtsPerSec'])

    Z = data['ZSnsr']*1e9  # to nm, initially in [m]
    DFL = data['Defl']*1e9  # to nm, initially in [m]
    # DFL = data['Defl']*1e9/Parsdict['InvOLS']   # to nA
    # plt.plot(Z, DFL)
    curve = np.column_stack([Z, DFL])
    Data = []
    Data.append([0, curve, 0, Parsdict['dT'], fnameshort])
    Data = np.asarray(Data, dtype=object)

    return Data, Parsdict, notedic
           
def import_multifiles_ibw(fileNames):
    Pars = Pars_gen()
    Pars.filedir.append(fileNames[0])
    Pars.TipType = 'Asylum ibw'
    # Pars.InvOLS = 1e-9
    # Pars.k = 1e9
    # Pars.dT = 1                 # Sampling time
    Pars.probe_shape = 'sphere'
    Pars.probe_dimension = 2000  # radius in nm
    Pars.Poisson = 0.5         # Poisson's ratio of the sample
    Pars.height = 0
    Pars.HeightfromZ = 0
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
    indx = 0
    for fileName in fileNames:
        DataC, Parsdict = import_ibw_curve(fileName)[0:2]
        DataC[0][0] = indx
        indx = indx+1
        Data = np.vstack((Data, DataC))
        # Data.append(DataC)
    # Data = np.asarray(Data, dtype=object)
    Results = make_Results(np.shape(Data)[0])
    Pars.InvOLS = Parsdict['InvOLS']  # parameters from the last curve
    Pars.InvOLS = 1  # since DFL in nm
    Pars.k = Parsdict['k']  # parameters from the last curve
    Pars.dT = Parsdict['dT']  # parameters from the last curve
    Pars.PL = np.zeros(np.shape(Data)[0])
    
    return Pars, Data, Results           
        
        
if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # import_opt='single'
    # import_opt='multi_from_folders'
    start_folder = 'D:/MailCloud/AFM_data/asylum/180909-4gels_for_Aplysia_CSC38_1Bn'
    # fileName = file_import_qt5(import_opt, file_types='*.ibw', window_title='select files', start_folder='D:/MailCloud/')
    fileNames = file_import_qt5('multi', file_types='*.ibw', window_title='select files', start_folder=start_folder)
    fileName = fileNames[0]
    # fileName = 'D:/MailCloud/BioMomentum/20201016_ecoflex_indentation/long_sample1_map2.txt'

    # Pars = Pars_gen()
    # vPars.filedir[0] = fileName
    # Pars.filedir.append(fileName)
    # Pars, Data, Results = import_specific_BM(Pars, num_regs)
    Data, Parsdict, notedic = import_ibw_curve(fileName)
    Pars, Data, Results = import_multifiles_ibw(fileNames)
    # runfile('D:/MEGAsync/My materials/python/Ting_code/Viscoindent_dataGUI.py', wdir='D:/MEGAsync/My materials/python/Ting_code', current_namespace=True)    # try:
    #     del app
    # except:
    #     print('noapp') 
    # app = QApplication(sys.argv)
    # ex = App()
    # app.exec()

