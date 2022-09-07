# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:16:01 2020

@author: Yuri
"""

import numpy as np
import pandas as pd
import re
import itertools
import matplotlib.pyplot as plt
import datetime

def Biomomentum_Import_common_fun(Dtype, filename):

    # file = open(filename, 'r')
    tempcount=0
    linenum=0


    fields_array = []
    pattern = '<*>'
    with open(filename, 'r') as file:
        line = file.readline().rstrip()
        while line:
            linenum=linenum+1
            if re.search(pattern, line):
                if not  re.search('<INFO>', line) and not re.search('<END INFO>', line):
                    fields_array.append((line, linenum))
            line = file.readline().rstrip()
    fields_array = pd.DataFrame(fields_array)
    # fields_names = fields_array.iloc[:,0].values
    # fields_names = list(filter(lambda x: x not in ['<Mach-1 File>', '<DATA>', '<END DATA>'], list(fields_array.iloc[:,0].values)))
    lines_array = []  # name, header start - end, data start - end
    for ii in range (1, fields_array.shape[0]):
        if not re.search('<DATA>', fields_array.iloc[ii][0]) and not re.search('<END DATA>', fields_array.iloc[ii][0]) and not re.search('<divider>', fields_array.iloc[ii][0]):
            lines_array.append((fields_array.iloc[ii][0], fields_array.iloc[ii-1][1]+1, fields_array.iloc[ii][1]-1, fields_array.iloc[ii+1][1]+1, fields_array.iloc[ii+2][1]-1))
    # get headers
    Headers = []
    lines_array = pd.DataFrame(lines_array)   # name, header start - end, data start - end
    
    fields_names = list(lines_array.iloc[:,0].values)
    
    for ii in range (lines_array.shape[0]):
        temp_header = []
        lines = range(lines_array.iloc[ii][1], lines_array.iloc[ii][2])
        with open(filename, 'r') as file:
            i = 0
            for line in file:
                if i in lines:
                    temp_header.append(line)
                # if i == lines_array.iloc[0][3]-1:
                #     Tablenames = line
                i += 1
        Headers.append(temp_header)
        with open(filename, 'r') as file:
            i = 0
            for line in file:
                if i == lines_array.iloc[0][3]-1:
                    Tablenames = line.split("\t")
                    break
                i += 1

    # both approaches provide same speed
    # data_cell = []
    # # try consecutive reading (combine all ranges) and then split?
    # for ii in range (lines_array.shape[0]):
    #     indexesP1 = range(0, lines_array.iloc[ii][3]-1)
    #     indexesP2 = range(lines_array.iloc[ii][4], lines_array.iloc[-1][4]+1)
    #     indexesP = itertools.chain(indexesP1, indexesP2)
    #     print (indexesP1, indexesP2)
    #     df = pd.read_csv(filename, skiprows=[i for i in indexesP], delimiter='\t')
    #     # print(df)
    #     data_cell.append(df)

    data_cell = []
    # try consecutive reading (combine all ranges) and then split?
    indexesP1 = range(0, lines_array.iloc[0][3]-1)
    indexesP = itertools.chain(indexesP1)
    for ii in range (lines_array.shape[0]-1):
        indexesP2 = range(lines_array.iloc[ii][4], lines_array.iloc[ii+1][3]-1)
        indexesP = itertools.chain(indexesP, indexesP2)
        # print (indexesP1, indexesP2)
    df = pd.read_csv(filename, skiprows=[i for i in indexesP], delimiter='\t')
    
    # print(df)
    # print (df.iloc[:,-1].dtype)
    # df.iloc[:,-1].astype(float)
    # df.iloc[:,-1].multiply(toN)
    
    
    dlenghts = []
    dcount = 0
    for ii in range (lines_array.shape[0]):
        print(ii)
        dlenghts.append(lines_array.iloc[ii][4] - lines_array.iloc[ii][3])
        # data_cell.append(df.iloc[dcount:dcount+dlenghts[ii]].astype(float).multiply(toN))  # do not multiply all
        data_cell.append(df.iloc[dcount:dcount+dlenghts[ii]].astype(float))  # do not multiply all
        dcount = dcount + dlenghts[ii]+1

    # data_cell.append(df)
    # data_cell.append([df.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)])


    # df = ('myfile.csv', sep=',',header=None)
    
    if len(Tablenames) < 8:  # now only for one column (may be make by column name)
        toN = 0.00980665
        for i, v in enumerate(data_cell):
            df = data_cell[i]
            df.iloc[:,-1] = df.iloc[:,-1].multiply(toN)
            df = df.rename({'Fz, gf': 'Fz, N'}, axis='columns')
            data_cell[i] = df
    else:
        toN = -1
        for i, v in enumerate(data_cell):
            df = data_cell[i]
            df[['Fz, N']] = df[['Fz, N']].multiply(toN)
            data_cell[i] = df
    Start_timeS1 = Headers[0][1]
    Start_timeS2 = Start_timeS1.split(":	")[1][0:-1]
    Start_time = datetime.datetime.strptime(Start_timeS2,'%H:%M:%S.%f')
    
    for i, v in enumerate(data_cell): # adjust time
            Loc_timeS1 = Headers[i][1]
            Loc_timeS2 = Loc_timeS1.split(":	")[1][0:-1]
            Loc_time = datetime.datetime.strptime(Loc_timeS2,'%H:%M:%S.%f')
            Add_time = (Loc_time-Start_time).total_seconds()
            df = data_cell[i]
            df.iloc[:,0] = df.iloc[:,0] + Add_time
    file.close()            
    return fields_array, lines_array, Headers, data_cell

if __name__ == '__main__':
    import time
    import sys
    sys.path.append('D:/MEGAsync/My materials/python')  # /Ting_code
    sys.path.append('D:/MEGAsync/My materials/python/Ting_code')
    from file_import_qt5simple import file_import_dialog_qt5 as file_import_qt5
    filename = 'D:/MailCloud/BioMomentum/20200814_MMM/10_90_s1bone_2.txt'
    # filename = 'D:/MailCloud/BioMomentum/20200623-ecoflex-large/largeS_test1_dmamulti4lowA.txt'
    # filename = 'D:/MailCloud/BioMomentum/20210415_gelatin/gelatin3%_compression_0.5N.txt'
    # filename = file_import_qt5(import_opt='single', file_types='*.txt', window_title='select files', start_folder='D:/MailCloud/BioMomentum')

    t1 = time.time()
    [fields_array, lines_array, Headers, data_cell] = Biomomentum_Import_common_fun('common', filename)
    t2 = time.time()
    print(t2-t1)
    Full_data = data_cell[0]
    for i, v in enumerate(data_cell): # adjust time
        if i>0:
            Full_data = pd.concat([Full_data, data_cell[i]], axis=0)

    Time = Full_data.iloc[:, 0].values
    Z = Full_data.iloc[:, 1].values
    F = Full_data.iloc[:, -1].values
    Numpy_data = np.column_stack([Time, Z, F])
    plt.figure(figsize=(10, 7))
    plt.plot(Time, Z)
    plt.show()

    from matplotlib.widgets import SpanSelector
    
    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(Z, (xmin, xmax))
        indmax = min(len(Z) - 1, indmax)
    
        thisx = Z[indmin:indmax]
        thisy = F[indmin:indmax]
        linfitcoeffs = np.polyfit(thisx, thisy, 1)
        linfitcurvefun = np.poly1d(linfitcoeffs)
        linfitcurve = linfitcurvefun(thisx)
        try:
            # ax.line2.remove()
            ax.lines[1].remove()
        except:
            pass
        # if 'line2' in ax.lines:
        #     line2.remove()
        line2 = ax.plot(thisx, linfitcurve, 'r')
        # line1.set_data(thisx, thisy)
        # ax.set_xlim(thisx[0], thisx[-1])  # zoom
        # ax.set_ylim(thisy.min(), thisy.max())
        # ax.axvspan(thisx[0], thisx[-1], color='y', alpha=0.5, lw=0)
        fig.canvas.draw()
    
    
    
    cellnum = 0
    Z = data_cell[cellnum].iloc[:, 1].values
    F = data_cell[cellnum].iloc[:, -1].values
    F = data_cell[cellnum].filter(regex='Fz').values
    fig, ax = plt.subplots(1,1)
    line1 = ax.plot(Z, F)
    # span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
    #                 rectprops=dict(alpha=0.3, facecolor='red'), span_stays='False')
    
    
    