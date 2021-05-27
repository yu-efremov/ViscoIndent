# -*- coding: utf-8 -*-
"""
@author: Yuri Efremov
import and data processing

"""
import numpy as np
import shelve
import pandas as pd
import seaborn as sns  # use boxplot, stripplot, swarmplot, violinplot, catplot
import matplotlib.pyplot as plt
import itertools

from Pars_class import Pars_gen
from file_import_qt5simple import file_import_dialog_qt5
from make_Results import make_Results


def open_AFM_data(filename):
    import shelve
    with shelve.open(filename) as my_shelf: 
        for key in my_shelf:
            print(key)
        Pars = my_shelf['Pars']
        Data = my_shelf['Data']
        Results = my_shelf['Results']
    my_shelf.close()
    return Pars, Data, Results

def save_AFM_data(savedict):
    filename = 'D:/MEGAsync/My materials/python/Ting_code/temp_spy_data/temp_shelve2'
    filename = file_import_dialog_qt5_simplest(3)[0]  #
    for key in savedict:
        print(key)
    with shelve.open(filename,'n') as my_shelf:
        for key, value in savedict.items():
            my_shelf[key] = value
    my_shelf.close()

filenames = file_import_dialog_qt5_simplest(2)  #

ResultsAll = make_Results(0)
ResultsAll['reginds'] = [];
counter = 0
for fname in filenames:
    Pars, Data, Results = open_AFM_data(fname[:-4])
    reginds = Pars.ROIPars.reg_nums_all[[Pars.ROIPars.reg_nums_all != 0]]
    reginds = reginds + counter
    Results['reginds'] = reginds
    counter = np.max(reginds)
    ResultsAll = ResultsAll.append(Results, ignore_index=True)  # remove .dat
counter2 = int(np.max(ResultsAll['reginds']))
ResultsMean = pd.DataFrame(columns=['Name', 'ROI#', 'mean', 'SD'], index=range(counter2))
col_names = ['Name', 'ROI#', 'mean', 'SD', 'Up_mean', 'Up_SD', 'Down_mean', 'Down_SD']
ResultsMean = pd.DataFrame(columns=col_names, index=range(counter2))
ResultsMean = ResultsMean.astype({'Name': str,
                     'ROI#': object,
                     'mean': np.float32,
                     'SD': np.float32,
                     'Up_mean': np.float32,
                     'Up_SD': np.float32,
                     'Down_mean': np.float32,
                     'Down_SD': np.float32,})

keyV = 'EHertzBEC'
for ii in range(1, counter2+1):
    indsc = ResultsAll.loc[ResultsAll['reginds'] == ii]
    indscUp = indsc.loc[indsc['Height'] > np.mean(indsc['Height'])]
    indscDown = indsc.loc[indsc['Height'] < np.mean(indsc['Height'])]
    ResultsMean['Name'].iloc[ii-1] = indsc['Name'].iloc[0]
    ResultsMean['ROI#'].iloc[ii-1] = ii
    ResultsMean['mean'].iloc[ii-1] = np.mean(indsc[keyV])
    ResultsMean['SD'].iloc[ii-1] = np.std(indsc[keyV])
    ResultsMean['Up_mean'].iloc[ii-1] = np.mean(indscUp[keyV])
    ResultsMean['Up_SD'].iloc[ii-1] = np.std(indscUp[keyV])
    ResultsMean['Down_mean'].iloc[ii-1] = np.mean(indscDown[keyV])
    ResultsMean['Down_SD'].iloc[ii-1] = np.std(indscDown[keyV])

data_len = len(ResultsMean.index)
# group_dividers = data_len // 2  # replace with correct dividers
group_sizes = [1, 2]  # replace with correct dividers
group_names = ['first', 'second'] # replace with correct names
# group_list_named = itertools.repeat(group_names, group_sizes)
group_list_named = list(itertools.chain.from_iterable(itertools.repeat(x, y) for x,y in zip(group_names,group_sizes)))
# group_list = ResultsMean['ROI#'].tolist()
# group_list_named = group_list.copy()
# for ii in range(len(group_list_named)):
#     if ii<group_dividers:
#         group_list_named[ii]=group_names[0]
ResultsMean['group_name'] = group_list_named


# ResultsMean['mean'].iloc[:3]    
# save_AFM_data({"ResultsMean" :ResultsMean})  # no better way to pass varnames
plt.figure()
# tips = sns.load_dataset("tips")
# ax = sns.swarmplot(x="total_bill", y="day", data=tips)
# ax = sns.swarmplot(x="reginds", y="EHertzBEC", data=ResultsAll)
ax = sns.swarmplot(x="group_name", y="Up_mean", data=ResultsMean)
ax = sns.boxplot(x="group_name", y="Up_mean", data=ResultsMean)
plt.show()

# ResultsMean[["group_name","Up_mean"]].groupby("group_name").mean()
# ResultsMean[["group_name","Up_mean"]].groupby("group_name").describe()
ResultsMean[["group_name","Up_mean"]].groupby("group_name").agg(['median', 'mean', 'std'])
ResultsMean[["group_name","mean"]].groupby("group_name").agg(['median', 'mean', 'std'])