# -*- coding: utf-8 -*-
"""
# Pars as object
"""
import numpy as np
import pandas as pd
# import re


class Pars_gen:
    def __init__(self):
        self.filedir = []
        self.fnameshort = []
        self.fixed_values = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)
        self.hydro = hydro()
        self.PLpars = PLpars()
        self.ROIPars = ROIPars()

    def class2CSV(self):
        Pars_dict = vars(self)
        Pars_dict_full = {}
        for key, value in Pars_dict.items():
            if str(type(Pars_dict[key])).find("__main__") > 0:
                print('object found')
                Pars_dict2 = vars(Pars_dict[key])
                for key2, value2 in Pars_dict2.items():
                    Pars_dict_full[key + '.' + key2] = value2
            else:
                Pars_dict_full[key] = value
                # Pars_dict[key] = Pars_dict2
        CSV ="\n".join([k+': '+''.join(str(v).replace("\n", ",")) for k,v in Pars_dict_full.items()])
        # You can store this CSV string variable to file as below
        # with open("filename.csv", "w") as file:
        # file.write(CSV)

        return Pars_dict_full, CSV

    def class2dict(self):
        Pars_dict = vars(self)
        Pars_dict_full = {}
        for key, value in Pars_dict.items():
            Pars_dict_full[key] = value
        Pars_dict_full['PLpars'] = 0
        print(Pars_dict_full['PLpars'])
        Pars_dict_hydro = vars(self.hydro)
        Pars_dict_full['hydro'] = Pars_dict_hydro
        Pars_dict_PLpars = vars(self.PLpars)
        Pars_dict_full['PLpars'] = Pars_dict_PLpars
        print(Pars_dict_PLpars)
        Pars_dict_ROIPars = vars(self.ROIPars)
        Pars_dict_full['ROIPars'] = Pars_dict_ROIPars

        return Pars_dict_full

    def class2list(self):
        Pars_dict = vars(self)
        Pars_list = []
        for key, value in Pars_dict.items():
            # if isinstance(value, object):
            if str(type(Pars_dict[key])).find("__main__") > 0:
                Pars_dict2 = vars(Pars_dict[key])
                Pars_list2 = []
                #print(Pars_dict2)
                for key2, value2 in Pars_dict2.items():
                    Pars_list2.append([key2, str(value2)])
                Pars_list.append([key, Pars_list2])
            else:
                Pars_list.append([key, str(value)])
        return Pars_list

    def class2df(self):  # bad
        Pars_dict = vars(self)
        Pars_dict_full = {}
        for key, value in Pars_dict.items():
            if str(type(Pars_dict[key])).find("__main__") > 0:
                Pars_dict2 = vars(Pars_dict[key])
                for key2, value2 in Pars_dict2.items():
                    Pars_dict_full[key+'.'+key2] = value2
            else:
                Pars_dict_full[key] = value
        df = pd.DataFrame.from_dict(Pars_dict_full, orient="index")
        # df.to_csv("Pars.csv")
        return df

    def CSV2class(self, CSV):
        # line = CSV.readline().rstrip()
        # while line:
        for line in CSV.splitlines():
            # linenum=linenum+1
            args = line.split(": ")
            if args[0] == 'fixed_values':
                # fvalsstr = re.sub('[[]]', "", args[1])
                fvalsstr = args[1].replace('[', "")
                fvalsstr = fvalsstr.replace(']', "")
                fvalsstr = fvalsstr.replace(',', "")
                print(fvalsstr)
                fixed_values = np.fromstring(fvalsstr, dtype=float, sep=' ')
                self.fixed_values = np.reshape(fixed_values, (2, 3))
                print(self.fixed_values)
            if args[0] == 'hydro.corr':
                self.hydro.corr = int(args[1])
            if args[0] == 'hydro.corr_type':
                self.hydro.corr_type = int(args[1])
                print(int(args[1]))
            if args[0] == 'hydro.speedcoef':
                self.hydro.speedcoef = int(args[1])
            if args[0] == 'PLpars.strategy':
                self.PLpars.strategy = int(args[1])
            if args[0] == 'PLpars.PLadj':
                self.PLpars.PLadj = int(args[1])
            if args[0] == 'ROIPars.ROImode':
                self.ROIPars.ROImode = int(args[1])
            if args[0] == 'ROIPars.multiselect':
                self.ROIPars.multiselect = int(args[1])
            if args[0] == 'ROIPars.h_level':
                self.ROIPars.h_level = float(args[1])
            if args[0] == 'ROIPars.delete_nonselected':
                self.ROIPars.delete_nonselected = int(args[1])
            if args[0] == 'ROIPars.num_of_ROIs':
                self.ROIPars.num_of_ROIs = int(args[1])

        pass

    def dict2class2(self, Pars_dict):
        # from Dict to Object
        for a, b in Pars_dict.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [dict2class2(self, x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict2class2(self, b) if isinstance(b, dict) else b)
        pass

    def dict2class(self, Pars_dict):
        # from Dict to Object
        for a, b in Pars_dict.items():
            setattr(self, a, b)
        if 'hydro' in Pars_dict:
            self.hydro = hydro()
            self.hydro.corr = Pars_dict['hydro']['corr']
            self.hydro.corr_type = Pars_dict['hydro']['corr_type']
            self.hydro.speedcoef = Pars_dict['hydro']['speedcoef']
        if 'PLpars' in Pars_dict:
            self.PLpars = PLpars()
            self.PLpars.strategy = Pars_dict['PLpars']['strategy']
            self.PLpars.PLadj = Pars_dict['PLpars']['PLadj']
        if 'ROIPars' in Pars_dict:
            self.ROIPars = ROIPars()
            self.ROIPars.strategy = Pars_dict['ROIPars']['strategy']
            self.ROIPars.ROImode = Pars_dict['ROIPars']['ROImode']
            self.ROIPars.multiselect = Pars_dict['ROIPars']['multiselect']
            self.ROIPars.h_level = Pars_dict['ROIPars']['h_level']
            self.ROIPars.delete_nonselected = Pars_dict['ROIPars']['delete_nonselected']
            if 'reg_nums_all' in Pars_dict['ROIPars']:
                self.ROIPars.reg_nums_all = Pars_dict['ROIPars']['reg_nums_all']
        pass


class hydro:
    def __init__(self):
        self.corr = 0
        self.corr_type = 2
        self.speedcoef = 0


class PLpars:
    def __init__(self):
        self.strategy = 0
        self.PLadj = 2


class ROIPars:
    def __init__(self):
        self.strategy = 0
        self.ROImode = 0
        self.multiselect = 1
        self.h_level = -1e10  # without pre_h_level
        self.delete_nonselected = 1
        self.num_of_ROIs = 1


if __name__ == '__main__':
    Pars = Pars_gen()
    Pars_dict = Pars.class2dict()
    Pars_CSV = Pars.class2CSV()[1]
    Pars_list = Pars.class2list()
    # Pars.CSV2class(Pars_CSV)
    Pars2 = Pars_gen()
    Pars2.dict2class(Pars_dict)
    # Pars_df = Pars.class2df()
    # import pickle
    # filehandler = open("Pars.obj","wb")
    # pickle.dump(Pars, filehandler)
    # filehandler.close()
    # opfile = open("Pars.obj",'rb')
    # object_file = pickle.load(opfile)
    # opfile.close()
