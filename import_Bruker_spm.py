# -*- coding: utf-8 -*-
"""
@author: Yuri
# 2019 septemebr - Yuri Efremov <yu.efremov@gmail.com>
# for import from single Bruker file

"""


import struct
import numpy as np
import matplotlib.pyplot as plt


class Bruker_import:
    """
    Class to import Bruker AFM force curves and force maps
    """

    def __init__(self, Pars, Data=[]):

        filedir = Pars.filedir[-1]  # last file in array if several
        fnameshort = filedir.split("/")
        fnameshort = fnameshort[-1]
        if not hasattr(Pars, "fnameshort"):
            Pars.fnameshort = []
        Pars.fnameshort.append(fnameshort)
        if filedir[-3:] == 'pfc':
            Brukertype = 'pfc'
        elif filedir[-3:] == 'spm':
            Brukertype = 'spm'
        else:
            print('file extension is not supported')
        offsets = []
        datalengths = []
        cols = []  # Samps/line
        Header = []
        Data_types = []
        
        tempcount = 0
        linenum = 0
        # file = open(filedir, 'r')

        with open(filedir, 'r', encoding='utf8', errors='ignore') as file:
            while True:
                linenum = linenum+1
                # print(linenum)  # errors='ignore' was added
                line = file.readline().rstrip()
                line = line[1:]
                Header.append(line)
                bad_inds=0
                # if line[0]=='*'
                if 'ZsensSens' in line and tempcount < 2:
                    args = line.split(": ")
                    if tempcount == 0:
                        args = args[1].split(" ")
                        Pars.ZsensSens = float(args[1])  # nm/V
                        tempcount = tempcount+1
                    elif tempcount == 1:
                        args = args[1].split(" ")
                        Pars.ZsensVSens = float(args[3][1:])  # V/LSB
                        tempcount = tempcount+1
                if 'Start context' in line and linenum < 10:
                    args = line.split(": ")
                    Pars.BrukerMode = args[1]  # force curve or force volume
                if '@Sens. DeflSens' in line:
                    args = line.split(": ")
                    args = args[1].split(" ")
                    Pars.DFLSens = float(args[1])  # nm/V
                if 'SoftHarmoniXSetpoint: V [Sens. ForceDeflSens]' in line:
                    args = line.split(": ")
                    args = args[1].split(" ")
                    Pars.DFLVSens = float(args[3][1:])  # V/LSB
                if 'Data offset' in line:
                    args = line.split(": ")
                    offsets.append(int(args[1]))
                if 'Data length' in line:
                    args = line.split(": ")
                    datalengths.append(int(args[1]))
                if 'Samps/line:' in line:
                    args = line.split(": ")
                    if ' ' in args[1]:
                        args = args[1].split(" ")
                        args[1] = int(args[0])+int(args[1])
                        if 900 <= int(args[1]) <= 1023:
                            bad_inds = 1024 - int(args[1])
                            args[1] = 1024  # repair incomplete data
                    cols.append(int(args[1]))
                if 'Scan rate:' in line:
                    args = line.split(": ")
                    Pars.Zfreq = float(args[1])
                if 'Date' in line:
                    args = line.split(": ")
                    Pars.date_acquired = args[1]  # date
                if 'Tip Serial Number' in line:
                    args = line.split(": ")
                    Pars.TipType = args[1]
                if 'Tip Radius:' in line:
                    args = line.split(": ")
                    Pars.probe_dimension = float(args[1])
                if 'Spring Constant:' in line:
                    args = line.split(": ")
                    Pars.k = float(args[1])
                if 'Scan Size' in line:
                    args = line.split(": ")
                    args = args[1].split(" ")
                    Pars.ScanSize = float(args[0])  # nm/V
                if 'Peak Force Amplitude:' in line:
                    args = line.split(": ")
                    Pars.PFTAmp = float(args[1])
                if 'PFT Freq:' in line:
                    args = line.split(" ")
                    Pars.PFTFreq = float(args[2])*1000
                if 'Sync Distance QNM:' in line and 'Auto ' not in line:
                    args = line.split(": ")
                    Pars.PFTSyncD = float(args[1])
                if 'Start context: ' in line:  # not "\Data type"
                    args = line.split(": ")
                    Data_types.append(args[1])
                # add Samps/line:
                if "File list end" in line:
                    datalengths = datalengths[1:]
                    cols = cols[-len(datalengths):]
                    break
        if Pars.BrukerMode == 'FVOL':
            Brukertype = 'spm'
        elif Pars.BrukerMode == 'FOL':
            Brukertype = 'spm_singlecurve'
        elif Pars.BrukerMode == 'OL1':
            Brukertype = 'spm_processed_map'
        pointspercurve = cols
        dT = 1./Pars.Zfreq/pointspercurve[0]  # dT indirect calculation
        Pars.dT = dT
        Pars.InvOLS = 1
        self.Pars = Pars
        self.Header = Header
        file.close()

        # file = open(filedir, 'rb')
        rawData = {}
        resTemp = np.zeros((int(max(datalengths)/4), len(offsets)))
        with open(filedir, 'rb') as file:
            for ik in range(len(offsets)):
                byte_length = datalengths[ik]
                off = offsets[ik]
                bpp = 4
                length = int(byte_length/bpp)
                file.seek(off)
                resTemp = np.array(
                struct.unpack("<"+str(length)+{2:'h', 4:'i', 8:'q'}[bpp], file.read(byte_length)),
                dtype='float64').reshape((cols[ik], int(length/cols[ik])), order='F')
                # .reshape((cols[ik], int(length/cols[ik])))
                if ik<2:  # cols[ik]>int(length/cols[ik]):
                    tl = int(resTemp.shape[0]/2)
                    for ij in range(resTemp.shape[1]):
                        resTemp[0:tl, ij] = np.flipud(resTemp[0:tl, ij])
                rawData[ik] = resTemp

            del Data_types[0]
            ids_topo = next((i for i, s in enumerate(Data_types) if s.startswith("OL")), None)
            #ids_topo = Data_types.index('AFM')  # was 2
            ids_Z = Data_types.index('FOL')+1  # was 1
            ids_DFL = Data_types.index('FOL')  # was 0
            rawData[ids_Z] = rawData[ids_Z]*Pars.ZsensSens*Pars.ZsensVSens  # Zscanner in nm
            rawData[ids_DFL] = rawData[ids_DFL]*Pars.DFLSens*Pars.DFLVSens  # DFL in nm (temporary)

            if Brukertype == 'spm':
                ppl = pointspercurve[-1]
                topo = rawData[ids_topo]*Pars.ZsensSens*Pars.ZsensVSens  # Zscanner in nm
                topo = np.transpose(topo)  # x-y adjust to make like in NanoScope
            # plt.imshow(topo, interpolation='nearest',cmap='viridis', origin='lower') #'jet'
            elif Brukertype == 'spm_singlecurve':
                topo = [0]
                
            Pars.topo = topo
            Numcurves = rawData[ids_Z].shape[1]
            sizeData = len(Data)
            nPix = []
            Data3 =[]

            for ij in range(Numcurves):
                currentcurve = np.array([rawData[ids_Z][:,ij], rawData[ids_DFL][:, ij]]).transpose()
                if currentcurve[0, 0] < -1000 or currentcurve[0, 0]==0:
                    goodindx = (currentcurve[:, 1] > -1000) & (currentcurve[:, 1]!=0)
                    currentcurve = currentcurve[goodindx]
                nPix.append(ij)
                # if type(Data) == list:
                #     Data.append(currentcurve)
                # else:
                #     np.append(Data, currentcurve)
                Data3.append([ij, currentcurve, 0, dT, fnameshort])

            Data3 = np.asarray(Data3, dtype=object)
            # if type(Data) == list:
            #     Data = np.asarray([nPix, Data], dtype=object)
            #     Data = Data.transpose()
            # if len(Data.shape) < 2:
            #     print('Data reformatted')

        if sizeData>0:
            Data3 = np.concatenate((Data, Data3), axis=0)

        self.Data = Data3  # contains pixel numbers and curve data
        # self.rawData = rawData  # raw data if needed
        file.close()


if __name__ == '__main__':
    from Pars_class import Pars_gen
    from file_import_qt5simple import file_import_dialog_qt5
    import_opt = 'single'  # single, 2 - multi
    start_folder = ''
    file_types = '*.dat'
    # filename = file_import_dialog_qt5(import_opt=import_opt, start_folder=start_folder) 
    # Pars = Pars_gen()
    # Pars.filedir.append(filename)
    # Bruker_data = Bruker_import(Pars)
    # Data = Bruker_data.Data
    # Pars = Bruker_data.Pars
    # Pars_dict = Pars.class2dict()
    # # rawData = Bruker_data.rawData
    # currentcurve = Data[0][1] # 0 - is pixelnumber, [1] is curve currentcurve[:,0] - z
    # if currentcurve[1, 0] < -1000:
    #     goodindx = currentcurve[:, 1] > -1000
    #     currentcurve = currentcurve[goodindx]
    # plt.plot(currentcurve[:, 0], currentcurve[:, 1])  # plot of a curve nm vs nm
    # plt.figure()
    # plt.imshow(Pars.topo, interpolation='nearest',cmap='viridis', origin='lower') #'jet'
    # plt.colorbar()
    # plt.show()

    # test multi import
    import_opt = 'multi'  # single, 2 - multi
    filenames = file_import_dialog_qt5(import_opt=import_opt, start_folder=start_folder)
    Data = []
    Pars = Pars_gen()
    for filename in filenames:
        Pars.filedir.append(filename)
        Bruker_data = Bruker_import(Pars, Data=Data)
        Data = Bruker_data.Data
        Pars = Bruker_data.Pars
    currentcurve = Data[0][1] # 0 - is pixelnumber, [1] is curve currentcurve[:,0] - z
    if currentcurve[1, 0] < -1000:
        goodindx = currentcurve[:, 1] > -1000
        currentcurve = currentcurve[goodindx]
    plt.plot(currentcurve[:, 0], currentcurve[:, 1])  # plot of a curve nm vs nm
