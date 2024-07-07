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

        filedir = Pars.filedir[0]
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
        Pars.dT = 1./Pars.Zfreq/pointspercurve[0]  # dT indirect calculation
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

            rawData[1] = rawData[1]*Pars.ZsensSens*Pars.ZsensVSens  # Zscanner in nm
            rawData[0] = rawData[0]*Pars.DFLSens*Pars.DFLVSens  # DFL in nm (temporary)

            if Brukertype == 'spm':
                ppl = pointspercurve[-1]
                topo = rawData[2]*Pars.ZsensSens*Pars.ZsensVSens  # Zscanner in nm
                topo = np.transpose(topo)  # x-y adjust to make like in NanoScope
            Pars.topo = topo
            Numcurves = rawData[0].shape[1]
            sizeData = len(Data)
            nPix = []
            Data3 =[]

            for ij in range(Numcurves):
                currentcurve = np.array([rawData[1][:,ij], rawData[0][:, ij]]).transpose()
                if currentcurve[0, 0] < -1000:
                    goodindx = currentcurve[:, 1] > -1000
                    currentcurve = currentcurve[goodindx]
                nPix.append(ij)
                Data.append(currentcurve)
                Data3.append([ij, currentcurve])

            Data3 = np.asarray(Data3, dtype=object)
            Data = np.asarray([nPix, Data], dtype=object)
            Data = Data.transpose()

            if len(Data.shape) < 2:
                print('Data reformatted')

        self.Data = Data3  # contains pixel numbers and curve data
        # self.rawData = rawData  # raw data if needed
        file.close()


if __name__ == '__main__':
    from Pars_class import Pars_gen
    from file_import_qt5simple import file_import_dialog_qt5
    import_opt = 'single'  # single, 2 - multi
    file_types = '*.dat'
    start_folder = ''
    filename = file_import_dialog_qt5(import_opt=import_opt, start_folder=start_folder) 
    Pars = Pars_gen()
    Pars.filedir.append(filename)
    Bruker_data = Bruker_import(Pars)
    Bruker_data.Header
    Data = Bruker_data.Data
    Pars = Bruker_data.Pars
    Pars_dict = Pars.class2dict()
    # rawData = Bruker_data.rawData
    currentcurve = Data[700][1] # 0 - is pixelnumber, [1] is curve currentcurve[:,0] - z
    if currentcurve[1, 0] < -1000:
        goodindx = currentcurve[:, 1] > -1000
        currentcurve = currentcurve[goodindx]
    plt.plot(currentcurve[:, 0], currentcurve[:, 1])  # plot of a curve nm vs nm
    plt.figure()
    plt.imshow(Pars.topo, interpolation='nearest',cmap='viridis', origin='lower') #'jet'
    plt.colorbar()
    plt.show()
