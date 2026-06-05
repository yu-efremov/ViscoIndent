# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:25:25 2026

@author: Yuri Efremov
import of jpk-force-map files
simplified version, only important parameters are screened in file
"""
import zipfile
import io
from struct import unpack
import numpy as np


def import_jpk_map(Pars, Data):

    fileName = Pars.filedir[0]
    fnameshort = fileName.split("/")
    fnameshort = fnameshort[-1]
    if fileName.endswith('jpk-force-map'):
        pass
        # JPKtype = 'ARDF'
        # elif filedir[-3:] == 'spm':
        #    Brukertype = 'spm'
    else:
        print('file extension is not supported (jpk-force-map required)')

    afm_file = zipfile.ZipFile(fileName, "r")
    Header = []
    linenum = 0
    with afm_file.open('header.properties', 'r') as binary_file:
        # Wrap the binary file to decode it into a text stream
        text_header = io.TextIOWrapper(binary_file, encoding="utf-8")

        for line in text_header:
            linenum = linenum+1
            # print(linenum)  # errors='ignore' was added
            Header.append(line)
            # print(line.strip())
            args = line.split("=")
            if 'force-scan-map.position-pattern.grid.ilength' in args[0]:
                # print(args[1])
                Xpixels = int(args[1])
            if 'force-scan-map.position-pattern.grid.jlength' in args[0]:
                # print(args[1])
                Ypixels = int(args[1])
            if 'force-scan-map.position-pattern.grid.ulength' in args[0]:
                Pars.ScanSize = float(args[1])*1e6  # um
                # print(Pars.ScanSize)
            if 'force-scan-map.settings.force-settings.extend-k-length' in args[0]:
                pts_per_approach = int(args[1])
                # print(pts_per_approach)
            if 'force-scan-map.settings.force-settings.extend-scan-time' in args[0]:
                time_per_approach = float(args[1])

        dT = time_per_approach/pts_per_approach
        # print(dT)

    channels = []
    with afm_file.open('shared-data/header.properties', 'r') as binary_file:  # find channels
        text_header2 = io.TextIOWrapper(binary_file, encoding="utf-8")
        for line in text_header2:
            args = line.split("=")
            if 'lcd-info' in args[0] and 'channel.name' in args[0]:
                args2 = args[0].split(".")
                channel_num = int(args2[1])
                channel_name = args[1]
                channels.append([channel_num, channel_name])
                if 'vDeflection' in channel_name:
                    vDeflChNum = str(channel_num)
                if 'measuredHeight' in channel_name:
                    ZChNum = str(channel_num)

    with afm_file.open('shared-data/header.properties', 'r') as binary_file:
        text_header2 = io.TextIOWrapper(binary_file, encoding="utf-8")
        for line in text_header2:
            linenum = linenum+1
            # print(linenum)  # errors='ignore' was added
            Header.append(line)
            # print(line.strip())
            args = line.split("=")
            if 'lcd-info.' + ZChNum + '.encoder.scaling.offset' in args[0]:
                # print('lcd-info.' + ZChNum + '.encoder.scaling.offset')
                # print(args[1])
                Z_offset = float(args[1])
            if 'lcd-info.' + ZChNum + '.encoder.scaling.multiplier' in args[0]:
                # print(args[1])
                Z_multiplier = float(args[1])
            if 'lcd-info.' + vDeflChNum + '.encoder.scaling.offset' in args[0]:
                # print(args[1])
                vDFL_offset = float(args[1])
            if 'lcd-info.' + vDeflChNum + '.encoder.scaling.multiplier' in args[0]:
                # print(args[1])
                vDFL_multiplier = float(args[1])
            if 'lcd-info.1.conversion-set.conversion.distance.scaling.multiplier' in args[0]:
                Pars.InvOLS = float(args[1])*1e9  # m/V probably
                # print(args[1])
            if 'lcd-info.1.conversion-set.conversion.force.scaling.multiplier' in args[0]:
                Pars.k = float(args[1]) # N/m probably 'SpringConstant'
            if 'force-scan-map.settings.force-settings.extend-scan-time' in args[0]:
                time_per_approach = float(args[1])
    # print('ok')

    # nPix = []
    topo=[]
    for name in afm_file.namelist():
        args = name.split("/")
        # print(args)
        if (name.startswith("index/") and len(args)==3 and len(args[2])==0):

            # print(args)
            pixN = int(args[1])
            apprFolder = name + "segments/0/"
            retrFolder = name + "segments/1/"
            # print(apprFolder + "channels/measuredHeight.dat")
            # read headers for approach and retraction for numpoints:
            header_approach = apprFolder + 'segment-header.properties'
            with afm_file.open(header_approach, 'r') as binary_file:
                text_headerC = io.TextIOWrapper(binary_file, encoding="utf-8")
                for line in text_headerC:
                    args = line.split("=")
                    if 'channel.vDeflection.data.num-points' in args[0]:
                        pts_per_approach = int(args[1])
                        break
            header_retraction = retrFolder + 'segment-header.properties'
            with afm_file.open(header_retraction, 'r') as binary_file:
                text_headerC = io.TextIOWrapper(binary_file, encoding="utf-8")
                for line in text_headerC:
                    args = line.split("=")
                    if 'channel.vDeflection.data.num-points' in args[0]:
                        pts_per_retraction = int(args[1])
                        break

            curveMH1 = unpack( ">" + str(pts_per_approach) + "i", afm_file.read(apprFolder + "channels/measuredHeight.dat"))
            curveMH2 = unpack( ">" + str(pts_per_retraction) + "i", afm_file.read(retrFolder + "channels/measuredHeight.dat"))
            curveDFL1 = unpack( ">" + str(pts_per_approach) + "i", afm_file.read(apprFolder + "channels/vDeflection.dat"))
            curveDFL2 = unpack( ">" + str(pts_per_retraction) + "i", afm_file.read(retrFolder + "channels/vDeflection.dat"))
            curveZ = -(np.concatenate([curveMH1, curveMH2])*Z_multiplier + Z_offset)*1e9  # to nm
            curveDFL = np.concatenate([curveDFL1 + curveDFL2])*vDFL_multiplier + vDFL_offset
            curve = np.column_stack([curveZ, curveDFL])  # *Pars.InvOLS
            topo.append(-max(curveZ))
            Data.append([pixN, curve, 0, dT, fnameshort])

    Data = np.asarray(Data, dtype="object")
    Pars.probe_dimension = np.nan  # not in the file
    Pars.probe_shape = 'sphere'
    Pars.fnameshort = []
    Pars.fnameshort.append(fnameshort)
    Pars.dT = dT
    Pars.Poisson = 0.5
    Pars.height = 0
    Pars.HeightfromZ = 0
    Pars.viscomodel = 'elastic'
    Pars.hydro.corr = 1
    Pars.hydro.corr_type = 2
    Pars.hydro.speedcoef = 4.0e-7
    Pars.cp_adjust = 1
    Pars.topo = np.array(topo).reshape(Xpixels, Ypixels, order='C')
    Pars.topo[1::2] = Pars.topo[1::2, ::-1]  # snake pattern (zig-zag, where the second line goes backward)
    indices = np.arange(Xpixels * Ypixels).reshape(Xpixels, Ypixels)
    indices[1::2] = indices[1::2, ::-1]
    snake_order = indices.flatten()
    for (index, element) in enumerate(Data):
        if element[0] != snake_order[index]:
            element[0] = int(snake_order[index])
    Data_sorted = np.array(sorted(Data, key=lambda x: x[0]), dtype=object)

    #Pars.topo = np.array(topo)
    Pars.TipType = 'unknown'
    print('jpk-force-map imported')

    return Pars, Data_sorted



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from Pars_class import Pars_gen
    # from file_import_qt5simple import file_import_dialog_qt5
    import_opt = 'single'  # single, 2 - multi
    start_folder = ''
    file_types = '*.jpk-force-map'
    fileName = "D:/yu_efremov/Yandex.Disk/python/Ting_code/user files/2026_Stefano/map-data-2026.04.24-11.43.42.224.jpk-force-map"
    fileName = "D:/yu_efremov/Yandex.Disk/python/Ting_code/user files/2026_Stefano/cell-substrate-data-2026.03.26-17.22.53.9123.jpk-force-map"
    fileName = "G:/YandexDisk/python/Ting_code/user files/2026_Stefano/map-data-2026.04.24-11.43.42.224.jpk-force-map"
    Data = []
    Pars = Pars_gen()
    Pars.filedir.append(fileName)
    [Pars, Data] = import_jpk_map(Pars, Data)
    currentcurve = Data[16][1] # 0 - is pixelnumber, [1] is curve currentcurve[:,0] - z
    plt.figure()
    z_start=4.9999999999999996E-6
    #plt.plot(currentcurve[:, 0]-3*z_start, currentcurve[:, 1])  # plot of a curve nm vs nm
    plt.plot(currentcurve[:, 0], currentcurve[:, 1])  # plot of a curve nm vs nm

    plt.figure()
    plt.imshow(Pars.topo, interpolation='nearest',cmap='viridis', origin='lower') #'jet'
    plt.colorbar()
    plt.show()
    #plt.plot(Pars.topo)
    #plt.imshow(Pars.PL, interpolation='nearest',cmap='viridis', origin='lower') #'jet'
    #plt.colorbar()




