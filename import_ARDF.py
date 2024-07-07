# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:59:21 2021

@author: Yuri

this code is based on the Matlab version:
ARDF to Matlab
version 5.0.0 (11.6 KB) by Matthew Poss
Matthew Poss (2021). ARDF to Matlab (https://www.mathworks.com/matlabcentral/fileexchange/80212-ardf-to-matlab),
MATLAB Central File Exchange. Retrieved September 1, 2021.
This version of the code is not fully tested, and there might be some problems 
since the ARDF file protocol is proprietary and may change
Reads and imports Asylum Research ARDF files to Python structures for force curve analysis.
2023.01: tested and modified for new ARDF files
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
from collections import namedtuple
import pprint

def readARDFpointer(fid, address):
    # Each pointer/header is 16 bytes
    if address != -1:
        # Navigate to address
        fid.seek(address, 0)

    # Initialize typePnt
    typePnt = np.zeros(4)
    byte_length = 4
    bpp = 4
    length = int(byte_length/bpp)
    # Read pointer
    checkCRC32 = struct.unpack('i', fid.read(4))[0]  # Read CRC-32 checksum
    sizeBytes = struct.unpack('i', fid.read(4))[0]  # Read byte size of section
    typePnt = struct.unpack('ssss', fid.read(4))  # Read 4-character pointer type
    typePnt = b''.join(typePnt)
    try:
        typePntdecoded = typePnt.decode("cp1252")  # not utf8
        typePntdecoded = typePnt.decode("utf8")  # not utf8
        typePnt = typePntdecoded
    except:
        typePntdecoded = typePnt
    # typePnt = ''.join(typePntdecoded)
    miscNum = struct.unpack('i', fid.read(4))[0]  # Read misc number

    return checkCRC32, sizeBytes, typePnt, miscNum

def readTOC(fid, address, ttype):
    # FTOC: File Table of Contents
    # TTOC: Text Table of Conents
    # Define null pointer title
    nullCase = "\0" * 4

    toc = namedtuple("toc", []);

    if address != -1:
        # Navigate to address
        fid.seek(address, 0)

    [dumCRC, dumSize, lastType, dumMisc] = readARDFpointer(fid, -1)
    # ARDF_checkType(lastType, ttype, fid)
    # Read remaining TOC header (assumes 32 byte header)
    toc.sizeTable = struct.unpack('i', fid.read(4))[0]  # [0]fread(fid,1,'uint64')
    toc.numbEntry = struct.unpack('i', fid.read(4))[0]  # [0]fread(fid,1,'uint32')
    toc.numbEntry = struct.unpack('i', fid.read(4))[0]  # [0]fread(fid,1,'uint32')
    toc.sizeEntry = struct.unpack('i', fid.read(4))[0]  # [0]fread(fid,1,'uint32')

    if toc.sizeEntry == 24:
        # FTOC, IMAG, VOLM
        toc.pntImag = []
        toc.pntVolm = []
        toc.pntNext = []
        toc.pntNset = []
        toc.pntThmb = []
    elif toc.sizeEntry == 32:
        # TTOC
        toc.idxText = []
        toc.pntText = []
    elif toc.sizeEntry == 40:
        # VOFF
        typecheck = 'VOFF'
        toc.pntCounter = []
        toc.linCounter = []
        toc.linPointer = []
        # IDAT  #TODO test for new files
        toc.data = []
        sizeRead = int((toc.sizeEntry - 16) / 4)
    else:
        # IDAT
        toc.data = []
        sizeRead = int((toc.sizeEntry - 16) / 4)

    # Initialize parameters for while loop
    done = 0
    numbRead = 1
    # Read TOC entries
    while (done == 0) and (numbRead <= toc.numbEntry):
        # Read entry header
        [dumCRC, dumSize, typeEntry, dumMisc] = readARDFpointer(fid, -1)
        # print(typeEntry)  # check typeEntry
        # Read remainder of entry
        if toc.sizeEntry == 24:
            # FTOC, IMAG, VOLM
            lastPointer = struct.unpack('i', fid.read(4))[0]  # fread(fid,1,'uint64');
            dum = struct.unpack('i', fid.read(4))[0]  # fread(fid,1,'uint64');
        elif toc.sizeEntry == 32:
            # TTOC
            lastIndex = struct.unpack('i', fid.read(4))[0]  # fread(fid,1,'uint64');
            lastPointer = struct.unpack('i', fid.read(4))[0]
            lastPointer = struct.unpack('i', fid.read(4))[0]  # 2 times
        elif toc.sizeEntry == 40:
            if typeEntry == 'VOFF':
                # VOFF
                lastPntCount = struct.unpack('i', fid.read(4))[0]  # fread(fid,1,'uint32');
                lastLinCount = struct.unpack('i', fid.read(4))[0]  # read(fid,1,'uint32');
                dum = struct.unpack('i', fid.read(4))[0]  # fread(fid,1,'uint64');
                dum = struct.unpack('i', fid.read(4))[0]  # fread(fid,1,'uint64');
                lastLinPoint = struct.unpack('i', fid.read(4))[0]  # fread(fid,1,'uint64');
                dum = struct.unpack('i', fid.read(4))[0]  # fread(fid,1,'uint64');
                # dum = struct.unpack('i', fid.read(4))[0]
            elif typeEntry == 'IDAT':
                lastData = struct.unpack('f'*sizeRead, fid.read(4*sizeRead))  # TODO check;
        else:
            # IDAT
            lastData = struct.unpack('f'*sizeRead, fid.read(4*sizeRead))  # fread(fid, sizeRead, 'single');
            # dum = struct.unpack('f', fid.read(4))[0]  # fread(fid,1,'uint64')

        if typeEntry == 'IMAG':
            toc.pntImag.append(lastPointer)
        elif typeEntry == 'VOLM':
            toc.pntVolm.append(lastPointer)
        elif typeEntry == 'NEXT':
            toc.pntNext.append(lastPointer)
        elif typeEntry == 'NSET':
            toc.pntNset.append(lastPointer)
        elif typeEntry == 'THMB':
            toc.pntThmb.append(lastPointer)
        elif typeEntry == 'TOFF':
            toc.idxText.append(lastIndex)
            # toc.idxText = [toc.idxText, lastIndex]
            toc.pntText.append(lastPointer)

        elif typeEntry == 'IDAT':  #TODO check
            toc.data.append(lastData)
            
        elif typeEntry == 'VOFF':
            toc.pntCounter.append(lastPntCount)
            toc.linCounter.append(lastLinCount)
            toc.linPointer.append(lastLinPoint)
        elif typeEntry == nullCase:
            if lastType == 'IBOX':
                toc.data.append(lastData)
            elif lastType == 'VTOC':
                toc.pntCounter.append(lastPntCount)
                toc.linCounter.append(lastLinCount)
                toc.linPointer.append(lastLinPoint)
            else:
                done = 1

        else:
            print(['ERROR: '+ str(typeEntry) +' not recognized!'])

        numbRead = numbRead + 1

    return toc

def readTEXT(fid, loc):
    txt1 = 0
    # Navigate to the note section
    fid.seek(loc, 0)
    # Read the notes header, verify type
    [dumCRC, dumSize, lastType, dumMisc] = readARDFpointer(fid,-1)
    # ardf_checkType(lastType, 'TEXT', fid)

    # Read the remainder of the header
    dumMisc = struct.unpack('i', fid.read(4))[0]  # fread(fid,1,'uint32');
    sizeNote = struct.unpack('i', fid.read(4))[0]  # fread(fid,1,'uint32');

    # Read the notes
    # txt = transpose( fread(fid, sizeNote, '*char') )
    txt1 = struct.unpack('s'*sizeNote, fid.read(sizeNote))
    txt2 = b''.join(txt1)
    # txt = txt2.decode("utf8")
    txt = txt2.decode("cp1252")  # correct

    return txt


def readDEF(fid, address, ttype):

    DEF = namedtuple("DEF", [])
    if address != -1:
        # Navigate to address
        fid.seek(address, 0)

    # Read DEF header, verify if correct type
    [dumCRC, sizeDEF, typeDEF, dumMisc] = readARDFpointer(fid, -1)
    # ardf_checkType(typeDEF, type, fid)
    # Read points & lines
    DEF.points = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32')
    DEF.lines = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32')

    # Set bytes to skip
    if typeDEF == 'IDEF':
        skip = 96
    elif typeDEF == 'VDEF':
        skip = 144

    # Read some bytes as dummy bytes
    dum = struct.unpack('s'*skip, fid.read(skip))[0]  # fread(fid, skip, '*char')

    # Read 32 bytes as text
    sizeText = 32
    txt1 = struct.unpack('s'*sizeText, fid.read(sizeText))
    txt2 = b''.join(txt1)
    DEF.imageTitle = txt2.decode("cp1252")  # transpose( fread(fid, sizeText, '*char') );

    # Read remaining bytes as dummy bytes
    sizeHead = 16
    remainingSize = sizeDEF - 8 - skip - sizeHead - sizeText
    dum = struct.unpack('s'*remainingSize, fid.read(remainingSize))  # fread(fid, remainingSize, '*char')
    return DEF


def readVSET(fid, address):

    vset = namedtuple("vset", [])
    if address != -1:
        # Navigate to address
        fid.seek(address, 0)

    # Read header, verify if correct type
    [dumCRC, lastSize, lastType, dumMisc] = readARDFpointer(fid, -1)
    # ardf_checkType(lastType, 'VSET', fid);

    # Read VSET data
    vset.force = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32');
    vset.line = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32');
    vset.point = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32');
    dum = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32')
    vset.prev = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint64')
    vset.next = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint64')
    dum = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32')
    return vset


def readVNAM(fid, address):

    vnam = namedtuple("vnam", [])
    if address != -1:
        # Navigate to address
        fid.seek(address, 0)

    # Read header, verify if correct type
    [dumCRC, lastSize, lastType, dumMisc] = readARDFpointer(fid, -1)
    # ardf_checkType(lastType, 'VNAM', fid);

    # Read data
    vnam.force = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32')
    vnam.line = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32')
    vnam.point = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32')
    vnam.sizeText = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32')
    sizeText = vnam.sizeText
    txt1 = struct.unpack('s'*sizeText, fid.read(sizeText))
    txt2 = b''.join(txt1)
    vnam.name = txt2.decode("cp1252")

    # Determine remaining size
    remainingSize = lastSize - 16 - vnam.sizeText - 16

    # Read remaining zeros to dummy variable
    dum = struct.unpack('s'*remainingSize, fid.read(remainingSize))
    return vnam


def readVDAT(fid, address):

    vdat = namedtuple("vdat", [])
    if address != -1:
        # Navigate to address
        fid.seek(address, 0)

    # Read header, verify if correct type
    [dumCRC, lastSize, lastType, dumMisc] = readARDFpointer(fid, -1)
    # ardf_checkType(lastType, 'VDAT', fid)

    # Read data
    vdat.force = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32');
    vdat.line = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32');
    vdat.point = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32');
    vdat.sizeData = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32'); % number of floats

    vdat.forceType = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32');
    vdat.pnt0 = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32');    % Pointers, presumably
    vdat.pnt1 = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32');
    vdat.pnt2 = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32');
    dum = struct.unpack('i'*2, fid.read(4*2))[0]  # fread(fid, 2, 'uint32');

    # Read data
    sizeData = vdat.sizeData
    vdat.data = struct.unpack('f'*sizeData, fid.read(4*sizeData))  # fread(fid, vdat.sizeData, 'single');

    return vdat


def readXDAT(fid, address):

    DEF = namedtuple("DEF", [])
    if address != -1:
        # Navigate to address
        fid.seek(address, 0)

    # Read header
    [dumCRC, lastSize, lastType, dumMisc] = readARDFpointer(fid, -1)

    # Verify if header correct type
#     if (~strcmp(lastType, 'XDAT')) && (~strcmp(lastType, 'VSET'))
#         error(['ERROR: No XDAT or VSET here!  Found: ' found '  Location:' num2str( ftell(fid)-16 )]);
#     end

    # Choose action depending on header type
    if lastType == 'XDAT':
        # Determine distance to step forward
        stepDist = lastSize - 16

        # Step forward that distance
        fid.seek(stepDist, 1)  # fseek(fid, stepDist, 'cof')  1 from current

    elif lastType == 'VSET':  # If VSET, step back 16 bytes
        # Step back 16 bytes (the size of ARDF header)
        fid.seek(-16, 1)  # fseek(fid, -16, 'cof')

    else:
        print('error in readXDAT')

    return DEF


def getARDFdata(FN):
    fid = open(FN, 'rb')
    # read file header
    [dumCRC, dumSize, lastType, dumMisc] = readARDFpointer(fid, 0)
    # ARDF_checkType(lastType, 'ARDF', fid);
    # FTOC: File Table of Contents
    F = namedtuple("F", [])
    ftoc = readTOC(fid, -1, 'FTOC')
    F.ftoc = ftoc
    # pprint.pprint(F.ftoc.__dict__)
    # TTOC: Text Table of Conents
    loc_TTOC = F.ftoc.sizeTable + 16
    F.ttoc = readTOC(fid, loc_TTOC, 'TTOC')
    # Read Main Notes
    F.ttoc.numbNotes = np.size(F.ttoc.pntText)
    noteMain = readTEXT(fid, F.ttoc.pntText[0])

    F.numbImag = np.size(F.ftoc.pntImag)
    D = namedtuple("D", [])
    D.Notes = noteMain
    D.imageList = []
    D.y = []
    # F.imagN = []
    F.imagN = []
    imagN = namedtuple("imagN", [])
    for n in range(0, F.numbImag):

        # Determine dynamic image structure name
        #imagN = ['imag', str(n)]

        # Read IMAG Table
        imagN = namedtuple("imagN", [])
        # print(n)
        # print(F.ftoc.pntImag[n])
        imagN = readTOC(fid, F.ftoc.pntImag[n], 'IMAG')
        # Reat IMAG-TTOC Table
        loc_IMAG_TTOC = F.ftoc.pntImag[n] + imagN.sizeTable
        imagN.ttoc = readTOC(fid, loc_IMAG_TTOC, 'TTOC')
        # IDEF header
        # Navigate to IDEF within IMAG
        loc_IMAG_IDEF = F.ftoc.pntImag[n] + imagN.sizeTable + imagN.ttoc.sizeTable
        imagN.idef = readDEF(fid, loc_IMAG_IDEF, 'IDEF')

        # Add to imageList
        D.imageList.append(imagN.idef.imageTitle)

        # Read all IBOX/IDAT entries
        idat = readTOC(fid, -1, 'IBOX')

        # Write IDAT data to image array
        D.y.append(idat.data)
        # Read closing IMAG header (GAMI), verify header type
        [dumCRC, dumSize, lastType, dumMisc] = readARDFpointer(fid,-1)
        # ardf_checkType(lastType, 'GAMI', fid);
        F.imagN.append(imagN)
        # Read the notes assocaited with each image (no)
        # numbImagText = size(F.(imagN).ttoc.pntText, 1)

    # VOLM: Force Curve Data
    F.numbVolm = np.size(F.ftoc.pntVolm)
    # Initialize data arrays
    D.channelList = []

    # Import header data and pointers for each volume
    F.volmN = []
    volmN = namedtuple("volmN", [])

    for n in range (0, F.numbVolm):
        # Determine dynamic volume structure name
        #volmN = ['volm' num2str(n)];

        # VOLM Header
        # Read all VOLM entries
        volmN = namedtuple("volmN", [])
        volmN = readTOC(fid, F.ftoc.pntVolm[n], 'VOLM')

        # VOLM-TTOC
        # Read all VOLM-TTOC entries
        loc_VOLM_TTOC = F.ftoc.pntVolm[n] + volmN.sizeTable
        volmN.ttoc = readTOC(fid, loc_VOLM_TTOC, 'TTOC')

        # VOLM-VDEF
        # Read VDEF entry
        loc_VDEF_IMAG = F.ftoc.pntVolm[n] + volmN.sizeTable + volmN.ttoc.sizeTable
        volmN.vdef = readDEF(fid, loc_VDEF_IMAG, 'VDEF')

        # VOLM-VCHN & VOLM-XDEF
        # Initialize local arrays
        volmN.vchn = []
        volmN.xdef = namedtuple("volmNxdef", [])

        # We unfortunately don't know how many VCHN entries to expect
        done = 0

        while done == 0:
            # Read header
            [dumCRC, lastSize, lastType, dumMisc] = readARDFpointer(fid, -1)
            # Read data differently depending on data type
            if lastType == 'VCHN':
                # Read 32 bytes of text
                textSize = 32
                txt1 = struct.unpack('s'*textSize, fid.read(textSize))
                txt2 = b''.join(txt1)
                theChannel = txt2.decode("cp1252")
                # theChannel = transpose( fread(fid, textSize, '*char') )
                # Append to channelList
                volmN.vchn.append(theChannel)
                # Read 32 dummy bytes
                remainingSize = lastSize - 16 - textSize
                dum = struct.unpack('s'*remainingSize, fid.read(remainingSize))  # fread(fid, remainingSize, '*char')
            elif lastType == 'XDEF':
                # Read additional header parameters
                # dum = fread(fid, 1, 'uint32')
                dum = struct.unpack('i', fid.read(4))[0]
                volmN.xdef.sizeTable = struct.unpack('i', fid.read(4))[0]  # fread(fid, 1, 'uint32')

                # Read text
                textSize = volmN.xdef.sizeTable
                txt1 = struct.unpack('s'*textSize, fid.read(textSize))
                txt2 = b''.join(txt1)
                volmN.xdef.text = txt2.decode("cp1252")  # transpose( fread(fid, F.(volmN).xdef.sizeTable, '*char') )

                # Read zero values
                remainingSize = lastSize - 16 - 8 - textSize
                dum = struct.unpack('s'*remainingSize, fid.read(remainingSize))
                # dum = fread(fid, lastSize - 16 - 8 - F.(volmN).xdef.sizeTable, '*char');
                done = 1

            else:
                print(['ERROR: '+ str(typeEntry) +' not recognized!'])

        # Write channel list data to structure
        # D.channelList = cat(3, D.channelList, F.(volmN).vchn)
        D.channelList.append(volmN.vchn)
        # VOLM-VTOC & VOLM-VOFF
        # Read Entire VTOC/VOFF Table
        # F.(volmN).idx = readTOC(fid, -1, 'VTOC')
        volmN.idx = readTOC(fid, -1, 'VTOC')

        # VOLM-MLOV
        # Verify that we have readed the end VOLM header, MLOV
        [dumCRC, lastSize, lastType, dumMisc] = readARDFpointer(fid, -1)
        # ardf_checkType(lastType, 'MLOV', fid);

        # VOLM-VSET
        # Read first and last VSET point to get trace/retrace, up/down information
        # Alternatively every VSET can be read, but this takes more time, space
        # for r = 1:F.(volmN).vdef.lines
        for r in range (0, volmN.vdef.lines):

            # Determine dynamic field name
            # vsetN = ['vset' str(r)]
            vsetN = namedtuple("vsetN", [])
            # Determine VSET address
            loc = volmN.idx.linPointer[r]

            # If the data exists
            if loc != 0:
                # Record VSET information
                vsetN = readVSET(fid, loc)

                # Record Scan Up/Down information
                if vsetN.line != (r - 1):
                    volmN.scanDown = 1
                else:
                    volmN.scanDown = 0
                # Record Trace/Retrace Information
                if vsetN.point == 0:
                      volmN.trace = 1
                else:
                      volmN.trace = 0
#     % =======================================
#     % Partial File Handling
#     %
#     % Remove zero data from partial image files
#     % Rewrite incorrect ScanDown note
#     % =======================================
#
#     % Find zero pointers to identify zero rows
#     idxZero = find( F.(volmN).idx.linPointer == 0 );
#     incMin = 1;
#     incMax = 0;
#
#     % If scanDown, then we need to flip the values of the idxZero array
#     if F.(volmN).scanDown == 1
#         idxZero = F.(volmN).vdef.lines - idxZero + 1;
#         incMin = 0;
#         incMax = 1;
#     end
        F.volmN.append(volmN)

    # end % end read all VOLM information
    # %% part with force curves
    # Trace/retrace selection
    # If we have two volumes, choose the desired one
    # if F.numbVolm > 1:
    #     if trace == F.volmN[0].trace:
    #         getVolm = 'volm1'
    #     else
    #         getVolm = 'volm2'
    #     end
    # else:
    #     getVolm = 'volm1'
    VN = 0  # volume numer - frist volume always
    # Get number of points
    numbPoints = F.volmN[0].vdef.points
    # If ScanDown, create an adjusted line index variable
    numbLines = F.volmN[0].vdef.lines
    G = namedtuple("G", [])
    G.ytemp = []
    counter = 1
    G.curves = []
    for getLine in range(0, numbLines):  # ALL lines

        if F.volmN[0].scanDown == 0:  # check what wrong
            adjLine = numbLines - getLine - 1
        else:
            adjLine = getLine

        # Determine the number of data channels
        # numbChannels = size(D.channelList, 1);
        numbChannels = np.size(F.volmN[0].vchn)
        # Get the desired data
        # Get location of first VSET in line
        locLine = F.volmN[0].idx.linPointer[adjLine]
        # If data exists
        if locLine != 0:  # MAIN PART
            # Navigate to the desired location
            fid.seek(locLine, 0) # fseek(fid, locLine, 'bof');
            # Initialize data arrays
            G.numbForce = []
            G.numbLine = []
            G.numbPoint = []
            G.locPrev = []
            G.locNext = []
            G.name = []
            G.y = []
            G.pnt0 = []
            G.pnt1 = []
            G.pnt2 = []
            # Read in the entire line
            for n in range(0, numbPoints):
                currbyte = fid.tell()
                # Read VSET info
                fid.seek(4, 1)
                vset = readVSET(fid, -1)

                # Write VSET info to arrays
                G.numbForce.append(vset.force)
                G.numbLine.append(vset.line)
                G.numbPoint.append(vset.point)
                G.locPrev.append(vset.prev)
                G.locNext.append(vset.next)
                currbyte = fid.tell()
                # Read & write VNAM info
                vnam = readVNAM(fid, -1)
                G.name.append(vnam.name)

                # Clear data matrix
                theData = []

                # Read VDAT info
                for r in range(0, numbChannels):
                    vdat = readVDAT(fid, -1)
                    theData.append(vdat.data)
                # plt.plot(theData)
                G.curves.append(theData)  # G.ytemp{counter} = theData;
                # Read XDAT if it exists Not sure what data is stored in XDAT
                currbyte = fid.tell()
                readXDAT(fid, -1)
                currbyte = fid.tell()
                # Concatenate data
                # If not the same number of rows, pad smaller data with zeros
                rowsGy  = np.size(G.y)
                rowsDat = np.size(theData)
                # if (rowsGy != rowsDat) and (n != 1):

                #     % Determine max number of rows
                #     maxRows = max([ rowsGy rowsDat ]);

                #     % If G.y less than max rows, pad it
                #     if rowsGy < maxRows

                #         % Get size of Gy
                #         sizeGy = size(G.y);
                #         % Set new number of rows
                #         sizeGy(1) = maxRows;
                #         % Copy old G.y
                #         oldGy = G.y;
                #         % Create new array
                #         G.y = zeros(sizeGy);
                #         % Copy depending on 2D or 3D size of array
                #         if max( size( sizeGy ) ) > 2
                #             G.y(1:rowsGy,:,:) = oldGy;
                #         else
                #             G.y(1:rowsGy,:) = oldGy;
          #               end

                #     % If theData less than max rows, pad it
                #     else

                #         % Get size of theData
                #         sizeDat = size(theData);
                #         % Set new number of rows
                #         sizeDat(1) = maxRows;
                #         % Copy old theData
                #         oldDat = theData;
                #         % Create new array
                #         theData = zeros(sizeDat);
                #         % Copy old to new
                #         theData(1:rowsDat,:) = oldDat;

                #     end % end if need to pad array

                # end % end if not equivalent sizes

                # Do a straight concatination
                # G.y = cat(3, G.y, theData);

                # Write VDAT pointers only for the final channel read
                G.pnt0.append(vdat.pnt0)   # Pointers, presumably
                G.pnt1.append( vdat.pnt1)
                G.pnt2.append(vdat.pnt2)
                counter = counter+1
                # print(counter)
            # Flip each array if retrace data
            if G.numbPoint[0] != 0:

                G.numbForce = np.flip(G.numbForce, 0)
                G.numbLine  = np.flip(G.numbLine, 0)
                G.numbPoint = np.flip(G.numbPoint, 0)
                G.locPrev   = np.flip(G.locPrev, 0)
                G.locNext   = np.flip(G.locNext, 0)
                G.name      = np.flip(G.name, 0)
                # G.y         = np.flip(G.y, 3) # Note 3rd dimension
                G.pnt0      = np.flip(G.pnt0, 0)
                G.pnt1      = np.flip(G.pnt1, 0)
                G.pnt2      = np.flip(G.pnt2, 0)


            # If only a point desired, return only the point

    fid.close()
    return F, D, G

def ARDF_import(Pars, Data = 'Data'):
    
    filedir = Pars.filedir[0]
    if filedir[-4:] == 'ARDF':
        Asylumtype = 'ARDF'
    # elif filedir[-3:] == 'spm':
    #    Brukertype = 'spm'
    else:
        print('file extension is not supported (ARDF required)')
    [F, D, G] = getARDFdata(filedir)
    Notes2 = dict(x.split(":", 1) for x in D.Notes.splitlines())
    Pars.Notes = Notes2
    fnameshort = filedir.split("/")
    fnameshort = fnameshort[-1]
    Pars.fnameshort = []
    Pars.fnameshort.append(fnameshort)
    Pars.InvOLS = float(Notes2['InvOLS'])*1e9
    Pars.k = float(Notes2['SpringConstant'])
    Pars.dT = 1/float(Notes2['NumPtsPerSec'])
    Pars.probe_dimension = np.nan  # not in the file
    Pars.topo = np.array(D.y[0])*1e9  # to nm
    Pars.TipType = 'unknown'
    # Pars.Poisson = 0.5
    Numcurves = len(G.curves)

    Data3 =[]

    for ij in range(Numcurves):
        currentcurve = np.array(G.curves[ij][0:2], dtype=float).transpose()
        currentcurve = currentcurve*1e9  # to nm
        currentcurve[:,1] = currentcurve[:,1]/Pars.InvOLS   # to nA
        # nPix.append(ij)
        # Data.append(currentcurve)
        Data3.append([ij, currentcurve])

    Data3 = np.asarray(Data3, dtype=object)
    # Data = np.asarray([nPix, Data])
    # Data = Data.transpose()

    return Pars, Data3

    


if __name__ == '__main__':
    import sys
    sys.path.append('D:/MEGAsync/My materials/python/Ting_code')
    from Pars_class import Pars_gen
    from file_import_qt5simple import file_import_dialog_qt5
    import_opt = 'single'  # single, 2 - multi
    file_types = '*.ARDF'
    start_folder = 'D:\MailCloud\AFM_data\asylum\gels\printed_gels\180119_printdropsinverted_CSC38_9A'
    filename = file_import_dialog_qt5(import_opt=import_opt, start_folder=start_folder) 
    Pars = Pars_gen()
    Pars.filedir.append(filename)
    [Pars, Data] = ARDF_import(Pars)
    topo = Pars.topo


    # from file_import_qt5simple import file_import_dialog_qt5
    # import_opt = 'single'  # single, 2 - multi
    # file_types = '*.dat'
    # start_folder = ''
    # filename = file_import_dialog_qt5(import_opt=import_opt, start_folder=start_folder)
    # filename = 'D:/MEGAsync/My materials/python/Ting_code/user files/dropL4se00.ARDF'

    # file = open(filename, 'rb')  # Reading. 'r' can be omitted
    # [checkCRC32, sizeBytes, typePnt, miscNum] = readARDFpointer(file, 0)
    # ftoc = readTOC(file, -1, 'FTOC')
    # pprint.pprint(ftoc.__dict__)
    # file.close() # Closing file
    # [F, D, G] = getARDFdata(filename)
    #print(F.ftoc)
    # topo2 = D.y[0]
    # plt.imshow(topo2, interpolation='nearest',cmap='viridis', origin='lower')
    # curves = G.curves
    # curve = curves[-1]
    # plt.plot(curve[0], curve[1])

    #pprint.pprint(F.ftoc.__dict__)