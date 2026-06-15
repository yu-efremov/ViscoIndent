# -*- coding: utf-8 -*-
"""
@author: Yuri Efremov
"""


# TODO add normalization to all fits?
import sys
if not sys.warnoptions:
    import warnings
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import r2_score
import math
from math import pi
from math import tan
# import pandas as pd

from bottom_effect_correction import bottom_effect_correction
from Ting_numerical import ting_numerical, JKR_transition, JKR
from timetofreqspectrumFun import timetofreqspectrumFun
from make_Results import make_Results
from fixedfit import fixedfit


def npmax(larray):
    max_idx = np.argmax(larray)
    max_val = larray[max_idx]
    return (max_idx, max_val)


def ForceCurvePLinc(x, a, b, c, x0, y0):
    # Function for determination of the contatct point position
    # a = parFCPL[0]
    # b = parFCPL[1]
    # x0 = parFCPL[2]
    # y0 = parFCPL[3]
    # c = exponent value for the current indenter shape
    y = np.zeros(x.shape[0])  # zeros(size(x));

    for i in range(x.shape[0]):
        if x[i] < x0:
            y[i] = y0 + b * (x[i]-x0)
        else:
            y[i] = y0+a*(x[i]-x0)**(c)
    return y


def norm2my(x):
    maxX = max(x)
    minX = min(x)
    MagX = maxX-minX
    y = x/MagX
    # y=y-mean(y)
    # y = y-max(y) + 0.5  # center at 0
    y = y-max(y) + 1  # center at 1
    return y, MagX


def smoothM(d, parS):
    y = d
    DL = len(d)-1
    for ij in range(len(d)-1):
        if np.isnan(y[ij]):
            k = 0
            while np.isnan(y[ij]):
                k = k+1
                y[ij] = y[ij+k]
    if parS > 1:
        y[1] = (d[1] + d[2] + d[3])/3
        y[-2] = (d[DL-2] + d[DL-1] + d[DL])/3
    if parS == 2 or parS == 3:  # for 2 and 3
        for ij in range(2, DL-2):
            y[ij] = (d[ij-1] + d[ij] + d[ij+1])/3
    if parS >= 4:  # for 4 and 5 and any more
        for n in range(2, DL-2):
            y[n] = (d[n-2] + d[n-1] + d[n] + d[n+1] + d[n+2])/5
    return y


def HertzBEC(h, K1, Poisson, Radius, power, Height, level0, modelprobe, ind):
    """ both EHertz and contact point are fitted
    h[0] is EHertz
    h[1] is contact point, indentation position
    level0 shoul be for used ind relative surfZ
    only for sphere"""
    if np.isnan(level0):
        if Height > 0:  # BEC for the purely elastic case
            BEC = bottom_effect_correction(Poisson, Radius, Height, modelprobe, ind-h[1])[0]
            # print('case1')
        else:
            BEC = np.ones(len(ind))
            # print('case2')
        # Force = abs(K1*h[0]*(ind-h[1])**(power)*BEC)
    else:
        BEC = bottom_effect_correction(Poisson, Radius, level0-h[1], modelprobe, ind-h[1])[0]
        # print('case3')
        
    if modelprobe =='sphere_correction1':
        BEC = BEC*(1-ind/(10*Radius))
    
    Force = K1*h[0]*abs(ind-h[1])**(power)*BEC

    return Force  # , BEC

def DMTBEC(h, K1, Poisson, Radius, power, Height, level0, modelprobe, ind):
    """ both EHertz and contact point are fitted
    h[0] is EHertz
    h[1] is contact point, indentation position
    h[2] is the adhesion force
    level0 shoul be for used ind relative surfZ
    only for sphere"""
    if np.isnan(level0):
        if Height > 0: #BEC for the purely elastic case
            BEC = bottom_effect_correction(Poisson, Radius, Height, modelprobe, ind-h[1])[0]
            # print('case1')
        else:
            BEC = np.ones(len(ind))
            # print('case2')
        #Force = K1*h[0]*abs((ind-h[1]))**(power)*BEC + h[2]
    else:
        BEC = bottom_effect_correction(Poisson, Radius, level0-h[1], modelprobe, ind-h[1])[0]
        print('case3')

    Force = K1*h[0]*abs((ind-h[1]))**(power)*BEC + h[2]
    return Force

def JKRnoBEC(h, K1, Poisson, Radius, power, Height, level0, modelprobe, ind):
    """ both EHertz and contact point are fitted
    h[0] is EHertz
    h[1] is contact point, indentation position
    h[2] is the adhesion
    no BEC yet
    only for sphere"""
    # Force = JKR(E_Hertz, adhesion_force, Poisson, probe_size, ind)
    Force = JKR(h[0], h[2], Poisson, Radius, ind-h[1])[0]
    return Force

def locate_position(pointV, arraydata):
    location = 0
    for k in range(len(arraydata)):  # arraydata.shape[0]
        if arraydata[k] > pointV:
            location = k-1
            break
    return location

# start of the main function


def tingsprocessingd1(Pars, curve_data):
    Results = make_Results(0)
    kk = 0
    pix = curve_data[0]
    # curve data: pixel, curve itselfs, DFL_corrs, dT, filename
    # contact_points = [0, 0, 0]
    # test = 1
    # if test == 1:
    DFL_corrs = [0, 0, 0]
    if len(curve_data)>=4:
        fnameshort = curve_data[4]
    else:
        fnameshort = Pars.fnameshort[kk]

    try:
        # kk = 0
        Results = make_Results(1)
        print('\nFC#', curve_data[0], ' ', sep='', end='')
        # global Pars
        Sens = Pars.InvOLS                  # sensetivity, nm/V
        Stiffness = Pars.k                  # cantilever spring constant, nN/nm (N/m)
        modelprobeStr = Pars.probe_shape    # 1-sphere/paraboloid, 2-cone, 3-pyramid, 4-cylinder
        Radius = Pars.probe_dimension       # radius (nm) or angle (degrees)
        Poisson = Pars.Poisson              # Poisson's ratio of the sample
        dT = Pars.dT                        # Sampling time
        try:
            dT = curve_data[3]  # dT in Data array position
            if dT == 0:
              dT = Pars.dT  
        except:
            dT = Pars.dT
        # print(dT)
        modelting = Pars.viscomodel         # 1 - PLR, 2 - SLS, 3 - sPLR, 4 - elastic only
        hydrodragcorr = Pars.hydro.corr  # hydrodinamic drag correction 0 or 1
        Height = Pars.height
        binomsearch = Pars.cp_adjust
        
        adhesion_model = Pars.adhesion_model  # none, DMT, JKR
        adhesion_region = Pars.adhesion_region # 'approach' 'retraction' 'both'
        adhesion = 0  # max adhesion force
        adhesion_pars = [adhesion_model, adhesion_region, -adhesion]

        try:
            HeightfromZ = Pars.HeightfromZ
        except:
            HeightfromZ = 0
        try:
            hydrotype = Pars.hydro.corr_type
        except:
            hydrotype = 2
        try:
            speedcoefmT = Pars.hydro.speedcoef
        except:
            speedcoefmT = 0
        try:
            DepthStart_percent = Pars.depth_start
            DepthEnd_percent = Pars.depth_end
        except:
            DepthStart_percent = 7
            DepthEnd_percent = 95
        try:
            Fixedpars = Pars.fixed_values
        except:
            Fixedpars = np.array([[0, 0, 0], [0, 0, 0]], dtype=float)
        try:
            Downsampling = Pars.downsampling
        except:
            Downsampling = 0
        try:
            right_border_cp1 = Pars.right_border_cp1
        except:   
            right_border_cp1 = 0.6

        if HeightfromZ == 1:
            SurfZ = np.reshape(Pars.PL, Pars.PL.size)[pix]  # changed from kk to pix
        else:
            SurfZ = np.nan

        modelprobe = modelprobeStr

        if modelprobe in {'sphere', 'sphere_correction1'}:
            K1 = 4*Radius**0.5/3
            power = 1.5
        elif modelprobe == 'cone' or modelprobe == 'pyramid':
            power = 2
            if modelprobe == 'cone':
                K1 = 2/pi*tan(Radius*pi/180)
            elif modelprobe == 'pyramid':
                K1 = 1.4906/2*tan(Radius*pi/180)  # Bilodeau, 1992
        elif modelprobe == 'cylinder':  # cylinder
            K1 = 2*Radius  # before integral no speed
            power = 1
        elif modelprobe == 'spheroid': # cylinder
            power = 1.5
            K1 = 4*Radius**0.5/3/2**1.5  # doubled deformation, radius should be same for all
            # K1 = K1/(Radius/Height)**0.5  # base diameter on measured height
            # K12 = Radius**0.5 

        K1 = K1/(1-Poisson**2)*1e-9
        
        # plt.imshow(Pars.PL'],interpolation='nearest',cmap='viridis', origin='Lower')
        currentcurve = curve_data[1][:, 0:2]  # or DataSel
        # plt.plot(currentcurve[:,0], currentcurve[:,1])
        Npoints = currentcurve.shape[0]

        if Npoints>1000:
            skip_data = np.floor(Npoints/1200).astype(int)
            skip_data2 = np.floor(Npoints/600).astype(int)
        else:
            skip_data = 1
            skip_data2 = 1

        if Downsampling == 2:
            skip_data = skip_data2
        if skip_data > 2 and Downsampling == 0:
            print('too many data poits, downsampling is recommended; ', sep='', end='')
        elif skip_data > 1 and Downsampling >= 1:
            print('downsampled x', skip_data, sep='', end='')
            index_select = range(1, Npoints, skip_data)
            # Data_old[kk] = Data[kk]
            currentcurveM = currentcurve
            currentcurveM[:, 0] = smoothM(currentcurve[:, 0], skip_data)
            currentcurveM[:, 1] = smoothM(currentcurve[:, 1], skip_data)
            currentcurve = currentcurveM[index_select, :]
            curve_data[1] = currentcurve
            dT = dT*skip_data
            curve_data[3] = dT

        checklength = int(currentcurve.shape[0]*0.05)
        if currentcurve[0, 0] > currentcurve[checklength, 0]:
            Displ = -currentcurve[:, 0]+currentcurve[0, 0]
            # CurveDir = -1  # decreasing curve
        else:  # default case
            Displ = currentcurve[:, 0] - currentcurve[0, 0]
            # CurveDir = 1  # increasing curve

        StartZ = currentcurve[0, 0]
        SurfZ = StartZ + SurfZ
        DFLIn = currentcurve[:, 1]
        FullLength = Displ.shape[0]
        [ApprLength, maxZ] = npmax(Displ)
        DisplAppr = Displ[0:ApprLength]
        DFLAppr0 = DFLIn[0:ApprLength]
        DFLAppr0Norm = norm2my(DFLAppr0)[0]*100
        DisplApprNorm = norm2my(DisplAppr)[0]*100
        # plt.plot(DFLIn)  plt.plot(DisplAppr, DFLAppr0) plt.plot(DisplApprNorm, DFLAppr0Norm)

        # ===First fit to locate the aproximate contatct point position.
        # define function for fitting
        # ForceCurvePLinc(x, a, b, c, x0, y0)
        ForceCurvePLincP = lambda x, a, b, x0, y0: ForceCurvePLinc(x, a, b, power, x0, y0)
        # fStartPoint = [1e-4,  1e-7,  60,  0]  # a,b,x0,y0
        fStartPoint = [1e-4,  0,  60,  0]  # a,b,x0,y0
        bounds0 = (0, -100, 0, -10), (100000, 20, 97, 97)
        c, cov = curve_fit(ForceCurvePLincP, DisplApprNorm, DFLAppr0Norm, fStartPoint, bounds=bounds0)
        # c,cov = curve_fit(ForceCurvePLincP,DisplAppr,DFLAppr0, [1e-4,  1e-7,  1000,  0])
        # yp = ForceCurvePLincP(DisplAppr,c[0],c[1],c[2],c[3])
        # plt.plot(DisplAppr, DFLAppr0); plt.plot(DisplAppr, yp)
        # yp = ForceCurvePLincP(DisplApprNorm, c[0], c[1], c[2], c[3])
        # plt.plot(DisplApprNorm, DFLAppr0Norm);  plt.plot(DisplApprNorm, yp)
        # print('R^2: '+str(r2_score(DFLAppr0,yp)))
        contactpoint0 = c[2]/100*norm2my(DisplAppr)[1]
        # find cp0 location
        locationcp0 = locate_position(contactpoint0, DisplAppr)
        firstE = c[0]/K1*(norm2my(DFLAppr0)[1]/100)/(norm2my(DisplAppr)[1]/100)**(power) #normalization should be accounted for
        # print(firstE)

        # hydrodynamic correction part
        DFL_corrs = [0, 0, 0]
        if hydrodragcorr == 0:  # speedcoefmT=0
            nohydrls = 0.15
            nohydrle = 0.75
            linearstart = math.floor(nohydrls*locationcp0)  # 0.15
            linearend = math.floor(nohydrle*locationcp0)  # 0.75
            AppX1 = DisplAppr[linearstart:linearend]
            AppY1 = DFLAppr0[linearstart:linearend]
            flinParams = np.polyfit(AppX1, AppY1, 1)
            slope1 = flinParams[0]
            intercept1 = flinParams[1]
            DFLc = DFLIn-slope1*Displ-intercept1  # all curve corrected light Approach
            DFL_corrs = [slope1, intercept1, 0]  # slope intercept, speedcoefmT
            # plt.plot(Displ, DFLc)
        elif hydrodragcorr == 1:
            startam = 1
            r01 = math.ceil(locationcp0*0.7)
            if hydrotype == 1:  # TODO not ready yet
                pass
            if hydrotype == 2:  # using provided speedcoefmT
                DisplSpeed = np.diff(Displ)/dT
                DisplSpeed = np.append(DisplSpeed, DisplSpeed[-1])
                DisplSpeed = smoothM(DisplSpeed, 5)
                DFLT = DFLIn-speedcoefmT*DisplSpeed/Sens/Stiffness
                # plt.plot(DisplSpeed), plt.plot(DisplSpeedS)
                # plt.plot(DFLIn), plt.plot(DFLT,'--'); plt.plot(Displ,DFLT)
                DisplP1 = Displ[startam:r01]
                DFLP1 = DFLT[startam:r01]
                flinParams = np.polyfit(DisplP1, DFLP1, 1)
                slopeH = flinParams[0]
                interceptH = flinParams[1]
                DFLc = DFLT-slopeH*Displ-interceptH
                DFL_corrs = [slopeH, interceptH, speedcoefmT]
                # print(DFL_corrs)
                # plt.plot(Displ, DFLc)
            if hydrotype == 3:  # for biomomentum
                zero_level_length = checklength  # around 5%
                zero_level_length = 10
                zero_level_DFL = np.mean(DFLIn[startam:zero_level_length])
                slopeH = 0
                interceptH = zero_level_DFL
                speedcoefmT = 0
                DFLc = DFLT-slopeH*Displ-interceptH
                DFL_corrs = [slopeH, interceptH, speedcoefmT]
                # print(DFL_corrs)
        else:
            DFLc = DFLIn  # no correction

        Indentation_noCP = Displ - Sens*DFLc  # indentation but no contact point provided        
        Forcefull1 = DFLc*Sens*Stiffness  # complete force
        ForcefullApproach = Forcefull1[0:ApprLength]

        # second fit for the contact point estimation
        left_border_point = math.floor(locationcp0*0.6)
        right_border_point = locationcp0+math.floor((ApprLength-locationcp0)*right_border_cp1)
        AppX1 = DisplAppr[left_border_point:right_border_point]
        AppY1 = ForcefullApproach[left_border_point:right_border_point]
        # plt.plot(AppX1,AppY1)
        fStartPoint = [1e-4,  0,  contactpoint0,  0]  # a,b,x0,y0
        c1, cov1 = curve_fit(ForceCurvePLincP, AppX1, AppY1, fStartPoint)  # do not edit! the most broad one
        # yp = ForceCurvePLincP(AppX1, c1[0], c1[1], c1[2], c1[3])
        # plt.plot(AppX1,AppY1,AppX1,yp)
        contactpoint01 = c1[2]
        locationcp01 = locate_position(contactpoint01, DisplAppr)
        IndentationFullApproach = Indentation_noCP[0:ApprLength] - contactpoint01
        [ApprLength2, maxF] = npmax(ForcefullApproach)

        IndentationApproach = IndentationFullApproach[locationcp01:ApprLength2]  # only contact region
        ForceApproach = Forcefull1[locationcp01:ApprLength2]  # only contact region
        # plt.plot(Displ, Forcefull1)
        # plt.plot(IndentationApproach, ForceApproach)

        DepthStart = DepthStart_percent*0.01*max(IndentationApproach)  # first 7% of ind ignored
        if DepthStart > 0:
            depth_start = locate_position(DepthStart, IndentationApproach)
            # TODO add safety check
        else:
            depth_start = 0
        DepthEnd = DepthEnd_percent*0.01*max(IndentationApproach)
        if DepthEnd > 0:
            depth_end = locate_position(DepthEnd, IndentationApproach)
            # TODO add safety check
        else:
            depth_end = len(IndentationApproach)

        IndAppSel = IndentationApproach[depth_start:depth_end]
        IndFSel = ForceApproach[depth_start:depth_end]
        # plt.plot(IndAppSel, IndFSel)
        # IndAppSelNorm = IndAppSel./max(IndAppSel)
        # IndFSelNorm = IndFSel./max(IndFSelNorm)
        # ============== elastic model fit, with or without BEC
        funHertzfit = lambda ind, a, b: HertzBEC([a, b], K1, Poisson, Radius, power, 0, np.nan, modelprobe, ind)
        # a=funHertzfit(IndentationApproach, 1, 2)
        fHbounds = ([1e-5, -300]), ([1e10, DepthStart-1])  
        # fHbounds = ([1e-5, -300]), ([1e10, 1e10])  # problem with bounds too, give infs or nans
        fHStartPoint = [1000, 0]  # firstE - cause some problems... [1000, 0]
        # print(fHStartPoint)
        warnings.simplefilter("ignore")
        # fHertz, fHcov = curve_fit(funHertzfit, IndAppSel, IndFSel, fHStartPoint, bounds=fHbounds, check_finite=False)
        try:
            fHertz, fHcov = curve_fit(funHertzfit, IndAppSel, IndFSel, fHStartPoint, bounds=fHbounds)
        except:
            fHertz, fHcov = curve_fit(funHertzfit, IndAppSel, IndFSel, fHStartPoint)
        warnings.simplefilter("default")
        # fHcurve = funHertzfit(IndentationApproach, fHertz[0], fHertz[1]) # for plot
        # plt.figure()
        # plt.plot(IndentationApproach,ForceApproach,IndentationApproach, fHcurve)
        # plt.plot(Indentation_noCP[locationcp01:ApprLength2], Forcefull1[locationcp01:ApprLength2], Indentation_noCP[locationcp01:ApprLength2], fHcurve)

        EHertz = fHertz[0]
        # print(fHertz)
        contactHertz = contactpoint01+fHertz[1]
        HeightH = -contactHertz-SurfZ
        RSq_Hertz = r2_score(IndFSel, funHertzfit(IndAppSel, fHertz[0], fHertz[1]))
        AdjRSq_Hertz = 1-(1-RSq_Hertz)*(len(IndFSel)-1)/(len(IndFSel)-2-1) # 1-(1-RSq)*(n-1)/(n-p-1)
        EHertzBEC = 0
        locationcpH = locate_position(contactHertz, DisplAppr)
        AdjRSq_HertzResults = AdjRSq_Hertz
        HeightHRes = HeightH
        contactHertzRes = contactHertz

        if (np.isnan(SurfZ) == 0 and HeightfromZ == 1) or Height > 0:  # BEC for elastic models
            warnings.simplefilter("ignore")
            if Height == 0:
                funHertzfitBEC = lambda ind, a, b: HertzBEC([a, b], K1, Poisson,
                        Radius, power, 0, -contactpoint01-SurfZ, modelprobe, ind)
            else:
                funHertzfitBEC = lambda ind, a, b: HertzBEC([a, b], K1, Poisson,
                                Radius, power, Height, np.nan, modelprobe, ind)
            # fHertzBEC, fHBECcov = curve_fit(funHertzfitBEC, IndAppSel, IndFSel,
            #                     fHStartPoint, bounds=fHbounds)  # check_finite = False
            fHertzBEC, fHBECcov = curve_fit(funHertzfitBEC, IndAppSel, IndFSel, fHStartPoint, bounds=fHbounds)
            warnings.simplefilter("default")
            # fHBECcurve = funHertzfitBEC(IndentationApproach, fHertzBEC[0], fHertzBEC[1])  # for plot
            # plt.plot(IndentationApproach, ForceApproach, IndentationApproach, fHBECcurve)
            EHertzBEC = fHertzBEC[0]
            contactHertzBEC = contactpoint01+fHertzBEC[1]
            HeightHBEC = -contactHertzBEC-SurfZ
            RSq_HertzBEC = r2_score(IndFSel, funHertzfitBEC(IndAppSel, fHertzBEC[0], fHertzBEC[1]))
            AdjRSq_HertzBEC = 1-(1-RSq_HertzBEC)*(len(IndFSel)-1)/(len(IndFSel)-2-1)  # 1-(1-RSq)*(n-1)/(n-p-1)
            locationcpH = locate_position(contactHertzBEC, DisplAppr)
            AdjRSq_HertzResults = AdjRSq_HertzBEC
            HeightHRes = HeightHBEC
            contactHertzRes = contactHertzBEC
        else:
            EHertzBEC = np.nan
        # TODO select what to save in Data. may be nothing, only contactpoints

        MaxDepth = np.max(IndentationApproach)
        hysterareasearch = 1  # approach-retraction hysteresis area
        if hysterareasearch == 1:
            indstart = int(locationcpH)
            MaxInd = int(ApprLength-indstart)  # points in Approach indentation
            indend = FullLength  # safety check
            for ala in range(indstart+MaxInd, FullLength):  # alternative search for indend based on Z value
                if Displ[ala] < Displ[indstart]:
                    indend = ala
                    break
            indentationfull = Displ[indstart:indend] - Displ[indstart] - Forcefull1[indstart:indend]/Stiffness
            Forcefull = Forcefull1[indstart:indend]
            areaappr = np.trapezoid(Forcefull[0:MaxInd], indentationfull[0:MaxInd])
            ForceR = np.array(Forcefull[MaxInd:])
            ForceR[ForceR < 0] = 0  # only positive points are used
            arearetr = -np.trapezoid(ForceR, indentationfull[MaxInd:])
            hystarea = areaappr - arearetr
            normhystarea = hystarea/areaappr
        contact_timeH = MaxInd*dT
        
        E_adhesion = 0
        adhesion = 0
        cp_adhesion=0
        # print(adhesion_model)
        # print(adhesion_region)
        if adhesion_model != 'none' and adhesion_region == 'approach' or adhesion_region == 'both':
            # max_adh_search = ApprLength
            max_adh_search = locationcp01+20
            min_adh_search = locationcp01-20
            # min_adh_search = 0
            adhFapproach = np.min(Forcefull1[min_adh_search:max_adh_search])
            adhFapprloc = np.argmin(Forcefull1[min_adh_search:max_adh_search]) + min_adh_search
            Indentappr_adh = Displ[adhFapprloc:ApprLength] - Displ[adhFapprloc] - Forcefull1[adhFapprloc:ApprLength]/Stiffness
            if DepthEnd > 0:
                depth_end = locate_position(DepthEnd, Indentappr_adh)
                # TODO add safety check
                if depth_end==0:
                    depth_end = len(Indentappr_adh)
            else:
                depth_end = len(Indentappr_adh)
        if adhesion_model == 'DMT':
            if adhesion_region == 'approach' or adhesion_region == 'both':
                # use region from the Hertz fit
                #plt.plot(Indentappr_adh,Forcefull1[adhFapprloc:ApprLength]) plt.plot(IndFSel) plt.plot(IndRetrSel)
                funDMTfit = lambda ind, a, b, c: DMTBEC([a, b, c], K1, Poisson, Radius, power, 0, np.nan, modelprobe, ind)
                fDMTbounds = ([1e-5, -300, -1e10]), ([1e10, DepthStart-1, -1e-10])  # max adh is 0
                fDMTbounds = ([1e-10, -1e5, -1e10]), ([1e10, 0, -1e-10])  # max adh is 0                
                fDMTStartPoint = [firstE, 0, adhFapproach]  # firstE - cause some problems...
                forceAdhfit = Forcefull1[adhFapprloc:adhFapprloc+depth_end]
                # fDMT, fHcov = curve_fit(funDMTfit, Indentappr_adh[0:depth_end], Forcefull1[adhFapprloc:adhFapprloc+depth_end], fDMTStartPoint, bounds=fDMTbounds)
                fDMT, fHcov = curve_fit(funDMTfit, Indentappr_adh[0:depth_end], Forcefull1[adhFapprloc:adhFapprloc+depth_end], fDMTStartPoint)  # no bounds
                # plt.plot(Indentappr_adh[0:depth_end], Forcefull1[adhFapprloc:adhFapprloc+depth_end])
                # plt.plot(Indentappr_adh[0:depth_end], funDMTfit(Indentappr_adh[0:depth_end], fDMT[0], fDMT[1], fDMT[2]))
                # plt.plot(Indentappr_adh[0:depth_end]-fDMT[1], funDMTfit(Indentappr_adh[0:depth_end], fDMT[0], fDMT[1], fDMT[2]))
                # plt.plot(Indentation_noCP[adhFapprloc:adhFapprloc+depth_end], Forcefull1[adhFapprloc:adhFapprloc+depth_end], Indentation_noCP[adhFapprloc:adhFapprloc+depth_end], funDMTfit(Indentappr_adh[0:depth_end], fDMT[0], fDMT[1], fDMT[2]))
                # plt.plot(Indentation_noCP[locationcp01:ApprLength2], Forcefull1[locationcp01:ApprLength2], Indentation_noCP[locationcp01:ApprLength2], funDMTfit(IndentationApproach, fHertz[0], fHertz[1], 0))               
                E_adhesionAppr = fDMT[0]
                adhesionApr = fDMT[2]
                cp_adhesion = fDMT[1] + Displ[adhFapprloc]
                locationcp_adhesion = locate_position(cp_adhesion, DisplAppr)
                # contactHertzRes = cp_adhesion
                adhesion=adhesionApr
                E_adhesion = E_adhesionAppr
                print(adhesionApr)
            if adhesion_region == 'retraction' or adhesion_region == 'both':
                Indentretraction_full = Displ[ApprLength+1:] - contactHertzRes - Forcefull1[ApprLength+1:]/Stiffness
                # find point of lowest force
                ind_adhesion = np.argmin(Forcefull1[ApprLength+1:])
                IndRetrSel = Indentretraction_full[:ind_adhesion]
                IndFSel = Forcefull1[ApprLength+1:ApprLength+ind_adhesion+1]
                #plt.plot(IndRetrSel,IndFSel) plt.plot(IndFSel) plt.plot(IndRetrSel)
                funDMTfit = lambda ind, a, b, c: DMTBEC([a, b, c], K1, Poisson, Radius, power, 0, np.nan, modelprobe, ind)
                fDMTbounds = ([1e-5, -300, -1e10]), ([1e10, DepthStart-1, 0])  # max adh is 0
                fDMTStartPoint = [1000, 0, -1]  # firstE - cause some problems...
                fDMT, fHcov = curve_fit(funDMTfit, IndRetrSel, IndFSel, fDMTStartPoint, bounds=fDMTbounds)
                E_adhesionRetr = fDMT[0]
                adhesionRetr = fDMT[2]
                print(adhesionRetr)
                if adhesion==0:
                    adhesion=adhesionRetr
                    E_adhesion = E_adhesionRetr
            adhesion_pars = [adhesion_model, adhesion_region, np.nanmin(Forcefull1)]
        if adhesion_model == 'JKR':
            if adhesion_region == 'approach' or adhesion_region == 'both':
                funJKRfit = lambda ind, a, b, c: JKRnoBEC([a, b, c], K1, Poisson, Radius, power, 0, np.nan, modelprobe, ind)
                fJKRbounds = ([1e-5, -300, -1e10]), ([1e10, DepthStart-1, -1e-10])  # max adh is 0
                fJKRbounds = ([1e-10, -1e5, -1e10]), ([1e10, 0, -1e-10])  # max adh is 0                
                fJKRStartPoint = [firstE, 0, adhFapproach]  # firstE - cause some problems...
                forceAdhfit = Forcefull1[adhFapprloc:adhFapprloc+depth_end]
                # fDMT, fHcov = curve_fit(funDMTfit, Indentappr_adh[0:depth_end], Forcefull1[adhFapprloc:adhFapprloc+depth_end], fDMTStartPoint, bounds=fDMTbounds)
                fJKR, fHcov = curve_fit(funJKRfit, Indentappr_adh[0:depth_end], Forcefull1[adhFapprloc:adhFapprloc+depth_end], fJKRStartPoint)  # no bounds
                # plt.plot(Indentappr_adh[0:depth_end], Forcefull1[adhFapprloc:adhFapprloc+depth_end])
                # plt.plot(Indentappr_adh[0:depth_end], funJKRfit(Indentappr_adh[0:depth_end], fJKR[0], fJKR[1], fJKR[2]))
                E_adhesion = fJKR[0]
                adhesion = fJKR[2]
                cp_adhesion = fJKR[1] + Displ[adhFapprloc]
                # contactHertzRes = cp_adhesion
                print(adhesion)
        if adhesion_model == 'JKR_transition' and modelprobe == 'sphere':
            if adhesion_region == 'approach' or adhesion_region == 'retraction' or adhesion_region == 'both':
                Indentretraction_full = Displ[ApprLength+1:] - contactHertzRes - Forcefull1[ApprLength+1:]/Stiffness
                # find point of lowest force
                ind_adhesion = np.argmin(Forcefull1[ApprLength+1:])
                IndRetrSel = Indentretraction_full[:ind_adhesion]
                IndFSel = Forcefull1[ApprLength+1:ApprLength+ind_adhesion+1]
                #plt.plot(IndRetrSel,IndFSel) plt.plot(IndFSel) plt.plot(IndRetrSel)
                funJKRfit = lambda ind, a, b, c: JKR_transition([a, b, c], K1, Poisson, Radius, power, 0, np.nan, modelprobe, ind)
                fDMTbounds = ([1e-5, -300, -1e10]), ([1e10, DepthStart-1, 0])  # max adh is 0
                fDMTStartPoint = [1000, 0, -1]  # firstE - cause some problems...
                fDMT, fHcov = curve_fit(funDMTfit, IndRetrSel, IndFSel, fDMTStartPoint, bounds=fDMTbounds)
                E_adhesion = fDMT[0]
                adhesion = fDMT[2]
                
        # print('\n')
        # print(kk)
        Results.loc[kk, 'Name'] = fnameshort
        Results.loc[kk, 'Pixel'] = curve_data[0]
        Results.loc[kk, 'model'] = modelting
        Results.loc[kk, 'EHertz'] = EHertz
        Results.loc[kk, 'hysteresis'] = normhystarea
        Results.loc[kk, 'max_force'] = np.nanmax(Forcefull1)
        Results.loc[kk, 'max_adh'] = np.nanmin(Forcefull1)
        Results.loc[kk, 'ind_depth'] = MaxDepth
        Results.loc[kk, 'AdjRsqHertz'] = AdjRSq_HertzResults
        Results.loc[kk, 'Height'] = HeightHRes
        if np.isnan(HeightHRes) and Height>0:
            Results.loc[kk, 'Height'] = Height
        Results.loc[kk, 'cpHertz'] = contactHertzRes
        Results.loc[kk, 'EHertzBEC'] = EHertzBEC
        Results.loc[kk, 'E_adhesion'] = E_adhesion
        Results.loc[kk, 'adhesion'] = adhesion
        Results.loc[kk, 'cp_adhesion'] = cp_adhesion
        Results.loc[kk, 'contact_timeH'] = contact_timeH

        indentationfull = Displ - contactHertzRes - Forcefull1/Stiffness
        warnings.simplefilter("ignore")
        if (np.isnan(SurfZ) == 0 and HeightfromZ == 1) or Height > 0:
            FitAppr = funHertzfitBEC(indentationfull, fHertzBEC[0], 0)
        else:
            FitAppr = funHertzfit(indentationfull, fHertz[0], 0)
        FitAppr[0:locationcpH] = np.nan  # remove extra
        FitAppr[ApprLength+1:] = np.nan  # remove extra
        warnings.simplefilter("default")
        # plt.plot(indentationfull,Forcefull1,indentationfull,FitAppr)
        # plt.plot(Displ, Forcefull1)
        currentcurve2 = np.c_[currentcurve, indentationfull, Forcefull1, FitAppr]
        curve_data[1] = currentcurve2



        # ============= part with viscoelastic processing ====================
        if normhystarea < 0.07:  # elastic warning
            print('small hysteresis area, mostly elastic response; ')

        if modelting != 'elastic':
            Resultstemp = np.full([15, 10], np.nan)
            Force_fit_arr = {}
            if len(IndentationApproach) < 70:
                print('too few indentation data points; ')
            pointright = 0  # shift in contact point position
            pr_step = np.floor(len(IndentationApproach)/10)  # step for adjustment of the contact point
            pointright = pointright - pr_step
            break_condition = 0
            modelting2 = modelting  # for test_models
            for ii in range(0, 15):  # max number of steps for adjustment of the contatct point
                pointright = pointright + pr_step
                indstart = int(locationcpH + pointright)
                MaxInd = int(ApprLength - indstart)  # points in Approach indentation
                for ala in range(indstart+MaxInd, FullLength):  # alternative search for indend based on Z value
                    if Displ[ala] < Displ[indstart]:
                        indend = ala
                        break
                indentationfull = Displ[indstart:indend] - Displ[indstart] - Forcefull1[indstart:indend]/Stiffness
                indentationfull[indentationfull < 0] = 0
                Forcefull = Forcefull1[indstart:indend]
                # plt.plot(indentationfull, Forcefull)
                NF = (max(Forcefull)-min(Forcefull))/10  # normalization to improve the fit quality
                Forcefulln = Forcefull/NF
                adhesion_parsn = [adhesion_pars[0], adhesion_pars[1], adhesion_pars[2]/NF]
                if adhesion_model != 'none':
                    adhesion_pars[2] = adhesionApr
                if adhesion_model == 'DMT' and adhesion_region == 'approach' or adhesion_region == 'both':
                    locationcpH = locationcp_adhesion
                #     Forcefulln = Forcefulln - adhesion/NF
                #     Forcefull = Forcefull - adhesion
                if modelting == 'sPLR':
                    par00 = [EHertz/NF, 0.15, EHertz*0.1/NF]  # E1,alpha,Einf
                    # parbounds = (EHertz*0.01/NF, 0, 0), (EHertz*1e4/NF, 1, EHertz*10/NF)
                    parboundsnp = np.array(((EHertz*0.001/NF, 0, 0), (EHertz*1e4/NF, 1, EHertz*10/NF)))
                elif modelting == 'SLS':
                    par00 = [EHertz*1.3/NF, 10, EHertz*0.7/NF]  # E0,tau,Einf
                    parboundsnp = np.array(((EHertz*0.001/NF, 1e-10, 0), (EHertz*1e5/NF, 1e3, EHertz*4/NF)))
                    parboundsnp = np.array(((EHertz*0.001/NF, 3, 0), (EHertz*1e5/NF, 1e3, EHertz*4/NF)))
                elif modelting == 'sPLReta' or modelting == 'sPLRetatest':
                    par00 = [EHertz/NF,0.2,1e-10/NF]  # E1, alpha, nu
                    parboundsnp = np.array(((EHertz*0.01/NF, 0, 0), (EHertz*1e4/NF, 1 ,1e-4/NF)))
                else:
                    par00 = [EHertz/NF, 0.15, EHertz*0.1/NF]  # E1,alpha,Einf
                    parboundsnp = np.array(((0, 0, 0), (EHertz*1e5/NF, 1e5, 1e5)))

                # sPLRetatest - nu from speed    
                if modelting == 'sPLRetatest' or modelting2 == 'sPLRetatest':
                    indspeed = np.diff(indentationfull)/dT
                    indspeed = np.append(indspeed, indspeed[-1])
                    maxSpeedloc = np.argmax(indspeed)
                    maxSpeed = indspeed[maxSpeedloc]
                    forcediff = np.diff(Forcefull)
                    peakminforcepos = np.argmin(forcediff[0:MaxInd])
                    peakminforce = forcediff[peakminforcepos]
                    peakmaxforcepos = np.argmax(forcediff[0:MaxInd])
                    peakmaxforce = forcediff[peakmaxforcepos]
                    # plt.plot(indspeed) plt.plot(forcediff)
                    # forcediff = smoothM(forcediff,5);
                    dF= np.mean([-peakminforce, peakmaxforce])
                    dInd = maxSpeed
                    if np.isnan(SurfZ) == 0 and HeightfromZ == 1:
                        Height = -Displ[indstart]-SurfZ
                    if Height>0:
                        [BEC, BECspeed] = bottom_effect_correction(Poisson, Radius, Height, modelprobe, indentationfull);
                    else:
                        BECspeed = np.ones(len(indentationfull))
                    nu = dF/K1/power/(indentationfull[peakminforcepos+1]**(power-1)*dInd)*3.7/BECspeed[maxSpeedloc-1]
                    par00 = [EHertz/NF,0.2,nu/NF] # E1,alpha,nu
                    parboundsnp = np.array(((EHertz*0.01/NF, 0, (nu/NF)*0.3), (EHertz*1e4/NF, 1 , nu/NF)))
                    Fixedpars = np.array([[0, 0, 0], [0, 0, nu]], dtype=float)
                    # Fixedpars = np.array([[0, 0, 1], [0, 0, nu]], dtype=float)
                    modelting = 'sPLReta'
                    modelting2 = 'sPLRetatest'

                FixedparsNF = np.copy(Fixedpars)
                FixedparsNF[1, [0, 2]] = FixedparsNF[1, [0, 2]]/NF
                fixedp = Fixedpars[0, :]
                fixedv = FixedparsNF[1, :]

                if np.isnan(SurfZ) == 0 and HeightfromZ == 1:
                    Height = -Displ[indstart]-SurfZ

                warnings.simplefilter("ignore")  # ignore warnings temporary
                # parbounds = tuple(map(tuple, parboundsnp))

                numfitpars = 3-sum(fixedp)
                # FixedparsNFtemp = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.double)
                if numfitpars > 0:
                    # x is indentationfull
                    funTing = lambda parf, x: ting_numerical(parf, adhesion_parsn, Poisson, Radius, dT, MaxInd, Height, modelting, modelprobe, x)[0]
                    Force_fitn, fitTingpar, fitTingcov = fixedfit(funTing, par00, parboundsnp, Fixedpars, indentationfull, Forcefulln)
                else:
                    Force_fitn = ting_numerical(fixedv, adhesion_parsn, Poisson, Radius, dT, MaxInd, Height, modelting, modelprobe, indentationfull)[0]
                    fitTingpar = []
                # f = plt.figure(2)
                # plt.plot(indentationfull, Forcefulln, indentationfull, Force_fitn) # plt.plot(indentationfull, Force_fitn)
                # plt.plot(t, Forcefulln, t, Force_fitn)

                parfit = np.asarray(fixedv, dtype=np.float64)
                parfit[fixedp == 0] = fitTingpar
                parfit[0] = parfit[0]*NF
                parfit[2] = parfit[2]*NF
                resid = Force_fitn - Forcefulln
                resid = resid*NF
                resnorm = np.linalg.norm(resid)
                [Force_fit, crad_fit, contact_time] = ting_numerical(parfit, adhesion_pars, Poisson, Radius, dT, MaxInd, Height, modelting, modelprobe, indentationfull)[0:3]
                RSq_Ting = r2_score(Forcefull, Force_fit)
                AdjRSq_Ting = 1-(1-RSq_Ting)*(len(Forcefull)-1)/(len(Forcefull)-numfitpars-1)  # 1-(1-RSq)*(n-1)/(n-p-1)
                Force_fit_arr[ii] = Force_fit
                # plt.plot(indentationfull, Forcefull, indentationfull, Force_fit)
                # t = np.linspace(0, dT*len(indentationfull), len(indentationfull))
                # plt.plot(t, Forcefull, t, Force_fit)

                Resultstemp[ii, 0] = kk
                Resultstemp[ii, 1] = EHertz
                Resultstemp[ii, 2] = np.real(parfit[0])
                Resultstemp[ii, 3] = np.real(parfit[1])
                Resultstemp[ii, 4] = np.real(parfit[2])
                Resultstemp[ii, 5] = resnorm
                Resultstemp[ii, 6] = np.mean(np.abs(resid))
                Resultstemp[ii, 7] = AdjRSq_Ting
                Resultstemp[ii, 8] = normhystarea
                Resultstemp[ii, 9] = pointright

                if binomsearch == 1:
                    resum_current = np.sum(resid)
                    resnorm_current = resnorm
                    endstep = np.nanargmin(Resultstemp[:, 5])  # lowest resnorm for the optimal CP
                if AdjRSq_Ting < 0.9 and ii == 1:  # step adjustment
                    pr_step = pr_step*2
                if break_condition >= 3:
                    break
                if ii > 0:
                    resnorm_prev = Resultstemp[ii-1, 5]
                    if (resnorm_prev < resnorm_current and break_condition == 0) or (resum_current < 0 and break_condition == 0):
                        pr_step = - np.floor(pr_step/2)
                        break_condition = break_condition + 1
                    elif break_condition >= 1:
                        if resum_current > 0:
                            pr_step = np.abs(np.round(pr_step/2))
                        elif resum_current < 0:
                            pr_step = -np.abs(np.round(pr_step/2))
                    break_condition = break_condition + 1

                    if ii == 1:
                        print('contact point adjustment, step (max=15): 1 ', end='')
                    else:
                        print(ii, end='')
                else:
                    endstep = np.nanargmin(Resultstemp[:, 5])  # lowest resnorm for the optimal CP
                if binomsearch == 0:
                    break

            # warnings.simplefilter("default")
            Freq = np.array([1/contact_time])
            Wpars = np.zeros([3, 1])
            Wpars[0] = Resultstemp[endstep, 2]  # E0
            Wpars[1] = Resultstemp[endstep, 3]  # Einf
            Wpars[2] = Resultstemp[endstep, 4]  # alpha/tau
            Ecomplex = timetofreqspectrumFun(Wpars, Freq*2*pi, modelting, Pars)  # TODO check function freqs to rad/s
            # Ecomplex = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            warnings.simplefilter("default")
            Results.loc[kk, 'contact_timeT'] = contact_time  # effective time (Ting) [s]
            Results.loc[kk, 'Freq'] = Freq  # effective frequency [Hz]
            Results.loc[kk, 'Estor'] = Ecomplex[0][0]  # effective storage modulus
            Results.loc[kk, 'Eloss'] = Ecomplex[0][1]  # effective loss modulus

            if Height != 0:
                pointright = Resultstemp[endstep, 9]
                indstart = int(locationcpH + pointright)
                MaxInd = ApprLength - indstart
                for ala in range(indstart+MaxInd, FullLength):  # alternative search for indend based on Z value
                    if Displ[ala] < Displ[indstart]:
                        indend = ala
                        break
                indentationfull = Displ[indstart:indend] - Displ[indstart] - Forcefull1[indstart:indend]/Stiffness
                indentationfull[indentationfull < 0] = 0
                Forcefull = Forcefull1[indstart:indend]
                # plt.plot(indentationfull, Forcefull)
                NF = (max(Forcefull)-min(Forcefull))/10  # normalization to improve the fit quality
                FixedparsNF = np.copy(Fixedpars)
                FixedparsNF[1, [0, 2]] = FixedparsNF[1, [0, 2]]/NF
                fixedp = Fixedpars[0, :]
                fixedv = FixedparsNF[1, :]

                Forcefulln = Forcefull/NF
                warnings.simplefilter("ignore")
                if numfitpars > 0:
                    funTing = lambda parf, x: ting_numerical(parf, adhesion_pars, Poisson, Radius, dT, MaxInd, 0, modelting, modelprobe, x)[0]
                    Force_fitn, fitTingpar, fitTingcov = fixedfit(funTing, par00, parboundsnp, Fixedpars, indentationfull, Forcefulln)
                    
                else:
                    Force_fitn = ting_numerical(fixedv, adhesion_pars, Poisson, Radius, dT, MaxInd, Height, modelting, modelprobe, indentationfull)[0]
                    fitTingpar = []

                warnings.simplefilter("default")
                parfit = fixedv
                parfit[fixedp == 0] = fitTingpar
                parfit[0] = parfit[0]*NF
                parfit[2] = parfit[2]*NF
                resid = Force_fitn - Forcefulln
                resid = resid*NF
                resnorm = np.linalg.norm(resid)
                Results.loc[kk, 'E0BEC'] = Resultstemp[endstep, 2]  # BEC
                Results.loc[kk, 'EinfBEC'] = Resultstemp[endstep, 4]  # BEC
                Results.loc[kk, 'alpha_tauBEC'] = Resultstemp[endstep, 3]  # BEC
                Results.loc[kk, 'E0'] = np.real(parfit[0])  # noBEC
                Results.loc[kk, 'Einf'] = np.real(parfit[2])  # noBEC
                Results.loc[kk, 'alpha_tau'] = np.real(parfit[1])  # noBEC
            else:
                Results.loc[kk, 'E0'] = Resultstemp[endstep, 2]  # BEC
                Results.loc[kk, 'Einf'] = Resultstemp[endstep, 4]  # BEC
                Results.loc[kk, 'alpha_tau'] = Resultstemp[endstep, 3]  # BEC
            pointright = Resultstemp[endstep, 9]
            indstart = int(locationcpH + pointright)
            Results.loc[kk, 'cpTing'] = Displ[indstart]  # +StartZ CurveDir*
            Results.loc[kk, 'resnorm'] = Resultstemp[endstep, 5]
            Results.loc[kk, 'S'] = Resultstemp[endstep, 6]
            Results.loc[kk, 'AdjRsq'] = Resultstemp[endstep, 7]
            FitTing = np.full(np.shape(Forcefull1), np.nan)
            FitTing[indstart:indstart+len(Force_fit_arr[endstep])] = Force_fit_arr[endstep]
            currentcurve2 = np.c_[currentcurve2, FitTing]
            # f = plt.figure(3)
            # plt.plot(currentcurve3[:,2], currentcurve3[:,3])
            # plt.plot(currentcurve3[:,2], currentcurve3[:,4])
            # plt.plot(currentcurve3[:,2], currentcurve3[:,5])
            # plt.plot(currentcurve3[:,2], currentcurve3[:,6])
        curve_data[1] = currentcurve2
    except BaseException as error:  # comment  to check errors
        Results.loc[kk, 'Name'] = fnameshort
        Results.loc[kk, 'Pixel'] = curve_data[0]
        Results.loc[kk, 'comment'] = str(error)
    if hasattr(Pars, "ROIPars"):
        if hasattr(Pars.ROIPars, "reg_nums_all"):
            Results.loc[kk, 'ROI'] = Pars.ROIPars.reg_nums_all[pix]

    return Results, curve_data, DFL_corrs


if __name__ == '__main__':
    # serach for Data in workspace
    import time
    import matplotlib.pyplot as plt
    from utils_ViscoIndent import load_AFM_data_pickle_short, curve_from_saved_pars
    # filename= 'D:/MailCloud/AFM_data/BrukerResolve/cytotoxicity/20211118_Ref52_ACR+NaOCL/control.0_000062.dat'
    filename= 'examples/Bruker_forcevolume_cells3.dat'
    # Pars, Data, Results = load_AFM_data_pickle_short(filename)
    # Pars.HeightfromZ = 0
    # Pars.height = 1000
    if 'Data' in globals():
        print('Data were found in workspace')
        global Pars, Results, Data
        # Pars.adhesion_model = "DMT"
        # Pars.adhesion_region = 'both'
        # Pars.viscomodel = 'SLS'
        # Pars.height = 1000000  # in nm
        start_time = time.perf_counter()
        kk = 0
        curve_data = Data[kk]
        curve=curve_data[1]
        # plt.plot(curve[:,0], curve[:,1])
        Results2, curve_data, DFL_corrs = tingsprocessingd1(Pars, curve_data)
        end_time = time.perf_counter()
        print(f"Processing time: {end_time - start_time:.4f} s")
        if Data.shape[1]<3:
            Data = np.hstack((Data,np.zeros((Data.shape[0],1))))
        Data[kk][2] = DFL_corrs
        # currentcurve3 = curve_from_saved_pars(Pars, Data[kk], Results.loc[kk, :])
        currentcurve3 = curve_from_saved_pars(Pars, Data[kk], Results2.loc[0, :])
        plt.plot(currentcurve3[:, 2], currentcurve3[:, 3])
        if currentcurve3.shape[1]>4:
            plt.plot(currentcurve3[:, 2], currentcurve3[:, 4])
        if currentcurve3.shape[1]>5:
            plt.plot(currentcurve3[:, 2], currentcurve3[:, 5])
        plt.figure()
        dT = curve_data[3]
        t = np.linspace(0, dT*len(currentcurve3[:, 2]), len(currentcurve3[:, 2]))
        plt.plot(t, currentcurve3[:, 3]); plt.plot(t, currentcurve3[:, 6])

