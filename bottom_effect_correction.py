# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:58:21 2020

@author: Yuri
    bottom effect correction coefficients
"""
import numpy as np
from numpy import pi

def bottom_effect_correction(Poisson, Probe_dimension, Height, modelprobe, indentationfull):
    BEC=np.ones(len(indentationfull))
    indentationfull[indentationfull<0]=np.nan #to remove warnings in sqrt
    if Poisson<=0.5 and Poisson>=0.45: # coefficients are from doi.org/10.1016/j.bpj.2018.05.012
        if modelprobe== 'sphere': #sphere
            Dpar1 =1.133
            Dpar2=1.497
            Dpar3=1.469
            Dpar4=0.755
            R = Probe_dimension
            h = Height
            BEC = 1+(Dpar1*(R*indentationfull)**(0.5)/h)+(Dpar2*((R*indentationfull)**(0.5)/h)**2)\
            +(Dpar3*((R*indentationfull)**(0.5)/h)**3)+(Dpar4*((R*indentationfull)**(0.5)/h)**4)
            
            BECspeed = 1+4/3*(Dpar1*(R*indentationfull)**(0.5)/h)+5/3*(Dpar2*((R*indentationfull)**(0.5)/h)**2)\
            +2*(Dpar3*((R*indentationfull)**(0.5)/h)**3)+7/3*(Dpar4*((R*indentationfull)**(0.5)/h)**4)
    
        elif modelprobe== 'cone' or modelprobe== 'pyramid' or modelprobe== 'cylinder':
    #in the curent version of the script there is no correction for the other probe geometries
            BEC = np.ones(len(indentationfull))
            BECspeed = np.ones(len(indentationfull))
            print ('BEC is not yet avilable')
    
    else:
        if modelprobe== 'sphere':
            # coefficients are from doi.org/10.1016/S0006-3495(02)75620-8
            alpha = -(1.2876-1.4678*Poisson+1.3442*Poisson^2)/(1-Poisson) #bonded case
            bettaD = (0.6387-1.0277*Poisson+1.5164*Poisson^2)/(1-Poisson) #bonded case
            Dpar1=2*alpha/pi;
            Dpar2=4*(alpha/pi)**2;
            Dpar3=(8/pi**3)*(alpha**3+(4*(pi**2)*bettaD/15));
            Dpar4=-(16*alpha/pi**4)*(alpha**3+(3*(pi**2)*bettaD/5));
            R = Radius
            h = Height
            BEC = 1+(Dpar1*(R*indentationfull)**(0.5)/h)+(Dpar2*((R*indentationfull)**(0.5)/h)**2)\
            +(Dpar3*((R*indentationfull)^(0.5)/h)**3)+(Dpar4*((R*indentationfull)**(0.5)/h)**4)
            
            BECspeed = 1+4/3*(Dpar1*(R*indentationfull)**(0.5)/h)+5/3*(Dpar2*((R*indentationfull)**(0.5)/h)**2)\
            +2*(Dpar3*((R*indentationfull)**(0.5)/h)**3)+7/3*(Dpar4*((R*indentationfull)**(0.5)/h)**4)

        elif modelprobe== 'cone' or modelprobe== 'pyramid' or modelprobe== 'cylinder':
    #in the curent version of the script there is no correction for the ohter probe geometries
            BEC=np.ones(len(indentationfull))
            BECspeed = np.ones(len(indentationfull))
            print ('BEC is not yet avilable')
            
    return BEC, BECspeed

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Poisson = 0.5
    Radius = 1000
    Height = 1000
    modelprobe = 'sphere'
    indentationfull = np.linspace(0,200,200)
    BEC = bottom_effect_correction(Poisson, Radius, Height, modelprobe, indentationfull)[0]
    plt.plot(indentationfull, BEC)
