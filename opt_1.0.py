#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 21:38:30 2018

@author: mariajesusmunozlopez
"""

"""Optimization and image reconstruction"""

import PM1
from scipy.fftpack import fftshift, ifftshift, fft2
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from scipy import ndimage
from scipy.integrate import simps
import boxes
import filters

"""Parameter settings for the microscope"""
imgSize =  513 # Choose an odd number to ensure rotational symmetry about the center

wavelength = 0.633 # microns
NA         = 1.2  # Numerical aperture of the objective
nImmersion = 1.515 # Refractive index of the immersion oil
pixelSize  = 0.1  # microns, typical for microscopes with 60X--100X magnifications

power      = 0.1 # Watts
Z0         = 376.73 # Ohms; impedance of free space

kMax = 2 * np.pi / pixelSize # Value of k at the maximum extent of the pupil function
kNA  = 2 * np. pi * NA / wavelength
dk   = 2 * np.pi / (imgSize * pixelSize)

"""Pupil function"""
kx = np.arange((-kMax + dk) / 2, (kMax + dk) / 2, dk)
ky = np.arange((-kMax + dk) / 2, (kMax + dk) / 2, dk)
KX, KY = np.meshgrid(kx, ky) # The coordinate system for the pupil function

maskRadius = kNA / dk # Radius of amplitude mask for defining the pupil
maskCenter = np.floor(imgSize / 2)
W, H       = np.meshgrid(np.arange(0, imgSize), np.arange(0, imgSize))
mask       = 1.0*(np.sqrt((W - maskCenter)**2 + (H- maskCenter)**2) < maskRadius)

amp   = np.ones((imgSize, imgSize)) * mask
phase = 2j * np.pi * np.ones((imgSize, imgSize))
pupil = amp * np.exp(phase)

"""Define phase mask"""
#Parameters
alphav = np.linspace(10**(-4),10**(2),10)
test = np.zeros((len(alphav),1))
mse = np.zeros((len(alphav),1))
for i, alpha in enumerate(alphav):
    beta = -3*alpha
    omega = np.pi/2
    coefficients = [0]*49
    coefficients[12] = 20  #set non-zero terms for the Zernike polynomials
    k = 0.5
    j = 1j
    #Choose phase mask and combine with pupil function (CPM, GCPM, SCMP, hexagon, square, tetrapod)
    phase_function = PM1.CPM(coefficients, alpha, beta, omega, k, maskRadius, maskCenter, imgSize, KX, KY)
    phase_mask = np.exp(j*phase_function)
    pupil_pm = np.multiply(phase_mask,pupil)

    """Defocus"""
    # Defocus from -1 micron to + 1 micron
    defocusDistance = np.linspace(-10.0, 10.0, 20) 
    defocusPSF      = np.zeros((imgSize, imgSize, defocusDistance.size))

    #Generate PSF
    #fftshift is necessary for fft2 to work correctly (puts data in the form required by fft2)
    for ctr, z in enumerate(defocusDistance):
        # Add 0j to ensure that np.sqrt knows that its argument is complex
        defocusPhaseAngle   = 1j * z * np.sqrt((2 * np.pi * nImmersion / wavelength)**2 - KX**2 - KY**2 + 0j)
        defocusKernel       = np.exp(defocusPhaseAngle)
        defocusPupil        = np.multiply(pupil_pm,defocusKernel)
        defocusPSFA         = fftshift(fft2(ifftshift(defocusPupil))) * dk**2
    
        # do power normalization in real space
        tmpPSF = np.real(defocusPSFA * np.conj(defocusPSFA))
        currentPower        = simps(simps(tmpPSF**2, dx = pixelSize), dx = pixelSize) / Z0
        normFac             = power/currentPower
        normedTmpPSF        = tmpPSF*np.sqrt(normFac)
        currentPower        = simps(simps(normedTmpPSF**2, dx = pixelSize), dx = pixelSize) / Z0
        defocusPSF[:,:,ctr] = normedTmpPSF

    """Reconstruction"""
    f = boxes.baby_example(defocusPSF.shape,(10,10,2),(70,5))
    Ff = filters.fft(f)

    #reconstruction of image
    #noise should be randomly sampled in the reconstruction (in the Fourier transform of g)
    #compute F_hat with wiener filter  

    FT_defocusPSF = filters.fft(defocusPSF)

    F_hat = filters.estimation(Ff, 0.01, FT_defocusPSF)
    f_hat = filters.ifft(F_hat)
    f_hat = np.abs(f_hat)

    """Optimization"""
    MSE = np.mean((np.abs(f_hat-f))**2)
    mse[i] = MSE
    test[i] = alpha



