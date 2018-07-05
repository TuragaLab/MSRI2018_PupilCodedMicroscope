#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 09:20:52 2018

@author: mariajesusmunozlopez
"""

"""PSF building"""

import PM1
from scipy.fftpack import fft2
from scipy.fftpack import fftshift, ifftshift
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from scipy import ndimage
from scipy.integrate import simps

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
alpha = 5*10**(-3)*np.pi 
beta = -3*alpha
omega = np.pi/2
coefficients = [0]*49
coefficients[12] = 20  #set non-zero terms for the Zernike polynomials
k = 0.5
j = 1j
#Choose phase mask and combine with pupil function (CPM, GCPM, SCMP, hexagon, square, tetrapod)
phase_function = PM1.hexagon(coefficients, alpha, beta, omega, k, maskRadius, maskCenter, imgSize, KX, KY)
phase_mask = np.exp(j*phase_function)
pupil_pm = np.multiply(phase_mask,pupil)

"""Defocus"""
# Defocus from -1 micron to + 1 micron
defocusDistance = np.arange(-1, 1.1, 0.5)
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

"""Plot PSF"""    
#maxIrradiance = np.max(defocusPSF[:,:,np.int(len(defocusDistance)/2)])

#Create X-Z and Y-Z projections
#X_PSF = np.sum(defocusPSF,axis=0)/imgSize
#Y_PSF = np.sum(defocusPSF,axis=1)/imgSize
#Rotated_Plot = ndimage.rotate(np.log(X_PSF), 90)

# Plot figure with subplots of different sizes
#fig = plt.figure(1)
# set up subplot grid
#gridspec.GridSpec(8,8)

# large subplot
#plt.subplot2grid((6,6), (0,0), colspan=4, rowspan=4)
#plt.gca().set_xlim((150, 350)) #Zoom in
#plt.gca().set_ylim((150, 350)) 
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
#plt.imshow((defocusPSF[:,:,np.int(len(defocusDistance)/2)]), interpolation='nearest')

# small subplot 1
#plt.subplot2grid((6,6), (0,4), colspan=2, rowspan=4)
#plt.gca().set_ylim((200, 300)) #Zoom in 
#plt.gca().set_xticks([])
#plt.gca().set_yticks([])
#plt.imshow((Y_PSF), interpolation = 'nearest')

#cb = plt.colorbar()
#cb.set_label('Irradiance, $W / \mu m^2$')

#plt.subplots_adjust(wspace=-0.999, hspace=-0.3)
#plt.show()


"""Check power"""
#print('Initial power of system:\t{:.7f} W'.format(power))
#print('Final power of system:\t\t{:.7f} W'.format(currentPower))

#Plot power at each value of depth z
#currentPowerStack = np.zeros((len(defocusDistance),1))
#for ctr in range(0,len(defocusDistance)):
#    currentPowerStack[ctr] = simps(simps(np.abs(defocusPSF[:,:,ctr])**2, dx = pixelSize), dx = pixelSize) / Z0

#plt.plot(defocusDistance,currentPowerStack)
#plt.ylabel('total power (W)')
#plt.ylim((0,1))
#plt.xlabel('z (micron)')
#plt.xlim((defocusDistance[1],defocusDistance[len(defocusDistance)-1]))
#plt.show()

"""Visualize phase mask and pupil"""
#fig, (ax0, ax1) = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (10,6))

#img = ax0.imshow(np.mod((phase_function),2*np.pi), extent = ((kx.min(), kx.max(), ky.min(), ky.max())))
#ax0.set_title('Phase mask')
#ax0.set_xlabel('kx, rad / micron')
#ax0.set_ylabel('ky, rad / micron')  

#img1 = ax1.imshow(np.real(pupil_pm * np.conj(pupil_pm)), extent = ((kx.min(), kx.max(), ky.min(), ky.max())))
#ax1.set_title('Pupil function')
#ax1.set_xlabel('kx, rad / micron')

#cb = plt.colorbar(img)
#plt.show()

"""Plot results at each depth of field z"""
#for ctr in range(0,len(defocusDistance)):
    # Show the image plane
#    img = plt.imshow(np.log(defocusPSF[:,:,ctr]),vmin = 0, vmax = np.log(maxIrradiance), interpolation='nearest')
#    cb = plt.colorbar(img)
 #   plt.gca().set_xlim((200, 400)) #Zoom in
 #   plt.gca().set_ylim((200, 400))  
#    plt.xlabel('x, pixels')
#    plt.ylabel('y, pixels')
#    cb.set_label('Irradiance, $W / \mu m^2$')
    
#    plt.show() 








