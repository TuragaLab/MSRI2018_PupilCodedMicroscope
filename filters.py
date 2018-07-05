#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 15:44:19 2018

@author: mariajesusmunozlopez
"""

import numpy as np

def fft(array):
    fft = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(array)))
    return(fft)

def ifft(array):
    ifft = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(array)))
    return(ifft)

def image_real(FT_sample,sigma, psf):
    FT_PSF=fft(psf)
    FT_g=np.multiply(FT_sample,FT_PSF)+ sigma**2 
    g=ifft(FT_g)
    g=np.absolute(g)
    return g[:,:,int(g.shape[2]/2)]

def wiener(FT_sample, sigma, FT_psf):
    Ff = FT_sample
    K = np.mean(np.multiply(Ff,np.conj(Ff)))
    wiener_top = K*np.conj(FT_psf) 
    wiener_bottom = (sigma**2)*np.ones(FT_psf.shape) + K*np.multiply(FT_psf,np.conj(FT_psf))
    wiener = np.divide(wiener_top,wiener_bottom)
    return wiener

def estimation(FT_sample,sigma, FT_psf):
    n = np.random.normal(0, sigma**2, FT_psf.shape) #measurement noise
    FT_g = np.multiply(FT_sample, FT_psf)+fft(n)
    W = wiener(FT_sample, sigma, FT_psf)
    F_hat = np.multiply(W,FT_g)
    return F_hat