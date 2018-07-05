#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:25:19 2018

@author: mariajesusmunozlopez
"""

"""Phase Mask Functions:
    (all take the same arguments (coefficients, alpha, beta, omega, k, maskRadius, maskCenter, imgSize, x, y)
    but only use the following)
    
    CPM(alpha, x, y): parameter alpha, x and y coordinates in Fourier space
    
    GCPM(alpha, beta, x, y): parameters alpha and beta, x and y coordinates in Fourier space
    
    SCPM(alpha, beta, omega, x, y): parameters alpha, beta, and gamma, 
                x and y coordinates in Fourier space
    
    hexagon(maskRadius, maskCenter, k, imgSize): maskRadius and maskCenter as defined for the
                pupil function, k slope of the ramps, imgSize number of pixels (can be different
                in each dimension, take the largest value)
    
    square(slope, num): slope of the ramps, number of pixels (can be different
                in each dimension, take the largest value) 
    
    tetrapod(coefficients, x, y): polynomial coefficients (number of terms), set as
                coefficient = [0]*number_of_terms
                coefficient[12] = 200 --- set non-zero terms;
                x and y coordinates in Fourier space
                
"""


import numpy as np

def CPM(_coefficients, alpha, _beta, _omega, _k, _maskRadius, _maskCenter, _imgSize, x, y):
    return alpha*(x**3 + y**3)

def GCPM(_coefficients, alpha, beta, _omega, _k, _maskRadius, _maskCenter, _imgSize, x, y):
    return alpha*(x**3 + y**3) + beta*(x**2*y + y**2*x)

def SCPM(_coefficients, alpha, beta, omega, _k, _maskRadius, _maskCenter, _imgSize,  x, y):
    return alpha*(x**3 + y**3) + beta*(np.sin(omega*x) + np.sin(omega*y))

def hexagon(_coefficients, _alpha, _beta, _omega, k, maskRadius, maskCenter, imgSize, _x, _y):
    D = np.int(imgSize/6*1.5)
    W_b = np.int((maskCenter - D/3)) 
    W_f = np.int((maskCenter + D/3))
    M = np.zeros((imgSize,imgSize))

    #Center Hexagon
    alpha = 0
    beta = 0
    for i in range(W_b, W_f):
        for j in range(W_b-1, W_f):
            M[i,j] = k*(alpha*i+beta*j)
    for i in range(W_f, W_f+D/3):
        for j in range(W_b, W_b+D/3):
            if i-j <= 2*D/3:
                M[i,j] = k*(alpha*i+beta*j)
    for i in range(W_f, W_f+D/3):
        for j in range(W_b+D/3, W_b+2*D/3):
            if i+j-(imgSize-1) <= 2*D/3:
                M[i,j] = k*(alpha*i+beta*j)
    for i in range(W_b-D/3, W_b):
        for j in range(W_b+D/3, W_b+2*D/3):
            if np.abs(i-j) <= 2*D/3:
                M[i,j] = k*(alpha*i+beta*j)
    for i in range(W_b-D/3, W_b):
        for j in range(W_b, W_b+D/3):
            if np.abs(i+j-(imgSize-1)) <= 2*D/3:
                M[i,j] = k*(alpha*i+beta*j)
            
    #Center-Left
    alpha = 0
    beta = -1
    for i in range(W_b, W_f):
        for j in range(W_b-2*D/3, W_f-2*D/3):
            M[i,j] = k*(alpha*i+beta*(j-maskCenter+D/3))
    for i in range(W_f, W_f+D/3):
        for j in range(W_b-2*D/3, W_b+D/3-2*D/3):
            if i-j-2*D/3 <= 2*D/3:
                M[i,j] = k*(alpha*i+beta*(j-maskCenter+D/3))
    for i in range(W_f, W_f+D/3):
        for j in range(W_b-D/3-5, W_b):
            if i+j <= imgSize-1:
                M[i,j] = k*(alpha*i+beta*(j-maskCenter+D/3))
    for i in range(W_b-D/3, W_b):
        for j in range(W_b-D/3, W_b):
            if i-j >= 0:
                M[i,j] = k*(alpha*i+beta*(j-maskCenter+D/3))
    for i in range(W_b-np.int(D/3), W_b):
        for j in range(W_b-2*D/3, W_b-D/3):
            if np.abs(i+j-(imgSize-1))/2 <= 2*D/3:
                M[i,j] = k*(alpha*i+beta*(j-maskCenter+D/3))
    
    #Center-Right            
    alpha = 0
    beta = 1
    for i in range(W_b, W_f):
        for j in range(W_b+2*D/3, W_f+2*D/3):
            M[i,j] = k*(alpha*i+beta*(j-maskCenter-D/3))
    for i in range(W_f, W_f+D/3):
        for j in range(W_b+2*D/3-1, W_b+D/3+2*D/3):
            if i-j+2*D/3 <= 2*D/3:
                M[i,j] = k*(alpha*i+beta*(j-maskCenter-D/3))
    for i in range(W_f, W_f+D/3):
        for j in range(W_b+3*D/3-1, W_b+4*D/3-1):
            if (i+j-(imgSize-1))/2 <= 2*D/3:
                M[i,j] = k*(alpha*i+beta*(j-maskCenter-D/3))
    for i in range(W_b-D/3, W_b):
        for j in range(W_b+2*D/3, W_b+3*D/3):
            if i+j-imgSize-1 >= 0:
                M[i,j] = k*(alpha*i+beta*(j-maskCenter-D/3))
    for i in range(W_b-np.int(D/3), W_b):
        for j in range(W_b+3*D/3, W_b+4*D/3-1):
            if j-i <= 4*D/3:
                M[i,j] = k*(alpha*i+beta*(j-maskCenter-D/3))        
            
    #Upper-Left
    alpha = 1
    beta = -1
    for i in range(W_b+D, W_f+D):
        for j in range(W_b-D/3, W_f-D/3+1):
            M[i,j] = k*(alpha*(i-maskCenter-2*D/3)+beta*(j-maskCenter))
    for i in range(W_b+2*D/3, W_b+D):
       for j in range(W_b, W_b+D/3):    
            if np.abs(i-j) >= 2*D/3:
                M[i,j] = k*(alpha*(i-maskCenter-2*D/3)+beta*(j-maskCenter))
    for i in range(W_b+2*D/3, W_b+D):
        for j in range(W_b-D/3, W_b):
            if i+j-(imgSize-1) >= 0:
                M[i,j] = k*(alpha*(i-maskCenter-2*D/3)+beta*(j-maskCenter))
                
    #Upper-Right
    alpha = 1
    beta = 1
    for i in range(W_b+D, W_f+D):
        for j in range(W_b+D/3, W_f+D/3):
            M[i,j] = k*(alpha*(i-maskCenter-2*D/3)+beta*(j-maskCenter))
    for i in range(W_b+2*D/3-2, W_b+D):
       for j in range(W_b+2*D/3, W_b+3*D/3+2):    
            if i-j >= 0:
                M[i,j] = k*(alpha*(i-maskCenter-2*D/3)+beta*(j-maskCenter))
    for i in range(W_b+2*D/3, W_b+D):
        for j in range(W_b+D/3, W_b+2*D/3):
            if i+j-(imgSize-1) >= 2*D/3:
                M[i,j] = k*(alpha*(i-maskCenter-2*D/3)+beta*(j-maskCenter))
                
    #Adjust - Upper
    
    ad = np.int(maskRadius/3)
    
    for i in range(W_b-D/3,W_f+D):
        for j in range(W_b-D/3,W_b+D/3):
            M[i-ad,j] = M[i,j]
    for i in range(W_b-2*D/3,W_f+D):        
        for j in range(W_b+D/3,W_b+3*D/3):
            M[i-ad,j] = M[i,j]
                
    #Bottom-Left
    alpha = -1
    beta = -1
    for i in range(W_b-3*D/3, W_f-3*D/3):
        for j in range(W_b-D/3, W_f-D/3):
            M[i,j] = k*(alpha*(i-maskCenter+2*D/3)+beta*(j-maskCenter))
    for i in range(W_f-3*D/3, W_f-2*D/3):
        for j in range(W_b-D/3, W_f-2*D/3):
            if j-i >= 0:
                M[i,j] = k*(alpha*(i-maskCenter+2*D/3)+beta*(j-maskCenter))
    for i in range(W_f-3*D/3, W_f-2*D/3):
        for j in range(W_b-2, W_b+D/3):
            if i+j-(imgSize-1) <= -2*D/3:
                M[i,j] = k*(alpha*(i-maskCenter+2*D/3)+beta*(j-maskCenter))
    
    #Bottom-Right
    alpha = -1
    beta = 1
    for i in range(W_b-3*D/3, W_f-3*D/3):
        for j in range(W_b+D/3, W_f+D/3):
            M[i,j] = k*(alpha*(i-maskCenter+2*D/3)+beta*(j-maskCenter))
    for i in range(W_f-3*D/3, W_f-2*D/3):
        for j in range(W_b+D/3, W_f):
            if np.abs(i-j) >= 2*D/3:
                M[i,j] = k*(alpha*(i-maskCenter+2*D/3)+beta*(j-maskCenter))
    for i in range(W_f-3*D/3, W_f-2*D/3):
        for j in range(W_b+2*D/3, W_b+3*D/3):
            if np.abs(i+j)-(imgSize-1) <= 1: #<=0
                M[i,j] = k*(alpha*(i-maskCenter+2*D/3)+beta*(j-maskCenter))
                
    #Adjust - Lower
    for i in range(W_f-2*D/3,W_b-3*D/3,-1):#,W_fh-2*D/3):
        for j in range(W_b+D/3,W_b+3*D/3):
            M[i+ad,j] = M[i,j]
    for i in range(W_f-2*D/3,W_b-3*D/3,-1):
        for j in range(W_b-D/3,W_f-D/3):
            M[i+ad,j] = M[i,j]
              
    return M           
                
def square(_coefficients, _alpha, _beta, _omega, slope, _maskRadius, _maskCenter, num, _x, _y):
    alpha = 1
    beta = 1
    
    n = int(2*num/5)
    M = np.ones((n,n))
    
    m = int(n/3)
    k = np.mod(n,m)
        
    M1 = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            M1[i,j] = alpha*i+beta*j
        
    M2 = np.zeros((m, m+k))
    for i in range(m):
        for j in range(m+k):
            M2[i,j] = alpha*i
        
    M3 = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            M3[i,j] = alpha*i-beta*j
        
    M4=np.zeros((m+k, m))
    for i in range(m+k):
        for j in range(m):
            M4[i,j] = beta*j
        
    M5 = np.zeros((m+k, m+k))
  
    M[0:m, 0:m] = -M1+np.max(M1)*np.ones(np.shape(M1))
    M[0:m, m:2*m+k] = -M2+np.max(M2)*np.ones(np.shape(M2))
    M[0:m, 2*m+k:n] = -M3+np.max(M3)*np.ones(np.shape(M3))
    M[m:2*m+k, 0:m] = -M4+np.max(M4)*np.ones(np.shape(M4))
    M[m:2*m+k, m:2*m+k] = M5
    M[m:2*m+k, 2*m+k:n] = M4
    M[2*m+k:n, 0:m] = M3-np.min(M3)*np.ones(np.shape(M3))
    M[2*m+k:n, m:2*m+k] = M2
    M[2*m+k:n, 2*m+k:n] = M1

    result = np.zeros((num,num))
    c = int((num-n)/2)
    result[c:num-c,c:num-c] = M
    
    return slope*result
    

def tetrapod(coefficients, _alpha, _beta, _omega, slope, _maskRadius, _maskCenter, _num, x, y):
    Z = [0]+coefficients
    r = x**2+y**2
    Z1  =  Z[1]  * 1
    Z2  =  Z[2]  * 2*x
    Z3  =  Z[3]  * 2*y
    Z4  =  Z[4]  * (2*r-1)
    Z5  =  Z[5]  * (x**2-y**2)
    Z6  =  Z[6]  * 2*x*y
    Z7  =  Z[7]  * (-2*x+3*x*r)
    Z8  =  Z[8]  * (-2*y+3*y*r)
    Z9  =  Z[9]  * (1-6*r-6*r*r)
    Z10 =  Z[10] * (x**3-3*x*y**2)
    Z11 =  Z[11] * (3*y*x**2-y**3)
    Z12 =  Z[12] * (-3*x**2+3*y**2+4*x**2*r-4*y**2*r)
    Z13 =  Z[13] * (-6*x*y+8*x*y*r)
    Z14 =  Z[14] * (3*x-12*x*r+10*x*r**2)
    Z15 =  Z[15] * (3*y-12*y*r+10*y*r**2)
    Z16 =  Z[16] * (-1+12*r-30*r**2+20*r**3)
    Z17 =  Z[17] * (x**4-6*x**2*y**2+y**4)
    Z18 =  Z[18] * (4*x**3*y-4*x*y**3)
    Z19 =  Z[19] * (-4*x**3+12*x*y**2+5*x**3*r-15*x*y**2*r)
    Z20 =  Z[20] * (-12*x**2*y+4*y**3+15*x**2*y*r-5*y**3*r)
    Z21 =  Z[21] * (6*x**2-6*y**2-20*x**2*r+20*y**2*r+15*x**2*r**2-15*y**2*r**2)
    Z22 =  Z[22] * (12*x*y-40*x*y*r+30*x*y*r**2)
    Z23 =  Z[23] * (-4*x+30*x*r-60*x*r**2+35*x*r**3)
    Z24 =  Z[24] * (-4*y+30*y*r-60*y*r**2+35*y*r**3)
    Z25 =  Z[25] * (1-20*r+90*r**2-140*r**3+70*r**4)
    Z26 =  Z[26] * (x**5-10*x**3*y**2+5*x*y**4)
    Z27 =  Z[27] * (5*x**4*y-10*x**2*y**3+y**5)
    Z28 =  Z[28] * (-5*x**4+30*x**2*y**2-5*y**4+6*x**4*r\
                    -36*x**2*y**2*r+6*y**4*r)
    Z29 =  Z[29] * (-20*x**3*y+20*x*y**3+24*x**3*y*r-24*x*y**3*r)
    Z30 =  Z[30] * (10*x**3-30*x*y**2-30*x**3*r+90*x*y**2*r\
                    +21*x**3*r**2-63*x*y**2*r**2)
    Z31 =  Z[31] * (30*x**2*y-10*y**3-90*x**2*y*r+30*y**3*r\
                    +63*x**2*y*r**2-21*y**3*r**2)
    Z32 =  Z[32] * (-10*x**2+10*y**2+60*x**2*r-60*y**2*r-105*x**2*r**2\
                    +105*y**2*r**2+56*x**2*r**3-56*y**2*r**3)
    Z33 =  Z[33] * (-20*x*y+120*x*y*r-210*x*y*r**2+112*x*y*r**3)
    Z34 =  Z[34] * (5*x-60*x*r+210*x*r**2-280*x*r**3+126*x*r**4)
    Z35 =  Z[35] * (5*y-60*y*r+210*y*r**2-280*y*r**3+126*y*r**4)
    Z36 =  Z[36] * (-1+30*r-210*r**2+560*r**3-639*r**4+252*r**5)
    Z37 =  Z[37] * (x**6-15*x**4*y**2+15*x**2*y**4-y**6)
    Z38 =  Z[38] * (6*x**5*y-20*x**3*y**3+6*x*y**5)
    Z39 =  Z[39] * (-6*x**5+60*x**3*y**2-30*x*y**4+7*x**5*r\
                    -70*x**3*y**2*r+35*x*y**4*r)
    Z40 =  Z[40] * (-30*x**4*y+60*x**2*y**3-6*y**5+35*x**4*y*r\
                    -70*x**2*y**3*r+7*y**5*r)
    Z41 =  Z[41] * (15*x**4-90*x**2*y**2+15*y**4-42*x**4*r+252*x**2*y**2*r\
                    -42*y**4*r+28*x**4*r**2-168*x**2*y**2*r**2+28*y**4*r**2)
    Z42 =  Z[42] * (60*x**3*y-60*x*y**3-168*x**3*y*r+168*x*y**3*r\
                    +112*x**3*y*r**2-112*x*y**3*r**2)
    Z43 =  Z[43] * (-20*x**3+60*x*y**2+105*x**3*r-315*x*y**2*r-168*x**3*r**2\
                    +504*x*y**2*r**2+84*x**3*r**3-252*x*y**2*r**3)
    Z44 =  Z[44] * (-60*x**2*y+20*y**3+315*x**2*y*r-105*y**3*r-504*x**2*y*r**2\
                    +168*y**3*r**2+252*x**2*y*r**3-84*y**3*r**3)
    Z45 =  Z[45] * (15*x**2-15*y**2-140*x**2*r+140*y**2*r+420*x**2*r**2\
                    -420*y**2*r**2-504*x**2*r**3+504*y**2*r**3\
                    +210*x**2*r**4-210*y**2*r**4)
    Z46 =  Z[46] * (30*x*y-280*x*y*r+840*x*y*r**2-1008*x*y*r**3+420*x*y*r**4)
    Z47 =  Z[47] * (-6*x+105*x*r-560*x*r**2+1260*x*r**3-1260*x*r**4+462*x*r**5)
    Z48 =  Z[48] * (-6*y+105*y*r-560*y*r**2+1260*y*r**3-1260*y*r**4+462*y*r**5)
    Z49 =  Z[49] * (1-42*r+420*r**2-1680*r**3+3150*r**4-2772*r**5+924*r**6)
    ZW = Z1 + Z2 +  Z3+  Z4+  Z5+  Z6+  Z7+  Z8+  Z9+ \
            Z10+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18+ Z19+ \
            Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28+ Z29+ \
            Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37+ Z38+ Z39+ \
            Z40+ Z41+ Z42+ Z43+ Z44+ Z45+ Z46+ Z47+ Z48+ Z49
    return ZW
