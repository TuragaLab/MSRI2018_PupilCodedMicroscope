#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:40:35 2018

@author: mariajesusmunozlopez
"""

import numpy as np
#import matplotlib.pyplot as plt

def baby_example(dim_data,dim_box,spacing):
    #x,y,z : dimension of big box
    #a,b,c : dimenstion of little boxes and each should be even
    #d1: xy spacing
    #d2: xz spacing
    
    x= dim_data[0]
    y= dim_data[1]
    z= dim_data[2]

    a=dim_box[0]
    b=dim_box[1]
    c=dim_box[2]
    
    d1=spacing[0]
    d2=spacing[1]
    
    B=np.zeros((x,y,z))
    lb=np.ones((a,b,c))
    
    #center coord of big box
    cx= int(x/2)
    cy= int(y/2)
    cz= int(z/2)
    
     #bottom
    bx= cx-d1
    by= cy-d1
    bz= cz-d2
    
    #top
    ux= cx+d1
    uy= cy+d1
    uz= cz+d2
    
    
    if bx+int(a/2)<=cx-int(a/2) and cx+int(a/2)<=ux-int(a/2)\
       and by+int(b/2)<=cy-int(b/2) and cy+int(b/2)<=uy-int(b/2)\
        and bz+int(c/2)<=cz-int(c/2) and cz+int(c/2)<=uz-int(c/2):
        B[cx-int(a/2):cx+int(a/2) , cy-int(b/2): cy+int(b/2) , cz-int(c/2):cz+int(c/2)]=lb
    
    
        B[bx-int(a/2):bx+int(a/2),by-int(b/2):by+int(b/2),bz-int(c/2):bz+int(c/2)]=lb
        
    
        B[ux-int(a/2):ux+int(a/2),uy-int(b/2):uy+int(b/2),uz-int(c/2):uz+int(c/2)]=lb
    
    else: 
        print("OOPS, mini boxes too big!")
        
    return B


"""Visualize example"""
#Box= baby_example((300,300,20),(10,10,2),(50,2))

#side views of boxes- not quite sure about the direction lol
#plt.imshow(np.sum(Box, axis=0))
#plt.xlim((200,600))
#plt.ylim((200,600))
#plt.show()

#plt.imshow(np.sum(Box, axis=1))
#plt.xlim((200,600))
#plt.ylim((200,600))
#plt.show()
#plt.imshow(np.sum(Box, axis=2))
#plt.xlim((200,600))
#plt.ylim((200,600))
#plt.show()  