# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:30:03 2019

@author: Babar kamal
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Create a black image
img=np.zeros((640,480))
# ... and make a white rectangle in it
img[100:-100,80:-80]=1

# See how it looks
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

# Rotate it for extra fun
img=ndimage.rotate(img,25,mode='constant')
# Have another look
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

# Get x-gradient in "sx"
sx = ndimage.sobel(img,axis=0,mode='constant')
# Get y-gradient in "sy"
sy = ndimage.sobel(img,axis=1,mode='constant')
# Get square root of sum of squares
sobel=np.hypot(sx,sy)

# Hopefully see some edges
plt.imshow(sobel,cmap=plt.cm.gray)
plt.show()

# Create a black image
img=np.zeros((640,480))
# ... and make a white rectangle in it
img[100:-100,80:-80]=1

# Define kernel for x differences
kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
# Define kernel for y differences
ky = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])
# Perform x convolution
x=ndimage.convolve(img,kx)
# Perform y convolution
y=ndimage.convolve(img,ky)
sobel=np.hypot(x,y)
plt.imshow(sobel,cmap=plt.cm.gray)
plt.show()