import cv2
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import morphology
from skimage import color
from skimage import segmentation
from skimage import feature
from skimage import filters

from  sklearn import cluster
from skimage.util import img_as_float
from skimage.util import img_as_ubyte

import regiongrowing as rg

import PIL.Image as Image
import pylab




def regionGrowing(image, seeds, pixelThreshold, regionThreshold, filename, matList):
    plt.close('all') # Close all remaining figures
    y, x = seeds.T

    labels = rg.regionGrowing(image, seeds, pixelThreshold, regionThreshold)
    plt.figure(1)
    plt.imshow(color.label2rgb(labels, image))
    plt.savefig('table.png')

    plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, image), labels))
    plt.plot(x, y, 'or', ms=3)
    plt.savefig(filename)
    matList.append(labels)





n_seeds=1
pTh= 5000
rTh = 2550
matList=[]


# img1 = Image.open("1/input3.jpg")
# img1.show()
plt.ion()

img1=Image.open("1/input4.jpg")
im_gray= img1.convert('L')
im=np.array(im_gray)
# plt.imshow(npimg1)
# print('show')
# pylab.show()


# im = io.imread("1/input3.jpg")
plt.figure(1)
plt.imshow(im,cmap='gray')
plt.savefig('gray.png')
print('Choose '+str(n_seeds)+ ' points')
markers = plt.ginput(n_seeds) # n points to choose as markers/seeds


print('Init done')

markers=np.asarray(markers) # Convert a Python list to a Numpy array
seeds=markers

print (seeds)

x_,y_ = seeds[0]
seeds[0]=[y_,x_]

print (im[0,0],":0")
print (im[28,64],":上衣")
print (im[81,57],":裤子")
print (im[60,22],":背景")
print (im[9,66],":身体")

if (im[0,0].dtype == 'uint8'):
    regionGrowing(im, seeds, 5, 5, "filename", matList)
else:
    regionGrowing(im, seeds, pTh, rTh, "filename", matList)

# img1=Image.open("1/input1.jpg")
# plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, image), labels))
# plt.plot(x, y, 'or', ms=3)
# plt.savefig(filename)



