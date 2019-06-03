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




def regionGrowing(image, seeds, pixelThreshold, regionThreshold, filename):
    plt.close('all') # Close all remaining figures
    y, x = seeds.T

    labels = rg.regionGrowing(image, seeds, pixelThreshold, regionThreshold)
    print(labels.shape)
    plt.figure(1)
    plt.imshow(color.label2rgb(labels, image))
    plt.savefig('table.png')

    plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, image), labels))
    plt.plot(x, y, 'or', ms=3)
    plt.savefig(filename)
    return labels

def changeClothes(img1, img2, labels, savePath):
    h,w,_ = img1.shape
    for ii in range(0,h):
        for jj in range(0,w):
            if labels[ii][jj] == True:
                img1[ii][jj][:] = img2[ii][jj][:]
    cv2.imwrite(savePath,img1)
    pass



n_seeds=1
pTh= 5000
rTh = 2550

plt.ion()

# img1=Image.open("./data/1/input2.jpg")
img1=Image.open("./input2.jpg")
im_gray= img1.convert('L')
im=np.array(im_gray)

img_shirt = cv2.imread('./input1.jpg')
img_skirt = cv2.imread('./input2.jpg')


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

if (im[0,0].dtype == 'uint8'):
    labels = regionGrowing(im, seeds, 30, 30, "filename")
else:
    labels = regionGrowing(im, seeds, pTh, rTh, "filename")

changeClothes(img_shirt, img_skirt, labels, './results.jpg')



