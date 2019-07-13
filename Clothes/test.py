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
    plt.savefig('../table.png')

    plt.imshow(segmentation.mark_boundaries(color.label2rgb(labels, image), labels))
    plt.plot(x, y, 'or', ms=3)
    plt.savefig("../"+filename)
    return labels

def changeClothes(img1, img2, labels, savePath):
    h,w,_ = img1.shape
    for ii in range(0,h):
        for jj in range(0,w):
            if labels[ii][jj] == True:
                img1[ii][jj][:] = img2[ii][jj][:]
    cv2.imwrite(savePath,img1)
    pass

# 定义4组输入的种子点
seed_list = [[438,449],[436,227],[214,285],[439,308]]
# 定义是否上衣覆盖下装  是：1；否：2
cover_list = [2, 1, 2, 1]
# 定义输入的组数
n_group = 4
# 定义输入组的名称
group_name = ["001", "002", "003", "004"]
# 定义阈值
thresh_list = [[30,30],[90,90],[35,35],[35,35]]
# 定义抠图通道
channel_list = ["b","b","b","b"]

n_seeds=1
pTh= 5000
rTh = 2550

plt.ion()

ii = 3
for ii in range(0,n_group):
    # 构造匹配服装的路径
    img1_path = "./%s_input4.jpg"%(group_name[ii])

    # 打开匹配服装文件
    # img1=Image.open(img1_path)
    temp = cv2.imread(img1_path)
    print(temp.shape)
    hlsImg = cv2.cvtColor(temp, cv2.COLOR_BGR2HLS)
    (h,l,s)=cv2.split(hlsImg)
    # (B, G, R) = cv2.split(temp)
    # print(B.shape,B.shape)
    cv2.imwrite("../hsl/%s_4h.jpg"%group_name[ii],h)
    cv2.imwrite("../hsl/%s_4l.jpg"%group_name[ii],l)
    cv2.imwrite("../hsl/%s_4s.jpg"%group_name[ii],s)

    # im_gray= np.array(img1.convert('L'))
    # cv2.imwrite("../hsl/%s_3h.jpg"%group_name[ii],im_gray)
# im=np.array(im_gray)
# # im=np.array(B)
# # 打开上衣的图片
# img_shirt = cv2.imread('./%s_input1.jpg'%group_name[ii])
# # 打开下装的图片
# img_skirt = cv2.imread('./%s_input2.jpg'%group_name[ii])


# # # im = io.imread("1/input3.jpg")
# plt.figure(1)
# plt.imshow(im,cmap='gray')
# plt.savefig('gray.png')
# print('Choose '+str(n_seeds)+ ' points')
# markers = plt.ginput(n_seeds) # n points to choose as markers/seeds


# print('Init done')

# markers=np.asarray(markers) # Convert a Python list to a Numpy array
# seeds=markers
# print (seeds)

# # 获取种子点
# x_,y_ = seeds[0]
# seeds[0]=[y_,x_]
# print (seeds)

# # 获取阈值
# pThresh,rThresh = thresh_list[ii]

# # 分割图像
# if (im[0,0].dtype == 'uint8'):
#     labels = regionGrowing(im, seeds, pThresh, rThresh, "filename")
# else:
#     labels = regionGrowing(im, seeds, pTh, rTh, "filename")

# # 图像合成
# # 如果下装覆盖上衣，则把下装加到上衣图片中
# if cover_list[ii] == 2:
#     changeClothes(img_shirt, img_skirt, labels, "output%d.jpg"%(ii+1))
# # 如果上衣覆盖下装，则把上衣加到下装图片中
# else:
#     changeClothes(img_skirt, img_shirt, labels, "output%d.jpg"%(ii+1))




