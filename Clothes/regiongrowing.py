#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:19:31 2017

@author: viper

Description : Test of a region growing algorithm
"""

#matplotlib.use('pgf') # Force Matplotlib back-end

# Modules....
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import segmentation
from skimage import color
from skimage import filters
from skimage import img_as_uint

def isAcceptable(img, point, candidate, pixelThreshold, regionThreshold, labels, seeds):
    x,y = point
    a,b = candidate
    i,j = seeds
    if (np.abs(int(img[x,y])-int(img[a,b]))<pixelThreshold) and (np.abs(int(img[i,j])-int(img[a,b]))<regionThreshold) and labels[a,b] != 1:
        return True
    else:
        return False

def regionGrowing(img, seeds, pthresh, rthresh):
    """
    Inputs :
        - image : a grayscale image
        - seeds : an array of seeds in (line, column) coordinates
        - pixelThreshold : threshold between 2 pixels to determine if the tested pixel is valid or not
        - regionThreshold : threhsold between a tested pixel and the seed of the neighbour
    Output :
        labels : a matrix of segmented regions
    Description : 
        regionGrowing implements a region-growing like algorithm to segment an image
    
    """
    # 获取图像的尺寸
    h,w = img.shape
    # 定义待扩展栈
    stack_list = []
    stack_list.append(seeds)
    # 定义已扩展区域
    labels = np.zeros([h,w])
    count = 0
    # 搜索状态空间
    while len(stack_list)>0:
        if count % 10000 == 0:
            print("%d"%len(stack_list))
        count += 1
        # 获取需要扩展的点的坐标
        cur_i,cur_j = stack_list.pop()

        # 标记当前点为已探索
        labels[cur_i][cur_j] = 1

        ### 四邻域扩展，可改为八邻域 ###
        # 判断边界条件，防止数组溢出
        if (cur_i == 0 or cur_i == h-1 or cur_j == 0 or cur_j == w-1):
            continue
        
        if isAcceptable(img, [cur_i,cur_j], [cur_i-1,cur_j], pthresh, rthresh, labels, seeds):
            stack_list.append((cur_i-1,cur_j))
            labels[cur_i-1][cur_j] = 1
        if isAcceptable(img, [cur_i,cur_j], [cur_i,cur_j-1], pthresh, rthresh, labels, seeds):
            stack_list.append((cur_i,cur_j-1))
            labels[cur_i][cur_j-1] = 1
        if isAcceptable(img, [cur_i,cur_j], [cur_i+1,cur_j], pthresh, rthresh, labels, seeds):
            stack_list.append((cur_i+1,cur_j))
            labels[cur_i+1][cur_j] = 1
        if isAcceptable(img, [cur_i,cur_j], [cur_i,cur_j+1], pthresh, rthresh, labels, seeds):
            stack_list.append((cur_i,cur_j+1))
            labels[cur_i][cur_j+1] = 1
    # 当待探索栈为空，返回区域标记
    return labels