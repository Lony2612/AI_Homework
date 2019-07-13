#coding:utf-8
import cv2
import numpy as np
import time
from Stack import *

# def stack_push(stack,)

def isAcceptable(img, point, candidate, pixelThreshold, regionThreshold, labels, seeds):
    x,y = point
    a,b = candidate
    i,j = seeds
    if (np.abs(int(img[x,y])-int(img[a,b]))<pixelThreshold) and \
        (np.abs(int(img[i,j])-int(img[a,b]))<regionThreshold) and \
        (a,b) not in labels:
        return True
    else:
        return False

def seed_fill(img, seeds, pthresh, rthresh):
    # 获取图像的尺寸
    h,w = img.shape
    # 定义待扩展栈
    stack = []
    stack.append(seeds)

    # 定义已扩展区域
    labels = set()

    # 搜索状态空间
    # while stack.get_length() > 0:
    while len(stack) > 0:
        # 获取需要扩展的点的坐标
        cur_i, cur_j = stack.pop()

        # 标记当前点为已探索
        labels.add((cur_i,cur_j))
        ### 四邻域扩展，可改为八邻域 ###
        # 判断边界条件，防止数组溢出
        if (cur_i == 0 or cur_i == h-1 or cur_j == 0 or cur_j == w-1):
            continue
        # 判断左边邻接点是否可扩展
        if isAcceptable(img, [cur_i,cur_j], [cur_i-1,cur_j], pthresh, rthresh, labels, seeds):
            stack.append((cur_i-1,cur_j))
            labels.add((cur_i-1, cur_j))
        # 判断下边邻接点是否可扩展
        if isAcceptable(img, [cur_i,cur_j], [cur_i,cur_j-1], pthresh, rthresh, labels, seeds):
            stack.append((cur_i,cur_j-1))
            labels.add((cur_i, cur_j-1))
        # 判断右边邻接点是否可扩展
        if isAcceptable(img, [cur_i,cur_j], [cur_i+1,cur_j], pthresh, rthresh, labels, seeds):
            stack.append((cur_i+1,cur_j))
            labels.add((cur_i+1, cur_j))
        # 判断上边邻接点是否可扩展
        if isAcceptable(img, [cur_i,cur_j], [cur_i,cur_j+1], pthresh, rthresh, labels, seeds):
            stack.append((cur_i,cur_j+1))
            labels.add((cur_i, cur_j+1))
    return labels

def changeClothes(img1, img2, labels, HorizontalShift, VerticalShift, savePath):
    for [ii,jj] in labels:
        img1[ii+VerticalShift][jj+HorizontalShift][:] = img2[ii][jj][:]
    cv2.imwrite(savePath,img1)
    pass

def get_boundry(labels):  
    boundry = 0
    for item in labels:
        if boundry < item[0]:
            boundry = item[0]
    return boundry

def get_axis(labels):
    sum = 0
    for item in labels:
        sum += item[1]
    return sum/len(labels)

if __name__ == '__main__':
    # 定义4组输入的种子点
    seed_list = [(1489,1425),(625,1421),(1861,1461),(625,1421),(1489,1425),(625,1421),(1861,1461),(625,1421)]
    seed_option = [(625,1421),(1489,1425)]
    # 定义输入的组数
    n_group = 4
    # 定义输入组的名称
    group_name = ["001", "002", "003", "004", "005", "006", "007", "008"]
    # 定义阈值
    thresh_list = [[40,100],[60,100],[60,100],[60,100],[60,100],[60,100],[60,100],[60,100]]
    # 定义衣物位移
    shift_list = [[0,0],[-8,0],[-115,0],[105,0],[0,0],[0,0],[0,0],[0,0]]

    start = time.time()
    for ii in range(0,n_group):
        # 构造匹配服装的路径
        img1_path = "./%s-input4.jpg"%(group_name[ii])
        img = cv2.imread(img1_path)
        height, width, _ = img.shape
        size = (int(width*0.1), int(height*0.1))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        (B, G, R) = cv2.split(img)
        
        # 获取预处理图片的种子
        seeds = (int(1110*0.1),int(1499*0.1))
        # 获取预处理图片的阈值
        pthresh,rthresh = [20,20]
        # 获取需要的位移
        HorShift,VerShift=shift_list[ii]

        # 匹配以获取标签
        labels = seed_fill(R, seeds, pthresh, rthresh)
        boundry = get_boundry(labels)
        if boundry > 160:
            cov = 1
        else:
            cov = 2
        
        # 构造匹配服装的路径
        img1_path = "./%s-input%d.jpg"%(group_name[ii],cov)
        img = cv2.imread(img1_path)
        (B, G, R) = cv2.split(img)

        # 获取对应图片的种子
        seeds = seed_list[ii]
        # seeds = seed_option[cov-1]
        # 获取对应图片的阈值
        pthresh,rthresh = thresh_list[ii]
        # 获取需要的位移
        HorShift,VerShift=shift_list[ii]

        # 进行区域扩展以获取衣物区域
        labels = seed_fill(R, seeds, pthresh, rthresh)
        print(get_axis(labels))
        # 打开上衣的图片
        img_shirt = cv2.imread('./%s-input1.jpg'%group_name[ii])
        # 打开下装的图片
        img_skirt = cv2.imread('./%s-input2.jpg'%group_name[ii])

        # 图像合成
        # 如果下装覆盖上衣，则把下装加到上衣图片中
        if cov == 2:
            changeClothes(img_shirt, img_skirt, labels, HorShift, VerShift, "%d-output.jpg"%(ii+1))
        # 如果上衣覆盖下装，则把上衣加到下装图片中
        else:
            changeClothes(img_skirt, img_shirt, labels, HorShift, VerShift, "%d-output.jpg"%(ii+1))
        print("The Group %d has been sovled"%(ii+1))
    end = time.time()
    print("time %d"%(end-start))