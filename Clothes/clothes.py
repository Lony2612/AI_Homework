import cv2
import numpy as np
import matplotlib.pyplot as plt

import regiongrowing as rg




def regionGrowing(image, seeds, pixelThreshold, regionThreshold, filename):
    plt.close('all') # Close all remaining figures
    labels = rg.regionGrowing(image, seeds, pixelThreshold, regionThreshold)
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
seed_list = [(1489,1425),(625,1421),(1861,1461),(625,1421),(1489,1425),(625,1421),(1861,1461),(625,1421)]
# 定义是否上衣覆盖下装  是：1；否：2
cover_list = [2, 1, 2, 1]
# 定义输入的组数
n_group = 4
# 定义输入组的名称
group_name = ["001", "002", "003", "004"]
# 定义阈值
thresh_list = [[60,100],[60,100],[60,100],[60,100]]

n_seeds=1

plt.ion()
for ii in range(n_group):
    # 构造匹配服装的路径
    img1_path = "./%s-input%d.JPG"%(group_name[ii],cover_list[ii])
    img = cv2.imread(img1_path)
    print(img)
    print(img1_path)
    (B, G, R) = cv2.split(img)

    # 打开上衣的图片

    # 获取种子点
    x_,y_ = seed_list[ii]
    seeds=(x_,y_)

    # 获取阈值
    pThresh,rThresh = thresh_list[ii]

    labels = regionGrowing(R, seeds, pThresh, rThresh, "filename")
    

    # 打开上衣的图片
    img_shirt = cv2.imread('./%s-input1.jpg'%group_name[ii])
    # 打开下装的图片
    img_skirt = cv2.imread('./%s-input2.jpg'%group_name[ii])

    # 图像合成
    # 如果下装覆盖上衣，则把下装加到上衣图片中
    if cover_list[ii] == 2:
        changeClothes(img_shirt, img_skirt, labels, "00%d-output.jpg"%(ii+1))
    # 如果上衣覆盖下装，则把上衣加到下装图片中
    else:
        changeClothes(img_skirt, img_shirt, labels, "00%d-output.jpg"%(ii+1))



