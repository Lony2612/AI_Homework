#coding:utf-8
import cv2
import numpy as np

def isAcceptable(img, point, candidate, pixelThreshold, regionThreshold, labels, seeds):
    x,y = point
    a,b = candidate
    i,j = seeds
    if (np.abs(int(img[x,y])-int(img[a,b]))<pixelThreshold) and (np.abs(int(img[i,j])-int(img[a,b]))<regionThreshold) and labels[a,b] != 1:
        return True
    else:
        return False

def seed_fill(img, seeds, pthresh, rthresh):
    # 获取图像的尺寸
    h,w = img.shape
    # 定义待扩展栈
    stack_list = []
    stack_list.append(seeds)
    # 定义已扩展区域
    labels = np.zeros([h,w])
    # 搜索状态空间
    while len(stack_list)!=0:
        # 获取需要扩展的点的坐标
        cur_i = stack_list[-1][0]
        cur_j = stack_list[-1][1]
        # 标记当前点为已探索
        labels[cur_i][cur_j] = 1
        # 从栈中弹出该点
        stack_list.remove(stack_list[-1])
        # print(len(stack_list))
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

def changeClothes(img1, img2, labels, HorizontalShift, VerticalShift, savePath):
    h,w,_ = img1.shape
    for ii in range(0,h):
        for jj in range(0,w):
            if labels[ii][jj] == True:
                img1[ii+VerticalShift][jj+HorizontalShift][:] = img2[ii][jj][:]
    cv2.imwrite(savePath,img1)
    pass

if __name__ == '__main__':
    # 定义4组输入的种子点
    seed_list = [(280,221),(95,211),(280,221),(93,211)]
    # 定义是否上衣覆盖下装  是：1；否：2
    cover_list = [2, 1, 2, 1]
    # 定义输入的组数
    n_group = 4
    # 定义输入组的名称
    group_name = ["001", "002", "003", "004"]
    # 定义阈值
    thresh_list = [[40,100],[60,100],[60,100],[60,100]]
    # 定义抠图通道
    channel_list = ["b","b","b","b"]
    # 定义衣物位移
    shift_list = [[0,0],[0,0],[-20,0],[14,0]]

    for ii in range(0,n_group):
        # 构造匹配服装的路径
        img1_path = "./%s_input%d.jpg"%(group_name[ii],cover_list[ii])
        img = cv2.imread(img1_path)
        print(img1_path)
        (B, G, R) = cv2.split(img)
        # B = cv2.imread("../b.jpg",0)
        # 获取对应图片的种子
        seeds = seed_list[ii]
        print(seeds)
        # 获取对应图片的阈值
        pthresh,rthresh = thresh_list[ii]
        print(pthresh,rthresh)
        # 获取需要的位移
        HorShift,VerShift=shift_list[ii]

        # 匹配以获取标签
        labels = seed_fill(R, seeds, pthresh, rthresh)

        # 打开上衣的图片
        img_shirt = cv2.imread('./%s_input1.jpg'%group_name[ii])
        # 打开下装的图片
        img_skirt = cv2.imread('./%s_input2.jpg'%group_name[ii])

        # 图像合成
        # 如果下装覆盖上衣，则把下装加到上衣图片中
        if cover_list[ii] == 2:
            changeClothes(img_shirt, img_skirt, labels, HorShift, VerShift, "output%d.jpg"%(ii+1))
        # 如果上衣覆盖下装，则把上衣加到下装图片中
        else:
            changeClothes(img_skirt, img_shirt, labels, HorShift, VerShift, "output%d.jpg"%(ii+1))
        print("The Group %d has been sovled"%(ii+1))