#coding:utf-8
import cv2
import numpy as np
import time

def isAcceptable(img, point, candidate, pixelThreshold, regionThreshold, labels, seeds):
    x,y = point
    a,b = candidate
    i,j = seeds
    if (np.abs(int(img[x,y])-int(img[a,b]))<pixelThreshold) and \
        (np.abs(int(img[i,j])-int(img[a,b]))<regionThreshold) and \
        labels[a,b] != 1:
        return True
    else:
        return False

def seed_fill(img, seeds, pthresh, rthresh):
    # 获取图像的尺寸
    h,w = img.shape
    # 定义待扩展栈
    stack_list = []
    head_list = []
    # stack_list.append(seeds)
    head_list.append(seeds)
    # 定义已扩展区域
    labels = np.zeros([h,w])
    # 定义扩展次数
    count = 0
    # 定义头部列表允许最大长度
    max_len_head = 375




    deg = 0

    # 搜索状态空间
    len_head = len(head_list)
    len_stack = len(stack_list)
    while len_stack + len_head != 0:
        if count % 10000 == 0:
            print("expand %8d, stack %8d head %8d"%(count, len_stack, len_head))
        # 扩展次数能整除头部列表最大长度时，将头部列表的一部分加到总体列表stack_list中
        if count % max_len_head == 0:
            # 获取当前头部列表长度
            len_head = len(head_list)
            # 当头部列表当前长度大于最大允许长度时，将head_list的一部分加到stack_list中去
            if len_head > max_len_head:
                stack_list += head_list[0:max_len_head]
                head_list = head_list[max_len_head:len_head]

        # 当head_list长度减少到0且stack_list的长度大于head_list的最大允许长度时，从stack_list取一部分给head_list
        len_head = len(head_list)
        len_stack = len(stack_list)
        if len_head == 0 and len_stack>=max_len_head:
            head_list = stack_list[len_stack-max_len_head:len_stack]
            stack_list = stack_list[0:len_stack-max_len_head]
        # 当head_list长度减少到0且stack_list的长度小于head_list的最大允许长度时，从stack_list取所有数据给head_list并将stack_list清0
        elif len_head == 0 and len_stack < max_len_head:
            head_list = stack_list
            stack_list = []
        

        # 获取需要扩展的点的坐标
        cur_i, cur_j = head_list[-1]
        # 标记当前点为已探索
        labels[cur_i][cur_j] = 1
        # 从栈中弹出该点
        head_list.remove(head_list[-1])
        count += 1
        ### 四邻域扩展，可改为八邻域 ###

        # 判断边界条件，防止数组溢出
        if (cur_i == 0 or cur_i == h-1 or cur_j == 0 or cur_j == w-1):
            continue
        # 判断左边邻接点是否可扩展
        if isAcceptable(img, [cur_i,cur_j], [cur_i-1,cur_j], pthresh, rthresh, labels, seeds):
            head_list.append((cur_i-1,cur_j))
            labels[cur_i-1][cur_j] = 1
            deg+=1
        # 判断下边邻接点是否可扩展
        if isAcceptable(img, [cur_i,cur_j], [cur_i,cur_j-1], pthresh, rthresh, labels, seeds):
            head_list.append((cur_i,cur_j-1))
            labels[cur_i][cur_j-1] = 1
            deg+=1
        # 判断右边邻接点是否可扩展
        if isAcceptable(img, [cur_i,cur_j], [cur_i+1,cur_j], pthresh, rthresh, labels, seeds):
            head_list.append((cur_i+1,cur_j))
            labels[cur_i+1][cur_j] = 1
            deg+=1
        # 判断上边邻接点是否可扩展
        if isAcceptable(img, [cur_i,cur_j], [cur_i,cur_j+1], pthresh, rthresh, labels, seeds):
            head_list.append((cur_i,cur_j+1))
            labels[cur_i][cur_j+1] = 1
            deg+=1
    # 当待探索栈为空，返回区域标记
        len_head = len(head_list)
        len_stack = len(stack_list)
    print("total %d"%deg)
    return labels

def changeClothes(img1, img2, labels, HorizontalShift, VerticalShift, savePath):
    h,w,_ = img1.shape
    for ii in range(0,h):
        for jj in range(0,w):
            if labels[ii][jj] == True:
                img1[ii+VerticalShift][jj+HorizontalShift][:] = img2[ii][jj][:]
    cv2.imwrite(savePath,img1)
    pass

def get_boundry(labels):
    h,w = labels.shape
    # for ii in range(0,h):
    #     for jj in range(0,w):
    #         if labels[ii][jj] == True:
    #             img1[ii][jj][:] = 0
    # cv2.imwrite(savePath,img1)
    
    boundry = 0
    for ii in range(w):
        for jj in range(h):
            if labels[jj][ii] == True:
                if boundry < jj:
                    boundry = jj
    return boundry

if __name__ == '__main__':
    # 定义4组输入的种子点
    seed_list = [(1489,1425),(625,1421),(1861,1461),(625,1421),(1489,1425),(625,1421),(1861,1461),(625,1421)]
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
        print("boundry %d, cover %d"%(boundry,cov))

        # 构造匹配服装的路径
        img1_path = "./%s-input%d.jpg"%(group_name[ii],cov)
        img = cv2.imread(img1_path)
        print(img1_path)
        (B, G, R) = cv2.split(img)

        # 获取对应图片的种子
        seeds = seed_list[ii]
        print(seeds)
        # 获取对应图片的阈值
        pthresh,rthresh = thresh_list[ii]
        print(pthresh,rthresh)
        # 获取需要的位移
        HorShift,VerShift=shift_list[ii]

        # 进行区域扩展以获取衣物区域
        labels = seed_fill(R, seeds, pthresh, rthresh)

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