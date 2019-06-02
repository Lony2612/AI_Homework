#coding:utf-8
import cv2  
import numpy as np
from matplotlib import pyplot as plt

# 说明：区域生长算法 
# 输入：原图像、种子点判断准则、生长准则 
# 返回：生长图像 


def Region_Grow(MatIn,iGrowPoint,iGrowJudge):
    #iGrowPoint为种子点的判断条件，iGrowJudge为生长条件  
    MatGrowOld=Get_Array(np.shape(MatIn)[0],np.shape(MatIn)[1])  
    MatGrowCur=Get_Array(np.shape(MatIn)[0],np.shape(MatIn)[1])  
    MatGrowTemp=Get_Array(np.shape(MatIn)[0],np.shape(MatIn)[1]) 
    #初始化原始种子点  
    for i in range(np.shape(MatIn)[0]):  
        for j in range(np.shape(MatIn)[1]):
            it=MatIn[i][j]  
            if it<=iGrowPoint:#选取种子点，自己更改  
                MatGrowCur[i][j]=255  
     
      
    DIR=[[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]] 

    MatTemp=MatGrowOld-MatGrowCur
    iJudge=cv2.countNonZero(MatTemp)  
    if not iJudge==0: #MatGrowOld!=MatGrowCur 判断本次和上次的种子点是否一样，如果一样则终止循环  
        MatGrowTemp=MatGrowCur  
        for i in range(np.shape(MatIn)[0]):  
            for j in range(np.shape(MatIn)[1]):  
                if MatGrowCur[i][j]==255 and not(MatGrowOld[i][j]==255):  
                    for iNum in range(8): 
                        iCurPosX=i+DIR[iNum][0]  
                        iCurPosY=j+DIR[iNum][1]  
                        if iCurPosX>0 and iCurPosX<(np.shape(MatIn)[0]-1) and iCurPosY>0 and iCurPosY<(np.shape(MatIn)[1]-1):  
                            if abs(MatIn[i][j]-MatIn[iCurPosX][iCurPosY])<iGrowJudge:  
                                MatGrowCur[iCurPosX][iCurPosY]=255       
        MatGrowOld=MatGrowTemp  
    return MatGrowCur

def Get_Array(x,y):
    return np.zeros((x,y),int)


MatIn1=cv2.imread('./input2.jpg')
MatIn= cv2.cvtColor(MatIn1, cv2.COLOR_BGR2GRAY)
# plt.imshow(MatIn,'gray'),plt.title('MatOut')
print(np.shape(MatIn1)[0]) 
print(np.shape(MatIn1)[1]) 

hist_cv = cv2.calcHist([MatIn1],[0],None,[256],[0,256])   
ret,thresh1 = cv2.threshold(MatIn1,125,255,cv2.THRESH_BINARY)

kernel = np.ones((3,3),np.uint8)
t=thresh1
for x in range(0,4):
    erosion = cv2.erode(t,kernel,iterations = 1)
    t=erosion

MatOut=Get_Array(np.shape(MatIn)[0],np.shape(MatIn)[1])   
MatOut=Region_Grow(MatIn,125,10)

plt.subplot(2,2,1),plt.imshow(MatIn,'gray'),plt.title('MatIn') ,plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(thresh1,'gray'),plt.title('thresh'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,3),plt.imshow(erosion,'gray'),plt.title('erosion'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,4),plt.imshow(MatOut,'gray'),plt.title('MatOut'),plt.xticks([]),plt.yticks([])
plt.show() 
     
