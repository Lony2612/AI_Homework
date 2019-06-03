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

def regionGrowing(image, seeds, pixelThreshold, regionThreshold, labels = None, sortDistances = False, noOrphans = False):
    """
    Inputs :
        - image : a grayscale image
        - seeds : an array of seeds in (line, column) coordinates
        - pixelThreshold : threshold between 2 pixels to determine if the tested pixel is valid or not
        - regionThreshold : threhsold between a tested pixel and the seed of the neighbour
        - labels : optionnal, give a preexisting array of labels. Must be the same size as image
        - sortDistances : sort the distance of potential candidates instead of direction, does not seem to be relevant
        - noOrphans : if some pixels are orphans (i.e. their value is 0), bind them to the closest labelled region.
    Output :
        labels : a matrix of segmented regions
    Description : 
        regionGrowing implements a region-growing like algorithm to segment an image
    
    """
    seeds=seeds.astype(int)
    toVisit=seeds.flatten()
    nIter=0
    nSeeds = len(seeds)
    nbRows, nbCols = image.shape
    i_n=0
    visited=[]
    if labels is None:
        labels = np.zeros(image.shape, dtype=np.int)
    else:
        # TODO : assert that the size of the label matrix is the same as the original image
        ()
    while toVisit != []:
        i_n+=1
        point=(toVisit[0], toVisit[1])
        x,y = int(toVisit[0]), int(toVisit[1])
        nIter+=1
        
        if i_n % 10000 ==0:
            print(i_n,":",toVisit)
            print ("value:",image[x,y])
        
        seed= labels[x,y]
        if nIter<=nSeeds:
            # Initialize the original seeds on matrix labels
            labels[x,y]=nIter
            toVisit = np.delete(toVisit,0)
            toVisit = np.delete(toVisit,0)
            toVisit = np.append(toVisit,(x,y))
        else:
            # print ("扩展节点:",x,y)
            visited.append((x,y))
            # Beginning of the treatment
            neighboursList=[(x,max(0,y-1)), (max(x-1,0),y), (x,min(nbCols-1,y+1)), (min(x+1,nbRows-1),y), (max(x-1,0),max(0,y-1)), (max(x-1,0),min(nbCols-1,y+1)), (min(x+1,nbRows-1),min(nbCols-1,y+1)), (min(x+1,nbRows-1),max(0,y-1))] # L, T, R, B, TL, TR, BR, BL
            # print ("neighbours:",neighboursList)
            distances=[]
            # Create a the list of distances in regards of neighboursList
            for candidate in neighboursList:
                a,b = candidate
                distances = np.append(distances, np.abs(int(image[x,y])-int(image[a,b])))
            if not sortDistances:
                toVisitNext=[]
                for i in range(len(neighboursList)):
                    candidate = neighboursList[i]
                    if (candidate not in visited) and isAcceptable(point, candidate, labels, image, seeds,pixelThreshold, regionThreshold) :
                        toVisitNext=np.append(toVisitNext, candidate)
                        a,b = candidate
                        labels[a,b]= seed
            else: # sortDistance = True
                toVisitNext=[]
                dataFrame=[]
                for i in range(len(neighboursList)):
                    candidate = neighboursList[i]
                    if (candidate not in visited) and isAcceptable(point, candidate, labels, image, seeds,pixelThreshold, regionThreshold):
                        dataFrame.append([distances[i],candidate])
                        a,b = candidate
                        labels[a,b]= seed
                dataFrame.sort()
                for _,candidate in dataFrame:
                    toVisitNext=np.append(toVisitNext, candidate)
            # Treatment of pixels to be visited        
            if toVisitNext is []: # i.e all neighbours are labeled
                # then remove the point being treated
                toVisit = np.delete(toVisit,0)
                toVisit = np.delete(toVisit,0) 
            else:
                # Add the candidates to be visited in the order chosen             
                toVisit = np.delete(toVisit,0)
                toVisit = np.delete(toVisit,0)
                toVisit = np.append(toVisit,toVisitNext)
                # print ("tovisit:",toVisit)

    if noOrphans:
        # To avoid point with label = 0 pop : array[:-1] 
        stack=[]
        for x in range(nbRows):
            for y in range(nbCols):
                if labels[x,y]==0: #if the point is not labelled
                    neighboursLabels=[labels[x,int(max(0,y-1))], labels[max(x-1,0),y], labels[x,min(nbCols-1,y+1)],labels[min(x+1,nbRows-1),y]]
                    seed = getLabelled(neighboursLabels)
                    if seed is None:
                        stack.append((x,y))
                    else:
                        while stack !=[]:
                            i,j = stack.pop(len(stack)-1)
                            labels[i,j]= seed
                        labels[x,y]=seed
                 
    return labels
            
def isAcceptable(point, candidate,labels, image, seeds,pixelThreshold, regionThreshold):
    """
    Inputs :
    Output : boolean
    Description : 
        isAcceptable returns if a candidate from a point is acceptable as a candidate
    
    """
    x,y = point
    a,b = candidate
    a,b,x,y= int(a),int(b),int(x),int(y)
    seed= labels[x,y] # label of tested point
    originalSeed = seeds[seed-1] # seed of the labeled point
    i,j = originalSeed
    i,j = int(i), int(j)
    if (np.abs(int(image[x,y])-int(image[a,b]))<pixelThreshold) and (np.abs(int(image[i,j])-int(image[a,b]))<regionThreshold) and labels[a,b] ==0:
        return True
    else:
        return False

def getLabelled(neighboursLabels):
    """
    Inputs :
        - neighboursLabels : a list of labels (positive integers)
    Output : value of the most common label in neighboursLabels except zero, None if zero.
    Description : 
        getLabelled returns the most common label in neighboursLabels except zero.
    """
    nbZeros = 0
    toRemove = []
    for k in range(len(neighboursLabels)):
        i = neighboursLabels[k]
        if i==0:
            nbZeros+=1
            toRemove.append(i)
    # If there are only zeros
    if nbZeros == len(neighboursLabels):
        return None
    elif nbZeros >0:
        # Remove all zeros
        for k in toRemove:
            neighboursLabels.remove(k)
        counts = np.bincount(neighboursLabels)

        return np.argmax(counts)
    else:
        counts = np.bincount(neighboursLabels)
        return neighboursLabels[np.argmax(counts)]

def labelExtractor(image):
    nbLabel = np.max(image)
    shape = image.shape
    matrixList=[]
    for i in range(nbLabel):
        matrixList.append(np.zeros(shape, dtype=np.int))
    nbRows, nbCols = shape
    for i in range(nbRows):
        for j in range(nbCols):
            label=image[i,j]
            mat = matrixList[label-1]
            mat[i,j]=label
    return matrixList
