from random import randint, shuffle
import numpy as np
from PIL import Image
import cv2 as cv2
import glob

from config import *

def minibatch(data, batchsize, filenames_5):
    length = len(data)
    epoch = i = 0
    tmpsize = None    
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            shuffle(data)
            i = 0
            epoch+=1        
        rtn = [read_image(data[j], filenames_5) for j in range(i,i+size)]
        i+=size
        tmpsize = yield epoch, np.float32(rtn)

def minibatchAB(dataA, batchsize, filenames_5):
    batchA=minibatch(dataA, batchsize, filenames_5)
    tmpsize = None    
    while True:        
        ep1, A = batchA.send(tmpsize)
        tmpsize = yield ep1, A

def read_image(fn, filenames_5):
    input_size = (111,148)
    cropped_size = (96,128)
    
    if isRGB:
    # Load human picture
        im = Image.open(fn).convert('RGB')
        im = im.resize( input_size, Image.BILINEAR )    
    else:
        im = cv2.imread(fn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        im = cv2.resize(im, input_size, interpolation=cv2.INTER_CUBIC)
    if apply_da is True:
        im = crop_img(im, input_size, cropped_size)
    arr = np.array(im)/255*2-1
    img_x_i = arr
    if channel_first:        
        img_x_i = np.moveaxis(img_x_i, 2, 0)
        
    # Load article picture y_i
    fn_y_i = fn[:-5] + "5.jpg"
    fn_y_i = fn_y_i[:fn_y_i.rfind("/")-1] + "5/" + fn_y_i.split("/")[-1]
    if isRGB:
        im = Image.open(fn_y_i).convert('RGB')
        im = im.resize(cropped_size, Image.BILINEAR )    
    else:
        im = cv2.imread(fn_y_i)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        im = cv2.resize(im, cropped_size, interpolation=cv2.INTER_CUBIC)
    arr = np.array(im)/255*2-1
    img_y_i = arr
    if channel_first:        
        img_y_i = np.moveaxis(img_y_i, 2, 0)
    
    # Load article picture y_j randomly
    fn_y_j = np.random.choice(filenames_5)
    while (fn_y_j == fn_y_i):
        fn_y_j = np.random.choice(filenames_5)
    if isRGB:
        im = Image.open(fn_y_j).convert('RGB')
        im = im.resize( cropped_size, Image.BILINEAR )
    else:
        im = cv2.imread(fn_y_j)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        im = cv2.resize(im, cropped_size, interpolation=cv2.INTER_CUBIC)
    arr = np.array(im)/255*2-1
    img_y_j = arr
    if randint(0,1): 
        img_y_j=img_y_j[:,::-1]
    if channel_first:        
        img_y_j = np.moveaxis(img_y_j, 2, 0)        
    
    if randint(0,1): # prevent disalign of the graphic on t-shirts and human when fplipping
        img_x_i=img_x_i[:,::-1]
        img_y_i=img_y_i[:,::-1]
    
    img = np.concatenate([img_x_i, img_y_i, img_y_j], axis=-1)    
    assert img.shape[-1] == 9
    
    return img

def load_data(file_pattern):
    return glob.glob(file_pattern)

def crop_img(img, large_size, small_size):
    # only apply DA to human images
    img_width = small_size[0]
    img_height = small_size[1]
    diff_size = (large_size[0]-small_size[0], large_size[1]-small_size[1])
    
    x_range = [i for i in range(diff_size[0])]
    y_range = [j for j in range(diff_size[1])]
    x0 = np.random.choice(x_range)
    y0 = np.random.choice(y_range)
    
    img = np.array(img)
    
    img = img[y0: y0+img_height, x0: x0+img_width, :]
    
    return img

def minibatch_demo(data, batchsize, fn_y_i=None):
    length = len(data)
    epoch = i = 0
    tmpsize = None
    shuffle(data)
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            shuffle(data)
            i = 0
            epoch+=1    
        rtn = [read_image(data[j], fn_y_i) for j in range(i,i+size)]
        i+=size
        tmpsize = yield epoch, np.float32(rtn)       

def minibatchAB_demo(dataA, batchsize, fn_y_i=None):
    batchA=minibatch_demo(dataA, batchsize, fn_y_i=fn_y_i)
    tmpsize = None    
    while True:        
        ep1, A = batchA.send(tmpsize)
        tmpsize = yield ep1, A

# from IPython.display import display
# def showX(X, rows=1):
#     assert X.shape[0]%rows == 0
#     int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')
    
#     if channel_first:
#         int_X = np.moveaxis(int_X.reshape(-1,3,128,96), 1, 3)
#     else:
#         if X.shape[-1] == 9:
#             img_x_i = int_X[:,:,:,:3]
#             img_y_i = int_X[:,:,:,3:6]
#             img_y_j = int_X[:,:,:,6:9]
#             int_X = np.concatenate([img_x_i, img_y_i, img_y_j], axis=1)
#         else:
#             int_X = int_X.reshape(-1,128,96, 3)
#     int_X = int_X.reshape(rows, -1, 128, 96,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
#     if not isRGB:
#         int_X = cv2.cvtColor(int_X, cv2.COLOR_LAB2RGB)
#     display(Image.fromarray(int_X))


# def showG(cycleA_generate, A):
#     def G(fn_generate, X):
#         r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
#         return r.swapaxes(0,1)[:,:,0]        
#     rA = G(cycleA_generate, A)
#     arr = np.concatenate([A[:,:,:,:3], A[:,:,:,3:6], A[:,:,:,6:9], rA[0], rA[1]])
#     showX(arr, 5) 