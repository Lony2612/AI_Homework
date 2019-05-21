from keras.models import Sequential, Model
from keras.layers import *
from keras.applications import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.optimizers import RMSprop, SGD, Adam
import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.distributions import Beta
from random import shuffle
from instance_normalization import InstanceNormalization
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt 


K.set_learning_phase(1)

channel_axis=-1 
channel_first = False 
 
#nc_in = 9 # number of input channel 
#nc_out = 4 # number of output channel 
 
# ========== Model config ========== 
ngf = 64 
ndf = 64 
use_lsgan = False 
位 = 10 if use_lsgan else 100 
nc_G_inp = 9  
nc_G_out = 4  
nc_D_inp = 6  
nc_D_out = 1  
gamma_i = 0.1 
use_instancenorm = True # False: batchnorm 
use_mixup = True 
linear_upsampling = False 
 
#========== Training config ========== 
mixup_alpha = 0.1 
imageSize = 256 
batchSize = 8 
lrD = 2e-4 
lrG = 2e-4

