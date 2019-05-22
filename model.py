from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal

from instance_normalization import InstanceNormalization
from config import *

from keras.applications import *
import tensorflow as tf

from keras.optimizers import RMSprop, SGD, Adam
# from IPython.display import clear_output
import numpy as np

def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization

# Basic discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer = conv_init, *a, **k)

def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                                   gamma_initializer = gamma_init)

def instance_norm():
    return InstanceNormalization(axis=channel_axis, epsilon=1.01e-5,
                                   gamma_initializer = gamma_init)

def BASIC_D(nc_in, ndf, max_layers=3, use_sigmoid=True):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """    
    if channel_first:
        input_a =  Input(shape=(nc_in, None, None))
    else:
        input_a = Input(shape=(None, None, nc_in))
    _ = input_a
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name = 'First') (_)
    _ = LeakyReLU(alpha=0.2)(_)
    
    for layer in range(1, max_layers):        
        out_feat = ndf * min(2**layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same", 
                   use_bias=False, name = 'pyramid.{0}'.format(layer)             
                        ) (_)
        _ = batchnorm()(_, training=1)        
        _ = LeakyReLU(alpha=0.2)(_)
    
    out_feat = ndf*min(2**max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4,  use_bias=False, name = 'pyramid_last') (_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)
    
    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name = 'final'.format(out_feat, 1), 
               activation = "sigmoid" if use_sigmoid else None) (_)    
    return Model(inputs=[input_a], outputs=_)


"""
Unet or Resnet, which one is better for pix2pix model? 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/117
Reply from the author:
    UNet gives slightly better results than Resnet in some of the pix2pix applications. 
    We haven't varied the depth of the UNet model, but it might be worth trying.
"""
def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True, use_batchnorm=True):
    
    s = isize if fixed_input_size else None
    _ = inputs = Input(shape=(s, int(s*.75), nc_in))
    x_i = Lambda(lambda x: x[:, :, :, 0:3], name='x_i')(inputs)
    y_i = Lambda(lambda x: x[:, :, :, 3:6], name='y_j')(inputs)
    xi_and_y_i = concatenate([x_i, y_i], name = 'xi_yi')
    xi_yi_sz64 = AveragePooling2D(pool_size=2)(xi_and_y_i)
    xi_yi_sz32 = AveragePooling2D(pool_size=4)(xi_and_y_i)
    xi_yi_sz16 = AveragePooling2D(pool_size=8)(xi_and_y_i)
    xi_yi_sz8 = AveragePooling2D(pool_size=16)(xi_and_y_i)
    layer1 = conv2d(64, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'layer1') (_)
    layer1 = LeakyReLU(alpha=0.2)(layer1)
    layer1 = concatenate([layer1, xi_yi_sz64]) # ==========
    layer2 = conv2d(128, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'layer2') (layer1)
    if use_instancenorm:
        layer2 = instance_norm()(layer2, training=1)
    else:
        layer2 = batchnorm()(layer2, training=1)
    layer3 = LeakyReLU(alpha=0.2)(layer2)
    layer3 = concatenate([layer3, xi_yi_sz32]) # ==========
    layer3 = conv2d(256, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'layer3') (layer3)
    if use_instancenorm:
        layer3 = instance_norm()(layer3, training=1)
    else:
        layer3 = batchnorm()(layer3, training=1)
    layer4 = LeakyReLU(alpha=0.2)(layer3)
    layer4 = concatenate([layer4, xi_yi_sz16]) # ==========
    layer4 = conv2d(512, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'layer4') (layer4)
    if use_instancenorm:
        layer4 = instance_norm()(layer4, training=1)
    else:
        layer4 = batchnorm()(layer4, training=1)
    layer4 = LeakyReLU(alpha=0.2)(layer4)
    layer4 = concatenate([layer4, xi_yi_sz8]) # ==========
    
    layer9 = Conv2DTranspose(256, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init, name = 'layer9')(layer4) 
    layer9 = Cropping2D(((1,1),(1,1)))(layer9)
    if use_instancenorm:
        layer9 = instance_norm()(layer9, training=1)
    else:
        layer9 = batchnorm()(layer9, training=1)
    layer9 = Concatenate(axis=channel_axis)([layer9, layer3])
    layer9 = Activation('relu')(layer9)
    layer9 = concatenate([layer9, xi_yi_sz16]) # ==========
    layer10 = Conv2DTranspose(128, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init, name = 'layer10')(layer9) 
    layer10 = Cropping2D(((1,1),(1,1)))(layer10)
    if use_instancenorm:
        layer10 = instance_norm()(layer10, training=1)
    else:
        layer10 = batchnorm()(layer10, training=1)
    layer10 = Concatenate(axis=channel_axis)([layer10, layer2])
    layer10 = Activation('relu')(layer10)
    layer10 = concatenate([layer10, xi_yi_sz32]) # ==========
    layer11 = Conv2DTranspose(64, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init, name = 'layer11')(layer10) 
    layer11 = Cropping2D(((1,1),(1,1)))(layer11)
    if use_instancenorm:
        layer11 = instance_norm()(layer11, training=1)
    else:
        layer11 = batchnorm()(layer11, training=1)
    layer11 = Activation('relu')(layer11)
        
    layer12 = concatenate([layer11, xi_yi_sz64]) # ==========
    layer12 = Activation('relu')(layer12)
    layer12 = Conv2DTranspose(32, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init, name = 'layer12')(layer12) 
    layer12 = Cropping2D(((1,1),(1,1)))(layer12)
    if use_instancenorm:
        layer12 = instance_norm()(layer12, training=1)
    else:
        layer12 = batchnorm()(layer12, training=1)
    
    layer12 = conv2d(4, kernel_size=4, strides=1, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'out128') (layer12)
    
    alpha = Lambda(lambda x: x[:, :, :, 0:1], name='alpha')(layer12)
    x_i_j = Lambda(lambda x: x[:, :, :, 1:], name='x_i_j')(layer12)
    alpha = Activation("sigmoid", name='alpha_sigmoid')(alpha)
    x_i_j = Activation("tanh", name='x_i_j_tanh')(x_i_j)
    out = concatenate([alpha, x_i_j], name = 'out128_concat')
    
    return Model(inputs=inputs, outputs=[out])   