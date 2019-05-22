
from keras.layers import *

from utils import *
from config import *
from model import *

import tensorflow as tf
import keras.backend as K
from keras.optimizers import RMSprop, SGD, Adam
import time

import numpy as np
from random import randint


# ## cycle_variables
def cycle_variables(netG1):
    """
    Intermidiate params:
        x_i: human w/ cloth i, shape=(128,96,3)
        y_i: stand alone cloth i, shape=(128,96,3)
        y_j: stand alone cloth j, shape=(128,96,3)
        alpha: mask for x_i_j, shape=(128,96,1)
        x_i_j: generated fake human swapping cloth i to j, shape=(128,96,3)
    
    Out:
        real_input: concat[x_i, y_i, y_j], shape=(128,96,9)
        fake_output: masked_x_i_j = alpha*x_i_j + (1-alpha)*x_i, shape=(128,96,3)
        rec_input: output of the second generator (generating image similar to x_i), shape=(128,96,3)
        fn_generate: a path from input to G_out and cyclic G_out
    """
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]

    # Legacy: how to split channels
    # https://github.com/fchollet/keras/issues/5474
    x_i = Lambda(lambda x: x[:,:,:, 0:3])(real_input)
    y_i = Lambda(lambda x: x[:,:,:, 3:6])(real_input)
    y_j = Lambda(lambda x: x[:,:,:, 6:])(real_input)
    alpha = Lambda(lambda x: x[:,:,:, 0:1])(fake_output)
    x_i_j = Lambda(lambda x: x[:,:,:, 1:])(fake_output)
    
    fake_output = alpha*x_i_j + (1-alpha)*x_i 
    concat_input_G2 = concatenate([fake_output, y_j, y_i], axis=-1) # swap y_i and y_j
    rec_input = netG1([concat_input_G2])
    rec_alpha = Lambda(lambda x: x[:,:,:, 0:1])(rec_input)
    rec_x_i_j = Lambda(lambda x: x[:,:,:, 1:])(rec_input)
    rec_input = rec_alpha*rec_x_i_j + (1-rec_alpha)*fake_output
    fn_generate = K.function([real_input], [fake_output, rec_input])
    return real_input, fake_output, rec_input, fn_generate, alpha

### Configuration
K.set_learning_phase(1)

# ## Define models
netGA = UNET_G(imageSize, nc_G_inp, nc_G_out, ngf)
#netGA.summary()

netDA = BASIC_D(nc_D_inp, ndf, use_sigmoid = not use_lsgan)
#netDA.summary()

real_A, fake_B, rec_A, cycleA_generate, alpha_A = cycle_variables(netGA)


# # Loss Function
if use_lsgan:
    loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
else:
    loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))


def D_loss(netD, real, fake, rec):
    #x_i, y_i, y_j = tf.split(real, [3, 3, 3], 3)    
    x_i = Lambda(lambda x: x[:,:,:, 0:3])(real)
    y_i = Lambda(lambda x: x[:,:,:, 3:6])(real)
    y_j = Lambda(lambda x: x[:,:,:, 6:])(real)
    x_i_j = fake  
    
    output_real = netD(concatenate([x_i, y_i])) # positive sample
    output_fake = netD(concatenate([x_i_j, y_j])) # negative sample
    output_fake2 = netD(concatenate([x_i, y_j])) # negative sample 2
    
    loss_D_real = loss_fn(output_real, K.ones_like(output_real))    
    loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
    loss_D_fake2 = loss_fn(output_fake2, K.zeros_like(output_fake2)) # New loss term for discriminator    
    if not use_nsgan:
        loss_G = loss_fn(output_fake, K.ones_like(output_fake))
    else:
        loss_G = K.mean(K.log(output_fake))
    
    loss_D = loss_D_real+(loss_D_fake+loss_D_fake2)
    loss_cyc = K.mean(K.abs(rec-x_i)) # cycle loss
    return loss_D, loss_G, loss_cyc

loss_DA, loss_GA, loss_cycA = D_loss(netDA, real_A, fake_B, rec_A)
loss_cyc = loss_cycA
loss_id = K.mean(K.abs(alpha_A)) # loss of alpha

loss_G = loss_GA + 1*(1*loss_cyc + gamma_i*loss_id)
loss_D = loss_DA*2

weightsD = netDA.trainable_weights
weightsG = netGA.trainable_weights

training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsD,[],loss_D)
netD_train = K.function([real_A],[loss_DA/2], training_updates)
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsG,[], loss_G)
netG_train = K.function([real_A], [loss_GA, loss_cyc], training_updates)



# # Load Image
# 
# Filenames:
# 
#     "./imgs/1/fileID_1.jpg" for human images.
#     "./imgs/5/fileID_5.jpg" for article images.

# Get filenames

data = "imgs"
train_A = load_data('./{}/1/*.jpg'.format(data))

filenames_1 = load_data('./{}/1/*.jpg'.format(data))
filenames_5 = load_data('./{}/5/*.jpg'.format(data))

assert len(train_A)


# ## Other utilities
# # Training
# 
# Show results every 50 iterations.
t0 = time.time()
niter = 150
gen_iterations = 0
epoch = 0
errCyc_sum = errGA_sum = errDA_sum = errC_sum = 0

display_iters = 50
train_batch = minibatchAB(train_A, batchSize, filenames_5)

#while epoch < niter: 
while gen_iterations < 5000:
    epoch, A = next(train_batch)   
    errDA  = netD_train([A])
    errDA_sum +=errDA[0]

    # epoch, trainA, trainB = next(train_batch)
    errGA, errCyc = netG_train([A])
    errGA_sum += errGA
    errCyc_sum += errCyc
    gen_iterations+=1
    if gen_iterations%display_iters==0:
        # if gen_iterations%(10*display_iters)==0: # clear_output every 500 iters
        #     clear_output()
        print('[%d/%d][%d] Loss_D: %f Loss_G: %f loss_cyc: %f'
        % (epoch, niter, gen_iterations, errDA_sum/display_iters,
           errGA_sum/display_iters, errCyc_sum/display_iters), time.time()-t0)        
        _, A = train_batch.send(4)
        showG(cycleA_generate, A)        
        errCyc_sum = errGA_sum = errDA_sum = errC_sum = 0

# # Demo
# 
# Show 8 results on the same target article.
len_fn = len(filenames_5)
assert len_fn > 0
idx = np.random.randint(len_fn)
fn = filenames_5[idx]

demo_batch = minibatchAB_demo(train_A, batchSize, fn)
epoch, A = next(demo_batch) 

_, A = demo_batch.send(8)
showG(cycleA_generate, A)