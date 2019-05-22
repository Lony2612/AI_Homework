channel_axis=-1
channel_first = False
nc_in = 9
nc_out = 4
ngf = 64
ndf = 64
use_lsgan = False
use_nsgan = False # non-saturating GAN
λ = 10 if use_lsgan else 100

# ========== CAGAN config ==========
nc_G_inp = 9 # [x_im y_im y_j]
nc_G_out = 4 # [alpha, x_i_j(RGB)]
nc_D_inp = 6 # Pos: [x_i, y_i]; Neg1: [G_out(x_i), y_i]; Neg2: [x_i, y_j]
nc_D_out = 1 
gamma_i = 0.1
use_instancenorm = True

loadSize = 143
imageSize = 128
batchSize = 16
lrD = 2e-4
lrG = 2e-4

isRGB = True
apply_da = True