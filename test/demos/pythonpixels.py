# Procedural textures are fun!
import numpy as np
import texfns
import texarray as tnp
from PIL import Image

def debug_image(x, normalize=True, filename='debug.png'):
    if len(x.shape) == 3 and x.shape[2] == 2:
        if normalize:
            x0 = np.min(x)
        else:
            x0 = 0
        x = np.concatenate([x, x0*np.ones([x.shape[0],x.shape[1],1])],axis=2)
    if normalize:
        x = x-np.min(x)
        x = x/np.max(x)
    x = x*255
    x_int = x.astype(np.uint8)
    im = Image.fromarray(x_int)
    print('Image save size:',x_int.shape)
    im.save(filename)

################################## Tech demos ##################################

def simple_veronoi():
    w = 768
    h = 512
    coords = tnp.ident_2D(h,w)
    min_dist, second_min_dist, dist_to_edge, index_i, index_j = \
      texfns.veronoi(coords, feature_size=1/32.0, jitter=1.0, pnorm=2.0, h0=128, w0=128)
    debug_image(min_dist)

def edge_veronoi():
    w = 768
    h = 512
    coords = tnp.ident_2D(h,w)
    min_dist, second_min_dist, dist_to_edge, index_i, index_j = \
      texfns.veronoi(coords, feature_size=1/32.0, jitter=1.0, pnorm=2.0, h0=128, w0=128,
                     compute_extras=True)
    debug_image(tnp.rgb(dist_to_edge, 0.7*min_dist, second_min_dist))

def index_veronoi():
    w = 768
    h = 512
    coords = tnp.ident_2D(h,w)
    scale = 12.0

    min_dist, second_min_dist, dist_to_edge, index_i, index_j = \
      texfns.veronoi(coords, feature_size=1/scale, jitter=1.0, pnorm=2.0, h0=128, w0=128,
                     compute_extras=True)
    debug_image(tnp.rgb(3*dist_to_edge, np.mod(index_i,3), np.mod(index_j,3)))

def hex_veronoi():
    w = 768
    h = 512
    coords = tnp.ident_2D(h,w)
    mat22 = np.asarray([[1,0],[0.5,0.866]])
    coords1 = tnp.matmul_2D(coords, mat22)
    jitter = 0.35
    min_dist1, second_min_dist1, dist_to_edge1, index_i1, index_j1 = \
      texfns.veronoi(coords1, feature_size=1/32.0, jitter=jitter, pnorm=2.0, h0=128, w0=128)
    min_dist2, second_min_dist2, dist_to_edge2, index_i2, index_j2 = \
      texfns.veronoi(coords, feature_size=1/32.0, jitter=jitter, pnorm=2.0, h0=128, w0=128,
                     matrix22=mat22)
    score = coords[:,:,0]+coords[:,:,1]
    weight = tnp.clamp(score*10,-0.5,0.5)+0.5
    debug_image(min_dist2*weight+min_dist1*(1.0-weight))

def simple_perlin():
    w = 768
    h = 512
    coords = tnp.ident_2D(h,w)
    grey = texfns.musgrave(coords, feature_size0=1/4.0, feature_size1=1/512.0, strength0=1, strength1=0.125, steps=8, h0=128, w0=128)
    debug_image(grey)

def brick_demo():
    w = 768*4
    h = 512*4
    coords = tnp.ident_2D(h,w)
    size = 0.04
    brick, ix_x, ix_y = texfns.brick_texture(coords, brick_width=size, brick_height=size*0.4, mortar=0.25, mortar_smooth=0.05,
                                             jitter_x=0.03, jitter_y=0.03)

    brick_herringbone, ix_x_h, ix_y_h = texfns.brick_texture(coords, brick_width=size, brick_height=size*0.5, mortar=0.7, mortar_smooth=0.35, herringbone=True)

    score = coords[:,:,0]+coords[:,:,1]
    weight = tnp.clamp(score*10,-0.5,0.5)+0.5

    intense = brick*(1.0-weight)+brick_herringbone*weight
    r = 0.5*np.mod(ix_x,6)*(1.0-weight)+np.mod(ix_x_h,3)*weight
    b = 0.5*np.mod(ix_y,6)*(1.0-weight)+np.mod(ix_y_h,3)*weight
    debug_image(tnp.rgb(intense*r,intense,intense*b))

def musgrave():
    w = 768
    h = 512
    coords = tnp.ident_2D(h,w)
    mode = 4
    if mode==0:
        grey = texfns.musgrave(coords, steps=10, feature_size0=1/4.0, feature_size1=1/512.0,
                               strength0=1, strength1=0.1, max_multiplier=3.0,
                               mult_cascade0=0.7, mult_cascade1=-0.5,
                               slope_minus = -0.25, slope_plus = 1.0,
                               h0=128, w0=128)
    elif mode==1:
        grey = texfns.musgrave(coords, steps=10, feature_size0=1/4.0, feature_size1=1/512.0,
                               strength0=1, strength1=0.01,
                               mult_cascade0=0.7, mult_cascade1=0.0,
                               slope_minus = -1.0, slope_plus = -1.0,
                               h0=128, w0=128)
    elif mode==2:
        grey = texfns.musgrave(coords, steps=6, feature_size0=1/4.0, feature_size1=1/512.0,
                               strength0=1, strength1=0.1,
                               mult_cascade0=0.0, mult_cascade1=0.0,
                               slope_minus = -1.0, slope_plus = -1.0,
                               h0=128, w0=128)
    elif mode==3:
        grey = texfns.musgrave(coords, steps=12, feature_size0=1/8.0, feature_size1=1/512.0,
                               strength0=0.8, strength1=0.01,
                               mult_cascade0=0.5, mult_cascade1=0.0, max_multiplier=4.0,
                               slope_minus = 1.0, slope_plus = 1.0,
                               h0=128, w0=128)
    elif mode==4:
        grey = texfns.musgrave(coords, steps=12, feature_size0=1/8.0, feature_size1=1/512.0,
                               strength0=0.8, strength1=0.01,
                               mult_cascade0=1.0, mult_cascade1=1.0, max_multiplier=24.0,
                               slope_minus =-1.0, slope_plus = 1.0,
                               h0=128, w0=128)
    #grey = np.abs(-grey)
    debug_image(grey)

def earthquake():
    w = 768
    h = 512
    coords = tnp.ident_2D(h,w)

    noisedistort0 = texfns.musgrave(coords, feature_size0=1/16.0, feature_size1=1/512.0,
                             strength0=1, strength1=0.05, steps=8, h0=128, w0=128)
    noisedistort1 = texfns.musgrave(coords, feature_size0=1/16.0, feature_size1=1/512.0,
                         strength0=1, strength1=0.05, steps=8, h0=128, w0=128)
    noisedistort = np.concatenate([np.expand_dims(noisedistort0,axis=2), np.expand_dims(noisedistort1,axis=2)],axis=2)

    distort = 0.0025
    coords = coords + distort*noisedistort

    sz = 0.03
    coords1 = texfns.fault_lines(coords, n_faultlines=16, sigma=0.5, sigma_slip=0.25)
    coords2 = texfns.fault_lines(coords1, n_faultlines=64, sigma=0.5, sigma_slip=0.05)

    #brick, ix_x, ix_y = texfns.brick_texture(coords2, brick_width=sz, brick_height=0.4*sz, mortar=0.25, mortar_smooth=0.05,
    #                                     jitter_x=0.03, jitter_y=0.03)
    grey = texfns.musgrave(coords2, feature_size0=1/4.0, feature_size1=1/512.0, strength0=1, strength1=0.125, steps=8, h0=128, w0=128)

    debug_image(grey)


################################### More realistic #############################

def marble():
    w = 768*1.0
    h = 512*1.0
    coords = tnp.ident_2D(h,w)
    noisedistort0 = texfns.musgrave(coords, feature_size0=1/16.0, feature_size1=1/512.0,
                             strength0=1, strength1=0.05, steps=8, h0=128, w0=128)
    noisedistort1 = texfns.musgrave(coords, feature_size0=1/16.0, feature_size1=1/512.0,
                         strength0=1, strength1=0.05, steps=8, h0=128, w0=128)

    noisedistort = np.concatenate([np.expand_dims(noisedistort0,axis=2), np.expand_dims(noisedistort1,axis=2)],axis=2)

    def slot(x, depth, power):
        depth = np.maximum(depth,0.0)
        return 1-np.minimum(depth,1.0)/(1.0+np.abs(x)**power)

    distort = 0.005
    coords1 = coords + distort*noisedistort

    noisecolor = texfns.musgrave(coords1, feature_size0=1/8.0, feature_size1=1/256.0,
                                 strength0=1, strength1=0.05, steps=8, h0=128, w0=128)


    noise_depth = texfns.musgrave(coords, feature_size0=1/16.0, feature_size1=1/256.0,
                                   strength0=1, strength1=0.25, steps=8, h0=128, w0=128)
    noise_depth1 = texfns.musgrave(coords, feature_size0=1/16.0, feature_size1=1/256.0,
                                   strength0=1, strength1=0.25, steps=8, h0=128, w0=128)

    score = np.abs(noisecolor)
    #print('stuff:',np.mean(noisescale), np.std(noisescale))
    #depth = 1.0/(1.0+np.exp(0.5*noisescale))
    #grey = slot(score*15*np.exp(noisescale*0.05))*depth + 1.0-depth
    grey = slot(score*3,1.0+0.5*noise_depth1, 2.5)+0.5*slot(score*7, noise_depth, 2.5)

    debug_image(grey)


#brick_demo()
#simple_veronoi()
#edge_veronoi()
#index_veronoi()
#simple_perlin()
#musgrave()
#hex_veronoi()
#earthquake()

#marble()
