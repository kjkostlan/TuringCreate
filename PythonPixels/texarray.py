# Lower level fns.
import numpy as np

############################## Simple array functions ##########################

def choose2(indicator, a, b, expand=False):
    if expand:
        indicator = np.expand_dims(indicator,axis=2)
    return a*(indicator<0.5)+b*(indicator>=0.5)

def rgb(r,g,b):
    return np.concatenate([np.expand_dims(c,axis=2) for c in [r,g,b]], axis=2)

def grey(rgb):
    return 0.5*rgb[:,1]+0.3*rgb[:,0]+0.2*rgb[:,2]

def clamp(x, lo, hi): # lo and hi can be scalars or arrays.
    return np.minimum(np.maximum(x,lo),hi)

########################## Working with [h,w,2] coordinante arrays #############

def ident_2D(h,w):
    # From -1 to 1 on the width dimension.
    ix_i = (np.arange(h)-0.5*h)/w
    ix_j = (np.arange(w)-0.5*w)/w
    hi, hj = np.meshgrid(ix_i, ix_j, sparse=False, indexing='ij')
    hi = np.expand_dims(hi,axis=2)
    hj = np.expand_dims(hj,axis=2)
    return np.concatenate([hi,hj],axis=2).astype(np.float64)

def ident_2D_ix(h,w):
    ix_i = (np.arange(h))
    ix_j = (np.arange(w))
    hi, hj = np.meshgrid(ix_i, ix_j, sparse=False, indexing='ij')
    hi = np.expand_dims(hi,axis=2)
    hj = np.expand_dims(hj,axis=2)
    return np.concatenate([hi,hj],axis=2).astype(np.float64)

def matmul_2D(coords, m22):
    # Linear is easy.
    i = coords[:,:,0].copy()
    j = coords[:,:,1].copy()
    out = np.zeros_like(coords)
    out[:,:,0] = i*m22[0,0] + j*m22[0,1]
    out[:,:,1] = i*m22[1,0] + j*m22[1,1]
    return out

def int_coords2D(coords):
    return (np.round(coords)).astype(np.int32)

def lookup_2D(coords,values):
    # Basis of Veronoi algorythim. Nearest neighbor interpolation.
    # Rounds to the nearest coordinate int.
    #coords is [h,w,2], values is [h0,w0,d] and is wrapardound. Output is [w,h,d]
    is1d = False
    if len(values.shape) <3:
        is1d = True
        values = np.expand_dims(values,axis=2)
    h = coords.shape[0]
    w = coords.shape[1]
    h0 = values.shape[0]
    w0 = values.shape[1]
    d = values.shape[2]
    coords_int = int_coords2D(coords)
    coords_int[:,:,0] = np.mod(coords_int[:,:,0]+0.5, h0)
    coords_int[:,:,1] = np.mod(coords_int[:,:,1]+0.5, w0)
    coords1_int = coords_int.reshape([h*w, 2], order='F')

    coords2_int = coords1_int[:,0]+coords1_int[:,1]*h0

    values1 = values.reshape([h0*w0, d], order='F')
    output1 = values1[coords2_int,:]

    out2 = np.reshape(output1, [h,w,d],order='F')
    if is1d:
        out2 =  np.squeeze(out2)
    return out2

def interpolate_2D(coords, values, smooth=True):
    # Way too many bilinear functions written.
    is1d = False
    if len(values.shape) <3:
        is1d = True
        values = np.expand_dims(values,axis=2)
    pole_i = np.zeros([1,1,2])
    pole_j = pole_i.copy()
    pole_i[0,0,0] = 1.001
    pole_j[0,0,1] = 1
    def g01(u):
        if smooth:
            u2 = u*u
            return 3.0*u2-2.0*u2*u
        else:
            return u
    coordsr = np.round(coords)
    remainder = coords-np.round(coords)
    rmi = np.expand_dims(remainder[:,:,0],axis=2)
    rmj = np.expand_dims(remainder[:,:,1],axis=2)

    center = lookup_2D(coordsr,values)
    low0 = lookup_2D(coordsr-pole_i,values)
    high0 = lookup_2D(coordsr+pole_i,values)
    low1 = lookup_2D(coordsr-pole_j,values)
    high1 = lookup_2D(coordsr+pole_j,values)
    low = lookup_2D(coordsr-pole_i-pole_j,values)
    high = lookup_2D(coordsr+pole_i+pole_j,values)
    low0_high1 = lookup_2D(coordsr-pole_i+pole_j,values)
    high0_low1 = lookup_2D(coordsr+pole_i-pole_j,values)

    # i-direction interp:
    mix_i = g01(np.abs(rmi*1.0))
    mix_j = g01(np.abs(rmj*1.0))

    output = center*(1.0-mix_i)*(1.0-mix_j)
    output = output+mix_i*(1.0-mix_j)*(low0*(rmi<0.0)+high0*(rmi>=0.0))
    output = output+mix_j*(1.0-mix_i)*(low1*(rmj<0.0)+high1*(rmj>=0.0))
    out2 = output+mix_i*mix_j*((rmi<0.0)*(rmj<0.0)*low+(rmi>=0.0)*(rmj>=0.0)*high+(rmi<0.0)*(rmj>=0.0)*low0_high1+(rmi>=0.0)*(rmj<0.0)*high0_low1)

    if is1d:
        out2 = np.squeeze(out2)
    return out2

def pnorm_2D(coords, pnorm):
    x = coords[:,:,0]
    y = coords[:,:,1]
    if pnorm==2.0:
        return np.sqrt(x*x+y*y)
    elif pnorm==1.0:
        return np.abs(x)+np.abs(y)
    elif pnorm>1e9:
        return np.max(np.abs(x),np.abs(y))
    else:
        return (x*pnorm+y*pnorm)**(1.0/pnorm)

def hodge_2D(coords):
    x = coords[:,:,[0]]
    y = coords[:,:,[1]]
    return np.concatenate([-y,x],axis=2)

def cross_2D(coordsA, coordsB):
    xa = coordsA[:,:,0]
    ya = coordsA[:,:,1]
    xb = coordsB[:,:,0]
    yb = coordsB[:,:,1]
    return xa*yb-ya*xb

def halfsize_2D(x): # Rounds down.
    is1d = False
    if len(x.shape) <3:
        is1d = True
        x = np.expand_dims(x,axis=2)
    i = int(0.5*x.shape[0])*2
    j = int(0.5*x.shape[1])*2
    x1 = 0.25*(x[0:i:2,0:j:2]+x[1:i:2,0:j:2]+x[0:i:2,1:j:2]+x[1:i:2,1:j:2])
    if is1d:
        x1 = np.squeeze(x1)
    return x1

def twicesize_2D(x):
    is1d = False
    if len(x.shape) <3:
        is1d = True
        x = np.expand_dims(x,axis=2)
    i = 2*x.shape[0]
    j = 2*x.shape[1]

    x_pad = np.pad(x,[(1,1),(1,1),(0,0)], mode='reflect')

    x1 = np.zeros([i,j, x.shape[2]])
    p55 = x_pad[1:-1, 1:-1, :]
    p54 = x_pad[1:-1, 0:-2, :]
    p45 = x_pad[0:-2, 1:-1, :]
    p44 = x_pad[0:-2, 0:-2, :]
    p65 = x_pad[2:, 1:-1, :]
    p56 = x_pad[1:-1, 2:, :]
    p66 = x_pad[2:, 2:, :]
    p46 = x_pad[0:-2, 2:, :]
    p64 = x_pad[2:, 0:-2, :]
    x1[0:i:2, 0:j:2] = 0.25*(p44+p45+p54+p55)
    x1[1:i:2, 0:j:2] = 0.25*(p54+p55+p64+p65)
    x1[0:i:2, 1:j:2] = 0.25*(p45+p46+p55+p56)
    x1[1:i:2, 1:j:2] = 0.25*(p55+p56+p65+p66)
    if is1d:
        x1 = np.squeeze(x1)
    return x1
