# Higher level fns, inspired by blender.
import numpy as np
import PythonPixels.texarray as tnp

def white_noise(coords):
    return np.random.randn(coords.shape[0],coords.shape[1])

def gradient(coords,direction, f):
    dot = direction[0]*coords[:,:,0]+ direction[1]*coords[:,:,1]
    return f(dot)


def veronoi(coords, feature_size=0.05, jitter=1.0, pnorm=2.0, h0=128, w0=128, compute_extras = False,
            matrix22=None):
    coords = (coords+1)/feature_size
    point_jitters = jitter*(np.random.random([h0,w0,2])-0.5)
    if matrix22 is None:
        matrix22 = np.identity(2)
    inv_matrix22 = np.linalg.inv(matrix22)
    coords_skew = tnp.matmul_2D(coords, matrix22)

    roll_i = [1,1,0,-1,-1,-1,0,1,0]
    roll_j = [0,1,1,1,0,-1,-1,-1,0]
    nr = len(roll_i)

    ez_one = np.ones(coords.shape[0:2])

    min_dist_edge = 0.4*ez_one
    index_i = 0.0*ez_one
    index_j = 0.0*ez_one
    min_point = np.zeros_like(coords)
    second_min_point = np.zeros_like(coords)

    h = coords.shape[0]
    w = coords.shape[1]
    relative_pt_locations = np.zeros([h, w, 2, nr])

    point_locations = point_jitters+tnp.ident_2D_ix(h0,w0)

    for ix in range(nr):
        point_locations_roll = np.roll(point_locations.copy(),[roll_i[ix],roll_j[ix],0],axis=[0,1,2])
        center_point = tnp.lookup_2D(coords_skew, point_locations_roll)
        cur_point = center_point-coords_skew # Delta from where we are.
        cur_point = tnp.matmul_2D(cur_point,inv_matrix22)

        cur_point[:,:,0] = np.mod(cur_point[:,:,0]+0.5*h0, h0)-0.5*h0
        cur_point[:,:,1] = np.mod(cur_point[:,:,1]+0.5*w0, w0)-0.5*w0
        #cur_point[:,:,0] = tnp.choose2(cur_point[:,:,0]<-0.5*h0, cur_point[:,:,0]+0.5*h0,cur_point[:,:,0])
        #cur_point[:,:,1] = tnp.choose2(cur_point[:,:,1]<-0.5*w0, cur_point[:,:,1]+0.5*w0,cur_point[:,:,1])
        relative_pt_locations[:,:,:,ix] = cur_point

    dists = np.zeros([h,w,nr])
    for ix in range(nr):
        dists[:,:,ix] = tnp.pnorm_2D(relative_pt_locations[:,:,:,ix],pnorm)

    min_dist = np.amin(dists,axis=2)
    second_min_dist = np.amin(dists+1e20*(dists==np.expand_dims(min_dist,axis=2)),axis=2)

    # Better options than a double loop?
    if compute_extras:
        coords_int = tnp.int_coords2D(coords)
        for ix in range(nr):
            indicator = (dists[:,:,ix]==min_dist)
            index_i = tnp.choose2((indicator>0.5), index_i, coords_int[:,:,0] - roll_i[ix])
            index_j = tnp.choose2((indicator>0.5), index_j, coords_int[:,:,1] - roll_j[ix])
        for ix in range(nr):
            for jx in range(nr):
                if jx == ix:
                    continue
                indicator = (dists[:,:,ix]==min_dist)*(dists[:,:,jx]==second_min_dist)
                if np.sum(indicator)>0:
                    min_point = relative_pt_locations[:,:,:,ix]
                    second_min_point = relative_pt_locations[:,:,:,jx]
                    point_avg = 0.5*(min_point+second_min_point)
                    point_diff = (second_min_point-min_point)#*np.expand_dims(tnp.pnorm_2D(second_min_point-min_point,pnorm)+1e-20,axis=2)
                    along_line = tnp.hodge_2D(point_diff)
                    dist_to_edge = np.abs(tnp.cross_2D(point_avg,along_line))
                    min_dist_edge = tnp.choose2((indicator>0.5), min_dist_edge, dist_to_edge)

    #if not compute_extras:
    #    min_dist_edge = (second_min_dist-min_dist) # TODO: better metric.
    return min_dist, second_min_dist, min_dist_edge, index_i, index_j

def musgrave(coords, steps=5, feature_size0=0.125, feature_size1=0.001953125, strength0=1, strength1=0.1,
                 mult_cascade0=0.0, mult_cascade1=0.0, max_multiplier=1e20,
                 slope_minus = -1.0, slope_plus = 1.0,
                 h0=128, w0=128):
    if steps<=1:
        steps = 1
        feature_size0 = np.sqrt(feature_size0*feature_size1)
        feature_size1 = feature_size0
        strength0 = np.sqrt(strength0*strength1)
        strength1 = strength0
    feature_sizes = np.exp(np.linspace(np.log(feature_size0), np.log(feature_size1), steps))
    strengths = np.exp(np.linspace(np.log(strength0), np.log(strength1), steps))
    mult_cascades = np.linspace(mult_cascade0, mult_cascade1, steps)
    angles = np.linspace(0, 0.25*np.pi, steps+1)
    output = 0.0
    noise_multiplier = 1.0

    norm_sample_1D = np.linspace(-4,4,256)
    mass = np.exp(-0.5*norm_sample_1D*norm_sample_1D)
    mass = mass/np.sum(mass)
    vals1d = norm_sample_1D*(-slope_minus)*(norm_sample_1D<0)+norm_sample_1D*slope_plus*(norm_sample_1D>=0)
    fancy_mean = np.sum(mass*vals1d)
    fancy_sigma = np.sqrt(np.sum(mass*vals1d*vals1d)-fancy_mean*fancy_mean)
    for i in range(steps):
        theta = angles[i]
        m22 = np.asarray([[np.cos(theta),np.sin(theta)], [-np.sin(theta),np.cos(theta)]])
        coords_i = tnp.matmul_2D(coords, m22)/feature_sizes[i]
        values = np.random.randn(h0,w0)
        unit_noise = tnp.interpolate_2D(coords_i, values, smooth=True)
        unit_noise = unit_noise*(-slope_minus)*(unit_noise<0)+unit_noise*slope_plus*(unit_noise>=0)
        output = output+strengths[i]*unit_noise*np.minimum(max_multiplier,noise_multiplier)
        noise_multiplier = noise_multiplier*np.exp(mult_cascades[i]*(unit_noise-fancy_mean)/fancy_sigma)
    return output

def brick_texture(coords, brick_width=0.0625, brick_height=0.03125, mortar=0.1, mortar_smooth=0.01,
                  herringbone=False, jitter_x=0.0, jitter_y = 0.0, h0=128, w0=128):

    aspect = brick_height/brick_width
    if herringbone: # this fn is a MESS!
        diag_score = (coords[:,:,1]-coords[:,:,0])/(brick_width)
        corner_score = (coords[:,:,1]+coords[:,:,0])/(brick_width)-0.5/brick_width
        corner_scaling = 2.0*aspect

        diag_int0 = np.round(diag_score-0.5).astype(np.int32)
        #abs_corner_shift =
        phase_shift = diag_int0
        corner_score = corner_scaling*(np.abs(np.mod((corner_score-phase_shift)/corner_scaling+0.75,1.0)-0.5)-0.25)
        #corner_score = corner_scaling*(np.abs(np.mod(0.0+0.75,1.0)-0.5)-0.25)
        #corner_score = corner_scaling*(-np.abs(np.mod(corner_score/corner_scaling,1.0)-0.5)-0.25)


        diag_int = np.round(diag_score+corner_score).astype(np.int32)

        brick_i_even = coords[:,:,0]/brick_height #0.5*diag_int*(brick_height+0.5*brick_width)/brick_height
        brick_i_odd = (coords[:,:,1]-0.5*brick_width-0.5*brick_height)/brick_height #0.5*diag_int*(brick_height+0.5*brick_width)/brick_height

        brick_i = tnp.choose2((np.mod(diag_int,2)==1), brick_i_even, brick_i_odd)
        brick_i_round = np.round(brick_i).astype(np.int32)

        brick_j_even = coords[:,:,1]/brick_width - aspect*brick_i_round
        brick_j_odd = (coords[:,:,0]-0.5*brick_width-0.5*brick_height)/brick_width - aspect*brick_i_round

        brick_j = tnp.choose2((np.mod(diag_int,2)==1), brick_j_even, brick_j_odd)
        brick_j_round = np.round(brick_j).astype(np.int32)

    else:
        brick_i = coords[:,:,0]/brick_height # integer at center.
        brick_j = coords[:,:,1]/brick_width
        brick_i_round = np.round(brick_i).astype(np.int32)
        brick_j = tnp.choose2((np.mod(brick_i_round,2)==1), brick_j, brick_j+0.5)
        brick_j_round = np.round(brick_j).astype(np.int32)

    if np.abs(jitter_x)+np.abs(jitter_y)>0:
        # Shift the values of brick_i,j don't recalculate? the rounded values.
        random_stuff = np.zeros([h0,w0,2])
        random_stuff[:,:,0] = jitter_x*(np.random.random([h0,w0])-0.5)
        random_stuff[:,:,1] = jitter_y*(np.random.random([h0,w0])-0.5)

        coords = np.zeros([brick_j.shape[0],brick_j.shape[1],2])
        coords[:,:,0] = brick_i_round
        coords[:,:,1] = brick_j_round

        rand_per_brick = tnp.lookup_2D(coords,random_stuff)
        brick_i = brick_i+rand_per_brick[:,:,0]
        brick_j = brick_j+rand_per_brick[:,:,1]

    brick_edge_i = 2*np.abs(brick_i-brick_i_round)
    brick_edge_j = 2*np.abs(brick_j-brick_j_round)
    mortar_i = mortar*0.5
    mortar_j = mortar_i*aspect # same absolute size, different relative size.
    mortar_score = np.maximum((brick_edge_i+mortar_i-1.0)*aspect, (brick_edge_j+mortar_j-1.0)) # >0 is mortar.

    brick_score = 1.0
    mortar_smooth = np.maximum(mortar_smooth,1e-9)
    brick_score = np.minimum(1.0, np.maximum(0.0, 0.5-(mortar_score/mortar_smooth)))

    return brick_score, brick_i_round, brick_j_round

def fault_lines(coords, n_faultlines=16, sigma=0.5, sigma_slip=0.125):
    # California is overdue...
    w = coords.shape[0]
    h = coords.shape[1]
    coords = coords.copy()
    for i in range(n_faultlines):
        xxyy = np.random.randn(4)
        origin = sigma*np.random.randn(2)
        line_normal = np.random.randn(2)
        line_normal = line_normal/np.sqrt(np.sum(line_normal*line_normal)+1e-20)
        slip = np.random.randn()*sigma_slip
        line_slide = np.expand_dims([0.5*slip*line_normal[1],-0.5*slip*line_normal[0]], axis=[0,1])

        coords1 = coords-np.expand_dims(origin,axis=[0,1])
        dot = coords1[:,:,0]*line_normal[0]+coords1[:,:,1]*line_normal[1]

        coords = tnp.choose2(dot>0, coords-line_slide, coords+line_slide, expand=True)

    return coords
