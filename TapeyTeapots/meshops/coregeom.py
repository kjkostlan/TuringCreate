# Core geometry operations.
import numba
import numpy as np

# kx3 = not vectorized. nx3 = vectorized.

def expand_dim1(x):
    if len(np.shape(x)) == 1:
        x = np.expand_dims(x,1)
    return x

def average_point(geom_3xk):
    # Pretty simple really.
    return np.mean(geom_3xk, axis=1)

def dists(geom_3xnA, geom_3xnB, sqrt=True):
    # Dists between points 1:1.
    geom_3xnA = expand_dim1(geom_3xnA)
    geom_3xnB = expand_dim1(geom_3xnB)
    delta = geom_3xnA-geom_3xnB
    norm2 = np.sum(delta*delta,axis=0)
    if sqrt:
        return np.sqrt(norm2)
    return norm2

def pairwise_dists(geom_3xkA, geom_3xjB, sqrt=True):
    # [A,B] pairs.
    deltass = np.expand_dims(geom_3xkA,2)-np.expand_dims(geom_3xjB,1) #[xyz, a, b]
    dists2 = np.sum(deltass*deltass,axis=0) #[a,b]
    if sqrt:
        return np.sqrt(dists2)
    return dists2

def _regression_line_plane(geom_3xk, is_plane):
    mean_point = average_point(geom_3xk)
    shifted_points = geom_3xk-np.expand_dims(mean_point,1)

    # Max or min the expected value of vTAv with eigenvalues:
    A = np.matmul(shifted_points, np.transpose(shifted_points))
    w,v = np.linalg.eig(A)
    if is_plane:
        target_ix = np.argmin(w)
    else: # line.
        target_ix = np.argmax(w)
    v_dir_or_normal = v[:,target_ix]
    # Sign convention: Sum is positive.
    if np.sum(v_dir_or_normal)<0:
        return mean_point, -v_dir_or_normal
    return mean_point, v_dir_or_normal

def regression_line(geom_3xk):
    # Regression line. Returns [origin, direction]. Origin at center of mass.
    return _regression_line_plane(geom_3xk, False)

def regression_plane(geom_3xk):
    # Regression plane. Returns [origin, normal].  Origin at center of mass.
    return _regression_line_plane(geom_3xk, True)

def project_to_line(origin, direction, query_3xn):
    query_3xn = query_3xn - np.expand_dims(origin, 1)
    proj_matrix = np.outer(direction,direction)/np.sum(direction*direction)
    return np.matmul(proj_matrix,query_3xn) + np.expand_dims(origin, 1)

def project_to_plane(origin, normal, query_3xn):
    query_3xn = query_3xn - np.expand_dims(origin, 1)
    projected_to_line = project_to_line([0,0,0], normal, query_3xn)
    return query_3xn - projected_to_line + np.expand_dims(origin, 1)

def line_line_closest(origin, direction, query_origin_3xn, query_direction_3xn):
    # Returns [closest_on_origin_3xn, closest_on_dest_3xn].
    query_origin_3xn = expand_dim1(query_origin_3xn)
    query_direction_3xn = expand_dim1(query_direction_3xn)

    #https://math.stackexchange.com/questions/1993953/closest-points-between-two-lines
    n = query_direction_3xn.shape[1]
    P1 = np.tile(np.expand_dims(origin,1),n)
    V1 = np.tile(np.expand_dims(direction,1),n)
    P2 = query_origin_3xn
    V2 = query_direction_3xn

    V3 = np.cross(V2, V1, axisa=0, axisb=0, axisc=0)
    # (P2-P1) = - t2 V2 + t3V3 + t1V1 = ([V1 -V2 V3])^Tt, solve for t.
    V123 = np.stack([np.transpose(V1), -np.transpose(V2), np.transpose(V3)], axis=2) # [n, xyz, v1v2v3]

    V123_inv = np.linalg.inv(V123+1e-100) # Also [n, v1v2v3,  xyz], not flip of last two dimnensions.
    t123 = np.einsum('iju,ui->ij', V123_inv, (P2-P1)) # [n, v1v2v3]

    t1 = np.transpose(t123[:,[0]]) #[1,n]
    t2 = np.transpose(t123[:,[1]]) #[1,n]
    t3 = np.transpose(t123[:,[2]])

    return [P1+t1*V1, P2+t2*V2]

def line_plane_intersection(plane_origin, plane_normal, query_origin_3xn, query_direction_3xn):
    # If the line is parallel to the plane the intersection point will be very far away.
    plane_normal = plane_normal/np.linalg.norm(plane_normal+1e-100)
    dots = np.sum((query_origin_3xn-np.expand_dims(plane_origin,1))*np.expand_dims(plane_normal,1),axis=0)
    slopes = np.sum(query_direction_3xn*np.expand_dims(plane_normal,1), axis=0)
    slopes[slopes==0] = 1e-15
    return query_origin_3xn + query_direction_3xn*np.expand_dims(-dots/slopes, 0)

def triangle_scaled_normals(triangle_xyz_v1v2v3_n):
    # Triangle_column_stacks is [x/y/z][v1/v2/v3][n].
    # Retruns [x/y/z][n].
    # Normal length is triangle area.
    if len(np.shape(triangle_xyz_v1v2v3_n)) < 3:
        triangle_xyz_v1v2v3_n = np.expand_dims(triangle_xyz_v1v2v3_n,2)
    deltas10 = triangle_xyz_v1v2v3_n[:,1,:]- triangle_xyz_v1v2v3_n[:,0,:]
    deltas20 = triangle_xyz_v1v2v3_n[:,2,:]- triangle_xyz_v1v2v3_n[:,0,:]
    return 0.5*np.cross(deltas10,deltas20, axisa=0, axisb=0, axisc=0)

def triangle_areas(triangle_xyz_v1v2v3_n):
    normals = triangle_scaled_normals(triangle_xyz_v1v2v3_n) #[x/y/z][n]
    return np.sqrt(np.einsum('ij,ij->j', normals, normals))

def barycentric(triangle_3columns, query_3xn):
    # Code adapted from: https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    # Returns [4, num_query]
    # The fourth coordinante is how far above or below the plane the point is in units of sqrt(area), making our coordinats like a triangular prism.
    q = triangle_3columns[:,0]
    u = triangle_3columns[:,1]-triangle_3columns[:,0]
    v = triangle_3columns[:,2]-triangle_3columns[:,0]
    n = np.cross(u, v, axis=0)
    s = np.sum(n*n)
    oneOver4ASquared = 1.0 / (s+1e-100)

    w = query_3xn - np.expand_dims(q,1)
    b2 = np.sum(np.cross(np.expand_dims(u,1), w, axisa=0, axisb=0, axisc=0) * np.expand_dims(n,1), axis=0) * oneOver4ASquared
    b1 = np.sum(np.cross(w, np.expand_dims(v,1), axisa=0, axisb=0, axisc=0) * np.expand_dims(n,1), axis=0) * oneOver4ASquared
    b0 = 1.0-b1-b2

    # Custom fourth coordinante:
    b3 = np.sum(w*np.expand_dims(n,1),axis=0)/(np.power(s,0.75)+1e-100)

    return np.transpose(np.stack([b0, b1, b2, b3], axis=1))

def unbarycentric(triangle_3columns, query_4xn):
    # Inverts the barycentric function.
    points_in_triangle = np.einsum('iu,uj->ij',triangle_3columns, query_4xn[0:3,:])

    # Normal:
    u = triangle_3columns[:,1]-triangle_3columns[:,0]
    v = triangle_3columns[:,2]-triangle_3columns[:,0]
    n = np.cross(u, v, axis=0)
    s = np.sum(n*n)
    n1 = n/(np.power(s,0.25)+1e-100) # the norm of n1 should be sqrt(area).

    return points_in_triangle + np.outer(n1, query_4xn[3,:])

def triangle_project(triangle_3columns, query_3xn):
    # Project to a point, edge, or face.
    bary_4xn = barycentric(triangle_3columns, query_3xn)

    # Make sure all components are at least zero, and that w is zero:
    for o in range(3):
        bary_4xn[o,:] = np.maximum(bary_4xn[o,:],0.0)
    bary_4xn[3,:] = 0.0

    # Normalize:
    bary_4xn = bary_4xn/(np.expand_dims(np.sum(bary_4xn,axis=0),0)+1e-100)

    return unbarycentric(triangle_3columns, bary_4xn)

@numba.jit(nopython=True)
def is_inside_loop_2D(edgeloop_2xk, query_2xn):
    n = query_2xn.shape[1]
    k = edgeloop_2xk.shape[1]
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ptx = query_2xn[0,i]
        pty = query_2xn[1,i]
        n_cross = 0
        for j in range(k):
            lx0 = edgeloop_2xk[0,j]
            lx1 = edgeloop_2xk[0,(j+1)%k]
            if lx0>ptx or lx1>ptx: # x in range.
                ly0 = edgeloop_2xk[1,j]
                ly1 = edgeloop_2xk[1,(j+1)%k]
                if (ly0 !=ly1): # Horiz lines can cause issues and are not needed.
                    if ((ly0<=pty) and (ly1>pty)) or ((ly0>=pty) and (ly1<pty)): # y in range.

                        inv_slope = (lx1-lx0)/(ly1-ly0)
                        intersect_x = lx0 + (pty-ly0)*inv_slope
                        if intersect_x > ptx:
                            n_cross = n_cross + 1
        if (n_cross % 2) == 1:
            out[i] = 1
    return out

def is_inside_loop(edgeloop_3xk0, query_3xn0):
    # Fit the loop into a regression plane.
    # Project all points down to the plane. Are we inside the polygon?
    # Note boolean convention: https://stackoverflow.com/questions/47996388/python-boolean-methods-naming-convention
    [origin, normal] = regression_plane(edgeloop_3xk0)

    edgeloop_3xk = project_to_plane(origin, normal, edgeloop_3xk0)
    query_3xn = project_to_plane(origin, normal, query_3xn0)

    # Flatten the polygon into a 2d plane, but have three choices to avoid singularities:
    normala = np.abs(normal)
    if normala[0] > 0.9*np.max(normala):
        ixs = [1,2]
    elif normala[1] > 0.9*np.max(normala):
        ixs = [0,2]
    else:
        ixs = [0,1]

    edgeloop_2xk = edgeloop_3xk[ixs,:]
    query_2xn = query_3xn[ixs,:]

    return is_inside_loop_2D(edgeloop_2xk, query_2xn)
