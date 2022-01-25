# Core geometry operations.
import numba
import numpy as np

# kx3 = not vectorized. nx3 = vectorized.

def average_point(geom_3xk):
    # Pretty simple really.
    return np.mean(geom_3xk, axis=1)

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
    v_biggest = v[:,target_ix]
    # Sign convention: Sum is positive.
    if np.sum(v_biggest)<0:
        return mean_point, -v_biggest
    return mean_point, v_biggest

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

#@numba.jit(nopython=True)
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


