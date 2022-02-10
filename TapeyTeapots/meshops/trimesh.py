# Operations to work with a mesh.
# Structure of a mesh:
#   mesh['verts'] = [3,nVert]
#   mesh['faces'] = [3,nFace]
#   mesh['uvs'] = [3,nFace,2,k], k>1 means multiple textures.
#   mesh['colors'] = [3-4, nVert]. Optional.
# Some operations may need us to copy some code from a "real" mesh library...
import numpy as np
import numba
from . import quat34

######## Nonmesh -> nonmesh [support functions] ###################
# Small support functions will go here.

############## Mesh -> Nonmesh [Extractive functions] #################

def volume(mesh, origin=None, normal=None, signed=True):
    # Standard volume. For non-watertight meshes changing the origin/normal will have an effect.
    # If all the faces are the "wrong way" the volume is the negative if signed is allowed to be true.
    if origin is None:
        origin = np.mean(mesh['verts'],axis=1)

    # Open meshes care about translation + rotation:
    verts = mesh['verts']-np.expand_dims(origin,1)

    if normal is not None:
        q = quat34.q_from_polarshift(np.asarray(normal), np.asarray([0,0,1]))
        verts = quat34.qv(q, verts)

    faces = mesh['faces'] #[3 n]
    #SIMILAR trick to https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up
    # But projecting down into a z=0 plane not a point.
    p1X = verts[0,faces[0,:]]; p2X = verts[0,faces[1,:]]; p3X = verts[0,faces[2,:]]
    p1Y = verts[1,faces[0,:]]; p2Y = verts[1,faces[1,:]]; p3Y = verts[1,faces[2,:]]
    p1Z = verts[2,faces[0,:]]; p2Z = verts[2,faces[1,:]]; p3Z = verts[2,faces[2,:]]
    triangle_area = 0.5*((p2X-p1X)*(p3Y-p1Y) - (p2Y-p1Y)*(p3X-p1X))
    mean_height = (p1Z+p2Z+p3Z)/3.0
    volume = np.sum(triangle_area*mean_height)
    if not signed:
        volume = np.abs(volume)
    return volume

def is_inside_mesh(mesh,query_3xn, ray_direction=None, relative_tol=0.0001):
    # For non-watertight meshes, the ray direction will matter.
    # Faces are oriented.
    if ray_direction is None:
        ray_direction = [1.0,1.0,1.0]
    verts_3xn = mesh['verts']
    faces_3xn = mesh['faces']
    TODO

def project_to_each_triangle(mesh, query_3xn):
    # Returns projected_pointss, distances
    # projected_pointss is [3, nFace, nVert]
    # distancess = [nFace, nVert].
    verts = mesh['verts']
    faces = mesh['faces']
    nVert = verts.shape[1]
    nFace = faces.shape[1]

    projected_pointss = np.zeros([3, nFace, nVert])

    for i in range(nFace):
        triangle_3columns = mesh['verts'][:,mesh['faces'][:,i]]
        projected_pointss[:,i,:] = coregeom.triangle_project(triangle_3columns, points_3xk)

    # Minimize distance:
    deltass = projected_pointss - np.expand_dims(query_3xn,1) # [3, nFace, nVert])
    distances2 = np.sum(deltass*deltass,axis=0) # [nFace, nVert]
    return projected_pointss, np.sqrt(distances2)

def project_to_mesh(mesh, query_3xn):
    # Returns [projected_points, face-ix].
    # Projeccts to the face that is closest.
    projected_pointss, distances = project_to_each_triangle(mesh, query_3xn)
    face_ixs = np.argmax(distances,axis=0) #[nVert]

    return np.transpose(projected_pointss[:,face_ixs,np.arange(nVert)]), face_ixs

@numba.jit(nopython=True)
def _accum1D(where, weight): # Is there a vanilla numpy way to do this?
    ix = np.max(where)
    out = np.zeros(ix)
    n = where.size
    for i in range(n):
        wi = where[i]
        out[wi] = out[wi] + weight[i]
    return out

def _accum2D(where0, where1, weight):
    stride = np.max(where0) + 1
    where = where0 + stride*where1
    unwrapped = _accum1D(where, weight) #[(0,0), (1,0), (2,0) ... (0,1), (1,1), (2,1) ...]
    return np.reshape(unwrapped, [np.max(where0), np.max(where1)], order='F')

def get_edges(mesh):
    # Edges are [2,n], and the bottom row is counter-clockwise to the top row around the face.
    # There are three times the number of edges as there are faces.
    faces = mesh['faces']
    n = faces.shape[1]
    edges = np.zeros([2,3*n],dtype=np.int64)
    edges[0,0:n] = faces[0,:]; edges[1,0:n] = faces[1,:]
    edges[0,n+1:2*n] = faces[1,:]; edges[1,n+1:2*n] = faces[2,:]
    edges[0,2*n+1:3*n] = faces[2,:]; edges[1,2*n+1:3*n] = faces[0,:]
    return edges

@numba.jit(nopython=True)
def _face_adj_matrix(faces_3xn, edge_only=True, exclude_diag=False):
    n = faces_3xn.shape[1]
    out = np.zeros([n,n],dtype=np.int64)
    for i in range(n):
        for j in range(n):
            v0 = faces_3xn[i,0]; v1 = faces_3xn[i,1]; v2 = faces_3xn[i,2]
            w0 = faces_3xn[j,0]; w1 = faces_3xn[j,1]; w2 = faces_3xn[j,2]
            eq0 = (v0==w0) or (v0==w1) or (v0==w2)
            eq1 = (v1==w0) or (v1==w1) or (v1==w2)
            eq2 = (v2==w0) or (v2==w1) or (v2==w2)
            if eq0:
                if (not edge_only) or eq1 or eq2:
                    out[i,j] = 1
            if eq1:
                if (not edge_only) or eq0 or eq2:
                    out[i,j] = 1
            if eq2:
                if (not edge_only) or eq1 or eq0:
                    out[i,j] = 1
            if exclude_diag and i==j:
                out[i,j] = 0
    return out

def get_face_adj(mesh, edge_only=True):
    # Gets faces adjacent to each face (sharing an edge).
    # Returns [face ix][adj faces], a jagged nested list.
    # Faces aren't adjacent to themselves.
    faces = mesh['faces']
    adj_mat = _face_adj_matrix(faces_3xn, edge_only=edge_only, exclude_diag=True)
    ixs = np.argwhere(adj_mat>0.5) # [k,2]
    out = [[] for _ in range(faces.shape[1])]
    for i in range(ixs.shape[1]):
        out[ixs[0]].append(ixs[1])
        #out[ixs[1]].append(ixs[0]) # Not necessary, adj_mat should be symmetric.
    return out

def leaky_verts(mesh):
    # Detects verts where the water would be able to leak past, assuming 1-way walls.
    # Includes: More than two faces meeting at an edge, disagreeing normals, and holes.

    nVert = mesh['verts'].shape[1]

    faces = mesh['faces']
    edges = get_edges(mesh) #[2 n]

    edge_matrix = _accum2D(edges[0,:], edges[1,:], weight) # [vert0, vert1].
    all_edges = np.zeros([nVert,nVert])
    all_edges[0:edge_matrix.shape[0],0:edge_matrix.shape[1]] = edge_matrix

    # If there is an edge form 4 to 5, there must be one from 5 to 4. And can't be more than one.
    too_many = edge_matrix>1.5
    no_reciprocity = (edge_matrix-np.transpose(edge_matrix))>0.5

    bad_pairs = np.argwhere(too_many+no_reciprocity>0.5) # [k 2]

    bad_ixs = np.asarray(set(np.reshape(bad_pairs,[bad_pairs.size])))
    return bad_ixs

################# Mesh -> Mesh [modification fns] ################

def flip_edges(mesh, vertex_pairs_2xk):
    # Returns the mesh.
    TODO

def flip_faces(mesh, face_ixs):
    TODO

def delete_faces(mesh, deleted_ixs):
    # No verts are deleted.
    TODO

def delete_verts(mesh, deleted_ixs):
    # Some faces may also be deleted.
    TODO

def remove_redundant_verts(mesh):
    TODO

def remove_redundant_faces(mesh):
    # Faces which share verts and have the normal pointed in the same direction.
    TODO

def meshcat(meshes):
    # Combines the meshes.
    TODO

def derive_intermediate_points(mesh, vert_ixs, weights_kxn):
    # Returns a mesh with no faces, but with vert pos, color, uvs, etc
    # that take from k vert_ixs and have n verts.
    # Weights is kxn.

    TODO

#################### Mesh + tweak -> nonmesh [Heuristic functions that don't return a mesh] #######################

def merge_nearby_points(points3xk, threshold=0.001):
    # Returns [points, old2new_ix]
    TODO

def poke_triangle(triangle_3columns, points_3xk, barycentric_rel_treshold=0.001):
    # Returns the faces.
    triangle_3columns = mesh['verts'][:,mesh['faces'][:,i]]
    coords = barycentric(triangle_3columns, pointsix_to_this_face) # [4, nToThisFace]
    TODO
    # Use relthreshold to determine whether or not points are on edges.
    # Pick the "best" point which minimizes putting other points near edges, makes nice triangles, etc.
    # Operate this function recursivly.
    for j in range(points.shape[1]):
        coords[:,j] = TODO
    TODO

########################## Mesh + tweak -> mesh [Heuristic modifcation fns] #########################

def merge_nearby(mesh, reldist = 0.0001):
    # Returns [mesh, vert_old2new]
    TODO

def flip_long_edges(mesh, cotan_ratio_treshold = 3.0, max_unflatness_final_radians = 1.0, max_unflatness_add_radians = 0.25):
    # Standard clean-a-mesh at your service!
    # cotan_ratio_threshold = only flip if it helps make angles be not near 0 or 180.
    # max_unflatness_addtan = Unflatness can get to 180 degrees.
    TODO

def add_points(mesh, points_3xk, relthreshold=0.0001, project_points=False):
    # Adds points to the mesh, in the face of edge or triangles, "poking" faces or edges.
    # Interpolates UV and preserves face orientation.
    # Does not add points that land ontop of points or duplicated points with eachother.
    # Returns the modified mesh.
    nFace = mesh['faces'].shape[1]
    nVert = mesh['verts'].shape[1]
    nProj = points_3xk.shape[1]

    # Project the points to the mesh to determine which face is closest:
    proj_points, face_ixs = project_to_mesh(mesh, points_3xk)

    # Make a jagged array of which points each face gets:
    pointsix_to_face0 = [[] for _ in range(nFace)] #[nFace][nPts to face]
    for i in range(nProj):
        pointsix_to_face0[face_ixs[i]].append(i)

    # Points very near the edge can spill onto adjacent faces:
    if relthreshold>1e-14:
        adj_facess = get_face_adj(mesh, edge_only=True) #[face_ix][adj_faces] jagged array.
        pointsix_to_face = [[] for _ in range(nFace)]
        for i in range(nFace):
            nToThisFace = len(points_to_face0[i])
            pointsix_to_this_face = proj_points[:, pointsix_to_face0[i]] #[3, nToThisFace]

            valid_faces = [i]+adj_facess[i]
            for f_ix in valid_faces:
                triangle_3columns = mesh['verts'][:,mesh['faces'][:,f_ix]]
                coords = barycentric(triangle_3columns, pointsix_to_this_face) # [4, nToThisFace]
                out_of_range = np.zeros(nToThisFace)
                for o in range(3):
                    out_of_range = out_of_range + (coords[:,o] < -relthreshold) + (coords[:,o] > 1+relthreshold)
                for j in range(nToThisFace): # For loops are OK as we are adding to a jagged vanilla python list.
                    if out_of_range[j] < 0.5:
                        pointsix_to_face[f_ix].append(pointsix_to_face0[i][j])
    else: # don't allow a threshold.
        pointsix_to_face = pointsix_to_face0

    points_to_face = [[] for _ in range(nFace)]

    dist_tol = np.mean(np.std(mesh['verts'], axis=1))*relthreshold
    for i in range(nFace):
        points0 = points_3xk[:,pointsix_to_face[i]]
        points = merge_nearby_points(points0, threshold=0.001)
        points_to_face[i] = points

    # Add new meshes to the mesh (and remove a few faces).
    new_meshes = []
    removed_faces = []
    for i in range(nFace):
        points_list = points_to_face[i] # List of 3-vectors.
        if len(points_list) > 0:
            points = np.transpose(points_list)
            new_faces = poke_triangle(triangle_3columns, points, barycentric_rel_treshold=relthreshold*0.999)
            weights = barycentric(triangle_3columns, points) # We are making quite a few redundent calls to this fns.
            new_mesh = derive_intermediate_points(mesh, mesh['faces'][:,i], weights[0:3,:])
            new_mesh['verts'] = points # Overwrite the position.
            new_mesh['faces'] = new_faces
            new_meshes.append(new_mesh)
            removed_faces.append(i) # No more i'th face.
    mesh = delete_faces(mesh, removed_faces)
    combined_add_mesh = meshcat(new_meshes)
    return meshcat([mesh, combined_add_mesh])
