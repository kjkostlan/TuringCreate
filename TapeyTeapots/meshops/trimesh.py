# Operations to work with a mesh.
# Structure of a mesh:
#   mesh['verts'] = [3,nVert]
#   mesh['faces'] = [3,nFace]
#   mesh['uvs'] = [3,nFace,2,k], k>1 means multiple textures. 2 can be >2 and code still works OK.
#   mesh['colors'] = [3-4, nVert]. Optional. 3-4 can be any number really.
#   mesh['mat'] = [nFace]. Optional, int index of materials.
# Some operations may need us to copy some code from a "real" mesh library...
import numpy as np
import numba
import scipy, scipy.spatial
from . import quat34, coregeom

######## Nonmesh -> nonmesh [support functions] ###################
# Small support functions will go here.

def _expand1(x): # TODO: duplicate code with fns in other file(s)
    x = np.asarray(x)
    if len(x.shape)<2:
        x = np.expand_dims(x,axis=1)
    return x

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

def is_inside_mesh(mesh,query_3xn, ray_direction3xn=None, relative_tol=0.0001):
    # For non-watertight meshes, the ray direction will matter.
    # Faces are oriented.
    if ray_direction3xn is None:
        ray_direction3xn = [1.0,1.0,1.0]
    ray_direction3xn = _expand1(ray_direction3xn)

    # Find the barycentric coords of line-plane intersection:
    n = query_3xn.shape[1]; nVert = mesh['verts'].shape[1]; nFace = mesh['faces'].shape[1]
    if ray_direction3xn.shape[1] == 1:
        ray_direction3xn = np.tile(ray_direction3xn, [1,n])
    closest_distance = 1e100*np.ones(n) # Closest intersection to plane facing the right way.
    closest_is_inside = np.zeros(n)
    for i in range(nFace):
        triangle3columns = mesh['verts'][:,mesh['faces'][:,i]]
        plane_origin = triangle3columns[:,0]
        outward_normal = np.cross(triangle3columns[:,1]-triangle3columns[:,0],triangle3columns[:,2]-triangle3columns[:,0])
        # Intersections blows up to a huge, but finite number if the line is parrallel to the plane.
        intersections = coregeom.line_plane_intersection(plane_origin, outward_normal, query_3xn, ray_direction3xn)
        # Barycentric time:
        baryxyzw = coregeom.barycentric(triangle3columns, query_3xn)
        in_range0 = (baryxyzw[0,:] > -relative_tol) * (baryxyzw[1,:] > -relative_tol) * (baryxyzw[2,:] > -relative_tol)
        in_range1 = (baryxyzw[0,:] < 1+relative_tol) * (baryxyzw[1,:] < 1+relative_tol) * (baryxyzw[2,:] < 1+relative_tol)
        inside_of_face = np.einsum('j,ji->i',outward_normal, ray_direction3xn) > 0
        distances = np.einsum('ji,ji->i',ray_direction3xn,intersections-query_3xn) # Dot product.
        record_set = (closest_distance<distances)*in_range0*in_range1

        closest_is_inside[record_set>0.5] = inside_of_face[record_set>0.5]
        closest_distance = np.minimum(closest_distance,distances)

    return closest_is_inside

def raycast_distance(mesh, ray_origin3xn, ray_direction3xn, relative_tol=0.0001, allow_negative=False):
    # Gets the collision distances, with >1e100 meaning no collision.
    verts = mesh['verts']; nVert = verts.shape[1]
    faces = mesh['faces']; nFace = faces.shape[1]

    norm = np.sqrt(np.sum(ray_direction3xn*ray_direction3xn,axis=0))
    ray_direction3xn = ray_direction3xn/np.expand_dims(norm+1e-100,axis=0)

    out = 1e101*np.ones(ray_origin3xn.shape[1])
    for i in range(nFace):
        tri3 = verts[:,faces[:,i]]
        plane_origin = tri3[:,0]
        plane_normal = np.cross(tri3[:,1]-tri3[:,0], tri3[:,2]-tri3[:,0])
        intersect3xn = coregeom.line_plane_intersection(plane_origin, plane_normal, ray_origin3xn, ray_direction3xn)
        bary4xn = coregeom.barycentric(tri3, intersect3xn)

        collision = (bary4xn[0,:]<=1.0+relative_tol)*(bary4xn[1,:]<=1.0+relative_tol)*(bary4xn[2,:]<=1.0+relative_tol)
        distances = np.sum((intersect3xn-ray_origin3xn)*ray_direction3xn,axis=0)
        if not allow_negative:
            distances[distances<0] = 1e200
            out[collision>0.5] = np.minimum(distances[collision>0.5],out[collision>0.5])
        else: # This code also works for positive, so the if statement is not strictly necessary.
            record_set = (collision>0.5)*(np.abs(distances)<np.abs(out))
            out[record_set>0.5] = distances[record_set>0.5]
    return out

def project_to_each_triangle(mesh, query_3xn):
    # Returns projected_pointss, distances
    # projected_pointss is [3, nFace, nQuery]
    # distancess = [nFace, nQuery].
    verts = mesh['verts']
    faces = mesh['faces']
    nQuery = query_3xn.shape[1]
    nFace = faces.shape[1]

    projected_pointss = np.zeros([3, nFace, nQuery])

    for i in range(nFace):
        triangle_3columns = mesh['verts'][:,mesh['faces'][:,i]]
        projected_pointss[:,i,:] = coregeom.triangle_project(triangle_3columns, query_3xn)

    # Minimize distance:
    deltass = projected_pointss - np.expand_dims(query_3xn,1) # [3, nFace, nVert])
    distances2 = np.sum(deltass*deltass,axis=0) # [nFace, nVert]
    return projected_pointss, np.sqrt(distances2)

def project_to_mesh(mesh, query_3xn):
    # Returns [projected_points, face-ix].
    # Projects to the face that is closest.
    projected_pointss, distances = project_to_each_triangle(mesh, query_3xn) #[3, nFace, nQuery], [nFace, nQuery]
    face_ixs = np.argmin(distances,axis=0); nVert = face_ixs.size #[nVert]
    return projected_pointss[:,face_ixs,np.arange(nVert)], face_ixs

@numba.jit(nopython=True)
def _accum1D(where, weight): # Is there a vanilla numpy way to do this?
    ix = np.max(where)
    out = np.zeros(ix+1)
    n = where.size
    for i in range(n):
        wi = where[i]
        out[wi] = out[wi] + weight[i]
    return out

def _accum2D(where0, where1, weight):
    stride = np.max(where0) + 1
    where = where0 + stride*where1
    unwrapped_short = _accum1D(where, weight) #[(0,0), (1,0), (2,0) ... (0,1), (1,1), (2,1) ...]
    unwrapped = np.zeros(stride*(np.max(where1)+1)); unwrapped[0:unwrapped_short.size] = unwrapped_short
    return np.reshape(unwrapped, [stride, np.max(where1)+1], order='F')

def get_halfedges(mesh):
    # Edges are [2,3*nFace], and the bottom row is counter-clockwise to the top row around the face.
    # There are three times the number of edges as there are faces, and usually edges come in pairs of a,b/b,a
    faces = mesh['faces']
    n = faces.shape[1]
    edges = np.zeros([2,3*n],dtype=np.int64)
    edges[0,0:n] = faces[0,:]; edges[1,0:n] = faces[1,:]
    edges[0,n:2*n] = faces[1,:]; edges[1,n:2*n] = faces[2,:]
    edges[0,2*n:3*n] = faces[2,:]; edges[1,2*n:3*n] = faces[0,:]
    return edges

@numba.jit(nopython=True)
def _get_halfedge_matrix_core(halfedges): #[to from], in counter-clockwise direction.
    he = halfedges; n = np.max(he) # half edges is [2, 3*nFace]
    mat = np.zeros((n+1,n+1),np.int64)
    for i in range(he.shape[1]):
        mat[he[1,i],he[0,i]] = mat[he[1,i],he[0,i]] + 1
    return mat

def get_halfedge_matrix(mesh):
    return _get_halfedge_matrix_core(get_halfedges(mesh))

def get_edges(mesh):
    # Half the number of half-edges in the mesh if the mesh is watertight.
    halfs0 = get_halfedges(mesh)
    stride = np.max(halfs0)+1
    halfs = np.zeros_like(halfs0)
    halfs[0,:] = np.min(halfs0,axis=0)
    halfs[1,:] = np.max(halfs0,axis=0)

    uhalfs = list(set(stride*halfs[0,:] + halfs[1,:])); uhalfs.sort()
    uhalfs = np.asarray(uhalfs)
    out = np.zeros([2,uhalfs.size],dtype=np.int64)
    tmp = uhalfs/stride; tmp = tmp.astype(np.int64); out[0,:] = tmp
    out[1,:] = uhalfs%stride
    return out

def get_vert2face(mesh):
    # Jagged array of which faces each vert is attached to.
    faces = mesh['faces']
    nVert = mesh['verts'].shape[1]
    out = [[] for _ in range(nVert)]
    for i in range(faces.shape[1]):
        v012 = faces[:,i]
        for o in range(3):
            out[v012[o]].append(i)
    return out

def get_face_counts(mesh):
    # Number of faces on each vert.
    v2f = get_vert2face(mesh)
    return np.asarray([len(v2fi) for v2fi in v2f])

@numba.jit(nopython=True)
def _face_adj_matrix(faces_3xn, edge_only=True, exclude_diag=False):
    n = faces_3xn.shape[1]
    out = np.zeros((n,n),dtype=np.int64)
    for i in range(n): # TODO: faster than O(n^2)
        for j in range(n):
            v0 = faces_3xn[0,i]; v1 = faces_3xn[1,i]; v2 = faces_3xn[2,i]
            w0 = faces_3xn[0,j]; w1 = faces_3xn[1,j]; w2 = faces_3xn[2,j]
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
    adj_mat = _face_adj_matrix(faces, edge_only=edge_only, exclude_diag=True)
    ixs = np.transpose(np.argwhere(adj_mat>0.5)) # [2,k]
    out = [[] for _ in range(faces.shape[1])]
    for i in range(ixs.shape[1]):
        out[ixs[0,i]].append(ixs[1,i])
        #out[ixs[1]].append(ixs[0]) # Not necessary, adj_mat should be symmetric.
    return out

def leaky_verts(mesh):
    # Detects verts where the water would be able to leak past, assuming 1-way walls.
    # Includes: More than two faces meeting at an edge, disagreeing normals, and holes.

    nVert = mesh['verts'].shape[1]
    if np.max(mesh['faces']) > nVert-1:
        raise Exception('Invalid mesh, faces ixs out of bounds (i.e. referring to verts that do not exist).')

    faces = mesh['faces']
    edges = get_halfedges(mesh) #[2 3*faces]

    edge_matrix = _accum2D(edges[0,:], edges[1,:], np.ones([edges.shape[1]])) # [vert0, vert1], is there an edge from vert0 to vert1.
    all_edges = np.zeros([nVert,nVert])
    all_edges[0:edge_matrix.shape[0],0:edge_matrix.shape[1]] = edge_matrix

    # If there is an edge form 4 to 5, there must be one from 5 to 4. And can't be more than one.
    too_many = edge_matrix>1.5
    no_reciprocity = (edge_matrix-np.transpose(edge_matrix))>0.5

    bad_pairs = np.argwhere(too_many+no_reciprocity>0.5) # [k 2]

    bad_ixs = np.asarray(list(set(np.reshape(bad_pairs,[bad_pairs.size]))))
    return bad_ixs

def derive_intermediate_points(mesh, vert_ixs, weights_kxn):
    # Returns a mesh with no faces uvs or mat (so not really a mesh), but with vert pos, color, uvs, etc
    # that take from k vert_ixs and have n verts.
    # Weights is kxn.
    out = dict()
    for k in ['verts', 'colors']:
        if k in mesh:
            out[k] = np.einsum('ik,kj->ij',mesh['verts'][:,vert_ixs],weights_kxn)
    return out

################# Mesh -> Mesh [modification fns] ################

def sort_faces(mesh):
    # Sorts the faces, by the lowest ix, and then by the ixs counter clockwise to said ix.
    # Then, within each face, it puts the lowest vert ix first.
    mesh = mesh.copy()

    faces = mesh['faces']
    nVert = np.max(faces) # Striding.
    min_face = np.min(faces,axis=0)
    min_ix = np.argmin(faces,axis=0)
    ar = np.arange(faces.shape[1])
    face_ccw = faces[(min_ix+1)%3,ar]; face_ccw2 = faces[(min_ix+2)%3,ar]
    score = (min_face*nVert*nVert + face_ccw*nVert + face_ccw2 +0.5).astype(np.int64)
    order = np.argsort(score)

    mesh['faces'] = mesh['faces'][:,order]
    if 'uvs' in mesh:
        mesh['uvs'] = mesh['uvs'][:,order,:,:,:]
    if 'mat' in mesh:
        mesh['mat'] = mesh['mat'][order]

    new_min_face = min_face[order]
    put1first = new_min_face == mesh['faces'][1,:]
    put2first = new_min_face == mesh['faces'][2,:]

    faces1 = np.copy(mesh['faces'])
    for o in range(3):
        faces1[o,put1first] = mesh['faces'][(o+1)%3,put1first]
        faces1[o,put2first] = mesh['faces'][(o+2)%3,put2first]
    mesh['faces'] = faces1
    if 'uvs' in mesh:
        uvs1 = np.copy(mesh['uvs'])
        for o in range(3):
            uvs1[o,put1first] = mesh['uvs'][(o+1)%3,put1first,:,:]
            uvs1[o,put2first] = mesh['uvs'][(o+2)%3,put2first,:,:]
    return mesh

@numba.jit(nopython=True)
def _flip_edge_core(faces, uvs, edges_2xk_sorted0, alter_uvs):
    # In-place modification of faces and uvs.
    # TODO: faster than this O(n^2) method (loop through faces and use binary search to find the edge).
    nFace = faces.shape[1]
    nEdge = edges_2xk_sorted0.shape[1]
    tmp_uvs4 = np.zeros((4, 2,uvs.shape[3])) # Average uvs, ccw starting with edge0. Used if alter_uvs is True.
    # Note: uvs is [3,nFace,2,k]
    for i in range(nEdge):
        e0 = edges_2xk_sorted0[0,i]; e1 = edges_2xk_sorted0[1,i]
        # Ixs go counter clockwise from e0:
        q0 = e0; q1 = -1; q2 = e1; q3 = -1; # ccw starting with edge0.
        f0 = -1; f1 = -1; # Face ixs. f0 is anterograde, f1 is retrograde.
        tmp_uvs4 = 0*tmp_uvs4 # Reset to zero.
        for j in range(nFace):
            for o in range(3):
                v0 = faces[(o+0)%3,j]; v1 = faces[(o+1)%3,j]; v2 = faces[(o+2)%3,j]
                uv0 = uvs[(o+0)%3, j, :, :]; uv1 = uvs[(o+1)%3, j, :, :]; uv2 = uvs[(o+2)%3, j, :, :]
                if v0==e0 and v1==e1: # Anterograde.
                    if f0>-1:
                        raise Exception('Multible faces on (anterograde) half edge')
                    f0 = j; q3 = v2
                    tmp_uvs4[0,:,:] = tmp_uvs4[0,:,:] + uv0
                    tmp_uvs4[2,:,:] = tmp_uvs4[2,:,:] + uv1
                    tmp_uvs4[3,:,:] = tmp_uvs4[3,:,:] + uv2
                elif v0==e1 and v1==e0: # Retrograde.
                    if f1>-1:
                        raise Exception('Multible faces on (retrograde) half edge')
                    f1 = j; q1 = v2
                    tmp_uvs4[2,:,:] = tmp_uvs4[2,:,:] + uv0
                    tmp_uvs4[0,:,:] = tmp_uvs4[0,:,:] + uv1
                    tmp_uvs4[1,:,:] = tmp_uvs4[1,:,:] + uv2
        if f0<0 or f1<0:
            raise Exception('The edge is not shared by two faces.')
        tmp_uvs4[0,:,:] = tmp_uvs4[0,:,:]*0.5; tmp_uvs4[2,:,:] = tmp_uvs4[2,:,:]*0.5 #average of two verts.

        # The actual switch itself:
        faces[0,f0] = q0; faces[1,f0] = q1; faces[2,f0] = q3;
        faces[0,f1] = q2; faces[1,f1] = q3; faces[2,f1] = q1;
        if alter_uvs:
            uvs[0,f0,:,:] = tmp_uvs4[0,:,:]; uvs[1,f0,:,:] = tmp_uvs4[1,:,:]; uvs[2,f0,:,:] = tmp_uvs4[3,:,:]
            uvs[0,f1,:,:] = tmp_uvs4[2,:,:]; uvs[1,f1,:,:] = tmp_uvs4[3,:,:]; uvs[2,f1,:,:] = tmp_uvs4[1,:,:]

def flip_edges(mesh, edges_2xk, try_to_preserve_textures=True):
    # Flips edges within their parallelagrams one at a time. Edges are pairs of vert ixs.
    # Only works on watertight edges and edges are flipped one by one, but can break watertightness.
    # try_to_preserve_textures does it's best to preserve UVs.
    #  By averaging the uvs of the two triangles that merge at en edge.
    mesh = mesh.copy()
    mesh['faces'] = np.copy(mesh['faces'])
    mesh['uvs'] = np.copy(mesh['uvs'])

    edges_2xk = np.asarray(edges_2xk)
    ixs = np.argsort(edges_2xk[0,:])
    edges_2xk_sorted0 = edges_2xk[:,ixs]
    _flip_edge_core(mesh['faces'], mesh['uvs'], edges_2xk_sorted0, try_to_preserve_textures)

    return mesh

@numba.jit(nopython=True)
def _shift_faces_core(faces, deleted_vert_ixs, nVert):
    # Some faces are removed.
    faces = np.copy(faces)
    is_delete = keep = np.zeros(nVert); is_delete[deleted_vert_ixs] = 1
    shifts = np.cumsum(is_delete)
    nFace = faces.shape[1]
    delete_face = np.zeros(faces.shape[1])
    for i in range(nFace):
        f012 = faces[:,i]
        if (is_delete[f012[0]] > 0.5) or (is_delete[f012[1]] > 0.5) or (is_delete[f012[2]] > 0.5):
            delete_face[i] = 1
        else:
            for o in range(3):
                faces[o,i] = f012[o]-shifts[f012[o]]
    return faces[:,delete_face<0.5]

def delete_verts(mesh, deleted_ixs):
    # Faces containing said verts will also be deleted.
    mesh = mesh.copy()
    nVert = mesh['verts'].shape[1]
    keep = np.ones(nVert); keep[deleted_ixs] = 0

    mesh['verts'] = mesh['verts'][:,keep>0.5]
    if 'colors' in mesh:
        mesh['colors'] = mesh['colors'][:,keep>0.5]
    mesh['faces'] = _shift_faces_core(mesh['faces'], np.asarray(deleted_ixs), nVert)
    return mesh

def remove_redundant_verts(mesh):
    # Verts with no faces attached.
    if np.max(mesh['faces']) > mesh['verts'].shape[1]:
        raise Exception('Mesh faces point to non-existant verts.')
    facesu = np.reshape(mesh['faces'],mesh['faces'].size)
    uses = _accum1D(facesu,np.ones_like(facesu))
    delete = np.nonzero(uses<0.5)[0]
    if delete.size>0:
        return delete_verts(mesh, delete)
    else:
        return mesh

def delete_faces(mesh, deleted_ixs, remove_redundant_vertices=True):
    mesh = mesh.copy(); nFace = mesh['faces'].shape[1]
    keep = np.ones(nFace); keep[deleted_ixs] = 0
    mesh['faces'] = mesh['faces'][:,keep>0.5]
    if 'mat' in mesh:
        mesh['mat'] = mesh['mat'][keep>0.5]
    if 'uvs' in mesh:
        if mesh['uvs'].shape[1] != nFace:
            raise Exception('Uvs does not match number of faces.')
        mesh['uvs'] = mesh['uvs'][:,keep>0.5,:,:]
    if remove_redundant_vertices:
        return remove_redundant_verts(mesh)
    return mesh

def remove_redundant_faces(mesh, distinguish_normals=True):
    # Faces which share verts.
    # OR: faces that are degenerate.

    f0 = mesh['faces'][0,:]; f1 = mesh['faces'][1,:]; f2 = mesh['faces'][2,:];

    non_degenerate = np.abs((f0-f1)*(f0-f2)*(f0-f2)) > 0.5

    # Use a vanilla python dictionary, not sure how to numba a high performance hashmap.
    nFace = f0.size
    used_up = set()
    redundant = np.zeros(nFace,dtype=np.int64)
    for i in range(nFace):
        txts = []
        txts.append(str(f0[i])+'_'+str(f1[i])+'_'+str(f2[i]))
        txts.append(str(f1[i])+'_'+str(f2[i])+'_'+str(f0[i]))
        txts.append(str(f2[i])+'_'+str(f0[i])+'_'+str(f1[i]))
        if not distinguish_normals:
            txts.append(str(f1[i])+'_'+str(f0[i])+'_'+str(f2[i]))
            txts.append(str(f2[i])+'_'+str(f1[i])+'_'+str(f0[i]))
            txts.append(str(f0[i])+'_'+str(f2[i])+'_'+str(f1[i]))

        for t in txts:
            if t in used_up:
                redundant[i] = 1
            else:
                used_up.add(t)

    delete_these = redundant+(non_degenerate<0.5) > 0.5
    deleted_ixs = np.nonzero(delete_these)[0]

    return delete_faces(mesh, deleted_ixs, remove_redundant_vertices=False)

def meshcat(meshes):
    # Combines the meshes.
    mesh_out = dict()

    if len(meshes) == 0:
        raise Exception('Empty mesh list.') # Or we could return an empty mesh instead?

    # Shift the face ixs:
    facess = []; ix_shift = 0
    for m in meshes:
        facess.append(m['faces']+ix_shift)
        ix_shift = ix_shift + m['verts'].shape[1]
    mesh_out['faces'] = np.concatenate(facess, axis=1)

    # But most things are simple concat:
    for k in ['verts','uvs','colors','mat']:
        example = None
        for m in meshes:
            if k in m:
                example = m[k]
        if example is not None: # Default values.
            valss = []; eshape = list(example.shape)
            for m in meshes:
                if k in m:
                    valss.append(m[k])
                else:
                    if k == 'verts' or k == 'colors':
                        eshape[1] = m['verts'].shape[1]
                    else:
                        eshape[1] = m['faces'].shape[1]
                    v = np.zeros(eshape, dtype=np.float64)
                    if k=='uvs': # Special default uvs. Note: some fns require uvs.
                        v[0,:,0,:] = 0.0; v[0,:,1,:] = 1.0
                        v[1,:,0,:] = 1.0; v[1,:,1,:] = 1.0
                        v[2,:,0,:] = 1.0; v[2,:,1,:] = 2.0
                    if k=='colors' and eshape[0] == 4:
                        v[3,:] = 0.5
                    valss.append(v)
            mesh_out[k] = np.concatenate(valss,axis=1)

    return mesh_out

#################### Mesh + tweak -> nonmesh [Heuristic functions that don't return a mesh] #######################

def merge_nearby_points(points3xk, threshold=0.001):
    # Returns [points, old2new_ix].
    TODO

def poke_triangle(triangle_3columns, points_3xk, triangle_uvs, is_edge12, is_edge02, is_edge01, convex_tweak = 0.0001, edge_assert = 1.0, check_edges=True):
    # Adds points to the triangle, returning the faces. The corners of the triangle are the first 3 points (k+3 points total).
    # Uses scipy.spatial.Delaunay, but modified so that is_point_on_edge points are on the edge.
    # The scipy algorythim makies a convex hull, so make a parabolic is_point_on_edge to put all them on the edge.
    # If not None, triangle_uvs is [3,"2",k].
    coords0 = coregeom.barycentric(triangle_3columns, points_3xk) # [4, k]

    if edge_assert<1: # Edge-marked points should be close to the corresponding edge.
        bad_points = is_edge12*(coords0[0,:]>edge_assert) + is_edge02*(coords0[1,:]>edge_assert) + is_edge01*(coords0[2,:]>edge_assert)
        if np.sum(bad_points)> 0:
            raise Exception('Edge error assert: Points marked as edge are not near said edge. This error will not be thrown if edge_assert is set to >=1.')

    coords_with_corners0 = np.concatenate([coregeom.barycentric(triangle_3columns,triangle_3columns), coords0],axis=1)

    non_edge = is_edge12 + is_edge02 + is_edge01 < 0.5

    def normalize_coords(coords):
        n = 1e-100+np.sum(coords[0:3,:],axis=0)
        coords = coords/(np.expand_dims(n,axis=0))
        return coords

    # Constrain all points to the triangle (usually moves points very little or not at all):
    # We add 2*convex_tweak so that they stay on the triangle.
    coords = normalize_coords(np.maximum(2.0*convex_tweak,np.minimum(coords0,1.0-2.0*convex_tweak)))

    # Project edge points to the actual edge:
    coords[0,is_edge12>0.5] = 0; coords[1,is_edge02>0.5] = 0; coords[2,is_edge01>0.5] = 0;
    coords = normalize_coords(coords)

    # Add a small convexity to ensure the triangulation respects edge-ness:
    coords[0,is_edge12>0.5] = (coords[1,is_edge12>0.5]*coords[1,is_edge12>0.5]+coords[2,is_edge12>0.5]*coords[2,is_edge12>0.5]-2)*convex_tweak
    coords[1,is_edge02>0.5] = (coords[0,is_edge02>0.5]*coords[0,is_edge02>0.5]+coords[2,is_edge02>0.5]*coords[2,is_edge02>0.5]-2)*convex_tweak
    coords[2,is_edge01>0.5] = (coords[0,is_edge01>0.5]*coords[0,is_edge01>0.5]+coords[1,is_edge01>0.5]*coords[1,is_edge01>0.5]-2)*convex_tweak
    coords = normalize_coords(coords) # Normalization is still necessary for triangles far from the origin.

    coords_with_corners = np.concatenate([coregeom.barycentric(triangle_3columns,triangle_3columns), coords],axis=1)
    points_plus_corners = coregeom.unbarycentric(triangle_3columns, coords_with_corners)

    c0 = triangle_3columns[:,0]; c1 = triangle_3columns[:,1]; c2 = triangle_3columns[:,2]; plane_n = np.cross(c1-c0,c2-c0)
    #Sign matters to ensure that a ccw triangle stays ccw.
    plane_u = (c1-c0)/np.linalg.norm(c1-c0); plane_v = -np.cross(c1-c0,plane_n)/np.linalg.norm(np.cross(c1-c0,plane_n))
    points2d = np.zeros([2,points_plus_corners.shape[1]])
    points2d[0,:] = np.einsum('j,ji->i',plane_u,points_plus_corners)
    points2d[1,:] = np.einsum('j,ji->i',plane_v,points_plus_corners)

    tri_obj = scipy.spatial.Delaunay(np.transpose(points2d), furthest_site=False, incremental=False, qhull_options=None)
    faces =  np.transpose(tri_obj.simplices)

    # Quality check on the Delauny boundary matching to what we want, which may break if the points are in unexpected places.
    # Following an edge around should get us to.
    if check_edges:
        halfe_mat = get_halfedge_matrix({'faces':faces});
        delauny_halfe = (halfe_mat-np.transpose(halfe_mat) > 0.5).astype(np.int64) # [to, from] only includes the edges.
        n_edges3 = [np.sum([is_edge01, is_edge12, is_edge02][o]) for o in range(3)]
        n_edge_total = np.sum(n_edges3)
        location_ix_gold = np.zeros(n_edge_total+3, dtype=np.int64); e_ix = 0; ix = 0 # 0 and counts up to 6.
        for o in range(3):
            location_ix_gold[ix] = e_ix
            e_ix = e_ix+1; ix = ix+1
            n_edg = n_edges3[o]
            location_ix_gold[ix:ix+n_edg] = e_ix
            e_ix = e_ix+1; ix = ix+n_edg

        vec = np.zeros(points_3xk.shape[1]+3,dtype=np.int64); vec[0] = 1
        ixs_green = np.zeros(n_edge_total+3, dtype=np.int64)
        for i in range(n_edge_total+3):
            nz = np.nonzero(vec)[0]
            if nz.size != 1:
                raise Exception('No single edge-loop found.')
            ixs_green[i] = nz[0]
            vec = np.matmul(delauny_halfe, vec)
        if vec[0] != 1:
            raise Exception('The edge-loop has the wrong number of verts.')
        location_ix_green = np.zeros(n_edge_total+3, dtype=np.int64)
        for i in range(n_edge_total+3):
            ix = ixs_green[i]
            if ix==0:
                es = 0
            elif ix==1:
                es = 2
            elif ix==2:
                es = 4
            elif is_edge12[ix-3]:
                es = 3
            elif is_edge02[ix-3]:
                es = 5
            elif is_edge01[ix-3]:
                es = 1
            else:
                es = -1 # error condition.
            location_ix_green[i] = es

        #print('Delauny faces:\n',faces)
        #print('Delauny halfe mat:\n',delauny_halfe)
        #print('Requested Edges:', [is_edge12, is_edge02, is_edge01])
        #print('v ixs green:',ixs_green)
        #print('loc ixs:', location_ix_gold, location_ix_green)

        if np.sum(np.abs(location_ix_gold-location_ix_green))>0:
            raise Exception('The assigned edges did not end up on the (correct) edge or the triangle corners ended up buried. Warning: this error cannot catch all bad edge cases.')

    # Interpolate the uvs:
    if triangle_uvs is not None:
        t3 = coords_with_corners0[0:3,:] #[3, nVert+3]
        vert_uvs = np.einsum('ti,tjk->ijk', t3, triangle_uvs) #[3, nVert+3],[3,"2", nMat] => [nVert+3, "2", nMat]
        out_uvs = np.zeros([3, faces.shape[1], triangle_uvs.shape[1], triangle_uvs.shape[2]]) #[3,nFace,"2",nMat]
        for o in range(3):
            out_uvs[o,:,:,:] = vert_uvs[faces[o,:],:,:]
    else:
        out_uvs = None

    return faces, out_uvs

########################## Mesh + tweak -> mesh [Heuristic modifcation fns] #########################

def flip_long_edges(mesh, cotan_ratio_treshold = 3.0, max_unflatness_final_radians = 1.0, max_unflatness_add_radians = 0.25):
    # Standard clean-a-mesh at your service!
    # cotan_ratio_threshold = only flip if it helps make angles be not near 0 or 180.
    # max_unflatness_addtan = Unflatness can get to 180 degrees.
    TODO

def glue_points(mesh, points_3xk, relthreshold=0.0001, project_points=False):
    # Adds points to the mesh, in the face of edge or triangles, "poking" faces or edges.
    # Interpolates UV and preserves face orientation.
    # Does not add points that land ontop of points or duplicated points with eachother.
    # Returns the modified mesh.
    nFace = mesh['faces'].shape[1]
    nVert = mesh['verts'].shape[1]
    nProj = points_3xk.shape[1]

    # Project the points to the mesh to determine which face is closest (even if we don't project points):
    proj_points, home_face_ixs = project_to_mesh(mesh, points_3xk)
    if project_points:
        points_3xk = proj_points
    pointsix_to_face = [[] for _ in range(nFace)]
    for i in range(nProj):
        pointsix_to_face[home_face_ixs[i]].append(i)

    # Accumilate various arrays.
    points_center = [[] for _ in range(nFace)] #[nFace][jagged points]. Points that are in the body of the face.
    points_edge = [[[],[],[]] for _ in range(nFace)] #[nFace][12/02/01][jagged points]. Points on each edge. Includes neighbors.
    points_corner = [[[],[],[]] for _ in range(nFace)] #[nFace][0/1/2][jagged points, corner does not include neighbors].
    fei2fei = dict() #Corresponding points (edges only), str(face_ix _ edge_ix _ ix within edge)=>[face_ix1, edge-ix1, ix within edge1]
    adj_facess = get_face_adj(mesh, True)
    def fei2str(f,e,i):
        return str(f)+'_'+str(e)+'_'+str(i)

    for i in range(nFace):
        triangle_3columns = mesh['verts'][:,mesh['faces'][:,i]]
        ptsHere = pointsix_to_face[i]; nHere = len(ptsHere)
        if nHere>0:
            points_to_this_face = proj_points[:, ptsHere] #[3, nToThisFace]
            coords = coregeom.barycentric(triangle_3columns, points_to_this_face)
            is_edge3 = [((coords[o,:] >= -relthreshold) & (coords[o,:] <= relthreshold)) for o in range(3)] #[3][nToThisFace]
            is_edge = is_edge3[0]+is_edge3[1]+is_edge3[2]
            is_corner3 = [(is_edge3[1]*is_edge3[2])>0.5, (is_edge3[0]*is_edge3[2])>0.5, (is_edge3[0]*is_edge3[1])>0.5]
            is_corner = is_corner3[0]+is_corner3[1]+is_corner3[2]
            # Populate center and corner arrays:
            for j in range(nHere):
                if is_edge[j]<0.5:
                    points_center[i].append(ptsHere[j])
                elif is_corner[j]>0.5:
                    for o in range(3):
                        if is_corner3[o][j]:
                            points_corner[i][o].append(ptsHere[j])

            # There are three adjacent faces for watertight meshes (if not watertight it doesn't really matter)
            # Find these faces.
            adj_faces = adj_facess[i]
            three_faces = [-1 for _ in range(3)] # adj to edges [1-2, 0-2, 0-1] of this face.
            three_faces_otheredges = [-1 for _ in range(3)] # Edge 1-2, 0-2, or 0-1 on the adj face.
            f012 = mesh['faces'][:,i]
            for adj in adj_faces: # Populate three_faces and three_faces_edges.
                f012_ = mesh['faces'][:,adj]
                pairs = [[f012[1],f012[2]], [f012[0],f012[2]], [f012[0],f012[1]]]
                pairs_ = [[f012_[1],f012_[2]], [f012_[0],f012_[2]], [f012_[0],f012_[1]]]
                for o in range(3):
                    pair = pairs[o]
                    for o_ in range(3):
                        pair_ = pairs_[o_]
                        if (pair_[0] == pair[0] and pair_[1] == pair[1]) or (pair_[0] == pair[1] and pair_[1] == pair[0]):
                            three_faces[o] = adj
                            three_faces_otheredges[o] = o_
            # Populate the edges and populate fei2str:
            for o in range(3): #o = edges of this face.
                edge_away = three_faces_otheredges[o]
                acc_home = points_edge[i][o]
                for j in range(len(ptsHere)):
                    adj_face = three_faces[o]
                    if is_corner[j]<0.5:
                        if is_edge3[o][j]>0.5:
                            if adj_face>-1: # watertight will always have >-1 here.
                                acc_away = points_edge[adj_face][edge_away]
                                if adj_face<i: # Match points to the lower face ix.
                                    fr = i; to = adj_face; fr_edge = o; to_edge = edge_away
                                    fr_kx = len(acc_home); to_kx = len(acc_away)
                                else:
                                    to = i; fr = adj_face; to_edge = o; fr_edge = edge_away
                                    to_kx = len(acc_home); fr_kx = len(acc_away)
                                fei2fei[fei2str(fr, fr_edge, fr_kx)] = [to, to_edge, to_kx]
                                acc_away.append(ptsHere[j]) # append to the home and away arrays after the fei2fei filling.
                            acc_home.append(ptsHere[j])

    # Calculate welded points, which include FEI points as well as corner points.
    ix_final_mesh = mesh['verts'].shape[1] # The new meshes go after the original mesh.
    fei2ix = dict()
    weldings = dict() # ix->ix, on the final mesh.
    for i in range(nFace):
        nCenter = len(points_center[i])
        nEdges = [len(pedge) for pedge in points_edge[i]]
        if nCenter+np.sum(nEdges)>0:
            for o in range(3): # Weld the corners.
                weldings[ix_final_mesh+o] = mesh['faces'][o,i]
            ix_final_mesh = ix_final_mesh + 3 + nCenter # Shift the ix by the triangle corners + central points.
            for o in range(3): # weld the edges.
                for j in range(len(points_edge[i][o])):
                    other_fei = fei2fei.get(fei2str(i,o,j),None)
                    if other_fei is not None:
                        ix_other = fei2ix[fei2str(other_fei[0],other_fei[1],other_fei[2])] # No keyError should be thrown (order of filling fei2ix).
                        weldings[ix_final_mesh] = ix_other
                    fei2ix[fei2str(i,o,j)] = ix_final_mesh
                    ix_final_mesh = ix_final_mesh + 1

    if not project_points: # Make the corners of faces corner points if there are corner points, as corner points otherwise are ignored.
        mesh_verts_moved_corner = np.copy(mesh['verts'])
        for i in range(nFace):
            corners = points_corner[i]
            for o in range(3):
                if len(corners[o]) > 0:
                    pt = points_3xk[:,corners[o][0]]
                    mesh_verts_moved_corner[:,mesh['faces'][o,i]] = pt
        mesh = mesh.copy()
        mesh['verts'] = mesh_verts_moved_corner

    # Make triangulations for each one:
    new_meshes = list(); removed_faces = list()
    for i in range(nFace):
        triangle_3columns = mesh['verts'][:,mesh['faces'][:,i]]
        ixs_center = points_center[i] #[n pts this face]
        ixs_edges = points_edge[i] #[12/02/01][n pts this face]
        if len(ixs_center) + len(ixs_edges[0]) + len(ixs_edges[1]) + len(ixs_edges[2]) > 0:
            points = np.concatenate([points_3xk[:,ixs_center], points_3xk[:,ixs_edges[0]], points_3xk[:,ixs_edges[1]], points_3xk[:,ixs_edges[2]]], axis=1)
            edge = np.concatenate([-1*np.ones(len(ixs_center),dtype=np.int64),
                                   np.zeros(len(ixs_edges[0]),dtype=np.int64),
                                   np.ones(len(ixs_edges[1]),dtype=np.int64),
                                   2*np.ones(len(ixs_edges[2]),dtype=np.int64)],axis=0)
            points_plus_corners = np.concatenate([triangle_3columns, points],axis=1)
            new_faces, new_uvs = poke_triangle(triangle_3columns, points, mesh['uvs'][:,i,:,:], edge==0, edge==1, edge==2, edge_assert=0.025)
            weights = coregeom.barycentric(triangle_3columns, points) # We are making quite a few redundent calls to this fns.
            new_mesh = derive_intermediate_points(mesh, mesh['faces'][:,i], weights[0:3,:])
            new_mesh['verts'] = points_plus_corners # Overwrite the position.
            new_mesh['faces'] = new_faces
            new_mesh['uvs'] = new_uvs
            if 'mat' in mesh:
                new_mesh['mat'] = mesh['mat'][i]*np.ones(1,dtype=np.int64)
            new_meshes.append(new_mesh)
            removed_faces.append(i) # Replace the one triangle with the multible triangles.

    #for i in range(nFace):
    #    ixs_center = points_center[i]; ixs_edges = points_edge[i]
    #    if len(ixs_center) + len(ixs_edges[0]) + len(ixs_edges[1]) + len(ixs_edges[2]) > 0:
    #        print('Poked Face:',i,[ixs_center, '|', ixs_edges[0],ixs_edges[1],ixs_edges[2]])
    #print('fei2fei:',fei2fei, 'fei2ix:',fei2ix)
    #print('Weldings is:', weldings)
    #for nm in new_meshes:
    #    print('New mesh vert size:', nm['verts'].shape, 'New mesh faces:\n', nm['faces'])

    if len(new_meshes)>0: # Sometimes, all points hit the corners and no structural changes are hit.
        mesh = delete_faces(mesh, removed_faces, remove_redundant_vertices=False)
        combined_add_mesh = meshcat(new_meshes)
        mesh = meshcat([mesh, combined_add_mesh])

    # welding_step:
    for i in range(mesh['faces'].shape[1]):
        for o in range(3):
            mesh['faces'][o][i] = weldings.get(mesh['faces'][o][i],mesh['faces'][o][i])
    return remove_redundant_verts(mesh) # Welded-away verts = redundent verts.
