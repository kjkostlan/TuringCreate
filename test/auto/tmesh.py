# Tests mesh operations.
import numpy as np
from TapeyTeapots.meshops import primitives, trimesh, meshchange, quat34
from . import tgeom, tquat43

def test_basics():
    # Tests primatives and the simplest mesh operations.
    # It is often easier to two test things against eachother.
    np.random.seed(7326)
    tests = []

    # Accum 2d test vs brute force:
    weights = [0.21,0.33,0.42,0.111]; ix = [0,1,0,2]; jx = [1,0,1,0]
    m_gold = np.zeros([np.max(ix)+1,np.max(jx)+1])
    for i in range(len(weights)):
        m_gold[ix[i],jx[i]] = m_gold[ix[i],jx[i]] + weights[i]
    m_green = trimesh._accum2D(np.asarray(ix), np.asarray(jx), np.asarray(weights))
    tests.append(tgeom.approx_eq(m_gold, m_green))

    # Tests of volume of a cube, and scaling by random 4x4 matrix:
    cube_mesh = primitives.cube()
    volume = trimesh.volume(cube_mesh, origin=None, normal=None, signed=True)
    tests.append(tgeom.approx_eq(volume, 8.0))
    leak = trimesh.leaky_verts(cube_mesh)
    tests.append(tgeom.approx_eq(leak, []))
    mat44 = tquat43.random_4x4_matrixes(1)[0]
    det = np.linalg.det(mat44[0:3,0:3])
    cube_mesh1 = cube_mesh.copy(); cube_mesh1['verts'] = quat34.m44v(mat44, cube_mesh1['verts'])
    volume1 = trimesh.volume(cube_mesh1, origin=None, normal=None, signed=True)
    tests.append(tgeom.approx_eq(volume*det, volume1))

    # Watertight on all other primatives:
    meshes = [primitives.sphere(), primitives.cylinder(), primitives.cone(), primitives.torus()]
    for mesh in meshes:
        leak = trimesh.leaky_verts(mesh)
        tests.append(np.asarray(leak).size==0)
        tests.append(float(trimesh.volume(mesh, signed=True))>0.001)

    return tgeom.is_all_true(tests)

def test_structural():
    # Structural mesh operations, such as deleting vertexes, etc.
    def vec3TOstr(vec3, hash_tol=0.001): # Hash_tol is a granulrity on the face.
        vec3_quantized = np.astype(vec3/tol, np.int64)
        return str(vec3_quantized)
    def face_centroid(mesh, face_ix):
        v012 = mesh['verts'][:,mesh['faces'][:,face_ix]]
        return np.mean(v012,axis=1)
    def face_normal(mesh, face_ix):
        v012 = mesh['verts'][:,mesh['faces'][:,face_ix]]
        v01 = v012[:,1]-v012[:,0]; v02 = v012[:,2]-v012[:,0]
        vn = np.cross(v01,v02)
        return vn/np.linalg.norm(vn)
    def face_area(mesh, face_ix):
        v012 = mesh['verts'][:,mesh['faces'][:,face_ix]]
        v01 = v012[:,1]-v012[:,0]; v02 = v012[:,2]-v012[:,0]
        vn = np.cross(v01,v02)
        return np.sum(vn*vn)*0.5
    def face2str(mesh, face_ix, hash_tol=0.001): # Hashes the face, including it's uvs, to a string.
        v012 = np.reshape(mesh['verts'][:,mesh['faces'][:,face_ix]],9)
        v012_quantized = [int(v/hash_tol) for v in v012]
        return str(v012_quantized)
    np.random.seed(1029)
    tests = []

    # Test get edges:
    sphere_mesh = primitives.sphere(resolution=32)
    edges_2xk = trimesh.get_halfedges(sphere_mesh)
    fulledges_2xk = trimesh.get_edges(sphere_mesh)
    tests.append(bool(edges_2xk.shape[1]==3*sphere_mesh['faces'].shape[1]))
    tests.append(bool(fulledges_2xk.shape[1]==int(1.5*sphere_mesh['faces'].shape[1]+0.5)))

    # Test flip edges:
    cube_mesh = primitives.cube()
    edges_2xk = trimesh.get_halfedges(cube_mesh)

    diagonal_edges = []
    for i in range(edges_2xk.shape[1]):
        e0 = edges_2xk[0,i]; e1 = edges_2xk[1,i]
        v0 = cube_mesh['verts'][:,e0]
        v1 = cube_mesh['verts'][:,e1]
        d = np.linalg.norm(v1-v0)
        if d>2.5 and e0<e1:
            diagonal_edges.append(i)
    tests.append(len(diagonal_edges)==6)
    edges_to_flip = edges_2xk[:,diagonal_edges]
    cube_mesh1 = trimesh.flip_edges(cube_mesh, edges_to_flip)
    tests.append(len(diagonal_edges)==6)
    tests.append(tgeom.approx_eq(trimesh.volume(cube_mesh, signed=True), trimesh.volume(cube_mesh1, signed=True)))

    bad_edges = edges_2xk[:,[0,0]] # The repeat makes the second edge invalid.
    try:
        cube_mesh_bad = trimesh.flip_edges(cube_mesh, bad_edges)
        tests.append(False)
    except:
        pass

    torus_mesh = primitives.torus(resolution=18, girth=0.25)
    edges_2xk = trimesh.get_edges(torus_mesh)
    flip_edge = np.random.random(edges_2xk.shape[1])>0.95 # Too many and will get the "too many edges" error.
    edges_to_flip = edges_2xk[:,flip_edge>0.5]
    n_vert = torus_mesh['verts'].shape[1]
    torus_mesh1 = trimesh.flip_edges(torus_mesh, edges_to_flip)

    nFace = torus_mesh['faces'].shape[1]
    old_area = float(np.sum([face_area(torus_mesh,i) for i in range(nFace)]))
    new_area =  float(np.sum([face_area(torus_mesh1,i) for i in range(nFace)]))
    tests.append((old_area>0.9*new_area) and (old_area<1.1*new_area))
    tests.append((old_area<0.9999*new_area) or (old_area>1.0001*new_area))

    # Test get_face_adj()
    torus_mesh = primitives.torus(resolution=16, girth=0.25)
    face2adj_moore = trimesh.get_face_adj(torus_mesh, edge_only=False)
    face2adj_vonNeumann = trimesh.get_face_adj(torus_mesh, edge_only=True)
    tests.append(bool(len(face2adj_moore[29])==12))
    tests.append(bool(len(face2adj_vonNeumann[13])==3))
    mutualMo = False
    for face in face2adj_moore[37]:
        for face2 in face2adj_moore[face]:
            if face2 == 37:
                mutualMo = True
    mutualVN = False
    for face in face2adj_vonNeumann[31]:
        for face2 in face2adj_vonNeumann[face]:
            if face2 == 31:
                mutualVN = True
    tests.append(mutualMo); tests.append(mutualVN)

    #Test delete verts/faces: Doesn't change non-deleted face centroids or cross-products.
    torus_mesh = primitives.torus(resolution=21, girth=0.125)
    deleted_face_ixs = np.nonzero(np.random.random(torus_mesh['faces'].shape[1])>0.75)[0]
    torus_mesh_f = trimesh.delete_faces(torus_mesh, deleted_face_ixs)
    deleted_vert_ixs = np.nonzero(np.random.random(torus_mesh['verts'].shape[1])>0.875)[0]
    torus_mesh_fv = trimesh.delete_verts(torus_mesh_f, deleted_vert_ixs)
    tests.append(bool(torus_mesh_f['faces'].shape[1] == torus_mesh['faces'].shape[1]-deleted_face_ixs.shape[0]))
    tests.append(bool(torus_mesh_fv['verts'].shape[1] == torus_mesh_f['verts'].shape[1]-deleted_vert_ixs.shape[0]))
    test_pass = False
    for tol in [0.001,0.00037]:
        old_set = set([face2str(torus_mesh, i, hash_tol=0.001) for i in range(torus_mesh['faces'].shape[1])])
        new_set = set([face2str(torus_mesh_fv, i, hash_tol=0.001) for i in range(torus_mesh_fv['faces'].shape[1])])
        if len(new_set-old_set)==0: # All new faces in the old faces.
            test_pass = True
    tests.append(test_pass)

    # Volumes should add. Face centroids preserved.
    torus = primitives.torus(resolution=21, girth=0.125)
    cone = primitives.cone(resolution=7, sharpness=1.0)
    cylinder = primitives.cylinder(resolution=7, tallness=1.0)
    meshes = [torus, cone, cylinder]
    mesh3 = trimesh.meshcat(meshes)
    volume_green = trimesh.volume(mesh3)
    volume_gold = np.sum([trimesh.volume(m) for m in meshes])
    tests.append(tgeom.approx_eq(volume_green,volume_gold))
    tests.append(bool(trimesh.leaky_verts(mesh3).size==0))

    # Remove redundent faces tests with duplicate meshes:
    cube_mesh = primitives.cube(); f = cube_mesh['faces']; u = cube_mesh['uvs']
    extra_cubes_mesh = cube_mesh.copy();
    extra_cubes_mesh['faces'] = np.concatenate([f,f,np.roll(f,1,axis=0),np.roll(f,2,axis=0)],axis=1)
    extra_cubes_mesh['uvs'] = np.concatenate([u,u,u,u],axis=1)
    cube_mesh1 = trimesh.remove_redundant_faces(extra_cubes_mesh, distinguish_normals=True)
    tests.append(tgeom.approx_eq(cube_mesh,cube_mesh1))

    f_flipped = np.stack([f[1,:],f[0,:],f[2,:]],axis=0)
    double_sided_cube_mesh = cube_mesh.copy();
    double_sided_cube_mesh['faces'] = np.concatenate([f,f_flipped],axis=1)
    double_sided_cube_mesh['uvs'] = np.concatenate([u,u],axis=1)
    still2side = trimesh.remove_redundant_faces(double_sided_cube_mesh, distinguish_normals=True)
    singleside = trimesh.remove_redundant_faces(double_sided_cube_mesh, distinguish_normals=False)
    tests.append(bool(still2side['faces'].shape[1]==24))
    tests.append(bool(singleside['faces'].shape[1]==12))

    # remove redundent verts welding test:
    cube_mesh_weld = primitives.cube()
    cube_mesh_weld['faces'][cube_mesh_weld['faces']==6] = 5 # Weld 6 and 5.
    mesh1 = trimesh.remove_redundant_verts(cube_mesh_weld)
    tests.append(bool(mesh1['verts'].shape[1]==7))

    return tgeom.is_all_true(tests)

def test_add_points():
    # This is by far the hardest function in trimesh, do to complex structural changes.
    tests = []; np.random.seed(444)
    cube = primitives.cube() # -1 to 1 all three axes, easy mesh to work with.
    #print(np.random.random())

    # poke_triangle(triangle_3columns, points_3xk, triangle_uvs, is_edge12, is_edge02, is_edge01
    # Warm up with poke triangles.
    def sortf(faces):
        mesh = trimesh.sort_faces({'faces':faces})
        return mesh['faces']

    triangle3 = np.identity(3)
    new_point = np.transpose([[1,1,1]])/3.0 # midpoint.

    # Center poke triangle test, single point:
    faces1, uvs1 = trimesh.poke_triangle(triangle3, new_point, None, np.asarray([0]), np.asarray([0]), np.asarray([0]))
    faces_gold = np.stack([[0,1,3],[0,3,2],[1,2,3]],axis=1)
    tests.append(tgeom.approx_eq(faces_gold,sortf(faces1)))

    # Edge poke triangle tests, single point:
    faces1, uvs1 = trimesh.poke_triangle(triangle3, new_point, None, np.asarray([1]), np.asarray([0]), np.asarray([0]), edge_assert=1.0)
    faces_gold = np.stack([[0,1,3],[0,3,2]],axis=1)
    tests.append(tgeom.approx_eq(faces_gold,sortf(faces1)))
    faces1, uvs1 = trimesh.poke_triangle(triangle3, new_point, None, np.asarray([0]), np.asarray([1]), np.asarray([0]), edge_assert=1.0)
    faces_gold = np.stack([[0,1,3],[1,2,3]],axis=1)
    tests.append(tgeom.approx_eq(faces_gold,sortf(faces1)))
    faces1, uvs1 = trimesh.poke_triangle(triangle3, new_point, None, np.asarray([0]), np.asarray([0]), np.asarray([1]), edge_assert=1.0)
    faces_gold = np.stack([[0,3,2],[1,2,3]],axis=1)
    tests.append(tgeom.approx_eq(faces_gold,sortf(faces1)))

    # 3 edges triangle poke test:
    new_points = np.transpose([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])
    faces_gold = np.stack([[0,5,4],[1,3,5],[2,4,3],[3,4,5]],axis=1)
    faces1, uvs1 = trimesh.poke_triangle(triangle3, new_points, None, np.asarray([1,0,0]), np.asarray([0,1,0]), np.asarray([0,0,1]))
    tests.append(tgeom.approx_eq(faces_gold,sortf(faces1)))

    # Cube mesh, poking one face with one point in the center:
    point_within_face = np.transpose([[0.45,0.25,1.0]]) # 6 squares = 12 triangles.
    mesh1 = trimesh.glue_points(cube, point_within_face, relthreshold=0.0001, project_points=False)
    tests.append(mesh1['faces'].shape[1] == 12+2)
    fc0 = trimesh.get_face_counts(cube); fc1 = trimesh.get_face_counts(mesh1)
    tests.append(len(trimesh.leaky_verts(mesh1))==0)

    # Cube mesh, poking one square face into a square pyramid (so at edge of two triangles):
    h = 0.35; point_within_face = np.transpose([[0,0,1+h]]) # 0.25 above the point.
    mesh1 = trimesh.glue_points(cube, point_within_face, relthreshold=0.0001, project_points=False)
    tests.append(mesh1['faces'].shape[1] == 12+2)
    fc0 = trimesh.get_face_counts(cube); fc1 = trimesh.get_face_counts(mesh1)
    tests.append(np.sum(fc1==4)>np.sum(fc0==4))
    tests.append(len(trimesh.leaky_verts(mesh1))==0)
    volume_gold = 8+1.0/3.0*4*h
    volume_green = trimesh.volume(mesh1, origin=None, normal=None, signed=True)
    tests.append(tgeom.approx_eq(volume_gold, volume_green))

    # Cube mesh, poking a corner.
    corner_point = np.transpose([[3,3,3]]) # Note: [1,1,1] is the corner, but we poke it.
    mesh1 = trimesh.glue_points(cube, corner_point, relthreshold=0.0001, project_points=False)
    tests.append(mesh1['faces'].shape[1] == 12)
    tests.append(trimesh.volume(mesh1)>trimesh.volume(cube)+1.0)
    tests.append(len(trimesh.leaky_verts(mesh1))==0)
    fc0 = trimesh.get_face_counts(cube); fc1 = trimesh.get_face_counts(mesh1)

    # Cube mesh, randomly add points, and set project to true:
    nPts = 64; pts = np.random.randn(3,nPts)
    mesh1 = trimesh.glue_points(cube, pts, relthreshold=0.000001, project_points=True)
    tests.append(len(trimesh.leaky_verts(mesh1))==0)
    tests.append(mesh1['faces'].shape[1] > 12 + nPts)
    tests.append(tgeom.approx_eq(trimesh.volume(mesh1),trimesh.volume(cube)))

    return tgeom.is_all_true(tests)
