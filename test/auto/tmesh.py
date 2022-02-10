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

#def test_project():
#    return False

#def extrude():
#    return False
