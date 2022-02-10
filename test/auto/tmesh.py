# Tests mesh operations.
from TapeyTeapots.meshops import primitives, trimesh, meshchange
from . import tgeom

def test_basics():
    # Primatives, etc.
    tests = []
    
    cube_mesh = primitives.cube()
    volume = trimesh.volume(cube_mesh, origin=None, normal=None, signed=True)
    tests.append(tgeom.approx_eq(volume, 8.0))
    print('WARNING: need more tests in tmesh/test_basics!')
    return tgeom.is_all_true(tests)

#def test_project():
#    return False

#def extrude():
#    return False
