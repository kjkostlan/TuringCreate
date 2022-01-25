# Operations to work with a mesh.
# Very complex operations may need us to copy some code from a "real" mesh library.

def derive_intermediate_points(mesh1, mesh2, vert_ixs1, vert_ixs2, weights2):
    # Returns a mesh with no faces but with verts that interpolate uvs, locations, etc.
    # These points can then be used in the mesh.
    TODO

def flip_edges(mesh, vertex_pairs_2xk):
    # Returns the mesh.
    TODO

def merge_nearby(mesh, reldist = 0.0001):
    # Returns [mesh, vert_old2new]
    TODO

def flip_long_edges(mesh, cotan_ratio_treshold = 3.0, max_unflatness_final_radians = 1.0, max_unflatness_add_radians = 0.25):
    # Standard clean-a-mesh at your service!
    # cotan_ratio_treshold = only flip if it helps make angles be not near 0 or 180.
    # max_unflatness_addtan = Unflatness can get to 180 degrees.
    TODO

def closest_barycentric(mesh, query_3xn, barycenter_weight_uvw = 4.0, barycenter_weight_z = 1.0):
    # Index of the closest triangle based on barycentric coordinants. 
    # barycenter_weight_uvw, barycenter_weight_z = Index of the closest triangle.
    TODO

def project_points(mesh, points_3xk, barycenter_weight_uvw = 4.0, barycenter_weight_z = 1.0):
    # See add_points for weight doc. Will project points onto the mesh.
    TODO

def add_points(mesh, points_3xk, relthreshold=0.0001, project_points=False,
                barycenter_weight_uvw = 4.0, barycenter_weight_z = 1.0):
    # Returns [mesh, vert_new2old, added_vert_ixs].
    TODO
