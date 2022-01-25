# Move some verts in a mesh. The way coordinants get dragged along is tricky to describe.

import numpy as np
def xform(mesh, vertex_ixs, new_vertex_positions, query_3xn, tangental_drag=True, relative_thickness=0.25):
    # Points get dragged along unless tangental_drag is False.
    # new_vertex_positions is 3x(length of vertex_ixs).
    verts = np.transpose(mesh['verts']) # Transpose so it's 3xn.
    
    # Closest triangle to the coords will be the one most in range barycenter wise:
    closest_triangle_ixs = TODO

    # All face ixs that have changed:
    all_ixs = list(sort(set(closest_triangle_ixs)))
    
    new_query_3xn = np.copy(query_3xn)
    for ix in all_ixs:
        xform

def invxform(mesh, vertex_ixs, new_vertex_positions, query_3xn, tangental_drag=True, relative_thickness=0.25):
    # Undoes xform, but only gaurenteed to be an inverse if the point is near the triangle.
    # TODO: make the inverse more accurate, but it is accurate in the limit of smaller steps.
    new_mesh = TODO
    old_vertex_positions = TODO
    return xform(new_mesh, vertex_ixs, old_vertex_positions, query_3xn, tangental_drag=True, relative_thickness=0.25)
