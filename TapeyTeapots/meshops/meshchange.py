# Mesh operations which can drag points along.
# Resolution sets how "good" the operations are for operations that rely on making approximations.
from . import trimesh, coregeom
from . import quat34 as qmv

class Meshxform(): # The ancestor class does nothing.

    ##### Always overload these:
    def __init__(self, geom_3xn=None, m44=None, new_geom_3xn=None, opts=None):
        self.geom_3xn = geom_3xn
        if m44 is None:
            m44 = np.identity([4,4])
        self.m44 = m44
        self.new_geom_3xn = new_geom_3xn
        self.opts = opts

    def xform_points(self, geom_3xn, resolution=32.0):
        # Transforms points forward, use for rendering.
        # Resolution sets how "good" the xform is for xforms 
        return geom_3xn

    def inv_xform_points(self, geom_3xn, resolution=32.0):
        # Used to figure out "where on the untransformed did we click", etc.
        # Inverse is APPROXIMATE! There are cases where it gets ill-defined, which should be mitigated.
        return geom_3xn

    ##### Sometimes overload these:
    def new_points(self, resolution=32.0):
        # What new points are needed to accuratly represent the mesh operation?
        # For example, an extrude operation needs the geometry defined.
        return np.zeros(shape=[3,0])

    def apply_to_mesh(self, mesh, resolution=32.0, relthreshold=0.0001):
        # Default how we apply it to a mesh, can always be changed at need.
        add_these_points_3xk = self.new_geometry(resolution=resolution)
        mesh1 = trimesh.add_points(mesh, add_these_points_3xk, relthreshold=relthreshold)
        mesh1['verts'] = self.xform_points(mesh1['verts'], resolution=resolution)
        return mesh1

    def get_constructor_arguments():
        # The arguments that can be passed into the constructor.
        # Used for generating code (note: code generation in Turing Create is very simplistic,m no hardcore metaprogramming!)
        return self.geom_3xn, self.m44, self.new_geom_3xn, self.opts

    def override_get_mesh_change(self, wigit):
        # Allows overriding the default behavior, i.e. for unusual curvy xforms.
        return None # None means do not override.

class Extrusion(Meshxform): # Pick a loop of points, pick a direction. Extrude.
    # TODO: use m44 instead of only displacement.
    def __init__(self, geom_3xn=None, m44=None, new_geom_3xn=None, opts=None):
        super().__init__()
        self.geom_3xn = geom_3xn
        self.relative_padding = 0.001 #The user can override these "hidden" options in the python script.
        self.displacement = np.asarray(opts['displacement'])

    def get_wigit(self):
        out = arrows.MoveArrows()
        mean_position = np.mean(edgeloop_3xk,1)
        m44 = qmv.m44_from_v(mean_position+displacement)
        TODO

    def _xform_core(edgeloop_3xk, geom_3xn, pad, delta, sign=1.0, resolution=32.0):
        dot_range = np.einsum('ui,u->i',edgeloop_3xk, delta)
        min_dot_allowed = np.min(dot_range)-pad
        max_dot_allowed = np.max(dot_range)+pad+1.0 # Add 1.0 so we extrude everything in range.

        plane_origin, plane_normal = regression_plane(edgeloop_3xk)
        n = np.shape(geom_3xn)[2]
        geom_skew_proj = coregeom.line_plane_intersection(plane_origin, plane_normal, query_origin_3xn, np.tile(np.expand_dims(delta,1), [1,n]))

        is_inside = coregeom.is_inside_loop(edgeloop, geom_skew_proj)
        dot_scores = np.einsum('ui,u->i',geom_3xn, delta)
        apply_xform = (is_inside>0.5)*(dot_scores>=min_dot)*(dot_scores<=max_dot)

        geom_3xn = np.copy(geom_3xn)
        geom_3xn[:,apply_xform] = geom_3xn[:,apply_xform]+delta*sign

        return geom_3xn

    def xform_points(self, geom_3xn, resolution=32.0):
        return _xform_core(self.geom_3xn, geom_3xn, self.relative_padding, self.displacement, sign=1.0, resolution=resolution)

    def inv_xform_points(self, geom_3xn, resolution=32.0):
        return _xform_core(self.geom_3xn, geom_3xn, self.relative_padding, self.displacement, sign=-1.0, resolution=resolution)

    def apply_to_mesh(self, mesh, resolution=32.0, relthreshold=0.0001):
        # Make relthreshold smaller b/c the extrude is sensitive.
        return super().apply_to_mesh(self, mesh, resolution=32.0, relthreshold=0.01*relthreshold)

    def new_points(self, resolution=32.0):
        # The make two rings of points, slightly inside and slightly outside the loop.
        relative_eps = 0.001
        boundary_pts = self.opts['edgeloop']
        _, plane_normal = coregeom.regression_plane(boundary_pts)
        boundary_pts_rollA = np.roll(boundary_pts, 1, axis=-1)
        boundary_pts_rollB = np.roll(boundary_pts, 1, axis=1)
        skipped_alongs = boundary_pts_rollB-boundary_pts_rollA
        deltas = relative_eps*np.cross(skipped_alongs, np.expand_dims(plane_normal,1),axisa=0,axisb=0,axisc=0)
        return np.concatenate([boundary_pts-deltas, boundary_pts+deltas], axis=0)

    def override_get_mesh_change(self, wigit):
        # TODO: use mat44 instead of setting opts['displacement'].
        default_meshchange = wigit.get_meshchange(allow_meshchange_override=False)
        _, displacement = qvm.m44TOm33v(default_meshchange.m44)
        return Extrusion(geom_3xn=self.geom_3xn, m44=None, new_geom_3xn=None, opts={'displacement':displacement})

    def get_constructor_arguments():
        return self.geom_3xn, None, None, {'displacement': self.opts['displacement']}

class Geomdrag(Meshxform): # Drag geometry. No smoothness.
    def __init__(self, geom_3xn=None, m44=None, new_geom_3xn=None, opts=None):
        super().__init__()
        self.geom_3xn = geom_3xn
        self.faces = opts['faces']
        self.new_geom_3xn = new_geom_3xn

    def _xform_core(verts, faces, new_verts, query_3xn):
        # For now simply do nearest (most in-face) triangle.
        # TODO: Later on we will have seperate systems for convex edges and verts.
        tri_ixs, barycentrics = trimesh.closest_barycentric({'verts':verts, 'faces':faces}, query_3xn)
        # Get the new points:
        output_verts = query_3xn
        TODO

    def xform_points(self, geom_3xn, resolution=32.0):
        return _xform_core(self.geom_3xn, self.faces, self.new_geom_3xn, geom_3xn)

    def inv_xform_points(self, geom_3xn, resolution=32.0):
        # Quite a crude approximation, but one that holds true if points are inside a triangle.
        return _xform_core(self.new_geom_3xn, self.faces, self.geom_3xn, geom_3xn)

    def get_constructor_arguments():
        return self.geom_3xn, None, self.new_geom_3xn, {'faces': self.opts['faces']}

class Globalxform(Meshxform):
    def __init__(self, geom_3xn=None, m44=None, new_geom_3xn=None, opts=None):
        super().__init__()
        if m44 is None:
            m44 = np.identity([4,4])
        self.m44 = np.copy(m44)

    # Global matrix 4x4.
    def xform_points(self, geom_3xn, resolution=32.0):
        return qmv.m44v(self.m44, geom_3xn)

    def inv_xform_points(self, geom_3xn, resolution=32.0):
        return qmv.m44v(np.linalg.inv(self.m44), geom_3xn)

    def get_constructor_arguments():
        return None, self.m44, None, {}

