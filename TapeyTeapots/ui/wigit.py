# The arrow thing that lets you move in one axis.
# (is there a better word for this?)
import numpy as np
import copy
import TapeyTeapots.meshops.primitives as primitives
import TapeyTeapots.meshops.quat34 as quat34

class Draggable():
    # Objects that can be dragged.
    def __init__(self, m44):
        self.m44 = m44

    def to_mesh(self):
        # Make a mesh that can be seen in the world.
        m = primatives.cube()
        m['verts'] = qmv.m44v(self.m44, m['verts'])
        return m

    def get_hitpoint(self, ray_origin, ray_direction, dig=0.0):
        # Collision detection! Returns None if fail. dig usually rangesf rom 0 to 1.
        inv_m44 = np.linalg.inv(self.m44)
        ray_origin0 = qmv.m44v(inv_m44, ray_origin)
        ray_direction0 = qmv.m44v(inv_m44, ray_direction)
        # Look for a collision with the unit bounding box [-1,-1,-1] to [1,1,1].
        TODO
        return qmv.m44v(self.m44, hitpoint)[:,0]

    def get_new_m44(self, camera44, screenx0, screenx1, screeny0, screeny1):
        # Gets the new m44.
        # x0, x1, y0, y1 range from -1 to 1 across a square screen.
        ray_origin, ray_direction = qmv.cam_ray(camera44, screenx0, screeny0)
        hitpoint0 = get_hitpoint(self, ray_origin, ray_direction, dig=0.5)
        if hitpoint0 is None or (screenx0==screenx1 and screeny0==screeny1):
            return self.m44 # Missed or didn't actually drag, no difference.
        normal = cam_plane_normal(camera44, hitpoint0)[:,0]
        ray_origin1, ray_direction1 = qmv.cam_ray(camera44, screenx1, screeny1)
        hitpoint1 = coregeom.line_plane_intersection(hitpoint0, normal, np.expand_dims(ray_origin1,1), np.expand_dims(ray_direction,1))[:,0]
        new_m44 = np.linalg.matmul(qmv.m44_from_v(hitpoint1-hitpoint0), self.m44)
        return new_m44

class Wigit():

    def __init__(self):
        self.m44 = np.identity([4,4])  # Most will have an m44, but some non-linear ones will not.
        self.draggables = [Draggable(self.m44)]

    def to_mesh(self):
        # Make a mesh that can be seen in the world.
        # The 'verts' of the mesh usually will be moved around by self.m44
        return trimesh.meshcat([d.to_mesh() for d in self.draggables])

    def after_click_and_drag(self, camera44, screenx0, screenx1, screeny0, screeny1):
        # Returns a new wigit object after click and drag is applied.
        # Does not modify the original one because mutation is generally harmful.
        # Should be overloaded by most fns.
        if screenx0==screenx1 and screeny0==screeny1: # No change.
            return self
        new_draggables = []
        new_m44s = []
        for d in draggables:
            new_m44 = d.get_new_m44(camera44, screenx0, screenx1, screeny0, screeny1)
            new_m44s.append(new_m44)
            d1 = copy.copy(d)
            d1.m44 = new_m44
            new_draggables.append(d1)
        self1 = copy.copy(self)
        self1.draggables = new_draggables
        self1.m44 = np.mean(np.stack([d.m44 for d in self.draggables], axis=2), axis=2)
        return self1

    def get_meshchange(self, ident_meshchange, allow_meshchange_override=True): # Used to effect a mesh change.
        # ident_meshchange is the mesh change when we don't really do anything.
        if allow_meshchange_override:
            override_meshchange = override_get_mesh_change(self)
            if override_meshchange is not None:
                return override_meshchange
        out = copy.copy(ident_meshchange)
        out.m44 = self.m44
        if out.geom_3xn is not None and out.new_geom_3xn is not None:
            out.new_geom_3xn = qmv.m44v(out.m44, out.geom_3xn)
        return out

class MoveArrows(Wigit):
    'TODO'

class RotateArrows(Wigit):
    'TODO'

class ScaleArrows(Wigit):
    'TODO'

class ShearArrows(Wigit):
    # This one isn't in blender. Here, planes are attached to arrows.
    'TODO'
