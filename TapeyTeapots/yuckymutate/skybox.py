# A nice-looking sky gradient that does not use textures.
# Most of this code is from: https://discourse.panda3d.org/t/color-gradient-scene-background/26946/2
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
import array

def create_gradient(sky_color, ground_color, horizon_color=None):

    vertex_format = GeomVertexFormat()
    array_format = GeomVertexArrayFormat()
    array_format.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
    vertex_format.add_array(array_format)
    array_format = GeomVertexArrayFormat()
    array_format.add_column(InternalName.make("color"), 4, Geom.NT_uint8, Geom.C_color)
    vertex_format.add_array(array_format)
    vertex_format = GeomVertexFormat.register_format(vertex_format)

    vertex_data = GeomVertexData("prism_data", vertex_format, GeomEnums.UH_static)
    vertex_data.unclean_set_num_rows(6)
    # create a simple, horizontal prism;
    # make it very wide to avoid ever seeing its left and right sides;
    # one edge is at the "horizon", while the two other edges are above
    # and a bit behind the camera so they are only visible when looking
    # straight up
    values = array.array("f", [
        -1000., -50., 86.6,
        -1000., 100., 0.,
        -1000., -50., -86.6,
        1000.,-50., 86.6,
        1000., 100., 0.,
        1000., -50., -86.6
    ])
    pos_array = vertex_data.modify_array(0)
    memview = memoryview(pos_array).cast("B").cast("f")
    memview[:] = values

    color1 = tuple(int(round(c * 255)) for c in sky_color)
    color3 = tuple(int(round(c * 255)) for c in ground_color)

    if horizon_color is None:
        color2 = tuple((c1 + c2) // 2 for c1, c2 in zip(color1, color3))
    else:
        color2 = tuple(int(round(c * 255)) for c in horizon_color)

    values = array.array("B", (color1 + color2 + color3) * 2)
    color_array = vertex_data.modify_array(1)
    memview = memoryview(color_array).cast("B")
    memview[:] = values

    tris_prim = GeomTriangles(GeomEnums.UH_static)
    indices = array.array("H", [
        0, 2, 1,  # left triangle; should never be in view
        3, 4, 5,  # right triangle; should never be in view
        0, 4, 3,
        0, 1, 4,
        1, 5, 4,
        1, 2, 5,
        2, 3, 5,
        2, 0, 3
    ])
    tris_array = tris_prim.modify_vertices()
    tris_array.unclean_set_num_rows(24)
    memview = memoryview(tris_array).cast("B").cast("H")
    memview[:] = indices

    geom = Geom(vertex_data)
    geom.add_primitive(tris_prim)
    node = GeomNode("prism")
    node.add_geom(geom)
    # the compass effect can make the node leave its bounds, so make them
    # infinitely large
    node.set_bounds(OmniBoundingVolume())
    prism = NodePath(node)
    prism.set_light_off()
    prism.set_bin("background", 0)
    prism.set_depth_write(False)
    prism.set_depth_test(False)

    return prism

def set_up_gradient_skybox(the_pivot, showbase):
    sky_color = (0, 0, 1., 1.)
    horizon_color = (.5, 0, .5, 1.)  # optional
    ground_color = (0, 1., 0, 1.)
    background_gradient = create_gradient(sky_color, ground_color)#, horizon_color)
    effect = CompassEffect.make(showbase.camera, CompassEffect.P_pos)#+CompassEffect.P_scale) #
    #the_pivot.set_effect(effect) # Applies to everything parented to pivot.
    background_gradient.set_effect(effect)

    background_gradient.reparent_to(the_pivot)
    # now the background model just needs to keep facing the camera (only
    # its heading should correspond to that of the camera; its pitch and
    # roll need to remain unaffected)
    effect = BillboardEffect.make(
        Vec3.up(),
        False,
        True,
        0.,
        NodePath(),
        # make the background model face a point behind the camera
        Point3(0., -10., 0.),
        False
    )
    #setLightOff
    #self.render.set_attrib(LightRampAttrib.makeHdr0())
    background_gradient.set_effect(effect)
    return background_gradient, effect
