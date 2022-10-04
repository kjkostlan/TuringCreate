# Builds various shapes (meshes, text, etc).
import numpy as np
from panda3d.core import *
from direct.gui.OnscreenText import OnscreenText

#################### Helper functions #########################
def vert_uvs(mesh, k=0):
    # Averages face uvs to get vert uvs, returns [2, nVert]
    # mesh['uvs'] = [3,nFace,2,k], we fix k and average over all faces going to a single vert.
    nVert = mesh['verts'].shape[1]
    moments = np.ones([2, nVert])*1e-100
    weights = moments*0.5
    faces = mesh['faces']; uvs = mesh['uvs'][:,:,:,k]
    for i in range(faces.shape[1]):
        for o in range(3):
            v_ix = faces[o,i]; uv = uvs[o,i,:]
            moments[:,v_ix] = uv[0:2] # uvs can be >2, i.e uvw maps, but we willnot use the w coord.
            weights[:,v_ix] = weights[:,v_ix] + 1 # count up the number of faces to said vert.
    return moments/weights

def get_relevant_keys(viz_type_set):
    # All keys that affect the object, except for mat44.
    # TODO: a bit clumsy for refactoring, etc.
    out = set()
    if 'light' in viz_type_set:
        out = out.union(set(['color']))
    if 'mesh' in viz_type_set:
        out = out.union(set(['verts','edges','faces','uvs','colors','is_vert_selected','selected_edges','is_face_selected','selected_verts']))
    if 'text' in viz_type_set:
        out = out.union(set(['text','font','small_caps_scale','slant','shadow','shadow_color','color','word_wrap','align','one_sided']))
    return out

#################### Small build functions #########################

def build_light(light, the_pivot):
    col = light.get('color', [16,16,16,1])
    # TODO: more kinds of lights.
    light_node = PointLight("point_light")
    light_node.setColor((col[0],col[1],col[2],col[3]))
    light_panda_obj = the_pivot.attach_new_node(light_node)
    light_node.attenuation = (1, 0, 1)
    return light_panda_obj

def build_mat44(mat44_np):
    mat44_np = np.transpose(mat44_np) # TODO: do we need this?
    mat44_panda = LMatrix4f(mat44_np[0,0],mat44_np[0,1],mat44_np[0,2],mat44_np[0,3],
                            mat44_np[1,0],mat44_np[1,1],mat44_np[1,2],mat44_np[1,3],
                            mat44_np[2,0],mat44_np[2,1],mat44_np[2,2],mat44_np[2,3],
                            mat44_np[3,0],mat44_np[3,1],mat44_np[3,2],mat44_np[3,3])
    return TransformState.makeMat(mat44_panda)

#################### Big functions #########################

# Mesh, camera, and GUI sync when a nice vanilla python object is updated.
def build_pointcloud(name, verts):
    nVert = verts.shape[1]
    vertex_format = GeomVertexFormat.get_v3c4()
    vertex_data = GeomVertexData(name, vertex_format, Geom.UH_static)
    pos_writer = GeomVertexWriter(vertex_data, "vertex")
    col_writer = GeomVertexWriter(vertex_data, "color")
    for i in range(nVert):
        pos_writer.add_data3(*verts[:,i])
        col_writer.add_data4(1.0,0.5,0.0,0.5)
    points_prim = GeomPoints(Geom.UH_static)
    points_prim.reserve_num_vertices(int(nVert+0.5))
    for i in range(nVert):
        points_prim.add_vertex(i)
    points_prim.close_primitive()
    # create a Geom and add the primitive to it
    geom = Geom(vertex_data)
    geom.add_primitive(points_prim)
    # finally, create a GeomNode, add the Geom to it and wrap it in a NodePath
    node = GeomNode(name)
    node.add_geom(geom)
    nodey = NodePath(node)
    #print(dir(nodey))
    nodey.set_render_mode_thickness(4)
    return nodey

def build_wireframe(name, wireframe):
    # 'verts' but 'edges' not faces.
    verts = wireframe['verts']
    edges = wireframe['edges']
    nVert = verts.shape[0]
    nEdge = edges.shape[0]
    vertex_format = GeomVertexFormat.get_v3c4()
    vertex_data = GeomVertexData(name, vertex_format, Geom.UH_static)
    pos_writer = GeomVertexWriter(vertex_data, "vertex")
    col_writer = GeomVertexWriter(vertex_data, "color")
    for i in range(nVert):
        pos_writer.add_data3(*verts[:,i])
        col_writer.add_data4(0.5,1.0,0.0,0.5)
    lines_prim = GeomLines(Geom.UH_static)
    lines_prim.reserve_num_vertices(int(2*nEdge+0.5))

    for i in range(nEdge):
        lines_prim.add_vertices(*edges[:,i])
    lines_prim.close_primitive()
    # create a Geom and add the primitive to it
    geom = Geom(vertex_data)
    geom.add_primitive(lines_prim)
    # finally, create a GeomNode, add the Geom to it and wrap it in a NodePath
    node = GeomNode(name)
    node.add_geom(geom)
    nodey = NodePath(node)
    #print(dir(nodey))
    nodey.set_render_mode_thickness(2)
    return nodey

def build_mesh(name, mesh):
    # Verts is [nVert,3], tris is [nFace,3] and is int not float.
    # uvs is [3,nFace,2,k] Note: Blender is {layer:[nFace, 3,2]}. For now we only care about k=0.
    # colors is [nVert, 4], float 0-1 rgba.
    verts = mesh['verts']
    faces = mesh['faces']
    uvs = mesh.get('uvs',None)
    if uvs is not None:
        uvs = uvs[:,:,:,0]
    colors = mesh.get('colors', None)
    nVert = verts.shape[1] # Panda3D will get an exploded mesh, giving it nFace*3 verts not nVert verts.
    nFace = faces.shape[1]

    # The mesh is exploded into individual triangles.
    # For now we flat shade with no normal interpolation, although it would not be too hard to change this.
    # See also: https://discourse.panda3d.org/t/new-procedural-geometry-samples/24882
    vertex_format = GeomVertexFormat.get_v3n3c4t2()
    vertex_data = GeomVertexData(name, vertex_format, Geom.UH_static)
    pos_writer = GeomVertexWriter(vertex_data, "vertex")
    normal_writer = GeomVertexWriter(vertex_data, "normal")
    if colors is not None:
        col_writer = GeomVertexWriter(vertex_data, "color")
    uv_writer = GeomVertexWriter(vertex_data, "texcoord")

    for i in range(nFace):
        v012 = verts[:,faces[:,i]]
        normal = np.cross(v012[:,2]-v012[:,0], v012[:,1]-v012[:,0])
        normal = normal/np.linalg.norm(normal+1e-100)
        for o in range(3):
            pos_writer.add_data3(*v012[:,o])
            if colors is not None:
                col_writer.add_data4(*colors[:,faces[o,i]])
            normal_writer.add_data3(*normal)
            if uvs is None:
                uv_writer.add_data2(o==0,o==1)
            else:
                uv_writer.add_data2(*uvs[o,i,:])

    tris_prim = GeomTriangles(Geom.UH_static)
    tris_prim.reserve_num_vertices(int(3*nFace+0.5))

    for i in range(nFace):
         tris_prim.add_vertices(*[3*i, 3*i+1, 3*i+2])
    tris_prim.close_primitive()

    # create a Geom and add the primitive to it
    geom = Geom(vertex_data)
    geom.add_primitive(tris_prim)

    # finally, create a GeomNode, add the Geom to it and wrap it in a NodePath
    node = GeomNode(name)
    node.add_geom(geom)
    nodey = NodePath(node)
    return nodey

def mesh3_keys():
    return ['point_mesh', 'edge_mesh', 'face_mesh']

def build_mesh3(name, mesh):
    # Returns meshes that show what is selected. Some or all returned meshes may be None.
    # The point and line meshes show what is selected.
    # is_vert_selected = is each vert selected, [nVert], Optional.
    #   Can also use selected_verts.
    # selected_edges = [*, 2] Optional.
       # Can NOT use is_edge_selected, as we do not have an edge array in the mesh.
    # is_face_selected = [nFace], optional.
    #   Can also use selected_faces.
    if mesh is None:
        return {'point_mesh':None, 'edge_mesh':None, 'face_mesh':None}
    nVert = mesh['verts'].shape[1]
    nFace = mesh['faces'].shape[1]

    sel_vert = mesh.get('is_vert_selected', np.zeros([nVert]))
    sel_edge = mesh.get('selected_edges', [])
    sel_face = mesh.get('is_face_selected', np.zeros([nFace]))

    if mesh.get('selected_verts', None) is not None:
        sel_vert[mesh['selected_verts']] = 1.0

    if np.sum(sel_face)>=0.5:
        # Colors are per vert, so verts with more selected faces get more yellow.
        colors = np.copy(mesh.get('colors', np.tile([0,0,1,1],[1,nVert])))

        sel_weight = np.zeros([nVert,1])
        for i in range(nFace):
            if sel_face[i] >= 0.5:
                for j in range(mesh['faces'].shape[0]): # One day we willl support non-tri faces.
                    sel_weight[mesh['faces'][j,i]] = sel_weight[mesh['faces'][j,i]] + 1
        sel_colors = np.transpose(np.tile([1,1,0,1],[nVert,1]))
        sel_porp = 1.0-1.0/(1.0+sel_weight)
        colors1 = sel_colors*sel_porp + colors*(1-sel_porp)
        mesh = mesh.copy()
        mesh['colors'] = colors1
    face_mesh = build_mesh(name, mesh)
    point_mesh = None
    edge_mesh = None
    if np.size(sel_vert)>0:
        sel_points = mesh['verts'][:,sel_vert>=0.5]
        point_mesh = build_pointcloud(name+'points', sel_points)
    if np.size(sel_edge)>0:
        edge_mesh = build_wireframe(name+'edges', {'verts':mesh['verts'], 'edges':sel_edge>=0.5})
    return {'point_mesh':point_mesh, 'edge_mesh':edge_mesh, 'face_mesh':face_mesh}

def build_text(text_dict):
    # In-world text.
    text_ob0 = TextNode('texty')
    text_ob0.set_text(text_dict['text'])
    if 'font' in text_dict: # Various optional properties.
        text_ob0.setFont(loader.loadFont(text_dict['font']))
    if 'small_caps_scale' in text_dict:
        text_ob0.setSmallCaps(True)
        text_ob0.setSmallCapsScale(text_dict['small_caps_scale'])
    if 'slant' in text_dict:
        text_ob0.setSlant(text_dict['slant'])
    if 'shadow' in text_dict:
        text_ob0.setShadow(text_dict['shadow'])
    if 'shadow_color' in text_dict:
        text_ob0.setShadowColor(text_dict['shadow_color'])
    if 'color' in text_dict:
        text_ob0.setTextColor(LVecBase4f(*text_dict['color']))
    if 'word_wrap' in text_dict:
        text_ob0.setWordwrap(text_dict['word_wrap'])
    align = text_dict.get('align', 'center')
    if align == 'center':
        text_ob0.setAlign(TextNode.ACenter)
    elif align=='left':
        text_ob0.setAlign(TextNode.ALeft)
    elif align=='right':
        text_ob0.setAlign(TextNode.ARight)
    text_ob = NodePath(text_ob0)
    if not text_dict.get('one_sided',False):
        text_ob.setTwoSided(True)
    #text_ob.setLightOff() # Option to display the color as-is.
    return text_ob

def build_onscreen_text(text_dict):
    # Onscreen, not in world. Most useful for quick debugs and mock-ups, as it is less versatile.
    pos = text_dict.get('pos',[0.0,0.0]); scale = text_dict.get('scale', 0.07)
    color = text_dict.get('color',[0,0,0,1])
    if 'xy' in text_dict:
        pos = text_dict['xy']
    if 'x' in text_dict:
        pos[0] = text_dict['x']
    if 'y' in text_dict:
        pos[1] = text_dict['y']
    text_ob = OnscreenText(text=text_dict['text'], pos=pos, scale=scale, fg=color)
    align = text_dict.get('align', 'center')
    if align == 'center':
        text_ob.setAlign(TextNode.ACenter)
    elif align=='left':
        text_ob.setAlign(TextNode.ALeft)
    elif align=='right':
        text_ob.setAlign(TextNode.ARight)
    return text_ob
