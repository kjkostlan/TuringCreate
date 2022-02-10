from panda3d.core import *
import numpy as np


# Mesh, camera, and GUI sync when a nice vanilla python object is updated.
def buildPointcloud(name, verts):
    nVert = verts.shape[0]
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

def buildWireframe(name, wireframe):
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


def buildMesh(name, mesh):
    # Verts is [nVert,3], tris is [nFace,3] and is int not float.
    # uvs is [nVert,2] Note: Different from blender which allows seams and is {layer:[nFace, 3,2]}
    # colors is [nVert, 4], float 0-1 rgba.
    verts = mesh['verts']
    tris = mesh['faces']
    uvs = mesh.get('UVs',None)
    colors = mesh.get('colors', None)
    nVert = verts.shape[0]
    nFace = tris.shape[0]

    # For now, use uvs into colors:
    # Anoter part from: https://discourse.panda3d.org/t/new-procedural-geometry-samples/24882
    vertex_format = GeomVertexFormat.get_v3n3c4t2()
    vertex_data = GeomVertexData(name, vertex_format, Geom.UH_static)
    pos_writer = GeomVertexWriter(vertex_data, "vertex")
    normal_writer = GeomVertexWriter(vertex_data, "normal")
    col_writer = GeomVertexWriter(vertex_data, "color")
    uv_writer = GeomVertexWriter(vertex_data, "texcoord")
    # the normal is the same for all vertices. TODO: fix this!
    normal = (0., -1., 0.)
    for i in range(nVert):
        pos_writer.add_data3(*verts[:,i])
        normal_writer.add_data3(normal)
        if colors is not None:
            col_writer.add_data4(*colors[:,i])
        if uvs is not None:
            uv_writer.add_data2(*uvs[:,i])

    tris_prim = GeomTriangles(Geom.UH_static)
    tris_prim.reserve_num_vertices(int(3*nFace+0.5))

    for i in range(nFace):
         tris_prim.add_vertices(*tris[:,i])
    tris_prim.close_primitive()

    # create a Geom and add the primitive to it
    geom = Geom(vertex_data)
    geom.add_primitive(tris_prim)
    # finally, create a GeomNode, add the Geom to it and wrap it in a NodePath
    node = GeomNode(name)
    node.add_geom(geom)
    nodey = NodePath(node)
    #print(dir(nodey))
    return nodey


def buildMesh3(name, mesh):
    # Returns point mesh, line mesh, face mesh.
    # The point and line meshes show what is selected.
    # is_vert_selected = is each vert selected, [nVert], Optional.
    #   Can also use selected_verts.
    # selected_edges = [*, 2] Optional.
       # Can NOT use is_edge_selected, as we do not have an edge array in the mesh.
    # is_face_selected = [nFace], optional.
    #   Can also use selected_faces.
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
    face_mesh = buildMesh(name, mesh)
    point_mesh = None
    edge_mesh = None
    if np.size(sel_vert)>0:
        sel_points = mesh['verts'][:,sel_vert>=0.5]
        point_mesh = buildPointcloud(name+'points', sel_points)
    if np.size(sel_edge)>0:
        edge_mesh = buildWireframe(name+'edges', {'verts':mesh['verts'], 'edges':sel_edge>=0.5})
    return point_mesh, edge_mesh, face_mesh

def np2panda_44(mat44_np):
    mat44_np = np.transpose(mat44_np) # TODO: do we need this?
    mat44_panda = LMatrix4f(mat44_np[0,0],mat44_np[0,1],mat44_np[0,2],mat44_np[0,3],
                            mat44_np[1,0],mat44_np[1,1],mat44_np[1,2],mat44_np[1,3],
                            mat44_np[2,0],mat44_np[2,1],mat44_np[2,2],mat44_np[2,3],
                            mat44_np[3,0],mat44_np[3,1],mat44_np[3,2],mat44_np[3,3])
    return TransformState.makeMat(mat44_panda)

def light2panda(light, the_pivot):
    pos = light.get('pos', [0,0,0])
    col = light.get('color', [1,1,1,1])
    # TODO: more kinds of lights.
    light_node = PointLight("point_light")
    light_node.setColor((col[0],col[1],col[2],col[3]))
    light_panda_obj = the_pivot.attach_new_node(light_node)
    light_panda_obj.setPos(pos[0], pos[1], pos[2])
    return light_panda_obj

################################################################################

def update_xforms(new_render_branch, mat44_ancestors, panda_objects_branch):
    # Even if everything stays the same, xforms can change due to changing root xforms.
    mat44 = np.matmul(new_render_branch.get('mat44', np.identity(4)), mat44_ancestors)
    xform = np2panda_44(mat44)
    if 'mesh' in panda_objects_branch:
        panda_objects_branch['mesh'].set_transform(xform)
    if 'text' in panda_objects_branch:
        panda_objects_branch['text'].set_transform(xform)
    if 'children' in new_render_branch:
        for ky in new_render_branch['children'].keys():
            update_xforms(new_render_branch['children'][ky], mat44, panda_objects_branch['children'][ky])

def sync_renders(old_render_branch, new_render_branch, mat44_ancestors, panda_objects_branch, pivot):
    # This relies on the vanilla side not mutating anything, just making shallow defensive copies.
    # We mutate panda_objects_branch of course.
    if old_render_branch is None:
        old_render_branch = {}
    if new_render_branch is None:
        new_render_branch = {}
    if old_render_branch is new_render_branch:
        return
    old_mesh = old_render_branch.get('mesh', None)
    new_mesh = new_render_branch.get('mesh', None)
    
    old_text = old_render_branch.get('text', None)
    new_text = new_render_branch.get('text', None)
    
    mat44 = np.matmul(new_render_branch.get('mat44', np.identity(4)), mat44_ancestors)

    change_mesh = old_mesh is not new_mesh
    change_text = old_text is not new_text
    change_xform = old_render_branch.get('mat44', np.identity(4)) is not new_render_branch.get('mat44',np.identity(4))

    kys3 = ['mesh_verts','mesh_edges','mesh_faces']
    myMeshes = [panda_objects_branch.get(k,None) for k in kys3]

    if change_mesh: #Thus updates myMeshes
        for myMesh in myMeshes:
            if myMesh is not None:
                myMesh.removeNode()
        if new_mesh is not None:
            #myMesh = buildMesh('meshy',new_mesh)
            myMeshes = buildMesh3('meshy', new_mesh)
            for i3 in range(3):
                myMesh = myMeshes[i3]
                panda_objects_branch[kys3[i3]] = myMesh
                if myMesh is not None:
                    myMesh.reparent_to(pivot)
                    #if i3==2:
                    #    myMesh.set_light(light)
        #elif 'mesh' in panda_objects_branch:
        else:
            for ky in kys3:
                if ky in panda_objects_branch:
                    del panda_objects_branch[ky]
    if (change_mesh or change_xform):
        xform = np2panda_44(mat44)
        for myMesh in myMeshes:
            if myMesh is not None:
                myMesh.set_transform(xform)
    
    text_ob = panda_objects_branch.get('text',None)
    if change_text:
        if text_ob is not None:
            text_ob.removeNode()
        if new_text is not None:
            text_ob0 = TextNode('texty')
            text_ob0.set_text(new_text['string'])
            if 'font' in new_text:
                text_ob0.setFont(loader.loadFont(text_ob['font']))
            text_ob = NodePath(text_ob0)
            text_ob.reparent_to(pivot)
            panda_objects_branch['text'] = text_ob
    if (change_text or change_xform) and (new_text is not None):
        xform = np2panda_44(mat44)
        text_ob.set_transform(xform)

    children_old = old_render_branch.get('children', {})
    children_new = new_render_branch.get('children', {})
    k_set = set(list(children_old.keys())+list(children_new.keys()))

    if 'children' not in panda_objects_branch:
        panda_objects_branch['children'] = {}

    for k in k_set:
        ch_old = children_old.get(k,None)
        ch_new = children_new.get(k,None)
        if ch_new is not None and k not in panda_objects_branch['children']:
            panda_objects_branch['children'][k] = {}
        panda_branch1 = panda_objects_branch['children'].get(k,{})
        if ch_old is not ch_new:
            sync_renders(ch_old, ch_new, mat44, panda_branch1, pivot)
        elif change_xform and ch_new is not None:
            update_xforms(ch_new, mat44, panda_branch1)

def sync(old_state, new_state, panda_objects, the_magic_pivot):
    old_render = old_state.get('render',{})
    new_render = new_state.get('render',{})
    lights_old = old_state.get('lights',[])
    lights_new = new_state.get('lights',[])
    
    if lights_new is not lights_old: # Any change will trigger all lights to be set to the root.
        for old_light in panda_objects.get('das_blinkin_lights',[]):
            old_light.removeNode()
        lights_panda_new = [light2panda(light, the_magic_pivot) for light in lights_new]
        for lightp in lights_panda_new:
            the_magic_pivot.set_light(lightp)
        panda_objects['das_blinkin_lights'] = lights_panda_new
    
    sync_renders(old_render, new_render, np.identity(4), panda_objects, the_magic_pivot)

    #{'pos':[0,-10,0],'look':[0,1,0],'fov':1.25}
    #panda_objects['cam'].set_x(new_state['camera']['pos'][0])
    #panda_objects['cam'].set_y(new_state['camera']['pos'][1])
    #panda_objects['cam'].set_z(new_state['camera']['pos'][2])

    #mat44_np = np.identity(4) #new_state['camera']['look'][0]
    xform_camera = np2panda_44(new_state['camera']['mat44'])
    panda_objects['cam'].set_transform(xform_camera)