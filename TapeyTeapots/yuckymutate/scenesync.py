from panda3d.core import *
import numpy as np
from TapeyTeapots.meshops import quat34
from direct.gui.OnscreenText import OnscreenText

# Helper functions:
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

# Mesh, camera, and GUI sync when a nice vanilla python object is updated.
def buildPointcloud(name, verts):
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
    uvs = mesh.get('uvs',None)
    if uvs is not None:
        vuvs = vert_uvs(mesh)
    colors = mesh.get('colors', None)
    nVert = verts.shape[1]
    nFace = tris.shape[1]

    # See also: https://discourse.panda3d.org/t/new-procedural-geometry-samples/24882
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
            uv_writer.add_data2(*vuvs[:,i])

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
    return nodey

def buildMesh3(name, mesh):
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
    face_mesh = buildMesh(name, mesh)
    point_mesh = None
    edge_mesh = None
    if np.size(sel_vert)>0:
        sel_points = mesh['verts'][:,sel_vert>=0.5]
        point_mesh = buildPointcloud(name+'points', sel_points)
    if np.size(sel_edge)>0:
        edge_mesh = buildWireframe(name+'edges', {'verts':mesh['verts'], 'edges':sel_edge>=0.5})
    return {'point_mesh':point_mesh, 'edge_mesh':edge_mesh, 'face_mesh':face_mesh}

def np2panda_44(mat44_np):
    mat44_np = np.transpose(mat44_np) # TODO: do we need this?
    mat44_panda = LMatrix4f(mat44_np[0,0],mat44_np[0,1],mat44_np[0,2],mat44_np[0,3],
                            mat44_np[1,0],mat44_np[1,1],mat44_np[1,2],mat44_np[1,3],
                            mat44_np[2,0],mat44_np[2,1],mat44_np[2,2],mat44_np[2,3],
                            mat44_np[3,0],mat44_np[3,1],mat44_np[3,2],mat44_np[3,3])
    return TransformState.makeMat(mat44_panda)

def light2panda(light, the_pivot):
    col = light.get('color', [16,16,16,1])
    # TODO: more kinds of lights.
    light_node = PointLight("point_light")
    light_node.setColor((col[0],col[1],col[2],col[3]))
    light_panda_obj = the_pivot.attach_new_node(light_node)
    pos = light.get('pos', [0,0,0]) # TODO: fit lights into the standard object tree.
    light_panda_obj.setPos(pos[0], pos[1], pos[2])
    light_node.attenuation = (1, 0, 1)
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

    mat44_this = np.identity(4)
    if 'mat44' in new_render_branch:
        mat44_this = new_render_branch['mat44']
    if 'pos' in new_render_branch: # Shortcut when no need to rotate, shear, or scale.
        mat44_tmp = np.identity(4); mat44_tmp[0:3,3] = new_render_branch['pos']
        mat44_this = np.matmul(mat44_tmp,mat44_this)
    mat44 = np.matmul(mat44_this, mat44_ancestors)

    change_mesh = old_mesh is not new_mesh
    change_text = old_text is not new_text
    ident_m44 = np.identity(4)
    change_xform = old_render_branch.get('mat44', ident_m44) is not new_render_branch.get('mat44', ident_m44)

    mesh_keys = list(buildMesh3('None',None).keys())
    mesh_panda_objs = [panda_objects_branch.get(k,None) for k in mesh_keys]

    if change_mesh:
        for mesh_obj in mesh_panda_objs:
            if mesh_obj is not None:
                mesh_obj.removeNode()
        if new_mesh is not None:
            #myMesh = buildMesh('meshy',new_mesh)
            mesh_panda_objs_new = buildMesh3('meshy', new_mesh)
            for k in mesh_keys:
                obj_new = mesh_panda_objs_new[k]
                panda_objects_branch[k] = obj_new
                if obj_new is not None:
                    obj_new.reparent_to(pivot)
        else:
            for ky in mesh_keys:
                if ky in panda_objects_branch:
                    del panda_objects_branch[ky]
    if (change_mesh or change_xform):
        xform = np2panda_44(mat44)

        for k in mesh_keys:
            if k in panda_objects_branch:
                if panda_objects_branch[k] is not None:
                    panda_objects_branch[k].set_transform(xform)

    text_ob = panda_objects_branch.get('text',None)
    if change_text:
        if text_ob is not None:
            text_ob.removeNode()
        if new_text is not None:
            text_ob0 = TextNode('texty')
            text_ob0.set_text(new_text['text'])
            if 'font' in new_text: # Various optional properties.
                text_ob0.setFont(loader.loadFont(new_text['font']))
            if 'small_caps_scale' in new_text:
                text_ob0.setSmallCaps(True)
                text_ob0.setSmallCapsScale(new_text['small_caps_scale'])
            if 'slant' in new_text:
                text_ob0.setSlant(new_text['slant'])
            if 'shadow' in new_text:
                text_ob0.setShadow(new_text['shadow'])
            if 'shadow_color' in new_text:
                text_ob0.setShadowColor(new_text['shadow_color'])
            if 'color' in new_text:
                text_ob0.setTextColor(LVecBase4f(*new_text['color']))
            if 'word_wrap' in new_text:
                text_ob0.setWordwrap(new_text['word_wrap'])
            align = new_text.get('align', 'center')
            if align == 'center':
                text_ob0.setAlign(TextNode.ACenter)
            elif align=='left':
                text_ob0.setAlign(TextNode.ALeft)
            elif align=='right':
                text_ob0.setAlign(TextNode.ARight)
            text_ob = NodePath(text_ob0)
            #text_ob.setLightOff() # Option to display the color as-is.
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

def sync_camera(cam44_old, cam44, cam_obj):
    # Camera math: Our camera xform is remove_w(norm_w(cam44*add_w(x)))
    # add_w adds w=1 to a 3 vector. norm_w divides by the w term.
    # Note: this is different than the standard 4x4 matrix for 3d xforms.
    # We set our own cam_xform on Panda's camera, which is the SAME as how 4x4 matrix.
    # We need to make Panda3d and our system equivalent:
    # remove_w(norm_w(cam44_panda*((cam_xform)^(-1)*add_w(x)))) = remove_w(norm_w(cam44*add_w(x)))
    # Which will be true if the matrix parts are the same:
    # cam44_panda*(cam_xform)^(-1) = cam44 => cam44_panda = cam44*cam_xform
    # So we have: cam_xform = (cam44^-1)*cam44_panda

    # The default camera points in the +y direction, while our ident q points in the -z direction:
    q = [np.sqrt(0.5),np.sqrt(0.5),0,0]
    lens = base.camLens;
    f = np.sqrt(2.0); lens.setFov(90.0); c = [0.01, 100]; lens.setNearFar(c[0], c[1])
    cam44_panda = quat34.qvfcyaTOcam44(q, [0,0,0],f,c)

    # TODO: sign of matrix makes bug where everything can disappear.
    cam_xform = np.matmul(np.linalg.inv(cam44),cam44_panda)

    cam_obj.set_transform(np2panda_44(cam_xform))

def sync_onscreen_text(panda_objects, old_state, new_state):
    # Onscreen text is a very system system that does not care about the camera, etc.
    old_text = old_state.get('onscreen_text',None)
    new_text = new_state.get('onscreen_text',None)
    txt_obj = panda_objects.get('onscreen_text', None)
    if old_text is not new_text: # Any change.
        if txt_obj is not None:
            txt_obj.destroy()
        if new_text is not None:
            pos = new_text.get('pos',[0.0,0.0]); scale = new_text.get('scale', 0.07)
            color = new_text.get('color',[0,0,0,1])
            if 'xy' in new_text:
                pos = new_text['xy']
            if 'x' in new_text:
                pos[0] = new_text['x']
            if 'y' in new_text:
                pos[1] = new_text['y']
            textObject = OnscreenText(text=new_text['text'], pos=pos, scale=scale, fg=color)
            align = new_text.get('align', 'center')
            if align == 'center':
                textObject.setAlign(TextNode.ACenter)
            elif align=='left':
                textObject.setAlign(TextNode.ALeft)
            elif align=='right':
                textObject.setAlign(TextNode.ARight)
            panda_objects['onscreen_text'] = textObject

def sync(old_state, new_state, panda_objects, the_magic_pivot):
    old_render = old_state.get('render',{})
    new_render = new_state.get('render',{})
    lights_old = old_state.get('lights',[])
    lights_new = new_state.get('lights',[])
    
    if new_state.get('show_fps',False):
        base.setFrameRateMeter(True)
    else:
        base.setFrameRateMeter(False)

    if lights_new is not lights_old: # Any change will trigger all lights to be set to the root.
        render.clear_light()
        if type(lights_new) is dict:
            lights_new = list(lights_new.values())
        for lightp in panda_objects.get('das_blinkin_lights',[]):
            render.clear_light(lightp)
            lightp.removeNode()
        lights_panda_new = [light2panda(light, the_magic_pivot) for light in lights_new]
        for lightp in lights_panda_new:
            render.set_light(lightp)
        panda_objects['das_blinkin_lights'] = lights_panda_new

    sync_renders(old_render, new_render, np.identity(4), panda_objects, the_magic_pivot)

    if 'camera' in old_state:
        old_camera = old_state['camera']['mat44']
    else:
        old_camera = None
    sync_camera(old_camera, new_state['camera']['mat44'], panda_objects['cam'])
    sync_onscreen_text(panda_objects, old_state, new_state)
