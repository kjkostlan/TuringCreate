# Outer level of the vanilla-pthon and pymesh application.
# No functions are allowed to mutate the app state.
import numpy as np
import scipy
import scipy.linalg
import c
import TapeyTeapots.pandawrapper as panda3dsetup
import TapeyTeapots.ui.nav3D as nav3D
from TapeyTeapots.meshops import quat34, primitives

def simple_init_state(colored_lights=True): # Start simple.
    q = quat34.q_from_polarshift([0,0,-1],[1,0,0])
    cam44 = quat34.qvfcyaTOcam44(q,[-5,0,0],f=2.0)
    the_mesh = {'mesh':makeRandomMesh(),'mat44':np.identity(4), 'children':{}}
    mat44_static  = np.identity(4)
    mat44_static[3,0:3] = [1,1,1]
    the_mesh_static = {'mesh':makeRandomMesh(),'mat44':mat44_static, 'children':{}}
    render = {'mat44':np.identity(4),
              'children': {'the_mesh':the_mesh, 'the_mesh_static':the_mesh_static}}
    if colored_lights:
        lights = [{'pos':[4.0, -32.0, 0.0],'color':[2048,0,0,1]},
                  {'pos':[0.0, -32.0, 4.0],'color':[0,2048,0,1]},
                   {'pos':[0.0, -32.0, 0.0],'color':[0,0,2048,1]}]
    else:
        lights = [{'pos':[0,-32,0],'color':[2048,2048,2048,1]}]
    return {'render':render,
            'lights':lights,
            'camera':{'mat44':cam44}}

def makeRandomMesh(nVert=None, nFace=None):
    if nVert is None:
        nVert = int(np.random.random()*128)+3
    if nFace is None:
        nFace = int(np.random.random()*512)+3

    verts = np.random.random([3,nVert])
    faces = (np.random.random([3,nFace])*(nVert-2)).astype(np.int32)
    uvs = np.random.random([2,nVert])-0.5
    colors = np.random.random([4,nVert])
    out = {'verts':verts,'faces':faces,'UV':uvs,'colors':colors}

    do_sel_verts = False
    do_sel_faces = False
    do_sel_edges = True
    if do_sel_verts:
        out['is_vert_selected'] = np.random.random([nVert])*0.7
    if do_sel_edges:
        sel_edge = np.zeros([2, int(nVert*0.5)])
        sel_edge[0,:] = np.arange(int(nVert*0.5))
        sel_edge[1,:] = np.arange(int(nVert*0.5))+1
        sel_edge = sel_edge.astype(np.int32)
        out['selected_edges'] = sel_edge

    if do_sel_faces:
        out['is_face_selected'] = np.random.random([nFace])*0.7

    return out

def make_random_text(prepend='RAND'):
    out = {}
    out['text'] = str(prepend)+str(np.random.random())+str('OM')
    return out

def print_inputs(mouse_state, key_state, mouse_clicks, key_clicks): # Debug.
    for k in mouse_state.keys():
        if mouse_state[k]:
            if k=='x' or k=='y' or k=='x_old' or k=='y_old':
                pass
            else:
                print('mouse:',k)
    x0 = mouse_state['x_old']
    x1 = mouse_state['x']
    y0 = mouse_state['y_old']
    y1 = mouse_state['y']
    if x0 != x1 or y0 != y1:
        print('Mouse move:',[x0,y0],'->',[x1,y1])
    for k in key_state.keys():
        if key_state[k]:
            print('key:',k)
    for k in mouse_clicks.keys():
        if mouse_clicks[k]:
            print('mclick:',k)
    for k in key_clicks.keys():
        if key_clicks[k]:
            print('kclick:',k)

def everyFrame_default(app_state, mouse_state, key_state, mouse_clicks, key_clicks, screen_state):
    app_state = app_state.copy()
    app_state['nframe'] = app_state.get('nframe',0)+1
    return app_state

def everyFrame_simple_demo(app_state, mouse_state, key_state, mouse_clicks, key_clicks, screen_state):
    app_state = everyFrame_default(app_state, mouse_state, key_state, mouse_clicks, key_clicks, screen_state)
    app_state['onscreen_text'] = {'text':'TEXTY','xy':[-0.9,0.9]}
    print_inputs(mouse_state, key_state, mouse_clicks, key_clicks)
    if np.mod(app_state['nframe'],85)==0:
        app_state = c.assoc_in(app_state, ['render', 'children', 'the_mesh', 'mesh'], makeRandomMesh())
        text = {'text':make_random_text()}
        text['mat44'] = quat34.m44_from_q(np.asarray([0.707,0,0,-0.707]))
        text['text']['color'] = [1,1,1,1]
        text['text']['slant'] = 0.3
        #text['text']['text'] = "Every day in \1slant\1every way\2 I'm \1red\1getting \1roman\1better \1slant\1and\2 better.\2\2"
        text['mat44'][0:3,0:3] = 0.125*text['mat44'][0:3,0:3]
        #text['mat44'][0:3,0:3] = text['mat44'][0:3,0:3]*8
        #print('random text mat44:', out['mat44'])
        app_state = c.assoc_in(app_state, ['render', 'children', 'some_text'], text)
        #app_state = c.assoc_in(app_state,['render', 'children'], make_cube_grid())
    return app_state

def make_cube_grid(n_x = 7, n_y = 7, n_z = 7, space_x = 1, space_y = 1, space_z = 1, radius = 0.05):
    # Put these into ['render', 'children'] of the world.
    cube_mesh = primitives.cube(); cube_mesh['verts'] = cube_mesh['verts']*radius
    obj = {'mat44': np.identity(4), 'mesh':cube_mesh}
    objs = {}
    for i in range(n_x):
        x = space_x*(i-(n_x-1)*0.5)
        for j in range(n_y):
            y = space_y*(j-(n_y-1)*0.5)
            for k in range(n_z):
                z = space_z*(k-(n_z-1)*0.5)
                m44 = np.identity(4); m44[0:3,3] = [x,y,z]
                obj1 = obj.copy(); obj1['mat44'] = m44
                cols = np.ones([4,12]);
                cols[0,:] = i/(n_x-0.999); cols[1,:] = j/(n_y-0.999); cols[2,:] = k/(n_z-0.999)
                obj1 = c.assoc_in(obj1,['mesh','colors'],cols)
                objs['cube_grid'+str(i)+'_'+str(j)+'_'+str(k)] = obj1
    return objs

def mark_corners(cam44, clip_z_value=0.0, margin_from_edge=0.05, relative_size = 0.1):
    # Makes 4 spherical meshes. Note: z=-1 is near, z=+1 is far clippin plane.
    four_corners = np.zeros([3,4])
    ed = 1.0-margin_from_edge
    four_corners[0,:] = [-ed,ed,-ed,ed]; four_corners[1,:] = [ed,ed,-ed,-ed] # Clockwise from top left.
    four_corners[2,:] = clip_z_value
    fc_world = quat34.cam44_invv(cam44, four_corners)
    dist = 0.707*(np.linalg.norm(fc_world[:,0]-fc_world[:,2])) # How far apart sets how large we draw them.
    meshes = {}
    #print('Four corners:\n',fc_world)
    for i in range(4):
        sphere_mesh = primitives.sphere(resolution=16)
        sphere_mesh['verts'] = sphere_mesh['verts']*0.5*relative_size*dist
        sphere_mesh['verts'] = sphere_mesh['verts'] + np.expand_dims(fc_world[:,i],axis=1)
        meshes['corner_sphere'+str(i)] = sphere_mesh
    return meshes

############################### Demos #################################
def panda3d_render_sync_test():
    # Tests whether or not the update system works properly, i.e. creating moving and deleting objects in the scene.
    tweak_m44 = quat34.m44_from_q(quat34.q_from_polarshift([0,0,1],[0.05,0,1]))

    def rand_place44(pos_scatter=8.0):
        v = pos_scatter*np.random.randn(3)*2.0
        q = np.random.randn(4)+1e-100; q = q/np.linalg.norm(q)
        return quat34.qrvTOm44(q,np.identity(3),v)

    def create_obj_tweak(app_state):
        meshes = [primitives.cube(), primitives.sphere(resolution=8), primitives.cylinder(resolution=12),
                  primitives.cone(resolution=12), primitives.torus(resolution=8)]
        mesh = meshes[np.random.randint(len(meshes))]
        new_obj = {'mesh':mesh,'mat44':rand_place44()}
        rand_k = str(np.random.random())
        return c.assoc_in(app_state,['render','children',rand_k], new_obj)

    def create_text_tweak(app_state):
        text = {'text':'text'+str(np.random.randn()),'mat44':rand_place44(),'color':np.random.random(4)}
        text['color'][3] = text['color'][3]**0.25
        text['mat44'][0:3,0:3] = text['mat44'][0:3,0:3]*3
        rand_k = str(np.random.random())
        new_text = {'text':text,'mat44':rand_place44()}
        return c.assoc_in(app_state,['render','children',rand_k], new_text)

    def create_light_tweak(app_state):
        light = {'color':np.random.random(4)*400.0,'pos':np.random.randn(3)*16}
        light['color'][3] = 1.0
        rand_k = str(np.random.random())
        #new_light = {'light':light,'pos':np.random.randn(3)*4}
        return c.assoc_in(app_state,['lights',rand_k], light)

    def mpower(m, pwr):
        return scipy.linalg.expm(pwr*scipy.linalg.logm(m))

    def move_obj_tweak(app_state, move_these='mesh'):
        if move_these != 'light':
            ch = app_state['render']['children']
        else:
            ch = app_state['lights']
        ch = ch.copy()
        #tweak_m44_inv = np.linalg.inv(tweak_m44)
        for k in list(ch.keys()):
            if move_these in ch[k]:
                tweak_m44_1 = tweak_m44
                if 'mat44' in ch[k]:
                    pos = ch[k]['mat44'][0:3,:]
                else:
                    pos = ch[k]['pos']
                r = np.linalg.norm(pos)
                #if r<24:
                #    tweak_m44_1 = tweak_m44_inv
                subk = 'mat44'
                if 'mat44' not in ch[k]:
                    subk = 'pos'
                rand_m44 = quat34.m44_from_q([1,0.02,0,0], normalize=True)
                rand_m44[0:3,3] = np.random.randn()*0.125
                #ch = c.update_in(ch, [k, subk], lambda m44:np.matmul(tweak_m44_1, m44))
                ch = c.update_in(ch, [k, subk], lambda m44:np.matmul(m44,rand_m44))
        if move_these != 'light':
            return c.assoc_in(app_state,['render','children'],ch)
        else:
            return c.assoc_in(app_state,['lights'],ch)

    def tweak_meshes(app_state):
        ch = app_state['render']['children']
        for k in list(ch.keys()):
            if 'mesh' in ch[k]:
                verts = ch[k]['mesh']['verts']
                verts = verts+np.random.randn(*verts.shape)*0.1
                ch = c.assoc_in(ch, [k, 'mesh', 'verts'], verts)
        return c.assoc_in(app_state,['render', 'children'], ch)

    def delete_obj_tweak(app_state, delete_these='mesh'):
        if delete_these != 'light':
            ch = app_state['render']['children']
        else:
            ch = app_state['lights']
        ch = ch.copy()
        for k in list(ch.keys()):
            if delete_these in ch[k] or delete_these == 'light':
                del ch[k]
                break # Only delete one thing.
        if delete_these != 'light':
            return c.assoc_in(app_state,['render','children'],ch)
        else:
            return c.assoc_in(app_state,['lights'],ch)

    def move_camera_tweak(app_state):
        q,v,f,_,_,_ = quat34.cam44TOqvfcya(app_state['camera']['mat44'])
        tweak_4q,tweak_r,tweak_v = quat34.m44TOqrv(tweak_m44)
        #print('tweak_m44:\n', tweak_m44)
        #print('Tweak m33 det:',np.linalg.det(tweak_m44[0:3,0:3]))
        #print('Tweak q:', tweak_4q)
        #print('Tweak r:\n', tweak_r)
        q1 = quat34.qq(tweak_4q, q)
        v1 = quat34.qv(tweak_4q, v)[:,0]
        new_m44 = quat34.qvfcyaTOcam44(q1,v1,f=f)
        #return app_state
        return c.assoc_in(app_state,['camera','mat44'],new_m44)

    packs = []
    packs.append(['Create object',create_obj_tweak, 45])
    packs.append(['Move object',lambda app_state:move_obj_tweak(app_state,move_these='mesh'), 60])
    packs.append(['Create text',create_text_tweak, 45])
    packs.append(['Move text',lambda app_state:move_obj_tweak(app_state,move_these='text'), 60])
    packs.append(['Create light',create_light_tweak, 15])
    #packs.append(['move light',lambda app_state:move_obj_tweak(app_state,move_these='light'), 60])
    packs.append(['Move camera',move_camera_tweak, 60])
    packs.append(['Tweak mesh verts',tweak_meshes, 60])
    packs.append(['Delete light',lambda app_state:delete_obj_tweak(app_state,delete_these='light'), 50])
    packs.append(['Delete object',lambda app_state:delete_obj_tweak(app_state,delete_these='mesh'), 50])
    packs.append(['Delete text',lambda app_state:delete_obj_tweak(app_state,delete_these='text'), 50])

    q_cam0 = quat34.q_from_polarshift([0,0,-1],[1,0,0])
    cam44_0 = quat34.qvfcyaTOcam44(q_cam0,v=[-16,0,0],f=2.0)
    q_,v_,f_,c_,y_,a_ = quat34.cam44TOqvfcya(cam44_0)
    init_state = {'camera':{'mat44':cam44_0}, 'show_fps':True,
                  'current_task':0,'frames_left':packs[0][2], 'slowdown':1,
                  'onscreen_text':{'text':'This starting text should only last one frame','xy':[-1.25,0.9],'align':'left'}}

    def _everyFrame_fn(app_state, mouse_state, key_state, mouse_clicks, key_clicks, screen_state):
        if app_state['frames_left'] <= 0: # Move on to next task.
            app_state = app_state.copy()
            app_state['current_task'] = (app_state['current_task']+1)%len(packs)
            app_state['frames_left'] = int(packs[app_state['current_task']][2]*app_state['slowdown'])
            if app_state['current_task']==0:
                app_state['slowdown'] = app_state['slowdown']+1.0
            if app_state['slowdown']>4:
                app_state['slowdown'] = 4
        current_task = app_state['current_task']
        app_state = packs[app_state['current_task']][1](app_state)
        app_state['frames_left'] = app_state['frames_left']-1
        txt_lines = ['This demo tests if Panda3D keeps up with "changes" to app_state.']
        txt_lines.append("SceneSync looks for changes using python's 'is'\nand will not detect inplace modification:")
        txt_lines.append("To make a change, don't mutate the app_state.\nInstead copy-on-modify while miminizing defensive copying using shallow copies when possible.")
        for i in range(len(packs)):
            line = packs[i][0]
            if 'Create object' in line:
                line = line+'('+str(len(app_state['render']['children'].keys()))+' obs+texts)'
            if i==app_state['current_task']:
                line = line+'<<<'
            txt_lines.append(line)
        txt = '\n'.join(txt_lines)
        app_state = c.assoc_in(app_state, ['onscreen_text','text'],txt)
        return app_state

    pre_run_frames = 32
    for i in range(pre_run_frames):
        init_state = _everyFrame_fn(init_state, None, None, None, None, None)

    x = panda3dsetup.App(init_state, _everyFrame_fn)

def panda3d_camerademo():
    # Is the camera working properly?
    #q = quat34.camq_from_look([-5, 2, -4]) # rotation of camera (3 DOF)
    #v = [5,-3,10] # location of center of camera (3 DOF)
    #f = 3.5 # f # kindof the f-number. A 90 degree FOV is an f of 1.0 and telophotos would be f>10. (1 DOF)
    #cl = [0.002, 1000] # the [near, far] clipping plane. For very weird cameras far can be nearer. (2 DOF)

    q = [1,0,0,0]
    v = [0,0,0]
    f = 1.414#*0.72
    cl = [0.01, 100]
    y = [1, 0] # [y-stretch, y shear in x direction] of the camera image. Usually [1,0] (2 DOF)
    a = [0.0,0,0,0] # Far clipping plane shear slope (applied after y), + is away from camera [near-x, near-y, far-x, far-y]. Usually all zero. (4 DOF)
    cam44 = quat34.qvfcyaTOcam44(q,v,f,c=cl,y=y,a=a)
    app_state = simple_init_state(); app_state['show_fps'] = True
    #all_meshes = {**make_cube_grid(),**corners}
    app_state = c.assoc_in(app_state,['render', 'children'], make_cube_grid())

    #apply_mouse_camera_fn(app_state, mouse_state, fn)
    fn_map_keys = {'a':nav3D.orbit,'b':nav3D.change_orbit_center,'c':nav3D.zoom_or_fov,'d':nav3D.rot_camera,
                   'e':nav3D.clip_plane,'f':nav3D.shear_view,'g':nav3D.shear_near_clipplane,
                   'h':nav3D.shear_far_clipplane,'escape':nav3D.reset}

    def every_frame_func(app_state, mouse_state, key_state, mouse_clicks, key_clicks, screen_state):
        f_camstate_deltax_deltay = None
        for k in key_state.keys():
            if key_state[k] and k in fn_map_keys:
                f_camstate_deltax_deltay = fn_map_keys[k]
        if f_camstate_deltax_deltay is not None:
            app_state = nav3D.apply_mouse_camera_fn(app_state, mouse_state, f_camstate_deltax_deltay, drag_only=False)
            objs = app_state['render']['children'].copy()
            cam44 = app_state['camera']['mat44']
            corners = mark_corners(cam44, clip_z_value=0.0, margin_from_edge=0.05, relative_size = 0.1)
            for ck in corners.keys():
                objs[ck] = corners[ck]
            for k in list(objs.keys()): # Test moving objects around.
                pass
                #objs[k] = c.assoc(objs[k],'mat44', np.random.random([4,4]))
                #if np.random.random()<0.01:
                #    del objs[k]
            app_state = c.assoc_in(app_state,['render','children'],objs)
            return app_state
        else:
            return nav3D.empty_everyframe(app_state, mouse_state, key_state, mouse_clicks, key_clicks, screen_state)

    #print('Cam44 default:',app_state['camera'])
    #q0,v0,f0,c0,y0,a0 = quat34.cam44TOqvfcya(app_state['camera']['mat44'])
    #print('App state camera: q:', q0, 'v:', v0, 'f:', f0, 'c:', c0, 'y:', y0, 'a:', a0)
    #app_state['camera'] = {'mat44':cam44}
    x = panda3dsetup.App(app_state, every_frame_func)
