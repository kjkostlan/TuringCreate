# Outer level of the vanilla-pthon and pymesh application.
# No functions are allowed to mutate the app state.
import numpy as np
import time
import scipy
import scipy.linalg
import c
import TapeyTeapots.pandawrapper as panda3dsetup
import TapeyTeapots.ui.nav3D as nav3D
from TapeyTeapots.meshops import quat34, primitives
from TapeyTeapots.ui import uicore

def simple_init_state(colored_lights=True, make_random_mesh=True): # Start simple.
    q = quat34.q_from_polarshift([0,0,-1],[1,0,0])
    cam44 = quat34.qvfcyaTOcam44(q,[-5,0,0],f=2.0)
    the_mesh = random_mesh_node()
    mat44_randmesh = np.identity(4); mat44_randmesh[3,0:3] = [1,1,1]
    render = {'mat44':np.identity(4),
              'bearcubs': {'the_mesh':the_mesh}}
    if not make_random_mesh:
        del render['bearcubs']['the_mesh']
    if colored_lights:
        lights = [{'pos':[4.0, -32.0, 0.0],'color':[1024,0,0,1]},
                  {'pos':[0.0, -32.0, 4.0],'color':[0,1024,0,1]},
                  {'pos':[0.0, -32.0, 0.0],'color':[0,0,1024,1]},
                  {'pos':[4.0, 32.0, 0.0],'color':[512,0,0,1]},
                  {'pos':[0.0, 32.0, 4.0],'color':[0,512,0,1]},
                  {'pos':[0.0, 32.0, 0.0],'color':[0,0,512,1]}]
    else:
        lights = [{'pos':[0,-32,0],'color':[1024,1024,1024,1]}]
    light_dict = {}
    for i in range(len(lights)):
        light_dict[str(i)] = {'pos':lights[i]['pos'],'color':lights[i]['color'],'viztype':'light'}
    render['bearcubs']['lights'] = {'bearcubs':light_dict}
    render['camera'] = {'mat44':np.identity(4)}
    return render

def random_mesh_node(nVert=None, nFace=None, mat44=None):
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

    if mat44 is not None:
        out['mat44'] = mat44
    out['viztype'] = 'mesh'
    return out

def make_random_text(prepend='RAND'):
    out = {}
    out['text'] = str(prepend)+str(np.random.random())+str('OM')
    return out

def print_inputs(inputs): # Debug.
    mouse_state = inputs['mouse']; key_state = inputs['keyboard']
    mouse_clicks = inputs['click']; key_clicks = inputs['type']
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

def everyFrame_default(app_state, inputs):
    app_state = app_state.copy()
    app_state['nframe'] = app_state.get('nframe',0)+1
    return app_state

def make_cube_grid(n_x = 7, n_y = 7, n_z = 7, space_x = 1, space_y = 1, space_z = 1, radius = 0.05):
    # Put these into ['bearcubs'] of the world.
    cube_mesh = primitives.cube(); cube_mesh['verts'] = cube_mesh['verts']*radius
    obj = {'mat44': np.identity(4), **cube_mesh,'viztype':'mesh'}
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
                obj1 = c.assoc_in(obj1,['colors'],cols)
                objs['cube_grid'+str(i)+'_'+str(j)+'_'+str(k)] = obj1
    return objs

def mark_on_screen(cam44, pos_screen, names, relative_size = 0.0625):
    # Marks a point on the screen.
    meshes = {}
    pos_world = quat34.cam44_invv(cam44, pos_screen)
    two_corners = np.zeros([3,2]);
    two_corners[:,0] = [-1,-1,pos_screen[2,0]]; two_corners[:,1] = [1,1,pos_screen[2,0]]
    two_corners_world = quat34.cam44_invv(cam44, two_corners)
    dist = 0.707*(np.linalg.norm(two_corners_world[:,0]-two_corners_world[:,1])) # How far apart sets how large we draw them.
    for i in range(pos_world.shape[1]):
        sphere_mesh = primitives.sphere(resolution=8)
        sphere_mesh['verts'] = sphere_mesh['verts']*0.5*relative_size*dist
        sphere_mesh['verts'] = sphere_mesh['verts'] + np.expand_dims(pos_world[:,i],axis=1)
        meshes[names[i]] = sphere_mesh
    return meshes

def mark_corners(cam44, clip_z_value=0.0, margin_from_edge=0.05, relative_size = 0.08):
    # Makes 4 spherical meshes. Note: z=-1 is near, z=+1 is far clippin plane.
    # Making new meshes every frame is inefficient, so the resolution is kept low.
    four_corners = np.zeros([3,4])
    ed = 1.0-margin_from_edge
    four_corners[0,:] = [-ed,ed,-ed,ed]; four_corners[1,:] = [ed,ed,-ed,-ed] # Clockwise from top left.
    four_corners[2,:] = clip_z_value

    names = ['corner_sphere'+str(i) for i in range(4)]
    return mark_on_screen(cam44, four_corners, names, relative_size = relative_size)

def sequential_task_everyframe(app_state, inputs, txt_lines, packs):
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
    txt_lines = txt_lines.copy()
    for i in range(len(packs)):
        line = packs[i][0]
        if 'Create object' in line:
            line = line+'('+str(len(app_state['bearcubs'].keys()))+' obs+texts)'
        if i==app_state['current_task']:
            line = line+'<<<'
        txt_lines.append(line)
    txt = '\n'.join(txt_lines)
    app_state = c.assoc_in(app_state, ['onscreen_text','text'],txt)
    return app_state

############################### Demos #################################

def mouse_key_input_demo():
    # Tests the panda3D input.
    txt_lines = ['This demo reports key and mouse inputs as text onscreen. There is no 3D scene.']
    txt_lines.append('It should report all three mouse buttons, and mouse move/drag.')
    txt_lines.append('It should treat normal and special keys the same.')
    txt_lines.append('Finally, it should respond to resizing the screen.')

    def every_frame(app_state, inputs):
        mouse_state = inputs['mouse']; key_state = inputs['keyboard']
        mouse_clicks = inputs['click']; key_clicks = inputs['type']
        screen_state = inputs['screen']
        txt_lines1 = txt_lines.copy()
        keys_pressed = []
        for k,v in key_state.items():
            if v:
                keys_pressed.append(k)
        txt_lines1.append('"keyboard": '+' '.join(keys_pressed))
        mouse_pressed = []
        for i in range(16):
            if i in mouse_state and mouse_state[i]:
                mouse_pressed.append(str(i))
        txt_lines1.append('"mouse": '+' '.join(mouse_pressed))
        mouse_string = ''
        for k in ['x_old','x','y_old','y','scroll_old','scroll']:
            mouse_string = mouse_string+k+'='+'{:.3f}'.format(mouse_state[k])+' '
        if mouse_state['scroll_old'] != mouse_state['scroll']:
            txt_lines1.append('Wheel just scrolled')
        txt_lines1.append(mouse_string)
        txt_lines1.append('"screen": '+str(screen_state))
        throttle = app_state.get('fps_throttle', False)
        if throttle:
            time.sleep(0.5)
        if 'tab' in key_clicks:
            throttle = not throttle
        app_state['fps_throttle'] = throttle
        throttle_text = 'Tab to enable 500ms lag'
        if throttle:
            throttle_text = 'LAG ON: Tab to disable lag'
        txt_lines1.append(throttle_text)
        if len(mouse_clicks)>0:
            txt_lines1.append('"click":'+str(mouse_clicks))
        if len(key_clicks)>0:
            txt_lines1.append('"type":'+str(key_clicks))
        txt = '\n'.join(txt_lines1)
        return c.assoc_in(app_state, ['onscreen_text','text'],txt)
    scene0 = {'camera':{'mat44':np.identity(4)}}
    txt = {'xy':[-1.325,0.9],'align':'left','text':'This text should only last one frame'}
    x = panda3dsetup.App(c.assoc_in(scene0, ['onscreen_text'],txt), every_frame)

def render_sync_demo():
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
        new_obj = {**mesh,'mat44':rand_place44(), 'viztype':'mesh'}
        rand_k = str(np.random.random())
        return c.assoc_in(app_state,['bearcubs','shapes','bearcubs',rand_k], new_obj)

    def create_text_tweak(app_state):
        text = {'text':'text'+str(np.random.randn()),'color':np.random.random(4)}
        text['color'][3] = text['color'][3]**0.25
        rand_k = str(np.random.random())
        new_text = {**text,'mat44':rand_place44(), 'viztype':'text'}
        return c.assoc_in(app_state,['bearcubs','texts','bearcubs',rand_k], new_text)

    def create_light_tweak(app_state):
        light = {'color':np.random.random(4)*40.0}
        light['color'][3] = 1.0
        rand_k = str(np.random.random())
        lnode = {**light, 'pos':np.random.randn(3)*16, 'viztype':'light'}
        #new_light = {'light':light,'pos':np.random.randn(3)*4}
        return c.assoc_in(app_state,['bearcubs','lights','bearcubs',rand_k], lnode)

    def mpower(m, pwr):
        return scipy.linalg.expm(pwr*scipy.linalg.logm(m))

    def move_obj_tweak(app_state, move_these='mesh'):
        if move_these == 'mesh':
            ky = 'shapes'
        elif move_these == 'light':
            ky = 'lights'
        elif move_these == 'text':
            ky = 'texts'
        else:
            raise Exception('move_these which kind of objects unrecognized:'+move_these)
        ch = app_state['bearcubs'].get(ky,{}).get('bearcubs',{}).copy()
        #tweak_m44_inv = np.linalg.inv(tweak_m44)
        for k in list(ch.keys()):
            tweak_m44_1 = tweak_m44
            if 'mat44' in ch[k]:
                pos = ch[k]['mat44'][0:3,:]
            else:
                pos = ch[k]['pos']
            r = np.linalg.norm(pos)
            #if r<24:
            #    tweak_m44_1 = tweak_m44_inv
            rand_m44 = quat34.m44_from_q([1,0.02,0,0], normalize=True)
            rand_m44[0:3,3] = np.random.randn()*0.125
            if ky == 'lights': # Move lights much more.
                rand_m44[0:3,3] = np.random.randn()*1.25
            if 'mat44' in ch[k]:
                ch = c.update_in(ch, [k, 'mat44'], lambda m44:np.matmul(m44,rand_m44))
            elif 'pos' in ch[k]:
                ch = c.update_in(ch, [k, 'pos'], lambda v:quat34.m44v(rand_m44,v)[:,0])
        return c.assoc_in(app_state,['bearcubs', ky, 'bearcubs'],ch)

    def tweak_meshes(app_state):
        shapes = app_state['bearcubs'].get('shapes',{}).get('bearcubs',{})
        for k in list(shapes.keys()):
            if shapes[k]['viztype'] == 'mesh':
                verts = shapes[k]['verts']
                verts = verts+np.random.randn(*verts.shape)*0.1
                shapes = c.assoc_in(shapes, [k, 'verts'], verts)
        return c.assoc_in(app_state,['bearcubs', 'shapes', 'bearcubs'], shapes)

    def delete_obj_tweak(app_state, delete_these='mesh'):
        if delete_these == 'mesh':
            ky = 'shapes'
        elif delete_these == 'light':
            ky = 'lights'
        elif delete_these == 'text':
            ky = 'texts'
        else:
            raise Exception('move_these which kind of objects unrecognized:'+move_these)
        ch = app_state['bearcubs'].get(ky,{}).get('bearcubs',{}).copy()
        for k in list(ch.keys()):
            del ch[k]; break # Only delete one thing.
        return c.assoc_in(app_state,['bearcubs',ky,'bearcubs'], ch)

    def move_camera_tweak(app_state):
        q,v,f,_,_,_ = quat34.cam44TOqvfcya(app_state['camera']['mat44'])
        tweak_4q,tweak_r,tweak_v = quat34.m44TOqrv(tweak_m44)
        q1 = quat34.qq(tweak_4q, q)
        v1 = quat34.qv(tweak_4q, v)[:,0]
        new_m44 = quat34.qvfcyaTOcam44(q1,v1,f=f)
        return c.assoc_in(app_state,['camera','mat44'],new_m44)

    packs = []
    packs.append(['Create object',create_obj_tweak, 45])
    packs.append(['Move object',lambda app_state:move_obj_tweak(app_state,move_these='mesh'), 60])
    packs.append(['Create text',create_text_tweak, 45])
    packs.append(['Move text',lambda app_state:move_obj_tweak(app_state,move_these='text'), 60])
    packs.append(['Create light',create_light_tweak, 15])
    packs.append(['Move light',lambda app_state:move_obj_tweak(app_state,move_these='light'), 120])
    packs.append(['Move camera',move_camera_tweak, 60])
    packs.append(['Tweak mesh verts',tweak_meshes, 20])
    packs.append(['Delete light',lambda app_state:delete_obj_tweak(app_state,delete_these='light'), 50])
    packs.append(['Delete object',lambda app_state:delete_obj_tweak(app_state,delete_these='mesh'), 50])
    packs.append(['Delete text',lambda app_state:delete_obj_tweak(app_state,delete_these='text'), 50])

    q_cam0 = quat34.q_from_polarshift([0,0,-1],[1,0,0])
    cam44_0 = quat34.qvfcyaTOcam44(q_cam0,v=[-16,0,0],f=2.0)
    init_state = {'camera':{'mat44':cam44_0}, 'show_fps':True,
                  'current_task':0,'frames_left':packs[0][2], 'slowdown':1,
                  'onscreen_text':{'text':'This starting text should only last one frame','xy':[-1.25,0.9],'align':'left'}}

    txt_lines = ['This demo tests if Panda3D keeps up with "changes" to app_state.']
    txt_lines.append("SceneSync looks for changes using python's 'is'\nand will not detect inplace modification:")
    txt_lines.append("To make a change, don't mutate the app_state.\nInstead copy-on-modify while miminizing defensive copying using shallow copies when possible.")

    def _everyFrame_fn(app_state, inputs):
        return sequential_task_everyframe(app_state, inputs, txt_lines=txt_lines, packs=packs)

    pre_run_frames = 0
    for i in range(pre_run_frames):
        init_state = _everyFrame_fn(init_state, inputs)

    x = panda3dsetup.App(init_state, _everyFrame_fn)

def tree_demo():
    # Tests the ability to manipulate scenes with parent-child relationships.
    def make_mesh():
        cube = primitives.cube()
        base_color = np.random.random(3)
        scatter = np.random.random()**2.5
        cube['colors'] = np.ones([4,cube['verts'].shape[1]])
        for o in range(3):
            cube['colors'][o,:] = base_color[o]
        cube['colors'][0:3,:] = cube['colors'][0:3,:]+scatter*np.random.randn(3, cube['verts'].shape[1])
        cube['colors'] = np.maximum(0.0, np.minimum(1.0, cube['colors']))
        return cube
    def make_object():
        scale_factor = 0.5+0.25*np.random.randn()
        v = 1.0*np.random.randn(3); q = np.random.randn(4); q = q/np.linalg.norm(q)
        m44 = quat34.qrvTOm44(q,scale_factor*np.identity(3), v)
        return {'mat44':m44,**make_mesh(),'viztype':'mesh'}

    def select_random_path(sub_tree, path0=None):
        # Random path to an object in the tree.
        if path0 is not None:
            path = path0
        else:
            path = []
        if len(path) == 0:
            stop_chance = 0.0625
        elif len(path) == 1:
            stop_chance = 0.6
        else:
            stop_chance = 0.333
        if np.random.random()<=stop_chance or 'bearcubs' not in sub_tree:
            return path
        if sub_tree['bearcubs'] is None:
            raise Exception('Subtree bearcubs set to none, likely bug in this demo.')
        kys = list(sub_tree['bearcubs'].keys())
        if len(kys) == 0:
            return path
        k = kys[np.random.randint(len(kys))]
        if 'light' in k: # Don't select lights.
            return path
        return select_random_path(sub_tree['bearcubs'][k], path+['bearcubs', k])

    def derive_m44(m44, move=True, rot=True, scale=True):
        #Randomally changes a matrix 44.
        target_scale = 0.5; scale_drift = 0.25; scale_random = 0.1
        target_r = 1.75; r_drift = 0.1; v_random = 0.1; rot_random = 0.1;
        q,rm,v = quat34.m44TOqrv(m44); scale = rm[0,0]
        if scale:
            scale = scale+scale_random*np.random.randn()-scale_drift*(scale-target_scale)
        if rot:
            q = q+rot_random*np.random.randn(); q = q/np.linalg.norm(q)
        if move:
            r = np.linalg.norm(v)
            v = np.random.randn()*v_random+v*(1.0-r_drift*(r-target_r))
        return quat34.qrvTOm44(q,scale*np.identity(3), v)

    def move_branch_tweak(app_state, move=True, rot=True, scale=True, substeps=16):
        # Moves a random branch randomally, but biasing the random walk to keep things from drifting too far.
        app_state = app_state.copy()
        substeps_left = app_state.get('move_substeps',0)
        if substeps_left == 0: # Every time substeps runs out, choose a new path and xform.
            app_state['mat44_path'] = select_random_path(app_state)+['mat44']
            app_state['mat44_begin'] = np.copy(c.get_in(app_state, app_state['mat44_path']))
            app_state['mat44_end'] = derive_m44(app_state['mat44_begin'],move=move, rot=rot, scale=scale)
            app_state['move_substeps'] = substeps - 1
            #time.sleep(1.0)
            #print('Changing path')
        else:
            app_state['move_substeps'] = app_state['move_substeps']-1
            weight1 = 1.0-app_state['move_substeps']/(substeps-1+1e-100)
            mat44 = app_state['mat44_begin']*(1.0-weight1) + app_state['mat44_end']*weight1
            #import copy; app_state = copy.deepcopy(app_state) # DEBUG:
            #app_state = app_state.copy() #DEBUG
            #app_state['bearcubs'] = app_state['bearcubs'].copy() #DEBUG
            #print('m44 diff norm:', np.sum(np.abs(mat44-c.get_in(app_state, app_state['mat44_path']))))
            return c.assoc_in(app_state, app_state['mat44_path'], mat44)
        return app_state

    def reparent_branch_tweak(app_state):
        # Randomally reparent a branch.
        tree = app_state
        n_try = 0
        while True:
            branch_path = select_random_path(tree) #Paths to objects, so will end in ['bearcubs', some_key]
            destination_path = select_random_path(tree)+['bearcubs',str(np.random.random())]
            if '/'.join(branch_path) in '/'.join(destination_path):
                n_try = n_try+1
                if n_try>512:
                    raise Exception('Probable infinite loop here (bug in the testing code).')
                continue # The destination cannot be inside the branch.
            branch = c.get_in(tree, branch_path)
            tree1 = c.update_in(tree, branch_path[0:-1], lambda one_above_branch:c.dissoc(one_above_branch, branch_path[-1]))
            tree2 = c.assoc_in(tree1, destination_path, branch)
            break
        time.sleep(0.25)
        return tree2

    packs = []
    packs.append(['Move branch',lambda app_state:move_branch_tweak(app_state,move=True, rot=False, scale=False), 128])
    packs.append(['Rotate branch',lambda app_state:move_branch_tweak(app_state,move=False, rot=True, scale=False), 128])
    packs.append(['Scale branch',lambda app_state:move_branch_tweak(app_state,move=False, rot=False, scale=True), 128])
    packs.append(['Change hierarchy (fps cap to 4)',reparent_branch_tweak, 8])

    q_cam0 = quat34.q_from_polarshift([0,0,-1],[1,0,0])
    cam44_0 = quat34.qvfcyaTOcam44(q_cam0,v=[-6,0,0],f=2.0)
    init_state = {'camera':{'mat44':cam44_0}, 'show_fps':True,
                  'current_task':0,'frames_left':packs[0][2], 'slowdown':1,
                  'onscreen_text':{'text':'This starting text should only last one frame','xy':[-1.25,0.9],'align':'left'}}
    init_objs = {}
    for _ in range(12):
        init_objs[str(np.random.random())] = make_object()
    init_state = {**init_state, **{'mat44':quat34.m33vTOm44(np.identity(3)*0.5),'bearcubs':init_objs}}
    def v2m44(v):
        return quat34.m33vTOm44(np.identity(3),v)
    init_state['bearcubs']['light0'] = {'mat44':v2m44([64.0, 0.0, 0.0]),'color':[1024,512,256,1],'viztype':'light'}
    init_state['bearcubs']['light1'] = {'mat44':v2m44([0.0, 64.0, 0.0]),'color':[512,768,512,1],'viztype':'light'}
    init_state['bearcubs']['light2'] = {'mat44':v2m44([0.0, 0.0, 64.0]),'color':[256,256,1024,1],'viztype':'light'}

    txt_lines = ['This demo tests updating a hierarchy.']
    txt_lines.append('Updates include moving branches around as well as changing the tree structure.')
    txt_lines.append('When a branch is moved, all branches below it should move accordingly.')
    txt_lines.append('It can be hard to tell what branches belong to what; the tree starts with 1 level only.')

    def _everyFrame_fn(app_state, inputs):
        return sequential_task_everyframe(app_state, inputs, txt_lines=txt_lines, packs=packs)

    x = panda3dsetup.App(init_state, _everyFrame_fn)

def camera_demo():
    # Is the camera working properly?
    app_state = simple_init_state(make_random_mesh=False); app_state['show_fps'] = True
    app_state = c.assoc_in(app_state,['bearcubs','cube_grid','bearcubs'], make_cube_grid())

    def sc_format(x): #Scalar format
        if x<=0 and x>-1e-10: # Annoying sign z-fighting, the = in <= IS needed
            return '0.0000'
        out = np.format_float_positional(x, unique=False, precision=4)
        return out

    txt_colors = np.ones([3,4]); txt_colors[0:3,0:3] = 500*np.identity(3)
    txt_txt = ['x=4', 'y=4', 'z=4']
    txt_locations = [[4,0,0],[0,4,0],[0,0,4]]
    for i in range(3):
        txt_m44 = quat34.m33vTOm44(np.identity(3),txt_locations[i])
        txt_dict = {'text':txt_txt[i],'color':txt_colors[i,:]}
        ob = {**txt_dict, 'mat44':txt_m44,'viztype':'text'}
        app_state = c.assoc_in(app_state,['bearcubs','text '+txt_txt[i]], ob)

    def every_frame_func(app_state, inputs):
        mouse_state = inputs['mouse']; key_state = inputs['keyboard']
        key_clicks = inputs['type']
        screen_state = inputs['screen']
        app_state = app_state.copy()
        if key_state['escape']:
            app_state = nav3D.apply_mouse_camera_fn(app_state, inputs, nav3D.reset)
        f_camstate_deltax_deltay = nav3D.blender_fn(inputs, include_oddballs=True)
        app_state = nav3D.blender_cam_every_frame(app_state, inputs, include_oddballs=True)
        cam44 = app_state['camera']['mat44']
        objs = app_state['bearcubs'].copy()
        corners = mark_corners(cam44, clip_z_value=63.0/64.0, margin_from_edge=0.05, relative_size = 0.1)
        cursor_screen = np.zeros([3,1]); cursor_screen[:,0] = [mouse_state['x'],mouse_state['y'],63.0/64.0]
        one_mesh_dict = mark_on_screen(cam44, cursor_screen, ['screen_mesh'], relative_size = 1.0/32.0)
        objs['screen_mesh'] = {**one_mesh_dict['screen_mesh'],'viztype':'mesh'}
        stretch = app_state.get('stretch_to_screen',False)
        if 's' in key_clicks:
            stretch = not stretch
            app_state['stretch_to_screen'] = stretch
        for ck in corners.keys():
            objs[ck] = {**corners[ck],'viztype':'mesh'}
        app_state = c.assoc_in(app_state,['bearcubs'],objs)
        lines = ['Move the camera like Blender! The camera itself is a 4x4 matrix.']
        lines.append('The 4 "corner" spheres should stay fixed when the camera moves, and one sphere follows mouse.')
        if ';' in key_clicks:
            app_state['show_controls'] = not app_state.get('show_controls', False)
        if app_state.get('show_controls', False):
            lines = lines + ['; to toggle controls', 'Middle click+drag to orbit', 'Scroll to zoom (shift for the other zoom)']
            lines = lines + ['Shift middle drag to strafe', 'Ctrl middle drag to roll', 'C+mousemove to change clipping planes']
            lines = lines + ['Y+mousemove to shear view', 'A+mousemove+(shift) to shear clipping planes']
            lines = lines + ['Escape to reset view']
        else:
            lines.append('; to toggle controls (all 15 DOFs can be modified)')
        lines.append('Screen size:'+ str(screen_state)+ ' (stretch to screen='+str(stretch)+' , toggle with s)')
        lines.append('The camera parameters are shown below:')
        q,v,f,c_,y,a = quat34.cam44TOqvfcya(cam44)
        p_new = [q,v,f,c_,y,a]; letters = ['q','v','f','cl','y','a']
        p_old = app_state.get('old_cam_param',p_new).copy()
        desc = ['(Quaterion applied to cam)']
        desc.append('(Camera center)'); desc.append('(Camera f-ratio, > 10 is telophoto)')
        desc.append('(Clipping [near, far] plane)'); desc.append('(Screen-based shear)')
        desc.append('(View frustum shear, unusual effect)')
        old_numstrs = []; new_numstrs = []
        old_changes = app_state.get('last_last_changes',[False]*len(p_new))
        for i in range(len(p_new)):
            maybe_d = ''
            if i==2: # Scalars
                old_numstr = sc_format(p_old[i])
                new_numstr = sc_format(p_new[i])
            else:
                old_numstr = str([sc_format(p) for p in p_old[i]])
                new_numstr = str([sc_format(p) for p in p_new[i]])
            old_numstr = old_numstr.replace("'",'')
            new_numstr = new_numstr.replace("'",'')
            if old_changes[i]:
                maybe_d = desc[i]
            lines.append(letters[i]+' = '+new_numstr+' '+maybe_d)
            old_numstrs.append(old_numstr); new_numstrs.append(new_numstr)
        any_mouse_action = np.abs(mouse_state['x']-mouse_state['x_old']) + np.abs(mouse_state['y']-mouse_state['y_old']) > 0.001
        any_mouse_action = any_mouse_action or np.abs(mouse_state['scroll']-mouse_state['scroll_old'])>0 or key_state['escape']
        if any_mouse_action and f_camstate_deltax_deltay is not None:
            app_state['last_last_changes'] = [old_numstrs[ii] != new_numstrs[ii] for ii in range(len(p_new))]
            app_state['old_cam_param'] = p_new

        txt = {'xy':[-1.325,0.9],'align':'left','text':'\n'.join(lines)}
        app_state = c.assoc_in(app_state, ['onscreen_text'],txt)
        return app_state

    x = panda3dsetup.App(app_state, every_frame_func, panda_config={'view-frustum-cull':0})

def ui_demo():
    # A simple UI.
    app_state = simple_init_state(make_random_mesh=False); app_state['show_fps'] = True
    txt = 'Shows buttons, sliders, and textfields (TODO get this demo working).\nBlender camera controls.'
    app_state['onscreen_text'] = {'xy':[-1.325,0.9],'align':'left','text':txt}


    q_cam0 = quat34.q_from_polarshift([0,0,-1],[1,0,0])
    cam44_0 = quat34.qvfcyaTOcam44(q_cam0,v=[-6,0,0],f=2.0)
    #app_state['camera'] = ['mat44':cam44_0]
    #init_state = {'camera':{'mat44':cam44_0}, 'show_fps':True,
    #              'current_task':0,'frames_left':packs[0][2], 'slowdown':1}


    cube = {**primitives.cube(), **{'viztype':'mesh'}}
    def ui_fn(branch, inputs):
        m44 = np.copy(branch.get('mat44', np.identity(4)))
        m44[0,3] = m44[0,3] + 1
        return c.assoc(branch,'mat44',m44)
    cube['UI'] = {'click_lev-0':ui_fn}

    app_state = c.assoc_in(app_state, ['bearcubs','cube_button'], cube)
    def every_frame_func(app_state, inputs):
        app_state = nav3D.blender_cam_every_frame(app_state, inputs)
        app_state = uicore.global_everyframe(app_state, inputs)
        return app_state

    x = panda3dsetup.App(app_state, every_frame_func)
