# Outer level of the vanilla-pthon and pymesh application.
# No functions are allowed to mutate the app state.
import numpy as np
import c
import TapeyTeapots.pandawrapper as panda3dsetup
from TapeyTeapots.meshops import quat34, primitives

def simple_init_state(colored_lights=True): # Start simple.
    cam44 = np.identity(4)
    # 4x4 matrixes, as column vectors, are [x, y, z, origin]
    cam44[:,3] = [0,-5,0,1] # position. Camera seems to point in the y direction.
    the_mesh = {'mesh':makeRandomMesh(),'mat44':np.identity(4), 'children':{}}
    mat44_static  = np.identity(4)
    mat44_static[3,0:3] = [1,1,1]
    the_mesh_static = {'mesh':makeRandomMesh(),'mat44':mat44_static, 'children':{}}
    render = {'mat44':np.identity(4),
              'children': {'the_mesh':the_mesh, 'the_mesh_static':the_mesh_static}}
    if colored_lights:
        lights = [{'pos':[4.0, -32.0, 0.0],'color':[1,0,0,1]},
                  {'pos':[0.0, -32.0, 4.0],'color':[0,1,0,1]},
                   {'pos':[0.0, -32.0, 0.0],'color':[0,0,1,1]}]
    else:
        lights = [{'pos':[0,-32,0],'color':[1,1,1,1]}]
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
    out['string'] = str(prepend)+str(np.random.random())+str('OM')
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
    print_inputs(mouse_state, key_state, mouse_clicks, key_clicks)
    if np.mod(app_state['nframe'],85)==0:
        app_state = c.assoc_in(app_state, ['render', 'children', 'the_mesh', 'mesh'], makeRandomMesh())
        app_state = c.assoc_in(app_state, ['render', 'children', 'some_text', 'text'], make_random_text())
    return app_state

def make_cube_grid(n_x = 8, n_y = 8, n_z = 8, space_x = 1, space_y = 1, space_z = 1, radius = 0.1):
    # Put these into ['render', 'children'] of the world.
    cube_mesh = primitives.cube(); cube_mesh['verts'] = cube_mesh['verts']*radius
    obj = {'mat44': np.identity(4), 'mesh':cube_mesh}
    objs = []
    for i in range(n_x):
        x = space_x*(i-n_x*0.5)
        for j in range(n_y):
            y = space_y*(j-n_y*0.5)
            for k in range(n_z):
                z = space_z*(k-n_z*0.5)
                m44 = np.identity(4); m44[0:3,3] = [x,y,z]
                obj1 = obj.copy(); obj1['mat44'] = m44
                cols = np.ones([4,12]);
                cols[0,:] = i/(n_x-0.999); cols[1,:] = j/(n_y-0.999); cols[2,:] = k/(n_z-0.999)
                obj1 = c.assoc_in(obj1,['mesh','colors'],cols)
                objs.append(obj1)
    return dict(zip(range(len(objs)),objs))

############################### Demos #################################
def panda3d_basic_render_ui_test():
    # Can panda3d draw what we want it to and capture user inputs?
    x = panda3dsetup.Demo(simple_init_state(), everyFrame_simple_demo)

def panda3d_camerademo():
    # Is the camera working properly?
    #quat34.cam44v(cam44, vectors_3xn)
    #quat34.cam44_invv(cam44, vectors_3xn)
    #cam44TOqvfcya(cam44)
    #qvfcyaTOcam44(q,v,f=1.0,c=None,y=None,a=None)
    q = quat34.camq_from_look([-5, 2, -4]) # rotation of camera (3 DOF)
    v = [5,-3,10] # location of center of camera (3 DOF)
    f = 3.5 # f # kindof the f-number. A 90 degree FOV is an f of 1.0 and telophotos would be f>10. (1 DOF)
    cl = [0.002, 1000] # the [near, far] clipping plane. For very weird cameras far can be nearer. (2 DOF)
    y = [1, 0] # [y-stretch, y shear in x direction] of the camera image. Usually [1,0] (2 DOF)
    a = [0,0,0,0] # Clipping plane shear slope (applied after y), + is away from camera [near-x, near-y, far-x, far-y]. Usually all zero. (4 DOF)
    cam44 = quat34.qvfcyaTOcam44(q,v,f=1.0,c=cl,y=y,a=a)
    app_state = simple_init_state()
    app_state = c.assoc_in(app_state,['render', 'children'], make_cube_grid())
    #app_state['camera'] = {'mat44':cam44}
    x = panda3dsetup.Demo(app_state, everyFrame_default)