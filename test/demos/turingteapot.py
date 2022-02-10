# Outer level of the vanilla-pthon and pymesh application.
# No functions are allowed to mutate the app state.
import numpy as np
import c
import TapeyTeapots.main as panda3dsetup

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
        sel_edge = np.zeros([int(nVert*0.5), 2])
        sel_edge[:,0] = np.arange(int(nVert*0.5))
        sel_edge[:,1] = np.arange(int(nVert*0.5))+1
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

def everyFrame_simple_demo(app_state, mouse_state, key_state, mouse_clicks, key_clicks, screen_state):
    app_state = app_state.copy()
    app_state['nframe'] = app_state.get('nframe',0)+1
    print_inputs(mouse_state, key_state, mouse_clicks, key_clicks)
    if np.mod(app_state['nframe'],85)==0:
        app_state = c.assoc_in(app_state, ['render', 'children', 'the_mesh', 'mesh'], makeRandomMesh())
        app_state = c.assoc_in(app_state, ['render', 'children', 'some_text', 'text'], make_random_text())

    return app_state

def panda3d_basic_render_ui_test():
    # Can panda3d draw what we want it to and capture user inputs?
    x = panda3dsetup.Demo(simple_init_state(), everyFrame_simple_demo)