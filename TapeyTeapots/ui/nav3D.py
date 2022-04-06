# 3D navigation basics.
import numpy as np
import c
from TapeyTeapots.meshops import quat34

def default_3D():
    return {'orbit_center':[0,0,0],'camera_pos':[10,0,0],'camera_f':2.0, 'camera_clip':[0.01,100],
            'view_rot_shear':[0.0,[1,0],[0,0,0,0]]}

def _constrain_3D(cam_3D):
    # In-place modification. Run after making a change.
    def limit_range(x, a, b):
        if x<a:
            return a
        if x>b:
            return b
        return x
    for v in [cam_3D['orbit_center'], cam_3D['camera_pos']]:
        for o in range(len(v)):
            v[o] = limit_range(v[o],-65536,65536)
    cam_3D['camera_f'] = limit_range(cam_3D['camera_f'],-65536,65536)
    cam_3D['view_rot_shear'][0] = limit_range(cam_3D['view_rot_shear'][0],-16,16)
    for o in range(2):
        cam_3D['camera_clip'][o] = limit_range(cam_3D['camera_clip'][o],1.0/65536,65536)
    for o in range(2):
        cam_3D['view_rot_shear'][1][o] = limit_range(cam_3D['view_rot_shear'][1][o],-16,16)
    for o in range(4):
        cam_3D['view_rot_shear'][2][o] = limit_range(cam_3D['view_rot_shear'][2][o],-16,16)

def get_orbit_distance(opts_3D):
    aim = np.asarray(opts_3D['orbit_center'])-opts_3D['camera_pos']
    return np.linalg.norm(aim)+1e-100

def get_camq(opts_3D):
    # Camera axes, unit length. +z is in the direction of the camera.
    aim = np.asarray(opts_3D['orbit_center'])-opts_3D['camera_pos']
    aim = aim/(np.linalg.norm(aim)+1e-100)
    if aim[0]*aim[0]+aim[1]*aim[1] < 1e-10: # Gimbal singularity.
        aim[0] = 0.0001
    camq_upright = quat34.camq_from_look(aim, up=None)
    radians = opts_3D['view_rot_shear'][0]
    rotate_camera_along_axis = quat34.axisangleTOq(aim, radians)
    return quat34.qq(rotate_camera_along_axis, camq_upright)

def get_cam_frame(opts_3D):
    # Orthogonal unit vectors that point in the cam x, y, z (ignores shears).
    camq = get_camq(opts_3D)
    return quat34.m33_from_q(camq)

def nav_to_camera(opts_3D):
    # Gets the cam44 matrix from the navigation options.
    camq = get_camq(opts_3D)
    oddball = opts_3D['view_rot_shear']
    v = opts_3D['camera_pos']; f = opts_3D['camera_f']; cl = opts_3D['camera_clip']
    return quat34.qvfcyaTOcam44(camq,v,f,c=cl,y=oddball[1],a=oddball[2])

########################## Camera controls ##################################

def reset(_o,_dx,_dy):
    return default_3D()

def orbit(opts_3D, delta_x, delta_y):
    # The most obvious one.
    frame = get_cam_frame(opts_3D); speed = -1.0 #Both + and - speed can work.
    new_aim = frame[:,2]+delta_x*speed*frame[:,0]+delta_y*speed*frame[:,1]
    q = quat34.q_from_polarshift(frame[:,2], new_aim)
    old_delta = opts_3D['camera_pos']-np.asarray(opts_3D['orbit_center'])
    new_delta = quat34.qv(q,np.expand_dims(old_delta,1))[:,0]
    return c.assoc(opts_3D, 'camera_pos', new_delta+opts_3D['orbit_center'])

def change_orbit_center(opts_3D, delta_x, delta_y):
    # The other most obvious one.
    frame = get_cam_frame(opts_3D); speed = -0.5
    dist = get_orbit_distance(opts_3D)
    shift = speed*dist*(frame[:,0]*delta_x+frame[:,1]*delta_y)
    opts_3D = opts_3D.copy()
    opts_3D['camera_pos'] = opts_3D['camera_pos']+shift
    opts_3D['orbit_center'] = opts_3D['orbit_center']+shift
    return opts_3D

def zoom_or_fov(opts_3D, delta_x, delta_y):
    # drag the y to move the camera (more common), or drag the x to change fov (less common).
    speed = 0.75
    opts_3D = opts_3D.copy()
    old_delta = opts_3D['camera_pos']-np.asarray(opts_3D['orbit_center'])
    new_delta = old_delta*np.exp(delta_y*speed)
    if np.linalg.norm(new_delta)<1e-5:
        new_delta = new_delta*1e-5/np.linalg.norm(new_delta)
    opts_3D['camera_pos'] = opts_3D['orbit_center']+new_delta
    opts_3D['camera_f'] = opts_3D['camera_f']*np.exp(delta_x*speed)
    return opts_3D

def rot_camera(opts_3D, delta_x, delta_y):
    # Only delta x matters.
    old_rot = opts_3D['view_rot_shear'][0]; speed = 1.0
    opts_3D = c.assoc_in(opts_3D, ['view_rot_shear',0], old_rot+delta_x*speed)
    return opts_3D

def clip_plane(opts_3D, delta_x, delta_y):
    # Delta x = near clipping plane. Delta y = far clipping plane.
    speed = 1.75
    opts_3D = opts_3D.copy()
    opts_3D['camera_clip'] = [opts_3D['camera_clip'][0]*np.exp(delta_x*speed),opts_3D['camera_clip'][1]*np.exp(delta_y*speed)]
    return opts_3D

def shear_view(opts_3D, delta_x, delta_y):
    # strech and shear, but no rotate.
    speed = 0.75
    old_y = opts_3D['view_rot_shear'][1]
    new_y = [old_y[0]+delta_y*speed, old_y[1]+delta_x*speed]
    opts_3D = c.assoc_in(opts_3D, ['view_rot_shear',1], new_y)
    return opts_3D

def shear_near_clipplane(opts_3D, delta_x, delta_y):
    # Very unusual effects arise from shearing the clipping planes.
    speed = 1.75
    old_a = opts_3D['view_rot_shear'][2]
    new_a = [old_a[0]+delta_x*speed, old_a[1]+delta_y*speed, old_a[2], old_a[3]]
    opts_3D = c.assoc_in(opts_3D, ['view_rot_shear',2], new_a)
    return opts_3D

def shear_far_clipplane(opts_3D, delta_x, delta_y):
    # Very unusual effects arise from shearing either of the clipping planes.
    speed = 1.75
    old_a = opts_3D['view_rot_shear'][2]
    new_a = [old_a[0], old_a[1], old_a[2]+delta_x*speed, old_a[3]+delta_y*speed]
    opts_3D = c.assoc_in(opts_3D, ['view_rot_shear',2], new_a)
    return opts_3D

############################## Combining fns ################################

def apply_mouse_camera_fn(app_state, mouse_state, f_camstate_deltax_deltay, drag_only=False):
    # Applies one of the "mouse drag" functions. drag_only can specify a mouse button or be any button.
    delta_x = mouse_state['x']-mouse_state['x_old']
    delta_y = mouse_state['y']-mouse_state['y_old']

    can_drag = True
    if drag_only is not False and drag_only is not None:
        TODO
    if can_drag:
        app_state = app_state.copy()
        app_state['nav3D_cam'] = f_camstate_deltax_deltay(app_state['nav3D_cam'], delta_x, delta_y)
        _constrain_3D(app_state['nav3D_cam'])
    app_state['camera'] = {'mat44':nav_to_camera(app_state['nav3D_cam'])}
    return app_state

def blender_fmap():
    # Maps blender's mouse and hotkey defaults to navigation fns.
    # Middle drag: orbit.
    # Shift+middle drag: pan (shift orbit center).
    # Scroll wheel: Zoom (by moving camera).
    TODO

#############################################################################

def empty_everyframe(app_state, mouse_state, key_state, mouse_clicks, key_clicks, screen_state):
     app_state = c.assoc(app_state,'nav3D_cam',app_state.get('nav3D_cam',default_3D()))
     app_state['camera'] = {'mat44':nav_to_camera(app_state['nav3D_cam'])}
     return app_state
