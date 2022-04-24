from panda3d.core import *
import numpy as np
from TapeyTeapots.meshops import quat34
from direct.gui.OnscreenText import OnscreenText
from . import shapebuild

def update_xforms(new_render_branch, mat44_ancestors, panda_objects_branch):
    # Even if everything stays the same, xforms can change due to changing root xforms.
    #mat44 = np.matmul(new_render_branch.get('mat44', np.identity(4)), mat44_ancestors) # Wrong order.
    mat44 = np.matmul(mat44_ancestors, new_render_branch.get('mat44', np.identity(4)))
    xform = shapebuild.build_mat44(mat44)
    for k in list(shapebuild.build_mesh3('',None).keys())+['text']:
        if k in panda_objects_branch and panda_objects_branch[k] is not None:
            panda_objects_branch[k].set_transform(xform)
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
    #mat44 = np.matmul(mat44_this, mat44_ancestors) # Wrong order.
    mat44 = np.matmul(mat44_ancestors, mat44_this)

    change_mesh = old_mesh is not new_mesh
    change_text = old_text is not new_text
    ident_m44 = np.identity(4)
    change_xform = old_render_branch.get('mat44', ident_m44) is not new_render_branch.get('mat44', ident_m44)

    mesh_keys = list(shapebuild.build_mesh3('None',None).keys())
    mesh_panda_objs = [panda_objects_branch.get(k,None) for k in mesh_keys]

    if change_mesh:
        for mesh_obj in mesh_panda_objs:
            if mesh_obj is not None:
                mesh_obj.removeNode()
        if new_mesh is not None:
            #myMesh = shapebuild.build_mesh('meshy',new_mesh)
            mesh_panda_objs_new = shapebuild.build_mesh3('meshy', new_mesh)
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
        xform = shapebuild.build_mat44(mat44)

        for k in mesh_keys:
            if k in panda_objects_branch:
                if panda_objects_branch[k] is not None:
                    panda_objects_branch[k].set_transform(xform)

    text_ob = panda_objects_branch.get('text',None)
    if change_text:
        if text_ob is not None:
            text_ob.removeNode()
        if new_text is not None:

            text_ob = shapebuild.build_text(new_text)
            text_ob.reparent_to(pivot)
            panda_objects_branch['text'] = text_ob
    if (change_text or change_xform) and (new_text is not None):
        xform = shapebuild.build_mat44(mat44)
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

def sync_camera(cam44_old, cam44, cam_obj, screen_state, stretch=False):
    # Camera math: Our camera xform is remove_w(norm_w(cam44*add_w(x)))
    # add_w adds w=1 to a 3 vector. norm_w divides by the w term.
    # Note: this is different than the standard 4x4 matrix for 3d xforms.
    # We set our own cam_xform on Panda's camera, which is the SAME as how 4x4 matrix.
    # We need to make Panda3d and our system equivalent:
    # remove_w(norm_w(cam44_panda*((cam_xform)^(-1)*add_w(x)))) = remove_w(norm_w(cam44*add_w(x)))
    # Which will be true if the matrix parts are the same:
    # cam44_panda*(cam_xform)^(-1) = cam44 => cam44_panda = cam44*cam_xform
    # So we have: cam_xform = (cam44^-1)*cam44_panda

    match_camera_more = True # Lighting has a bug when this is disabled.

    # The default camera points in the +y direction, while our ident q points in the -z direction:
    q = [np.sqrt(0.5),np.sqrt(0.5),0,0]; v = [0,0,0]
    lens = base.camLens;
    if match_camera_more:
        q_unused,v_unused,f,cl,_,_ = quat34.cam44TOqvfcya(cam44)
    else:
        f = 1.0; cl = [0.01, 100]

    w = screen_state[0]; h = screen_state[1]
    theta = 2.0*np.arctan(1.0/f)*180.0/np.pi
    if stretch:
        lens.setFov(theta,theta);
    else:
        f1 = f*min(h,w)/max(h,w)
        theta1 = theta1 = 2.0*np.arctan(1.0/f1)*180.0/np.pi
        if w<=h:
            lens.setFov(theta,theta1)
        else:
            lens.setFov(theta1,theta)

    lens.setNearFar(cl[0], cl[1])
    cam44_panda = quat34.qvfcyaTOcam44(q, v, f, cl)

    cam_xform = np.matmul(np.linalg.inv(cam44),cam44_panda)
    if cam_xform[3,3]<0: # Sign fix.
        cam_xform = -cam_xform

    cam_obj.set_transform(shapebuild.build_mat44(cam_xform))

def sync_onscreen_text(panda_objects, old_state, new_state):
    # Onscreen text is a very system system that does not care about the camera, etc.
    old_text = old_state.get('onscreen_text',None)
    new_text = new_state.get('onscreen_text',None)
    txt_obj = panda_objects.get('onscreen_text', None)
    if old_text is not new_text: # Any change.
        if txt_obj is not None:
            txt_obj.destroy()
        if new_text is not None:
            panda_objects['onscreen_text'] = shapebuild.build_onscreen_text(new_text)

def sync(old_state, new_state, screen_state, panda_objects, the_magic_pivot):
    # We render a nested sequence of notes, and the state has 'type' and maybe 'children'.
    # 'type' can be string or a function (TODO).
    # TODO: 'lights', 'camera', etc are currently on the global level. We can put them
    #  at other levels i.e. to have a streetlight or drone object that has a light/camera in it.
    if old_state is None:
        old_state = {}
    if new_state is None:
        new_state = {}
    lights_old = old_state.get('lights',[]) # Lights are seperate from the tree. TODO: change this.
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
        lights_panda_new = [shapebuild.build_light(light, the_magic_pivot) for light in lights_new]
        for lightp in lights_panda_new:
            render.set_light(lightp)
        panda_objects['das_blinkin_lights'] = lights_panda_new

    sync_renders(old_state, new_state, np.identity(4), panda_objects, the_magic_pivot)

    if 'camera' in old_state:
        old_camera = old_state['camera']['mat44']
    else:
        old_camera = None
    sync_camera(old_camera, new_state['camera']['mat44'], panda_objects['cam'], screen_state, stretch=new_state.get('stretch_to_screen',False))
    sync_onscreen_text(panda_objects, old_state, new_state)
