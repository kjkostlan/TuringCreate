from panda3d.core import *
import numpy as np
from TapeyTeapots.meshops import quat34
from direct.gui.OnscreenText import OnscreenText
from . import shapebuild

def _mutate2(x, k1, k2, v2):
    # modifies x in place, setting x[k1][k2], even if k1 isn't in x.
    if k1 not in x:
        x[k1] = {}
    x[k1][k2] = v2

def update_xforms(new_render_branch, mat44_ancestors, panda_objects_branch, modified_light_map, path):
    # Even if everything stays the same, xforms can change due to changing root xforms.
    #mat44 = np.matmul(new_render_branch.get('mat44', np.identity(4)), mat44_ancestors) # Wrong order.
    mat44 = np.matmul(mat44_ancestors, new_render_branch.get('mat44', np.identity(4)))
    if 'light' in new_render_branch: # Light update the mat44
        _mutate2(modified_light_map, '.'.join(path), 'mat44', mat44)
    xform = shapebuild.build_mat44(mat44)
    for k in list(shapebuild.build_mesh3('',None).keys())+['text']:
        if k in panda_objects_branch and panda_objects_branch[k] is not None:
            panda_objects_branch[k].set_transform(xform)
    if 'bearcubs' in new_render_branch:
        for ky in new_render_branch['bearcubs'].keys():
            update_xforms(new_render_branch['bearcubs'][ky], mat44, panda_objects_branch['bearcubs'][ky], modified_light_map, path+[ky])

def sync_renders(old_render_branch, new_render_branch, mat44_ancestors, panda_objects_branch, pivot, modified_light_map, path):
    # This relies on the vanilla side not mutating anything, just making shallow defensive copies.
    # We mutate panda_objects_branch of course.
    # modified_light_map is modified in-place.
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

    old_light = old_render_branch.get('light',None)
    new_light = new_render_branch.get('light',None)

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
    change_light = old_light is not new_light
    ident_m44 = np.identity(4)
    change_xform = False
    for k_o in ['mat44', 'pos']:
        if k_o in new_render_branch or k_o in old_render_branch:
            if old_render_branch.get(k_o, ident_m44) is not new_render_branch.get(k_o, ident_m44):
                change_xform = True
            else:
                break
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

    if change_light:
        if new_light is not None:
            modified_light_map['.'.join(path)] = new_light
        else:
            modified_light_map['.'.join(path)] = None # Mark as deleted.
    if (change_xform or change_light) and new_light is not None: # Update light poistion this-level.
        _mutate2(modified_light_map, '.'.join(path), 'mat44', mat44)
    bearcubs_old = old_render_branch.get('bearcubs', {})
    bearcubs_new = new_render_branch.get('bearcubs', {})
    k_set = set(list(bearcubs_old.keys())+list(bearcubs_new.keys()))

    if 'bearcubs' not in panda_objects_branch:
        panda_objects_branch['bearcubs'] = {}

    for k in k_set:
        ch_old = bearcubs_old.get(k,None)
        ch_new = bearcubs_new.get(k,None)
        if ch_new is not None and k not in panda_objects_branch['bearcubs']:
            panda_objects_branch['bearcubs'][k] = {}
        panda_branch1 = panda_objects_branch['bearcubs'].get(k,{})
        if ch_old is not ch_new:
            sync_renders(ch_old, ch_new, mat44, panda_branch1, pivot, modified_light_map, path+[k])
        elif change_xform and ch_new is not None:
            update_xforms(ch_new, mat44, panda_branch1, modified_light_map, path+[k])

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

def sync_lights(panda_objects, light_modifications, the_magic_pivot):
    # Fun with lights.
    # render.clear_light() # Empty is global clear light, if we need this than oops.
    light_objs = panda_objects.get('scenesync.das_blinkin_lights',{})
    for mod_k in light_modifications.keys():
        lmod = light_modifications[mod_k]
        delete_light = lmod is None or lmod is False
        big_mod_light = False; light_obj = light_objs.get(mod_k,None)
        if type(lmod) is dict:
            kys = lmod.keys()
            for ky in lmod:
                if ky != 'mat44':
                    big_light_mod = True
        if mod_k in light_objs and (delete_light or big_mod_light):
            render.clear_light(light_obj)
            light_obj.removeNode()
        if big_mod_light or (mod_k not in light_objs and not delete_light):
            light_obj = shapebuild.build_light(lmod, the_magic_pivot)
            render.set_light(light_obj); light_objs[mod_k] = light_obj
        if light_obj is not None and lmod is not None and 'mat44' in lmod:
            pos = lmod['mat44'][0:3,3];
            light_obj.setPos(pos[0], pos[1], pos[2])
    panda_objects['scenesync.das_blinkin_lights'] = light_objs

def sync(old_state, new_state, screen_state, panda_objects, the_magic_pivot):
    # We render a nested sequence of notes, and the state has 'type' and maybe 'bearcubs'.
    # 'type' can be string or a function (TODO).
    # TODO: 'lights', 'camera', etc are currently on the global level. We can put them
    #  at other levels i.e. to have a streetlight or drone object that has a light/camera in it.
    if old_state is None:
        old_state = {}
    if new_state is None:
        new_state = {}

    if new_state.get('show_fps',False):
        base.setFrameRateMeter(True)
    else:
        base.setFrameRateMeter(False)

    light_modifications = {}
    sync_renders(old_state, new_state, np.identity(4), panda_objects, the_magic_pivot, light_modifications, [])
    sync_lights(panda_objects, light_modifications, the_magic_pivot)
    if 'camera' in old_state:
        old_camera = old_state['camera']['mat44']
    else:
        old_camera = None
    sync_camera(old_camera, new_state['camera']['mat44'], panda_objects['cam'], screen_state, stretch=new_state.get('stretch_to_screen',False))
    sync_onscreen_text(panda_objects, old_state, new_state)
