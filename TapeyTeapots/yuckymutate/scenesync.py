from panda3d.core import *
import numpy as np
import copy
from TapeyTeapots.meshops import quat34
import TapeyTeapots.yuckymutate.shadow as shadow
from direct.gui.OnscreenText import OnscreenText
from . import shapebuild

####################### Panda fns #######################

def logged_fn_call(log, f_name, module, f_obj, *args):
    # Log iff log is not None.
    if log is not None:
        log.append(['f_name', module, f_obj]+list(args))
    return f_obj(*args)

########################################################

def in_place_add_m44s_to_shadow(app, m44_shadow, m44globalkey):
    # Adds m44s to the shadow in place.
    I44 = np.identity(4)
    def f(branch, shadow_branch):
        if 'bearcubs' in shadow_branch:
            for ky in shadow_branch['bearcubs']:
                sbranch1 = shadow_branch['bearcubs'][ky]
                m44_bc = branch['bearcubs'][ky].get('mat44', I44)
                sbranch1[m44globalkey] = np.matmul(shadow_branch[m44globalkey], m44_bc)

    m44_shadow[m44globalkey] = app.get('mat44', I44)
    shadow.multiwalk([app, m44_shadow], m44_shadow, f, postwalk=False)

def make_m44_shadows(old_app, new_app):#, m44globalkey, filter=True):
    # Makes the m44 shadows.
    # The "standard" shadows are computed with: shadow.make_shadow([old_state, new_state], digf='diff')
    # But changing an m44 changes all children recursivly, so this will not do.
       # (we give Panda3D the locations of the ).

    moved_branch_ids = set() # Everything that was moved (that existed before).
    def diff_dig_plus(old_branch, new_branch):
        if old_branch is not None and new_branch is not None:
            change_m44 = new_branch.get('mat44',None) is not old_branch.get('mat44',None)
            if change_m44: # Mark all changed_m44 branches.
                moved_branch_ids.add(id(new_branch))
        return shadow.diff_dig([old_branch, new_branch], ck='bearcubs', ixs=[1])
    m44shadows = shadow.make_shadow([old_app, new_app], digf=diff_dig_plus)

    branch_shadow_pairs = [] # Branch and cooresponding shadow, for the paths that must be fully dug.
    def fill_pairs(new_branch, shadow_branch):
        if id(new_branch) in moved_branch_ids:
            branch_shadow_pairs.append([new_branch, shadow_branch])
    shadow.multiwalk([new_app, m44shadows], m44shadows, fill_pairs)

    tmp_key = 'scensync___bearcubs_TMP'
    for pair in branch_shadow_pairs: # Pairs should not overlap.
        new_branch = pair[0]; shadow_branch = pair[1]
        fullshadow = shadow.make_shadow([new_branch], ck='bearcubs', digf=True)
        if 'bearcubs' in fullshadow:
            shadow_branch[tmp_key] = fullshadow['bearcubs']
    for pair in branch_shadow_pairs:
        new_branch = pair[0]; shadow_branch = pair[1]
        if tmp_key in shadow_branch:
            shadow_branch['bearcubs'] = shadow_branch[tmp_key]
            del shadow_branch[tmp_key]

    #print('Phase1 shadow:', phase1_shadow, 'Final shadow:', out, 'len shadowmakr:', len(shadow_mark_pairs))

    # Filter them:
    #TODO # What does filter do?
    #if filter:
    #    def filter_f(old_branch, new_branch, shadow_branch):
    #        change = False
    #        if old_branch is None:
    #            old_render = {}
    #        if new_branch is None:
    #            new_branch = {}
    #        for k in set(list(old_branch.keys())).union(set(list(new_branch.keys()))):
    #            if k != 'bearcubs' and old_branch.get(k,None) is not new_branch.get(k,None):
    #                change = True
    #        if not change:
    #            del shadow_branch[m44globalkey]
    #    shadow.multiwalk([old_app, new_app, out], phase1_shadow, filter_f)

    return m44shadows

####################### Panda mutation functions #######################

def sync_objects1(log, old_render_branch, new_render_branch, panda_branch, pivot, modified_light_map):
    # Single sync step. Does not handle the m44s here.
    # Does NOT: do anything to the bearcubs. Set the m44 transform.

    if old_render_branch is None:
        old_render_branch = {}
    if new_render_branch is None:
        new_render_branch = {}

    # Prelim step: we don't worry about mat44 here.
    change = False
    if 'viztype' in old_render_branch or 'viztype' in new_render_branch:
        for k in set(list(old_render_branch.keys())+list(new_render_branch.keys())):
            if k != 'bearcubs' and k != 'mat44':
                if old_render_branch.get(k,None) is not new_render_branch.get(k,None):
                    change = True
    if not change:
        return

    # What types of objects do we have?
    viz_type_old = old_render_branch.get('viztype',None)
    viz_type_new = new_render_branch.get('viztype',None)
    if type(viz_type_old) is not list and type(viz_type_old) is not tuple:
        viz_type_old = [viz_type_old]
    if type(viz_type_new) is not list and type(viz_type_new) is not tuple:
        viz_type_new = [viz_type_new]
    viz_type_old = set(viz_type_old); viz_type_new = set(viz_type_new)
    viz_old_and_new_type = viz_type_old.union(viz_type_new)

    # Structural change check (i.e. it changes the structure of the mesh, etc)
    relevant_keys = shapebuild.get_relevant_keys(viz_old_and_new_type)
    all_keys = set(old_render_branch.keys()).union(new_render_branch.keys())
    changed_keys = []
    for ky in all_keys:
        if old_render_branch.get(ky,None) is not new_render_branch.get(ky,None):
            changed_keys.append(ky)
    big_update = len(relevant_keys.intersection(set(changed_keys)))>0

    change_mesh = 'mesh' in viz_old_and_new_type and big_update
    change_text = 'text' in viz_old_and_new_type and big_update
    change_light = 'light' in viz_old_and_new_type and big_update

    mesh_keys = shapebuild.mesh3_keys()
    #mesh_panda_objs = [panda_branch.get(k,None) for k in mesh_keys]

    if change_mesh:
        for k in mesh_keys:
            if k in panda_branch:
                mesh_obj = panda_branch[k]
                if mesh_obj is not None:
                    logged_fn_call(log, 'mesh_obj.removeNode', mesh_obj, mesh_obj.removeNode)
                del panda_branch[k]
        if 'mesh' in viz_type_new:
            mesh_panda_objs_new = logged_fn_call(log, 'shapebuild.build_mesh3', shapebuild, shapebuild.build_mesh3, 'meshy', new_render_branch)
            for k in mesh_keys:
                obj_new = mesh_panda_objs_new[k]
                panda_branch[k] = obj_new
                if obj_new is not None:
                    logged_fn_call(log, 'obj_new.reparent_to', obj_new, obj_new.reparent_to, pivot)

    text_ob = panda_branch.get('text',None)
    if change_text:
        if text_ob is not None:
            logged_fn_call(log, 'text_ob.removeNode', text_ob, text_ob.removeNode)
        if 'text' in viz_type_new:
            text_ob = logged_fn_call(log, 'shapebuild.build_text', shapebuild, shapebuild.build_text, new_render_branch)
            logged_fn_call(log, 'text_ob.reparent_to', text_ob, text_ob.reparent_to, pivot)
            panda_branch['text'] = text_ob

    if change_light:
        if 'light' in viz_type_new:
            modified_light_map[id(new_render_branch)] = new_render_branch
        else:
            modified_light_map[id(new_render_branch)] = None # Mark as deleted.

def sync_objects(log, old_render, new_render, panda, oldnew_shadow, pivot, modified_light_map):
    # Call this BEFORE the mat44 updating.
    if panda is None:
        raise Exception('The outer level cannot be None since it is mutated')
    def f(old_render_branch, new_render_branch, panda_branch, oldnew_shadow):
        if 'bearcubs' in oldnew_shadow:
            if 'bearcubs' not in panda_branch:
                panda_branch['bearcubs'] = {}
            for k in oldnew_shadow['bearcubs'].keys():
                if k not in panda_branch['bearcubs']:
                    panda_branch['bearcubs'][k] = {}
        sync_objects1(log, old_render_branch, new_render_branch, panda_branch, pivot, modified_light_map)
    shadow.multiwalk([old_render, new_render, panda, oldnew_shadow], oldnew_shadow, f, postwalk=False)

def update_xforms(log, panda_objects, new_render, m44_shadow, modified_light_map, m44globalkey):
    # Use m44_shadow to update the xforms.
    # Call AFTER the updating but BEFORE sync_lighs (since it populated modified_light_map)

    in_place_add_m44s_to_shadow(new_render, m44_shadow, m44globalkey)

    def update_xforms1(panda_branch, new_render_branch, m44_shadow_branch):
        if panda_branch is None:
            raise Exception('None panda branch here.')
        if m44globalkey in m44_shadow_branch:
            mat44 = m44_shadow_branch[m44globalkey]
            if 'light' in new_render_branch and id(new_render_branch) not in modified_light_map: # Small mat44 light update.
                modified_light_map[id(new_render_branch)] = {'mat44':mat44}
            xform = logged_fn_call(log, 'shapebuild.build_mat44', shapebuild, shapebuild.build_mat44, mat44)
            for k in shapebuild.mesh3_keys()+['text']: # Empty call to build mesh doesn't log.
                if k in panda_branch and panda_branch[k] is not None:
                    logged_fn_call(log, 'panda_branch[k].set_transform', panda_branch[k], panda_branch[k].set_transform, xform)

    shadow.multiwalk([panda_objects, new_render, m44_shadow], m44_shadow, update_xforms1)

def sync_camera(log, cam44_old, cam44, cam_obj, screen_state, stretch=False):
    # Camera math: Our camera xform is remove_w(norm_w(cam44*add_w(x)))
    # add_w adds w=1 to a 3 vector. norm_w divides by the w term.
    # Note: this is different than the standard 4x4 matrix for 3d xforms.
    # We set our own cam_xform on Panda's camera, which is the SAME as how 4x4 matrix.
    # We need to make Panda3d and our system equivalent:
    # remove_w(norm_w(cam44_panda*((cam_xform)^(-1)*add_w(x)))) = remove_w(norm_w(cam44*add_w(x)))
    # Which will be true if the matrix parts are the same:
    # cam44_panda*(cam_xform)^(-1) = cam44 => cam44_panda = cam44*cam_xform
    # So we have: cam_xform = (cam44^-1)*cam44_panda

    if cam44_old is cam44: # No change.
        return

    q0, v0, f0, cl0, y0, a0 = quat34.cam44TOqvfcya(cam44)
    #manual_q = False # Set traits manually instead of setting the cam44.
    #manual_v = False
    manual_fov = True
    manual_cl = True
    #if manual_q:
    #    q = q0
    #else:
    q = [np.sqrt(0.5),np.sqrt(0.5),0,0]
    #if manual_v:
    #    v = v0
    #else:
    v = [0,0,0]
    if manual_fov:
        f = f0
    else:
        f = 1.0
    if manual_cl:
        cl = cl0
    else:
        cl = [0.01,100]

    # Aspect ratio et al.
    w = screen_state[0]; h = screen_state[1]
    theta = 2.0*np.arctan(1.0/f)*180.0/np.pi
    fov_xy = [theta, theta]
    if not stretch:
        f1 = f*min(h,w)/max(h,w)
        theta1 = theta1 = 2.0*np.arctan(1.0/f1)*180.0/np.pi
        if w<=h:
            fov_xy = [theta,theta1]
        else:
            fov_xy = [theta1,theta]

    lens = base.camLens;
    logged_fn_call(log, 'lens.setFov', lens, lens.setFov, fov_xy[0], fov_xy[1])
    logged_fn_call(log, 'lens.setNearFar', lens, lens.setNearFar, cl[0], cl[1])
    pandas_cam44 = quat34.qvfcyaTOcam44(q, v, f, cl) # No y or a term.


    #cam44_panda = quat34.qvfcyaTOcam44(q, v, f, cl)

    nodem44 = np.matmul(np.linalg.inv(cam44),pandas_cam44)
    if nodem44[3,3]<0: # Sign fix.
        nodem44 = -nodem44
    nodem44 = nodem44/nodem44[3,3]

    setPos = False
    if setPos:
        nodem44[0:3,3] = [0,0,0]
    xform_obj = logged_fn_call(log, 'shapebuild.build_mat44', shapebuild, shapebuild.build_mat44, nodem44)
    logged_fn_call(log, 'cam_obj.set_transform', cam_obj, cam_obj.set_transform, xform_obj)
    #print('setting position:', v0)
    if setPos:
        logged_fn_call(log, 'cam_obj.setPos', cam_obj, cam_obj.setPos, v0[0],v0[1],v0[2])

def sync_onscreen_text(log, panda_objects, old_state, new_state):
    # Onscreen text is a very system system that does not care about the camera, etc.
    old_text = old_state.get('onscreen_text',None)
    new_text = new_state.get('onscreen_text',None)
    txt_obj = panda_objects.get('onscreen_text', None)
    if old_text is not new_text: # Any change.
        if txt_obj is not None:
            logged_fn_call(log, 'txt_obj.destroy', txt_obj, txt_obj.destroy)
        if new_text is not None:
            panda_objects['onscreen_text'] = logged_fn_call(log, 'shapebuild.build_onscreen_text', shapebuild, shapebuild.build_onscreen_text, new_text)

def sync_lights(log, panda_objects, light_modifications, the_magic_pivot):
    # light_modifications can be simple modifications with only 'mat44' or more complex modifications.
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
            logged_fn_call(log, 'render.clear_light', render, render.clear_light, light_obj)
            logged_fn_call(log, 'light_obj.removeNode', light_obj, light_obj.removeNode)
        if big_mod_light or (mod_k not in light_objs and not delete_light):
            light_obj = logged_fn_call(log, 'shapebuild.build_light', shapebuild, shapebuild.build_light, lmod, the_magic_pivot)
            logged_fn_call(log, 'render.set_light', render, render.set_light, light_obj); light_objs[mod_k] = light_obj
        if light_obj is not None and lmod is not None and 'mat44' in lmod:
            pos = lmod['mat44'][0:3,3];
            logged_fn_call(log, 'light_obj.setPos', light_obj, light_obj.setPos, pos[0], pos[1], pos[2])
    panda_objects['scenesync.das_blinkin_lights'] = light_objs

def sync(old_state, new_state, screen_state, panda_objects, the_magic_pivot, log):
    # We render a nested sequence of notes, and the state has 'type' and maybe 'bearcubs'.
    # 'type' can be string or a function (TODO).
    # TODO: 'lights', 'camera', etc are currently on the global level. We can put them
    #  at other levels i.e. to have a streetlight or drone object that has a light/camera in it.
    if old_state is None:
        old_state = {}
    if new_state is None:
        new_state = {}

    old_show_fps = old_state.get('show_fps',False)
    new_show_fps = new_state.get('show_fps',False)

    if old_show_fps != new_show_fps:
        logged_fn_call(log, 'base.setFrameRateMeter', base, base.setFrameRateMeter, new_show_fps)

    global_m44_key = 'scenesync.m44global'

    core_shadow = shadow.make_shadow([old_state, new_state], digf='diff')
    light_modifications = {}
    sync_objects(log, old_state, new_state, panda_objects, core_shadow, the_magic_pivot, light_modifications)

    m44globalkey = 'mat44_global'
    m44_shadow = make_m44_shadows(old_state, new_state)#, m44globalkey)
    update_xforms(log, panda_objects, new_state, m44_shadow, light_modifications, m44globalkey)

    sync_lights(log, panda_objects, light_modifications, the_magic_pivot)
    if 'camera' in old_state:
        old_camera = old_state['camera']['mat44']
    else:
        old_camera = None
    sync_camera(log, old_camera, new_state['camera']['mat44'], panda_objects['cam'], screen_state, stretch=new_state.get('stretch_to_screen',False))
    sync_onscreen_text(log, panda_objects, old_state, new_state)
