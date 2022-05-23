# Runs the UI for everything.
#We describe the tree with 'bearcubs' to denote the next branch.
# ui functions are put into the 'UI' key within this tree.
# Functions are of the form: {'click_lev+3':f(branch, mouse_state, key_state, mouse_clicks, key_clicks, screen_state)=>branch}
# 'click' 'type' supported for now TODO: allow more, including everyframe events with some sort of performance option.
# 'click_lev-3' = Operate on the branch 3 levels out, error if farther out than the root.
  # Note: two levels out in the tree is really only one level out in terms of object parenting.
# 'click_lev=3' = Operate on the branch 3 levels from the root toward us, set to zero to operate on the root.
import numpy as np
import c
import TapeyTeapots.meshops.quat34 as quat34

# TODO: hitboxes.
def get_fnlevs(ui, current_lev, inputs):
    # Returns [[f, lev],[f, lev],...] pairs of functions and the levels they operate on.
    # Errors are thrown if the requested level does not match.
    # A 'click' will trigger iff mouse_clicks has anything, etc.
    pairs = []
    kys = list(ui.keys()); kys.sort()
    active_kys = {'click':len(inputs['click'])>0,'type':len(inputs['type'])>0}
    for k in kys:
        pieces = k.split('_'); pieces.append('lev-0')
        type = pieces[0];
        if active_kys.get(type,False):
            level_tweak = pieces[1]
            lev_num = int(level_tweak.replace('lev','').replace('-','').replace('=',''))
            if '-' in level_tweak:
                lev_num = current_lev - lev_num
            pairs.append([ui[k], lev_num])
    return pairs

def cam_ray(app_state, mouse_state):
    # near, far cam ray.
    cam44 = app_state['camera']['mat44']
    cursor_screen = np.zeros([3,1]); cursor_screen[:,0] = [mouse_state['x'],mouse_state['y'],-1.0]
    pos_world_near = quat34.cam44_invv(cam44, cursor_screen)[:,0]
    cursor_screen[2,0] = 1.0
    pos_world_far = quat34.cam44_invv(cam44, cursor_screen)[:,0]
    return pos_world_near, pos_world_far

def _walk_everyframe(branch_old, app_state_new, path, inputs):
    # The path is the path in app_state_old.
    if 'UI' in branch_old:
        fn_levs = get_fnlevs(branch_old['UI'], len(path), inputs)
        not_clobbered = c.get_in(app_state_new, path, not_found=None) is not None # An earlier UI function may have clobbered us, making our actions moot.
        if len(fn_levs) > 0 and not_clobbered:
            for flv in fn_levs:
                f = flv[0]; lv = flv[1]
                piece = c.get_in(app_state_new, path[0:lv])
                piece_new = f(piece, inputs)
                if type(piece_new) is not dict:
                    raise Exception('the function did not return a dict.')
                app_state_new = c.assoc_in(app_state_new, path[0:lv], piece_new)
    if 'bearcubs' in branch_old:
        for k in branch_old['bearcubs'].keys():
            path1 = path + ['bearcubs', k]
            branch_old1 = branch_old['bearcubs'][k]
            app_state_new = _walk_everyframe(branch_old1, app_state_new, path1, inputs)
    return app_state_new

def global_everyframe(app_state, inputs):
    any_mouse_or_key = len(inputs['click']) + len(inputs['type']) > 0
    click_near, click_far = cam_ray(app_state, inputs['mouse']) # TODO: use this for hitboxes.
    if not any_mouse_or_key: # Optimization: avoid walking through the tree every frame.
        return app_state
    return _walk_everyframe(app_state, app_state, [], inputs)
