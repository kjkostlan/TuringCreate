# Runs the UI for everything.
#We describe the tree with 'bearcubs' to denote the next branch.
# ui functions are put into the 'UI' key within this tree.
# Functions are of the form: {'click_lev-3':f(branch, inputs)=>branch}. Inputs includes 'call_path'
# 'click' 'type' supported for now TODO: allow more, including everyframe events with some sort of performance option.
# 'click_lev-3' = Operate on the branch 3 levels out, error if farther out than the root.
  # Note: two levels out in the tree is really only one level out in terms of object parenting.
# 'click_lev=3' = Operate on the branch 3 levels from the root toward us, set to zero to operate on the root.
import numpy as np
import c
import TapeyTeapots.meshops.quat34 as quat34

################ Helper functions ###################

memoized = dict() #Store old app states and other precomputations here.

def all_event_types():
    # All valid event types.
    return ['click', 'unclick', 'type', 'untype', 'move', 'drag', 'everyframe', 'window']

def needsfocus_event_types():
    # Only triggered on a component if it has focus.
    # Overrided with 'needs_focus'.
    return ['type', 'untype']

def needs_collision_event_types():
    # Overrided with 'needs_collision'
    return ['click', 'unclick', 'move', 'drag']

def setsfocus_event_types():
    # These set the focus.
    return ['click']


################ Raytracing our click ###################

def hitbox2mesh(xxyyzz):
    # Hitbox to cube mesh. Less efficient than making dedicated hitbox code but oh well.
    TODO

def get_hit_mesh(branch):
    # Local coords. None if it does not have a hitbox.
    TODO

def cam_ray(app_state, mouse_state):
    # near, far cam ray.
    cam44 = app_state['camera']['mat44']
    cursor_screen = np.zeros([3,1]); cursor_screen[:,0] = [mouse_state['x'],mouse_state['y'],-1.0]
    pos_world_near = quat34.cam44_invv(cam44, cursor_screen)[:,0]
    cursor_screen[2,0] = 1.0
    pos_world_far = quat34.cam44_invv(cam44, cursor_screen)[:,0]
    return pos_world_near, pos_world_far

################ Single UI functions ###################

def is_mouse_over(click_local_near, click_local_far, branch):
    # widgets can collide with hitboxe(s) or with the mesh.
    # TODO: report where on the component as well.
    mesh = get_hit_mesh(branch)
    distance = trimesh.raycast_distance(mesh, click_local_near, click_local_far-click_local_near, relative_tol=0.0001)
    max_dist = np.linalg.norm(click_local_near-click_local_far)
    return distance<=max_dist

def responds_to1(ui):
    #ui = branch.get('UI',{})
    # Everything that the current branch responseds to, not including any 'bearcubs'
    # Does not account for focus or hitbox.
    # Returns a set.
    TODO

def get_fnlevs(ui, current_lev, inputs):
    #ui = branch.get('UI',{})
    # Returns [[f, lev],[f, lev],...] pairs of functions and the levels they operate on.
    # This fn does not filter by the hitbox or by whether the focus is on this component.
    pairs = []
    kys = list(ui.keys()); kys.sort()
    all_keys = all_event_types()
    active_ky_set = responds_to1(ui)
    for k in kys:
        pieces = k.split('_'); pieces.append('lev-0')
        type = pieces[0];
        if type in active_ky_set:
            focus = False
            level_tweak = pieces[1]
            lev_num = int(level_tweak.replace('lev','').replace('-','').replace('=',''))
            if '-' in level_tweak:
                lev_num = current_lev - lev_num
            pairs.append([ui[k], lev_num])
    return pairs

################ Tree UI functions ###################

def in_place_update_shadow(old_app, new_app, ui_shadows):
    # Returns the updated shadow which stores where we respond to all active keys.
    # Ui_shadow ['UI'] = True or False.
    # Uses diffs between the old_branch and the new_branch.
    if old_app==old_app:
        return ui_shadows # No changes.
    #ui_types = ui_shadows.keys() # All valid keys are here.
    #vals = ui_shadows.values()
    #nUI = len(ui_types)

    # TODO: we call this fn twice (once in thescenesync and once here):
    diff_shadow = shadow.make_shadow([old_app, new_app], digf='diff') # Digs into additions and deletions.

    link_key = 'uicore__correspondingUIshadow'
    ui_key = 'uicore__uikeyset'
    any_ui_diff = [False] # if there is ANY UI diff.
    #shadow.add_tree_link(ui_shadows, diff_shadow, link_key)
    def set_ui_keyset1(old_branch, new_branch, dshadow_branch):
        if 'UI' in old_branch or 'UI' in new_branch: # Skip both not having UI (presumably most objects).
            active_ky_set_old = responds_to1(shadow.get1(old_branch, 'UI', {}))
            active_ky_set = responds_to1(shadow.get1(new_branch, 'UI', {}))
            if active_ky_set_old != active_ky_set:
                any_ui_diff[0] = True
            dshadow_branch[ui_key] = active_ky_set
    shadow.multiwalk([old_branch, new_app, diff_shadow], diff_shadow, add_ui_keyset1)

    if any_ui_diff[0]: # Optimization to skip non-ui changes.
        def update_shadow1(dshadow_branch, ui_shadow_branch, ui_ty):
            # Populate bearcubs:
            if 'bearcubs' in dshadow_branch:
                ui_shadow_branch['bearcubs'] = ui_shadow_branch.get('bearcubs',{})
                for v in dshadow_branch['bearcubs'].values():
                    if v not in ui_shadow_branch['bearcubs']:
                        ui_shadow_branch['bearcubs'][v] = {'UI':False} # A deeper branch may set to True.
            has_ui = ui_ty in dshadow_branch.get(ui_key, set())
            ui_shadow_branch['UI'] = has_ui
        ui_types = list(ui_shadows.keys())

        def clean_ui_shadow(ui_shadow):
            # Removes pathways to nowhere.
            any_cubs = False # Recursivly this digs deeper.
            if 'bearcubs' in ui_shadow:
                for k in list(ui_shadow['bearcubs']).keys():
                    if not clean_ui_shadow(ui_shadow['bearcubs'],k):
                        del ui_shadow['bearcubs'][k]
                    else:
                        any_cubs = True
                if not any:
                    del ui_shadow['bearcubs']
            x = bool(ui_shadow.get('UI', False))
            return x

        for ui_type in ui_types:
            upd1 = lambda dshadow_branch, ui_shadow_branch: update_shadow1(dshadow_branch, ui_shadow_branch, ui_ty)
            shadow.multiwalk([diff_shadow, ui_shadows[ui_type], ui_ty], diff_shadow, upd1, postwalk=False) #postwalk must be false.
            clean_ui_shadow(ui_shadows[ui_type])

def add_navigation_links(shadow, ancestor_key, us_bearcub_key):
    # Link to the ancestor within ui_shadow.
    # Call AFTER the update shadow on each shadow that is used. Puts ancestors of new_app in ui_shadow.
    TODO

def add_global_m44s(new_app, ui_shadow, mat44_key):
    # Adds m44s to the UI shadow.
    mat44_update1(ancestor_mat44, branch)
    TODO

def clear_ui_fns(ui_shadow):
    # Call before add_ui_fns.
    TODO

def add_ui_fns(new_app, ui_shadow, ui_fn_key):
    # Adds functions *at the proper level*.
    get_fnlevs(ui, current_lev, inputs)
    #responds_to1?
    TODO

def add_collisions_local(new_app, ui_shadow, inputs, collision_key):
    #Adds the collisions, in local space.
    # No collisions = None.
    # Call after add_globals_m44

    TODO

def apply_uis(new_app, ui_shadows, ui_fn_key, ancestor_key, us_bearcub_key, collision_key, current_focus_id):
    # Call almost last. Returns the modified state.
    # Applies filters from collisions and focus.
    def_need_focus = set(needsfocus_event_types())
    def_need_coll = set(needs_collision_event_types())
    #print('UI shadow:', ui_shadows)
    'needs_focus'
    needs_collision_event_types
    'needs_collision'
    'call_path'
    TODO

def get_focus_id(new_app, last_focusing_camera, last_focusing_input, ui_shadows, current_focus_id):
    # Id of the object bieng focused.
    # Call AFTER apply ui.
    # TODO: If there are multible focii we use a heuristic to pick one. It is not strictly speaking the nearest collision.

    setsfocus_event_types
    TODO
    cam_ray(app_state, mouse_state)

################ The main function ################

def global_everyframe(app_state, inputs, memoize_id=None):
    # If memoize_id is set, avoids checking everything by storing the app_state as old_app as well as other precomputes in said ID.
    # Otherwise it has to check everything!
    kold = 'old_app'; kshd = 'shadows'; kbc = 'key_to_us1'; kan = 'ancestor_shadow'
    k44 = 'global_mat44'; kui = 'callback_fn'; kcl = 'local_collision'
    kfocus = 'uicore.focus';
    if type(memoize_id) is dict:
        memo = memoize_id
    elif memoize_id is None:
        memo = {} # A dev/null of sorts.
    else:
        memo = memoized.get(memoize_id, {})

    memo[kshd] = in_place_update_shadow(memo.get(kold,None), app_state, memo.get(kshd,{}))
    ui_shadows = memo[kshd]

    vals = list(ui_shadows.values()); vals.sort()
    for ui_shadow in vals:
        add_navigation_links(ui_shadow, kan, kbc)
        add_global_m44s(app_state, ui_shadow, k44)
        clear_ui_fns(ui_shadow)
        add_ui_fns(app_state, ui_shadow, kui)
        add_collisions_local(app_state, ui_shadow, inputs, kcl)

    app_state = apply_uis(app_state, ui_shadows, kui, kan, kbc, kcl, app_state.get(kfocus, None))
    last_focus = app_state.get(kfocus, {}).copy()
    new_focus_id = get_focus_id(app_state, last_focus.get('last_focusing_camera', None), last_focus.get('last_focusing_input', None), shadows, last_focus.get('focus_id', None))
    last_focus['focus_id'] = new_focus_id

    memo[kold] = app_state #Current for this frame = old for next frame.

    return app_state

def clear_memoized():
    for k in memoized:
        del memoized[k]

################ Core fns (using only our keys) ################


'TODO' # Stuff below deprecated.

def _walk_everyframe_old(branch_old, app_state_new, path, mat44_ancestors, inputs):
    # The path is the path in app_state_old.
    mat44 = np.matmul(mat44_ancestors, branch_old.get('mat44',np.identity(4))) # mat44 of the this branch in global space.

    click_near, click_far = inputs['mouse']['ray_global']
    inputs['mouse']['ray_local'] = [np.matmul(inv44, click_near), np.matmul(inv44, click_far)]

    if 'UI' in branch_old:
        fn_levs = []
        if branch_old.get('uicore.focus',False):
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
            app_state_new = _walk_everyframe_old(branch_old1, app_state_new, path1, mat44, inputs)
    return app_state_new

def ui_walk(branch, shadow_branch, path, inputs):
    m44_key = 'uicore.mat44global'

def set_focuses(app_state, inputs):
    # Sets the 'uicore.focus' key.
    # Mouse events set focuses.
    # 'event_bubbles' will block events.

    tmp_key = '_uicore.mouse_distance' # Mutate with this.


    if type=='click':
        ray0, ray1 = inputs['mouse']['ray_local']
        mesh = get_hit_mesh()
        TODO
        is_mouse_over(branch, click_local_near, click_local_far, mesh)
    TODO

################ Mutation functions ################

def filter_shadow(event_tys, shadow_branches):
    # Only use the branches that have changed.
    {ty: shadow_branches[ty] for ty in event_tys}
    return TODO

#def set_bearcub_mangle(branch, shadow_branch):
    # Renames bearcubs => uicore.bearcubs when we don't want to dig deeper.
#    TODO

#def clear_bearcub_mangle(branch):
#    TODO

#def removem44s(branch, shadow_branch, m44_key):
    # Shadow branch keeps paths that go to the branch.

#    if m44_key in x_branch:
#        del x_branch[m44_key]

#    if 'bearcubs' in shadow_branch:
#        TODO

#    def f(x_branch):
#        if m44_key in x_branch:
#            del x_branch[m44_key]
#    def digf(_, x_branch):

#    #scenesync.fpwalk(trees, f, f_ixs=(0,1), digf='diff', digf_ixs=(0,1), digf_after=True, ck='bearcubs')
#    scenesync.mutate_walk(None, new_tree, f, digf=None, retarget_here=None, digf_after=True)

def global_everyframe_old(app_state, inputs, memoize_id=None):
    # If memoize_id is set, avoids checking everything by keeping track of the app_state.
    if memoize_id is not None:
        memo = memoized.get(memoize_id, {})
        memoized[memoize_id] = memo
    old_app_state = memo.get('old_app_state',{})

    # Shadow has one key per evt ty.
    shadows = memo.get('ui_shadows',{}); memo['ui_shadows'] = shadow

    for evt_ty in all_event_types(): # Update
        shadows[evt_ty] = {}
        shadow[evt_ty] = updated_shadow(old_branch, new_branch, event_ty, shadow[evt_ty])

    evt_tys_trigger = triggered_evt_types(inputs) # Always everyframe. Sometimes other event types as well.

    # Add the transforms:
    m44_key = 'uicore.mat44global'
    shadow_filtered = filter_shadow(evt_tys_trigger, shadows)
    set_bearcub_mangle(app_state, shadow_filtered)
    scenesync.add_global_m44s(None, app_state, m44_key, do_everything=True)

    TODO # UI step here.

    scenesync.add_global_m44s(None, app_state, m44_key, do_everything=True)


    #
    #get_fnlevs(ui, current_lev, inputs)

    #TODO # stuff below is out of date needs to be replaced.

    #every_frames = mutant_memoize.get('every_frame')


    # mutant_memoize will mutate and store speedups.
    #any_mouse_or_key = len(inputs['click']) + len(inputs['type']) > 0
    #click_near, click_far = cam_ray(app_state, inputs['mouse']) # TODO: use this for hitboxes.
    #app_state = set_focuses(app_state, inputs)
    #inputs['mouse']['ray_global'] = [click_near, click_far]

    #if not any_mouse_or_key: # Optimization: avoid walking through the tree every frame.
        # TODO: dynamically adjust for everyFrame events with mutation.
    return app_state
    #return _walk_everyframe(app_state, app_state, [], np.identity(4), inputs)
