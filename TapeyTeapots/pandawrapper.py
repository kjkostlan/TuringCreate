#!/usr/bin/env python

import traceback
import time
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.interval.IntervalGlobal import *
from direct.filter.CommonFilters import CommonFilters
from panda3d.core import ConfigVariableString

import time
from TapeyTeapots.yuckymutate import scenesync, mousekey
import reload
import numpy as np

class App:

    def __init__(self, initial_state, every_frame_fn, panda_config=None):

        self.appstate = initial_state
        self.every_frame_fn = every_frame_fn
        self.appstate['elapsed_frames'] = 0
        self.old_appstate = {}
        self.showbase = showbase = ShowBase()
        self.panda_stuff_that_mutates = {}
        showbase.disable_mouse() # Disables the default mouse camara control.
        
        self.task_mgr = showbase.task_mgr
        self.mouse_watcher = showbase.mouseWatcherNode
        #self.gui_mouse_watcher = gui_mouse_watcher
        #showbase.camera.setFov(90)
        self.panda_stuff_that_mutates['cam'] = showbase.camera
        #print('lens:', showbase.camera.getLens())

        # The pivot is very strong: it holds the weight of the other meshes and text, and lights/camera/
        self.pivot = showbase.render.attach_new_node("pivot")
        self._task = showbase.task_mgr.add(self._every_frame, "every_frame")

        self.mouse_state = {}
        self.key_state = {}
        self.is_mouse_clicks = {}
        self.is_key_clicks = {}
        self.screen_state = [800,600]
        mousekey.set_up_mouse(showbase, self.mouse_state, self.is_mouse_clicks)
        mousekey.set_up_keys(showbase, self.key_state, self.is_key_clicks)

        #https://docs.panda3d.org/1.10/python/programming/render-to-texture/common-image-filters
        filters = CommonFilters(base.win, base.cam)
        filters.setSrgbEncode()

        #https://docs.panda3d.org/1.10/python/programming/configuration/accessing-config-vars-in-a-program
        if panda_config is not None:
            print('Custom panda config (warning: will last until app restart):', panda_config)
            for k,v in panda_config.items():
                var_ob = ConfigVariableString(k, 'NONE')
                #print('Specified in config file: ', var_ob.getValue())
                var_ob.setValue(str(v))
                #print('Value we will use: ', var_ob.getValue())

        # https://discourse.panda3d.org/t/userexit-without-kill-the-python-interpreter/10683
        try:
            showbase.run()
        except SystemExit as e:
            showbase.task_mgr.remove(self._task)
            showbase.destroy()

    def _every_frame(self, task):

        self.screen_state = [base.win.getProperties().getXSize(), base.win.getProperties().getYSize()]
        mousekey.update_mouse_pos(self.mouse_state, self.screen_state, stretch_to_screen=self.appstate.get('stretch_to_screen',False))

        try:
            mouse_clicks, key_clicks = mousekey.convert_mouse_and_key(self.is_mouse_clicks, self.is_key_clicks)
            ui = {'mouse':self.mouse_state,'keyboard':self.key_state,'click':mouse_clicks,'type':key_clicks, 'screen':self.screen_state}
            appstate1 = self.every_frame_fn(self.appstate, ui)
        except Exception:
            print('Every Frame Error:')
            print(traceback.format_exc())
            time.sleep(0.25)
            appstate1 = None
        if appstate1 is None or type(appstate1) is not dict or len(appstate1.keys())==0:
            print('Every Frame returns None or empty app-state, which will be not used.')
            appstate1 = self.appstate
            time.sleep(0.25)

        if appstate1 is not None:
            self.appstate = appstate1
        for k in self.is_mouse_clicks.keys():
            self.is_mouse_clicks[k] = False
        for k in self.is_key_clicks.keys():
            self.is_key_clicks[k] = False

        # Sync the app state:
        oldstate = self.old_appstate
        newstate = self.appstate
        try:
            scenesync.sync(self.old_appstate, self.appstate, self.screen_state, self.panda_stuff_that_mutates, self.pivot)
        except Exception:
            print('Appstate synchronization error:')
            print(traceback.format_exc())
            time.sleep(0.25)
        self.old_appstate = self.appstate

        self.appstate['elapsed_frames'] = self.appstate['elapsed_frames']+1
        if np.mod(self.appstate['elapsed_frames'], 60) < 0.5:
            try:
              reload.reload_user_py_modules()
            except Exception:
                print('Importlib reload Error:')
                print(traceback.format_exc())
                time.sleep(0.25)

        return task.cont
