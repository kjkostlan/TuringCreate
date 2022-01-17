#!/usr/bin/env python

import traceback
import time
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
from direct.interval.IntervalGlobal import *

import high_vanilla
import time
from yuckymutate import scenesync, mousekey, reload
import numpy as np

class Demo:

    def __init__(self):

        self.appstate = high_vanilla.init_state()
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
        showbase.task_mgr.add(self.__every_frame, "every_frame")

        self.mouse_state = {}
        self.key_state = {}
        self.mouse_clicks = {}
        self.key_clicks = {}
        self.screen_state = [800,600]
        mousekey.set_up_mouse(showbase, self.mouse_state, self.mouse_clicks)
        mousekey.set_up_keys(showbase, self.key_state, self.key_clicks)

        showbase.run()

    def __every_frame(self, task):

        self.screen_state = [base.win.getProperties().getXSize(), base.win.getProperties().getYSize()]
        mousekey.update_mouse_pos(self.mouse_state, self.screen_state)

        try:
            appstate1 = high_vanilla.everyFrame(self.appstate, self.mouse_state, self.key_state, self.mouse_clicks, self.key_clicks, self.screen_state)
        except Exception:
            print('Every Frame Error:')
            print(traceback.format_exc())
            time.sleep(3)
        if appstate1 is None or type(appstate1) is not dict or len(appstate1.keys())==0:
            print('Every Frame returns None or empty app-state, which will be not used.')
            appstate1 = self.app_state
            time.sleep(3)

        self.appstate = appstate1
        for k in self.mouse_clicks.keys():
            self.mouse_clicks[k] = False
        for k in self.key_clicks.keys():
            self.key_clicks[k] = False

        # Sinc the app state:
        oldstate = self.old_appstate
        newstate = self.appstate
        try:
            scenesync.sync(self.old_appstate, self.appstate, self.panda_stuff_that_mutates, self.pivot)
        except Exception:
            print('Appstate synchronization error:')
            print(traceback.format_exc())
            time.sleep(3)
        self.old_appstate = self.appstate

        self.appstate['elapsed_frames'] = self.appstate['elapsed_frames']+1
        if np.mod(self.appstate['elapsed_frames'], 60) < 0.5:
            try:
              reload.reload_user_py_modules()
            except Exception:
                print('Importlib reload Error:')
                print(traceback.format_exc())
                time.sleep(3)

        return task.cont
if __name__ == '__main__':
    demo = Demo()
