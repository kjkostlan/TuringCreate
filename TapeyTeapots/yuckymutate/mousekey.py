# Mouse and keyboard events to interact with. Based on polling and counts.
from panda3d.core import *

def set_up_keys(showbase, key_state, key_clicks):
    def update_key_map(controlName, controlState):
        key_state[controlName] = controlState
        if controlState:
            key_clicks[controlName] = True

    for simple in "abcdefghijklmnopqrstuvwxyz0123456789-=[]\\/.,;`'":
        key_clicks[simple] = False
        key_state[simple] = False
        showbase.accept(simple,update_key_map, [simple, True])
        showbase.accept(simple+"-up",update_key_map, [simple, False])
    complexes = ["escape", "f1", "f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12",
                 "print_screen","scroll_lock", "backspace", "insert", "home", "page_up", "num_lock",
                 "tab",  "delete", "end", "page_down", "caps_lock", "enter", "arrow_left",
                 "arrow_up", "arrow_down", "arrow_right", "shift", #"lshift", "rshift",
                 "control", "alt", #"lcontrol", "lalt",
                 "space"] #"ralt", "rcontrol"]
    for complex in complexes:
        key_clicks[complex] = False
        key_state[complex] = False
        showbase.accept(complex,update_key_map, [complex, True])
        showbase.accept(complex+"-up",update_key_map, [complex, False])

def set_up_mouse(showbase, mouse_state, mouse_clicks):
    def update_mouse_map(controlName, controlState):
        mouse_state[controlName] = controlState
        if controlState:
            mouse_clicks[controlName] = True
    for i in range(5):
        mouse_clicks[str(i)] = False
        mouse_state[str(i)] = False
        showbase.accept('mouse'+str(i),update_mouse_map, [str(i), True])
        showbase.accept('mouse'+str(i)+"-up",update_mouse_map, [str(i), False])
    for pk in ['x_old','x','y_old','y']:
        mouse_state[pk] = 0.0

def update_mouse_pos(mouse_state, screen_state):
        mouseWatcher = base.mouseWatcherNode
        screenX = base.win.getProperties().getXSize()
        screenY = base.win.getProperties().getYSize()
        mouse_state['x_old'] = mouse_state['x']
        mouse_state['y_old'] = mouse_state['y']

        if mouseWatcher.hasMouse():
            mousy = mouseWatcher.getMouse()
            xy = [mousy.getX(), mousy.getY()]
            if screenY>screenX: # Cameras extend the FOV of the biggest dimension.
                xy[1] = xy[1]*screenY/screenX
            else:
                xy[0] = xy[0]*screenX/screenY
            mouse_pos = xy
            mouse_state['x'] = mouse_pos[0]
            mouse_state['y'] = mouse_pos[1]
