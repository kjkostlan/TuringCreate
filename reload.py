# Reload the python scripts.
import importlib
import os, traceback

file2contents = dict()

def list_py_files():
    py_files = []
    for root, subdirs, files in os.walk("."):
        if '__pycache__' not in root:
            for file in files:
                if file.endswith('.py') and 'reload.py' not in file:
                    py_files.append(root.replace('\\','/')+'/'+file)
    return py_files

def file2module(filename, throw_errors=True):
    no_dot_py = filename[0:-3]
    no_dot_slash =no_dot_py.replace("./",'')
    dots_not_slashes = no_dot_slash.replace('/','.')
    
    if throw_errors:
        #https://stackoverflow.com/questions/301134/how-to-import-a-module-given-its-name-as-string
        return importlib.import_module(dots_not_slashes)
    else:
        try:
            return importlib.import_module(dots_not_slashes)
        except Exception:
            return None

def reload_user_py_modules(print_reload=True, throw_errors=True):
    py_files = set(list_py_files())

    for f in py_files:
        old_contents = file2contents.get(f,None)
        with open (f, "r") as myfile:
            new_contents=myfile.read()
        if old_contents != new_contents:
            # https://stackoverflow.com/questions/437589/how-do-i-unload-reload-a-python-module
            if print_reload and old_contents is not None:
                print('reloading:',f)
            m = file2module(f, throw_errors=throw_errors)
            if m is not None:
                if old_contents is not None:
                    importlib.reload(m)
                file2contents[f] = new_contents

def reload_user_py_catcherr(print_reload=True): # Catches and prints errors, does not throw errors.
    try:
        reload_user_py_modules(print_reload=True)
    except Exception:
            print('error when load/reloading modules:')
            traceback.print_exc()
