# Reload the python scripts.
import importlib
import os

file2contents = dict()

def list_py_files():
    py_files = []
    for root, subdirs, files in os.walk("."):
        if '__pycache__' not in root:
            for file in files:
                if file.endswith('.py') and 'reload.py' not in file:
                    py_files.append(root.replace('\\','/')+'/'+file)
    return py_files

def file2module(filename):
    no_dot_py = filename[0:-3]
    no_dot_slash =no_dot_py.replace("./",'')
    dots_not_slashes = no_dot_slash.replace('/','.')
    
    try:
        #https://stackoverflow.com/questions/301134/how-to-import-a-module-given-its-name-as-string
        return importlib.import_module(dots_not_slashes)
    except Exception:
        raise Exception('Module eval error for: '+ dots_not_slashes)

def reload_user_py_modules(print_reload=True):
    py_files = set(list_py_files())
    modules = [file2module(py_file) for py_file in py_files]

    #if print_reload:
    #    print('big reload step')
    for f in py_files:
        m = file2module(f)
        old_contents = file2contents.get(f,None)
        with open (f, "r") as myfile:
            new_contents=myfile.read()
        if old_contents != new_contents:
            file2contents[f] = new_contents
            # https://stackoverflow.com/questions/437589/how-do-i-unload-reload-a-python-module
            if print_reload:
                print('reloading:',m)
            importlib.reload(m)
