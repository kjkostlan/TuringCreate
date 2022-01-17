# Demos and autos.
import functool

module_list_demos = ['test.demos.pythonpixels']
module_list_auto = []

def _mergedicts(dict_list):
    out = {}
    for d in dict_list:
        out = {**out, **d}
    return out

def _module_to_method_objs(module_string): # also imports the module.
    module = importlib.import_module(module_string)
    leaves = dir(module)
    fullpath_strs = [module_string + '.' method_name for method_name in leaves]
    fn_objs = [getattr(module_string, leaf) for leaf in leaves]

    return dict(zip(fullpath_strs, fn_objs))

def report_broken(print_reports = True):
    # Automatic tests the compute can tell if it works or not.
    #method_strings = _vcat([_list_methods_strings(m) for m in module_list_auto])
    auto_fnss = [_module_to_method_objs(m_string) for m_string in module_list_demos]
    fn_name2obj = _mergedicts(auto_fnss)

    if print_reports:
        print("**Running all units tests**")

    failures = []
    for fname, fobj in fn_name2obj.items():
        try:
            result = fobj()
            if not result:
                failures.append(fname)
        except Exception:
            if print_reports:
                traceback.print_exc()
            failures.append(fname)
    return failures

def list_demos():
    # Map from int to fn.

def run_demo(ix):
    # Runs the ix's demo.
    int2demo = list_demos()
    return int2demo(ix)
