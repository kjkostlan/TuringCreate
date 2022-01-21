# Automatic testing fns either work or break.
# Demo testing fns require human judgement.
import importlib, inspect, traceback

module_list_demos = ['test.demos.pythonpixels', 'test.demos.turingteapot']
module_list_auto = ['test.auto.nomutate']

def _mergedicts(dict_list):
    out = {}
    for d in dict_list:
        out = {**out, **d}
    return out

def _module_to_method_objs(module_string, include_private=False): # also imports the module.
    module_obj = importlib.import_module(module_string)
    leaves0 = dir(module_obj)
    if not include_private:
        leaves0 = list(filter(lambda fname: fname[0] != '_', leaves0))
    fullpath_strs = []
    fn_objs = []
    for i in range(len(leaves0)):
        fn_obj = getattr(module_obj, leaves0[i])
        if callable(fn_obj) and str(inspect.signature(fn_obj)) == '()': # zero argument fn.
            fullpath_strs.append(module_string + '.' + leaves0[i])
            fn_objs.append(fn_obj)
    return dict(zip(fullpath_strs, fn_objs))

def report_broken(print_reports = True):
    # Automatic tests the compute can tell if it works or not.
    #method_strings = _vcat([_list_methods_strings(m) for m in module_list_auto])
    auto_fnss = [_module_to_method_objs(m_string) for m_string in module_list_auto]
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
    if print_reports:
        if len(failures)>0:
            for f in failures:
                print('Failed:', f)
        else:
            print('all tests passed!')
    return failures

def list_demos():
    # Map from fname to fn.
    demo_fnss = [_module_to_method_objs(m_string) for m_string in module_list_demos]
    fn_name2obj = _mergedicts(demo_fnss)
    fname2int = dict(zip(fn_name2obj.keys(), range(len(fn_name2obj))))
    return fname2int, fn_name2obj

def run_demo(ix):
    # Runs the ix's demo.
    fname2int, fn_name2obj = list_demos()
    int2fname = dict(zip(fname2int.values(),fname2int.keys()))

    return fn_name2obj[int2fname[ix]]()
