# Automatic testing fns either work or break.
# Demo testing fns require human judgement.
import importlib, inspect, traceback, time
import reload

module_list_demos = ['test.demos.pythonpixels', 'test.demos.turingteapot']
module_list_auto = ['test.auto.tnomutate', 'test.auto.tgeom', 'test.auto.tmesh', 'test.auto.tquat43']

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

def run_f_catch_err(f_obj, print_reports=True):
    # Runs f, catching and reporting any errors. Returns false if an error is thrown, else the fns result.
    result = False
    try:
        result = f_obj()
    except Exception:
        if print_reports:
            traceback.print_exc()
    return result

def report_broken(print_reports = True):
    # Automatic tests the compute can tell if it works or not.
    #method_strings = _vcat([_list_methods_strings(m) for m in module_list_auto])
    auto_fnss = [_module_to_method_objs(m_string) for m_string in module_list_auto]
    fn_name2obj = _mergedicts(auto_fnss)

    if print_reports:
        print("**Running all unit tests**")

    failures = []
    for fname, fobj in fn_name2obj.items():
        result = run_f_catch_err(fobj)
        if not result:
            failures.append(fname)
    if print_reports:
        if len(failures)>0:
            for f in failures:
                print('Failed:', f)
        else:
            print('all', len(fn_name2obj), 'tests passed!')
    return failures

def broken_record(delay_seconds = 1.0, empty_lines = 6):
    # Keeps running the broken tests, waiting delay_seconds every loop.
    # It only reruns the tests broken on the first run.
    # Use while working on a test to avoid having to constantly run the test.
    auto_fnss = [_module_to_method_objs(m_string) for m_string in module_list_auto]
    fn_name2obj = _mergedicts(auto_fnss)
    failures = report_broken(print_reports = True)
    if len(failures) == 0:
        print('No need to re-try, all tests working.')
        return
    while True:
        for _ in range(empty_lines):
            print('')
        reload.reload_user_py_catcherr()
        auto_fnss = [_module_to_method_objs(m_string) for m_string in module_list_auto] # TODO: only in the module(s) that have failures.
        fn_name2obj = _mergedicts(auto_fnss) # Need to refresh this every time reload is called.
        all_passed = True
        for f in failures:
            f_obj = fn_name2obj[f]
            if not run_f_catch_err(f_obj):
                print('Test failure:', f)
                all_passed = False
        if all_passed:
            print('All broken tests fixed, exiting test loop.')
            break
        time.sleep(delay_seconds)

def list_demos(m_string=None):
    # Map from fname to fn.
    if m_string is not None:
        f_name2obj = _module_to_method_objs(m_string)
    else:
        demo_fnss = [_module_to_method_objs(m_string) for m_string in module_list_demos]
        f_name2obj = _mergedicts(demo_fnss)
    int2fname = dict(zip(range(len(f_name2obj)),f_name2obj.keys()))
    return int2fname, f_name2obj

def run_demo(ix, m_string=None):
    # Runs the ix's demo.
    int2fname, f_name2obj = list_demos(m_string)
    return f_name2obj[int2fname[ix]]()
