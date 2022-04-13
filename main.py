# A simple interface that allows running code.
import test.summary as tests
import reload
import sys, traceback

def _k_or_v2k(x, k_or_v):
    # Returns the k that matches key or value.
    if k_or_v in x:
        return k_or_v
    elif type(k_or_v) is str:
        for k in x.keys():
            if k_or_v.lower() in x[k].lower():
                return k
    else:
        return False

def _to_int(x):
    try:
        return int(x)
    except:
        return str(x)

def _nonfalse(x): # There is probably an easier way.
    if (x != False and x is not None) or (type(x) is int):
        return True
    return False

if __name__ == "__main__":
    print('Turing create launched...')
    while True:
        for i in range(4):
            print('')
        x = str(input('Select what to run, or "h" for help:')).lower().strip()
        if x=='q':
            quit()
        reload.reload_user_py_catcherr()
        if x=='':
            x = 'h'
        try:
            if x=='h':
                print('"t" to run all unit tests ("t r" to repeat), "d" to list demos, "d 123" to run the 123\'rd demo, "e xyz" to eval xyz, q to quit.')
            elif x=='t':
                tests.report_broken()
            elif x=='t r':
                tests.broken_record()
            elif x[0]=='d':
                pieces = x.split(' ')
                module_dict = dict(zip(range(len(tests.module_list_demos)), tests.module_list_demos))
                if len(pieces) == 1:
                    print('Demo modules:\n')
                    print(module_dict)
                elif len(pieces) == 2: # Module specified, list fns within module.
                    k1 = _k_or_v2k(module_dict, _to_int(pieces[1]))
                    if _nonfalse(k1):
                        fname2int, fn_name2obj = tests.list_demos(module_dict[k1])
                        print('Demos within module: '+module_dict[k1]+'\n',fname2int)
                    else:
                        print('Unrecognized module or out-of-bounds int ix:',pieces[1])
                elif len(pieces) == 3: # Run the function.
                    k = _k_or_v2k(module_dict, _to_int(pieces[1]))
                    if _nonfalse(k):
                        fname2int, fn_name2obj = tests.list_demos(module_dict[k])
                        k1 = _k_or_v2k(fname2int, _to_int(pieces[2]))
                        if _nonfalse(k1):
                            f_obj = fn_name2obj[fname2int[k1]]
                            f_obj()
                        else:
                            print('Unrecognized fn within module or out-of-bounds int ix:',pieces[2])
                    else:
                        print('Unrecognized module or out-of-bounds int ix:',pieces[1])
                else:
                    print('Too many arguments given.')
            elif x[0]=='e':
                eval(x[1:])
            else:
                print('Not recognized option, use "h" for help.')
        except Exception:
                print('error:')
                traceback.print_exc()
