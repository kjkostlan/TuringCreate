# A simple interface that allows running code.
import test.summary as tests
import reload
import sys, traceback

if __name__ == "__main__":
    print('Turing create launched...')
    while True:
        for i in range(4):
            print('')
        x = str(input('Select what to run, or "h" for help:')).lower().strip()
        if x=='':
            x = 'h'
        try:
            if x=='h':
                print('"t" to run all unit tests, "d" to list demos, "d 123" to run the 123\'rd demo, "e xyz" to eval xyz, q to quit.')
            elif x=='t':
                tests.report_broken()
            elif x=='q':
                quit()
            elif x=='d':
                fname2int, fn_name2obj = tests.list_demos()
                print(fname2int)
            elif x[0] == 'd':
                ix = int(x[1:])
                tests.run_demo(ix)
            elif x[0]=='e':
                eval(x[1:])
            else:
                print('Not recognized option, use "h" for help.')
        except Exception:
                print('error:')
                traceback.print_exc()
        try:
            reload.reload_user_py_modules(print_reload=True)
        except Exception:
                print('error when load/reloading modules:')
                traceback.print_exc()