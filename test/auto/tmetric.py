# Test metrics for unit-testing.
import numpy as np, math

def _dict2list(d):
    # Sort the keys!
    kys1 = list(d.keys()); kys1.sort()
    l = []
    for ky in kys1:
        l.append(ky)
        l.append(d[ky])
    return l

def _inty(x, precision=1e-4):
    # Numbers to integers (divided by precision).
    # Tuples to lists.
    # Dicts to lists with sorted keys.
    if x is None:
        return x
    if type(x) is list or type(x) is tuple:
        return [_inty(xi) for xi in x]
    if type(x) is dict:
        return _inty(_dict2list(x))
    if type(x) is np.ndarray:
        return _inty(list(x))
    if type(x) is str or (x is False) or (x is True):
        return x
    if math.isnan(x):
        return 'nan'
    return int(np.round(x/precision))

def approx_eq(x1, x2, precision=1e-4):
    # Uses inty, but also runs the precision at lower values incase we have a bad bounrady condition.
    # Works on tree datastructures, allowing more flexibility than numpy.allclose.
    phi = 0.5*(1.0+np.sqrt(5.0))

    precision1 = precision
    while precision1 > 1e-15:
        if str(_inty(x1)) == str(_inty(x2)):
            return True
        precision1 = precision1/phi
    return False

def is_all_true(x):
    for xi in x:
        if not xi:
            return False
    return True
