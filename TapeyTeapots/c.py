#Handles datastructures such as trees in a non-mutation way.
# Uses "is" to check for updates.

def assoc(x, k, v): # Dict-only.
    if x.get(k,None) is v:
        return x
    x1 = x.copy()
    x1[k] = v
    return x1

def get_in(x,ks, not_found=None):
    if x is None:
        return not_found
    if len(ks)==0:
        return x
    return get_in(x.get(ks[0],None),ks[1:],not_found)

def assoc_in(x,ks, v):
    if x is None:
        x = {}
    if len(ks) == 0:
        return v
    k0 = ks[0]
    branch1 = assoc_in(x.get(k0,None),ks[1:],v)
    if branch1 is x.get(k0,None):
        return x
    x1 = x.copy()
    x1[k0] = branch1
    return x1

def update(x,k,f):
    val = x.get(k,None)
    val1 = f(val)
    if val1 is val:
        return x
    x1 = x.copy()
    x1[k] = val1
    return x1

def update_in(x,ks,f):
    v = get_in(x,ks)
    v1 = f(v)
    return assoc_in(x,ks,v1)
