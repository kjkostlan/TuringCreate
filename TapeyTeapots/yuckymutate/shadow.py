# Collections can shadow eachother.
def rmkey(x, k):
    if type(x) is dict:
        if k in x:
            del x[k]

def get1(x,k, default=None):
    # Treats None like an empty dict.
    if x is None:
        return default
    return x.get(k, default)

def get_branches(trees, key, ck='bearcubs'):
    branches = []
    for tree in trees:
        branches.append(get1(tree,ck,{}).get(key,None))
    return branches

def diff_dig(trees, ck='bearcubs', ixs='all'):
    # Dig on the keys that are different, based on "is".
    # Goes on all bearcubs of all trees, unless ixs limits which tree(s) to pathdown.
    any_change = False
    if len(trees) == 0:
        return []
    for tree in trees:
        if tree is not trees[0]:
            any_change=True
    if not any_change:
        return []
    bearcubss = [get1(tree,ck,{}) for tree in trees]
    different_cubs = False
    for bearcubs in bearcubss:
        if bearcubs is not bearcubss[0]:
            different_cubs = True
    if not different_cubs:
        return []

    kys = set()
    if ixs=='all':
        ixs = list(range(len(trees)))
    for i in ixs:
        for bearcubs in bearcubss:
            kys = kys.union(set(bearcubss[i].keys()))

    out = []
    for k in kys:
        branches = get_branches(trees, k, ck='bearcubs')
        for branch in branches:
            if branch is not branches[0]:
                out.append(k)
    return out

def _dig_to_list(dig, walk_tree):
    dig_list = []
    if dig is True:
        dig_list = list(walk_tree[ck].keys())
    if type(dig) is list or type(dig) is tuple:
        dig_list = list(dig)
    return dig_list

def get_dig_fn(digf, ck='bearcubs'):
    if digf == 'diff': # Default look for distances.
        def digf(*trees):
            return diff_dig(trees, ck=ck)
    elif digf == True:
        def digf(*trees):
            bearcubss = [get1(tree,ck,{}) for tree in trees]
            kys = set()
            for bearcubs in bearcubss:
                kys = kys.union(set(bearcubs.keys()))
            return kys
    return digf

def _make_shadow_core(trees, digf, ck, format_as_tree):
    dig_list = digf(*trees)
    branchess = [get_branches(trees, k, ck) for k in dig_list]
    shadows = [_make_shadow_core(branches, digf, ck, format_as_tree) for branches in branchess]
    if format_as_tree and len(dig_list)>0:
        return {'bearcubs':dict(zip(dig_list, shadows))}
    return dict(zip(dig_list, shadows))

def make_shadow(trees, digf='diff', ck='bearcubs', format_as_tree=True):
    # Makes a shadow from trees that stores where we dig. Shadows are nested dicts with empty dicts at there end.
    digf = get_dig_fn(digf, ck=ck)
    return _make_shadow_core(trees, digf, ck, format_as_tree)

def multiwalk(trees, the_shadow, f, postwalk=True, ck='bearcubs', shadow_format_as_tree=True):
    # Calls f(*trees) at each level. Walks along the shadow (set shadow to trees[0] to walk along trees[0]).
    # Returns the result in tree-format and does not change the shadow.
    # Shadows are formatted as trees, i.e. ['bearcubs']['myKey']['bearcubs']['myKey2'][...]
      # Unless shadow_format_as_tree is False then we can simplify to ['myKey']['myKey2'].
    if shadow_format_as_tree:
        the_shadow = the_shadow.get(ck,{})
    if not postwalk:
        out = f(*trees)
    kys = the_shadow.keys()
    branchess = dict(zip(kys,[get_branches(trees, k, ck) for k in kys]))
    walks1 = dict(zip(kys,[multiwalk(branchess[k], the_shadow[k], f, postwalk, ck, shadow_format_as_tree) for k in kys]))
    if postwalk:
        out = f(*trees)
    if out is not None and len(kys>0): # f must return None or dict.
        out[ck] = walks1
    return out

def add_tree_link(tree, the_shadow, link_key, ck='bearcubs'):
    # Adds a link (shadow->tree) to the shadow. In-place modification.
    # None if the tree does not exist.
    def f(tree1, shadow1):
        shadow1[link_key] = tree1
    multiwalk([tree, the_shadow], the_shadow, f, ck=ck)

def add_ancestor_link(the_shadow, ancestor_key, ck='bearcubs'):
    # Adds links to the ancestor of shadow at each step. In-place modification.
    # Does not add it to root of tree. Shadow can also be a non-shadow tree.
    def f(shadow1):
        if ck in shadow1:
            for ky in shadow1[ck].keys():
                shadow1[ck][ky][ancestor_key] = shadow1
    multiwalk([the_shadow], the_shadow, f, ck=ck)

    #return fpwalk(trees, None, f_ixs=None, digf='diff', digf_ixs=None, ck='bearcubs', outputf=None, outputf_ixs=None, recordf=None, ancestor_key=None, dict_to_store_dig=None, root=True)

#def clean(x, clean_key, ck='bearcubs'):
    # Removes clean_key from x recursivly.
#    fpwalk([x], lambda x: rmkey(x, clean_key), digf=lambda x: clean_key in x, postwalk=True, ck=ck)

#def tree_acc(trees, f, f_ixs=None, digf='diff', digf_ixs=None, postwalk=True, ck='bearcubs', recordf=None):
#    # Generates a tree based on f(parents, bearcubs).
#    tmp_ancestor_key = 'shadow._parent_tmp'
#    tmp_dig_key = ''
#    #trees_plus_1 = trees.copy(); trees_plus_1.append()
#    if f_ixs is None:
#        travel_ix = 0
#    else:
#        travel_ix = f_ixs[0]
#
#    def calldigf(*trees):
#        TODO
#    def callf(trees):
#
#       return f()
#    postwalk=True
#    TODO

#def propagate_shadow(trees):
#    TODO

#def create_shadow(trees, TODO=TODO, ck='bearcubs'):
    # Updates one.
#    TODO
