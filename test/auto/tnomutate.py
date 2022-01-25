# Tests the non-mutation of c.
import c

def _pairwise(pairs):
    for p in pairs:
        if str(p[0]) != str(p[1]):
            return False
    return True

def test_one_level():
    # Tests one-level modifications and ensures they don't actually mutate.
    eq_pairs = []
    
    if _pairwise([[[1,2,3],[1,2,4]]]): # Test the pairwise itself a little.
        return False

    x = [1,2,3]
    x1 = c.assoc(x, 0, 10)
    x2 = c.update(x, 1, lambda x: x*16)
    eq_pairs.append([x1,[10,2,3]])
    eq_pairs.append([x,[1,2,3]])
    eq_pairs.append([x2,[1,32,3]])

    y = {"one":1, "two":2}
    y1 = c.assoc(y,"one",10)
    y2 = c.assoc(y1,"three",333)
    eq_pairs.append([y, {"one":1, "two":2}])
    eq_pairs.append([y1, {"one":10, "two":2}])
    eq_pairs.append([y2, {"one":10, "two":2, "three":333}])

    return _pairwise(eq_pairs)

def test_multi_level():
    # Multible levels at once. Mutation is copy on modify and shallow.
    eq_pairs = []

    x = [[1,2,3],{"ten":10, "twenty":20}]
    x1 = c.assoc_in(x,[0,0],[10, 20])
    x2 = c.assoc_in(x,[1,"twenty"],22.2)
    x3 = c.update_in(x,[0,0],lambda y: y+2)

    eq_pairs.append([x, [[1,2,3],{"ten":10, "twenty":20}]])
    eq_pairs.append([x1, [[[10, 20],2,3],{"ten":10, "twenty":20}]])
    eq_pairs.append([x2, [[1,2,3],{"ten":10, "twenty":22.2}]])
    eq_pairs.append([x3, [[3,2,3],{"ten":10, "twenty":20}]])

    return _pairwise(eq_pairs)
