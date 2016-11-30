"""
Utilities for manipulating/preprocessing/evaluating parse trees.
"""

from __future__ import division

from nltk import Tree


def oneline(t):
    "Write tree to a single line."
    return t._pformat_flat(nodesep='', parens='()', quotes=False)


def remove_trace(t):
    """Remove trace constituents and coarsen the corresponding labels.

    >>> t = Tree.fromstring('(S (S-TPC-2 (NP-SBJ (PRP (-LBR- -LBR-) a (-RBR- -RBR-)))) (VP (VBD b) (S (-NONE- *T*-2))) (. .))')
    >>> print remove_trace(t)
    (S (S (NP (PRP (-LBR- -LBR-) a (-RBR- -RBR-)))) (VP (VBD b)) (. .))

    """
    if is_terminal(t):
        assert isinstance(t, basestring), t
        return t
    label = t.label()
    if label == '-NONE-':  # trace
        return None
    if not label.startswith('-'):    # skip special labels like -LRB-
        label = label.split('-')[0]
        label = label.split('=')[0]
    new = filter(None, [remove_trace(c) for c in t])
    if not new:        # if no more children (and not originally a leaf)
        return None    # remove this item
    return Tree(label, new)


def binarize(t, right=False):
    """
    Left-binarization by default

      >>> print binarize(Tree.fromstring('(A a b c)'))
      (A (@A a b) c)

      >>> print binarize(Tree.fromstring('(A a (B b c (D d) e) f)'))
      (A (@A a (B (@B (@B b c) (D d)) e)) f)

    Example of right binarization:

      >>> print binarize(Tree.fromstring('(A a (B b c (D d) e) f)'), right=1)
      (A a (@A (B (@B (@B b c) (D d)) e) f))

    """

    if is_terminal(t):
        #assert isinstance(t, basestring), t
        return t

    label = t.label()
    children = list(t)
    new = [binarize(c) for c in children]
    if len(children) <= 2:
        return Tree(label, new)
    else:
        if right:
            curr = Tree('@' + label, [new[-2], new[-1]])
            for c in reversed(new[1:-2]):
                curr = Tree('@' + label, [c, curr])
            return Tree(label, [new[0], curr])
        else:
            curr = Tree('@' + label, [new[0], new[1]])
            for c in new[2:-1]:
                curr = Tree('@' + label, [curr, c])
            return Tree(label, [curr, new[-1]])


def was_binarized(t):
    return isinstance(t, Tree) and t.label().startswith('@')


def unbinarize(t):
    if is_terminal(t):
        return t
    assert not was_binarized(t)
    new_children = []
    for c in t:
        if was_binarized(c):
            _unbinarize_helper(c, new_children)
            continue
        new_children.append(c)
    return Tree(t.label(), [unbinarize(c) for c in new_children])


def _unbinarize_helper(t, flat):
    #assert is_binary(t), t
    assert was_binarized(t)
    for c in t:
        if was_binarized(c):
            _unbinarize_helper(c, flat)
        else:
            flat.append(c)


def list2tree(t):
    """Convert list representation of a tree into an actual tree.

    >>> print list2tree([1, [2, 3, 4]])
    (1 (2 3 4))

    >>> print list2tree([1, 2])
    (1 2)

    """
    if not isinstance(t, list):
        return t
    return Tree(t[0], map(list2tree, t[1:]))


def check_binary_tree(t):
    """Check if `Tree` `t` is binary.

    >>> is_binary(Tree.fromstring('(A a a a)'))
    False

    >>> is_binary(Tree.fromstring('(A a (A a a))'))
    True

    Warning null tree is not binary:

      >>> is_binary(Tree.fromstring('()'))
      False

    """
    if is_terminal(t):
        return True
    else:
        return len(t) <= 2 and all(check_binary_tree(c) for c in t)


def is_item_tree(t):
    if is_terminal(t):
        return True
    else:
        return isinstance(t.label(), tuple) and all(is_item_tree(c) for c in t)


def item_tree(t, i=0, terminals=True):
    """Convert to item tree.

    >>> t = Tree.fromstring('(A a (B b) c)')
    >>> item_tree(t)
    Tree(('A', 0, 3), [('a', 0, 1), Tree(('B', 1, 2), [('b', 1, 2)]), ('c', 2, 3)])

    >>> item_tree(t, terminals=False)
    Tree(('A', 0, 3), ['a', Tree(('B', 1, 2), ['b']), 'c'])

    """
    if is_terminal(t):
        if terminals:
            return (t, i, i+1)
        else:
            return t
    else:
        cs = []
        I = i
        k = i + 1
        for c in t:   # each child
            x = item_tree(c, i=i, terminals=terminals)
            if isinstance(x, Tree):
                (_, _, k) = x.label()
            else:
                k = i + 1
            i = k                          # shift
            cs.append(x)
        return Tree((t.label(), I, k), cs)


def item_time_tree(t, i=0, terminals=True):
    """Convert to item tree with time-step.

    >>> t = Tree.fromstring('(A a (B1 (B0 b)) c)')
    >>> item_time_tree(t)
    Tree(('A', 0, 3, 0), [('a', 0, 1, None), Tree(('B1', 1, 2, 1), [Tree(('B0', 1, 2, 0), [('b', 1, 2, None)])]), ('c', 2, 3, None)])

    """
    if is_terminal(t):
        if terminals:
            return (t, i, i+1, None)
        else:
            return t
    else:
        cs = []
        I = i
        k = i + 1
        T = None
        for c in t:   # each child
            x = item_time_tree(c, i=i, terminals=terminals)
            if isinstance(x, Tree):
                (_, _, k, T) = x.label()
            else:
                k = i + 1
            i = k                          # shift
            cs.append(x)
            if len(cs) > 1:
                time = 0
            else:
                assert len(cs) == 1
                if T is None:
                    time = 0
                else:
                    time = T + 1
        return Tree((t.label(), I, k, time), cs)


def item_tree_get_items(t):
    "Extract labels ((X,I,K) triples) from item tree."
    if not hasattr(t, 'subtrees'):
        return set()
    return {s.label() for s in t.subtrees()}


def item_tree_to_label(t, terminals=True):
    if not isinstance(t, Tree):
        if terminals:
            assert isinstance(t, tuple) and len(t) == 3, t
            return t[0]
        else:
            return t
    return Tree(t.label()[0], map(item_tree_to_label, t))


def tree_edges(t):
    """
    Extract hyperedges from a tree (derivation).

    >>> for x in tree_edges(Tree.fromstring('(A a (B b (C c)) d)')):
    ...     print x
    ('A', ['a', 'B', 'd'])
    ('B', ['b', 'C'])
    ('C', ['c'])

    """
    if is_terminal(t):
        return []
    return [(s.label(), [b.label() if isinstance(b, Tree) else b for b in s]) for s in t.subtrees()]


def fix_terminals(t, sentence):
    """Replace symbols at the leaves of tree `t` with tokens from `sentence`.

    Arguments:
     `t`: item tree
     `sentence`: sequence of tokens.

    >>> item_tree(Tree.fromstring('(A a (B b (C c)) d)'))
    Tree(('A', 0, 4), [('a', 0, 1), Tree(('B', 1, 3), [('b', 1, 2), Tree(('C', 2, 3), [('c', 2, 3)])]), ('d', 3, 4)])

    >>> print fix_terminals(item_tree(Tree.fromstring('(A a (B b (C c)) d)')), [1,2,3,4])
    (A 1 (B 2 (C 3)) 4)

    >>> print fix_terminals(item_tree('ROOT'), [1,2,3,4])
    (ROOT )

    """

    if not isinstance(t, Tree):   # handle empty parse which comes in as
        [root, _, _] = t
        return Tree(root, [])

    def swap(item):
        (_,i,_) = item
        return sentence[i]

    return Tree(t.label()[0], [fix_terminals(c, sentence) if isinstance(c, Tree) else swap(c) for c in t])


def recall_tree(gold_items, tree):
    """Compute recall of tree's item's wrt to gold_items.

    This implementation wants a regular tree (not an "item tree").

    >>> t = Tree.fromstring('(A a (B b (C c)) d)')
    >>> recall_tree(set([('A', 0, 4), ('B', 1, 3), 'impossible', 'also-impossible']), t)
    (2, 3, 4)

    """
    C, G,_ = _recall_tree(gold_items, tree, I=0)
    return C, G, len(gold_items)


def _recall_tree(gold_items, tree, I=0):
    "helper"
    if not isinstance(tree, Tree):
        return 0, 0, I + 1
    C = 0           # Size of the intersection of relevant and retrieved
    B = 0           # Size of retrieved item set
    K = I           # End position of the item (unknown until recurison returns)
    for s in tree:
        c, b, K = _recall_tree(gold_items, s, K)
        B += b
        C += c
    label = tree.label()
    return int((label, I, K) in gold_items) + C, B + 1, K


def collapse_unary_chains(tree, trail=True):
    """Removes unaries chains A -> B -> ... -> C, replacing them with A -> C. This
    transformation does not remove unary rules, but does ensure that only a
    single unary rule is needed.

    With trail:

      >>> print collapse_unary_chains(Tree.fromstring('(A (B (C (D d))))'), trail=1)
      (A-B-C-D d)

      >>> print collapse_unary_chains(Tree.fromstring('(A (B (C (D d e))))'), trail=1)
      (A-B-C (D d e))

      >>> print collapse_unary_chains(Tree.fromstring('(A (B (C c (D d e))))'), trail=1)
      (A-B (C c (D d e)))

      >>> print collapse_unary_chains(Tree.fromstring('(A (B b) (C (C (C c))))'), trail=1)
      (A (B b) (C-C-C c))

      >>> print collapse_unary_chains(Tree.fromstring('(A (B b) (C (C (C (E e f)))))'), trail=1)
      (A (B b) (C-C-C (E e f)))

    No trail:

      >>> print collapse_unary_chains(Tree.fromstring('(A (B (C (D d))))'), trail=0)
      (A d)

      >>> print collapse_unary_chains(Tree.fromstring('(A (B (C (D d e))))'), trail=0)
      (A (D d e))

      >>> print collapse_unary_chains(Tree.fromstring('(A (B (C c (D d e))))'), trail=0)
      (A (C c (D d e)))

      >>> print collapse_unary_chains(Tree.fromstring('(A (B b) (C (C (C c))))'), trail=0)
      (A (B b) (C c))

      >>> print collapse_unary_chains(Tree.fromstring('(A (B b) (C (C (C (E e f)))))'), trail=0)
      (A (B b) (C (E e f)))

    """

    def transform(t):

        if is_terminal(t):
            return t

        elif is_unary(t):
            l = t.label()
            [c] = t
            d = transform(c)
            if is_unary(d):         # max-unary-chain length of one.
                if trail:
                    d.set_label(l + '-' + d.label())
                else:
                    d.set_label(l)
                return d
            else:
                return Tree(l, [d])

        elif is_binary(t):
            l = t.label()
            [a, b] = t
            return Tree(l, [transform(a), transform(b)])

        else:
            raise AssertionError('Expected binary tree. %r' % t)

    return transform(tree)



def top_symbol(tree, trail=True):
    """Removes unaries chains A -> B -> ... -> C, replacing them with A -> C. This
    transformation does not remove unary rules, but does ensure that only a
    single unary rule is needed.

    With trail:

      >>> print top_symbol(Tree.fromstring('(ROOT (S (NP (NN Jewelry)) (VP (NNS Makers) (S (VP (VBP Copy) (NP (@NP (NNS Cosmetics) (NNS Sales)) (NNS Ploys)))))))'), trail=1)
      (ROOT-S
        (NP-NN Jewelry)
        (VP
          (NNS Makers)
          (S-VP
            (VBP Copy)
            (NP (@NP (NNS Cosmetics) (NNS Sales)) (NNS Ploys)))))

    """

    def transform(t):

        if is_terminal(t):
            return t

        elif is_unary(t):
            l = t.label()
            [c] = t
            d = transform(c)
            if not is_terminal(d):   # one symbol per span
                if trail:
                    d.set_label(l + '-' + d.label())
                else:
                    d.set_label(l)
                return d
            else:
                return Tree(l, [d])

        elif is_binary(t):
            l = t.label()
            [a, b] = t
            return Tree(l, [transform(a), transform(b)])

        else:
            raise AssertionError('Expected binary tree. %r' % t)

    return transform(tree)


def collapse_self_transition(tree):
    """Remove useless chains A -> A -> A -> B C ==> A -> B C

    >>> print collapse_self_transition(Tree.fromstring('(A (A (A (A A))))'))
    (A A)

    >>> print collapse_self_transition(Tree.fromstring('(A (A (A (A A A))))'))
    (A A A)

    >>> print collapse_self_transition(Tree.fromstring('(A (A (B (A (A A)))))'))
    (A (B (A A)))

    """

    def transform(t):

        if is_terminal(t):
            return t

        elif is_unary(t):
            l = t.label()
            [c] = t
            d = transform(c)
            if not is_terminal(d) and d.label() == l:
                return d
            return Tree(l, [d])

        elif is_binary(t):
            l = t.label()
            [a, b] = t
            return Tree(l, [transform(a), transform(b)])

        else:
            raise AssertionError('Expected binary tree. %r' % t)

    return transform(tree)



def is_unary(d):
    return isinstance(d, Tree) and len(d) == 1

def is_binary(d):
    return isinstance(d, Tree) and len(d) == 2

def is_terminal(d):
    return not isinstance(d, Tree)

def is_preterm(d):
    return len(d) == 1 and is_terminal(d[0])


def test_binarize_unbinarize():
    from ldp.parsing.ptb import PTB_ROOT, PTB
    from arsenal.terminal import yellow
    print yellow % 'basic binarize/unbinarize tests.'
    _test_binarize_unbinarize(map(Tree.fromstring,
                                  ['(A a)',
                                   '(A a b)',
                                   '(A a b c)',
                                   '(A a (B b) c)',
                               ]))

    print yellow % 'binarize/unbinarize PTB.'
    ptb = PTB.standard_split(PTB_ROOT)
    #_test_binarize_unbinarize(ptb.load_fold(ptb.train))
    _test_binarize_unbinarize(ptb.load_fold(ptb.dev))
    _test_binarize_unbinarize(ptb.load_fold(ptb.test))


def _test_binarize_unbinarize(trees):
    from arsenal.iterextras import iterview
    for t in iterview(list(trees)):
        b = binarize(t)

        # check that tree is indeed binary
        assert check_binary_tree(b), b

        # check roundtrip 'original tree'->'binary tree'->'original tree'
        u = unbinarize(b)
        assert u == t

        # unbinarize on an unbinarized tree should do nothing (other than copy
        # the tree)
        assert unbinarize(t) == t


if __name__ == '__main__':
    test_binarize_unbinarize()

    import doctest
    doctest.testmod()
