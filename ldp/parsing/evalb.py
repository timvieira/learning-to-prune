"""
Wrapper around EVALB script.
"""

import re
import numpy as np
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from ldp.parsing.util import item_tree, is_preterm, oneline
from nltk import Tree


def fpr(want_and_got, got, want):
    "Compute F-measure, precition, and recall from |want & got|, |got|, |want|."
    if want == 0 and got == 0:
        # Wanted empty set, got empty set.
        r = p = f = 1
    elif want == 0 or got == 0 or want_and_got == 0:
        # Assuming previous case failed, then we have zero f1.
        f = 0
        r = 1.0 if want == 0 else want_and_got * 1.0 / want
        p = 1.0 if got == 0 else want_and_got * 1.0 / got
    else:
        r = want_and_got * 1.0 / want
        p = want_and_got * 1.0 / got
        f = 2 * p * r / (p + r)
    return f,p,r


_ignore_labels = frozenset({"''", '``', '.', ':', ',', '.', '?', '!'})

def _relevant_items(t):
    return {s.label() for s in t.subtrees()
            # no credit for preterminals
            if (not is_preterm(s)
                # no credit for punctuation
                and s.label()[0] not in _ignore_labels
                # Not a binarized item (including check here means we don't have to
                # unbinarize trees before calling).
                and not s.label()[0].startswith('@'))
    }


def _recall_tree(gold_items, tree, I=0):
    """Help routine for computing F1 of `tree` against a set of `gold_items`.

    Returns a tripe of (C,B,K).
      C: The size of the intersection of relevant and retrieved
      B: The size of retrieved item set
      K: The end position of the item, which unknown until recurison returns
         (because tree is a label tree not an item tree).

    """
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
    if label in _ignore_labels or is_preterm(tree) or label.startswith('@'):
        return C, B, K
    else:
        return int((label, I, K) in gold_items) + C, B + 1, K


def evalb_unofficial(want, got):
    """Unofficial EVALB score. Seems to be correct, but should always run the
    official script to report results. However, we don't always run it because
    it's slow to run (due to system calls and IO).

    """
    if isinstance(want, basestring):
        # if `want` is a string, convert to tree
        want = Tree.fromstring(want)
    if isinstance(want, Tree):
        # if `want` is a tree, convert to a set of relevant items.
        want = _relevant_items(item_tree(want))
    if not isinstance(got, Tree):
        # if `got` is a string we failed to parse.
        return 0, 0, len(want)
    # Calling `relevant_items` is slow because it creates several temporary data
    # structures, it might be useful for debugging or inspecting error types
    # more closely.
    #got = _relevant_items(item_tree(got))
    got_and_want, got, _ = _recall_tree(want, got)
    return got_and_want, got, len(want)


def evalb(expect, got, verbose=0, executable='bin/EVALB/evalb'):
    """Compute evalb score by running the official script.

    Note: This function is fairly slow because it using a system call to the
    official script.

    """
    if len(got) == 0 or isinstance(got, basestring):
        return 0.0

    assert isinstance(expect, basestring)

    # Note: We write to /tmp so we don't overload any networked filesystems.
    with NamedTemporaryFile(prefix='/tmp/evalb-gld-', dir='tmp') as f, \
         NamedTemporaryFile(prefix='/tmp/evalb-out-', dir='tmp') as g:

        f.write(expect)
        f.write('\n')
        f.flush()

        g.write(oneline(got))
        g.write('\n')
        g.flush()

        p = Popen([executable, f.name, g.name], stdout=PIPE, stderr=PIPE)

        [x, y] = p.communicate()

    # Check that no (error) messages were written to stderr.
    assert not y, y

    if verbose:
        print x

    # Note: evalb will match this pattern twice, we want the first match.
    [f, _] = re.findall('Bracketing FMeasure\s+=\s+(.*)\n', x)

    # Don't return NaN! Return zero!
    f = float(f) / 100.0
    return 0.0 if np.isnan(f) else f


def evalb_many(expect, got, executable='bin/EVALB/evalb'):

    if len(got) == 0:
        return 0.0

    # Note: We write to /tmp so we don't overload any networked filesystems.
    with NamedTemporaryFile(prefix='/tmp/evalb-gld-', dir='tmp') as f, \
         NamedTemporaryFile(prefix='/tmp/evalb-out-', dir='tmp') as g:

        for t in expect:
            f.write(oneline(t))
            f.write('\n')
            f.flush()

        for t in got:
            g.write(oneline(t))
            g.write('\n')
            g.flush()

        p = Popen([executable,
                   f.name, g.name], stdout=PIPE, stderr=PIPE)

        [x, y] = p.communicate()

    #from arsenal.terminal import red
    #print red % '[evalb] %s' % x
    #print red % '[evalb] %s' % y
    assert not y, y

    [f, _] = re.findall('Bracketing FMeasure\s+=\s+(.*)\n', x)

    return float(f) / 100.0


def test_evalb_unofficial():
    "Compare evalb_unofficial to evalb (official)."
    from ldp.parsing.util import binarize
    from ldp.prune.example import Setup
    from ldp.parse.benchmark import Parser
    from ldp.parse import leftchild
    from arsenal.iterview import iterview
    for grammar in ['medium']:
        s = Setup(grammar=grammar, maxlength=30, train=0, dev=50)
        parser = Parser(leftchild, s.grammar, chomsky=0)
        for e in iterview(s.dev):
            m = e.mask
            state = parser(e, m)
            ucoarse = parser.decode(e, state.derivation)

            # TODO: Technically, average evalb([x_i]) over sentences is
            # *NOT* the same as evalb([x_1...x_n]) on the corpus.
            #
            # This is a "macro v. micro average" problem.

            unofficial = lambda a,b: fpr(*evalb_unofficial(a, b))[0]

            fb = unofficial(e.gold_unbinarized, binarize(ucoarse))
            f = unofficial(e.gold_unbinarized, ucoarse)
            h = evalb(e.gold_unbinarized, ucoarse)
            assert abs(fb - f) < 1e-8, "binarization shouldn't affect scores."
            assert abs(f - h) < 1e-4
    print '[test/evalb unofficial] pass'


if __name__ == '__main__':
    unofficial = lambda a,b: fpr(*evalb_unofficial(a, b))[0]
    assert abs(1.0 -
               evalb('(A (B b) (C c))',
                     Tree.fromstring('(A (B b) (C c))'))) <= 0.0
    assert abs(0.6667 -
               evalb('(A (B (B b)) (C (C c)))',
                     Tree.fromstring('(A0 (B (B b)) (C (C c)))'))) <= 0.0001

    assert abs(1.0 -
               unofficial('(A (B b) (C c))',
                          Tree.fromstring('(A (B b) (C c))'))) <= 0.0
    assert abs(0.6667 -
               unofficial('(A (B (B b)) (C (C c)))',
                          Tree.fromstring('(A0 (B (B b)) (C (C c)))'))) <= 0.0001

    test_evalb_unofficial()
