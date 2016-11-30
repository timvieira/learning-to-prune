"""Penn treebank loader.

From Stanford-NLP documentation
http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/trees/BobChrisTreeNormalizer.html

We perform the following tree normalizations

1. [X] Nonterminals are stripped of alternants, functional tags and
   cross-reference codes

   Truncate all nonterminal labels before characters introducing annotations
   according to TreebankLanguagePack (traditionally, -, =, |)

2. [X] Empty elements (ones with nonterminal label "-NONE-") are deleted from
   the tree

3. [X] The null label at the root node is replaced with the label "ROOT".

4. [X] Recursively delete any nodes that do not dominate any words

5. [X] Delete X-over-X nodes.

   These occasionally occur as a result other normalizations.

6. [X] Relabel "PRT" and "PRT|ADVP" -> "ADVP"

Notes:

 - Handling of out-of-vocabulary words (OOV) happens elsewhere (see
   `ldp.parsing.unk.signature`).

"""
import re, sys
from path import path
from nltk import Tree
#from arsenal.terminal import yellow
#from arsenal.misc import open_diff
from ldp.parsing.util import remove_trace, binarize, collapse_self_transition


PTB_ROOT = path('data/LDC99T42/treebank_3/parsed/mrg/wsj/').expanduser()


class PTB(object):
    """
    Penn treebank loader.
    """

    def __init__(self, train, dev, test, other):
        self.train = train
        self.dev = dev
        self.test = test
        self.other = other

    @staticmethod
    def load_fold(dirs):
        # [2015-02-13] I tried using nltk.corpus.treebank. It was at least 2x
        # slower, but it's probably more robust.
        for d in sorted(dirs):
            for filename in sorted(d.glob('*.mrg')):
                with file(filename) as f:
                    contents = f.read()
                # split individual trees
                for chunk in re.compile(r'^\(\s*\(', re.M).split(contents):
                    if not chunk.strip():
                        continue
                    chunk = '( (' + chunk
                    yield Tree.fromstring(chunk, remove_empty_top_bracketing=True)

    @classmethod
    def standard_split(cls, root):
        """
        Use standard train/dev/test/other splits 2-21/22/23/24, respectively.
        """
        train = []; dev = []; test = []; other = []
        for d in path(root).listdir():
            if d.isdir():
                number = int(d.basename())
                # for some reason we drop sections < 2.
                if 2 <= number <= 21:
                    train.append(d)
                elif number == 22:
                    dev.append(d)
                elif number == 23:
                    test.append(d)
                elif number == 24:
                    other.append(d)
        train.sort()
        assert len(train) == 20 and len(test) == 1 and len(dev) == 1
        return cls(train, dev, test, other)


def preprocess(t0):
    "Binarize and remove traces."
    b = binarize(Tree('ROOT', [remove_trace(t0)]))
    t = collapse_self_transition(b)
#    for s in t.subtrees():
#        if s.label().endswith('PRT|ADVP') or s.label() == 'PRT':
#            #s.set_label('PRT')
#            s.set_label('ADVP')

#    if 0:
#    #if b != t:
#        print yellow % '-----'
#        print yellow % '"interesting" collapse happened'
#        print
#        print b
#        print
#        print t
#        print
#        open_diff(b, t)
#
#    if 0:
#        #t = collapse_unary_chains(b)
#        if t != b:
#            interesting = False
#            for s in t.subtrees():
#                if '-' in s.label() and not s.label().startswith('-'):
#                    if len(s.leaves()) not in [1, len(t.leaves())]:
#                        interesting = True
#                        break
#
#            if interesting:
#                print yellow % '-----'
#                print yellow % '"interesting" collapse happened'
#                #print t0
#                #print
#                #print b
#                #print
#                for s in t.subtrees():
#                    if '-' in s.label() and not s.label().startswith('-'):
#                        s.set_label(yellow % s.label())
#                #print
#                print t
#                print yellow % '-----'

    return t


def ptb(names='train', n=None, maxlength=None, minlength=None):
    """
    Binarize and remove traces. Can also filter by length.

    names: can be a list or string. For lists we union the folds.

    """
    if maxlength is None:
        maxlength = sys.maxint
    if minlength is None:
        minlength = 0
    if n is None:
        n = sys.maxint
    d = PTB.standard_split(PTB_ROOT)
    if isinstance(names, basestring):
        names = [names]
    i = 0
    for name in names:
        if i == n:
            break
        for t0 in d.load_fold(getattr(d, name)):
            b = preprocess(t0)
            s = b.leaves()
            # Note: we run `preproces` before checking length because it often
            # change the length of the sentence by deleting trace nodes.
            if minlength <= len(s) <= maxlength:
                yield (' '.join(s), b)
                i += 1
                if i == n:
                    break


def test_load():
    from arsenal.timer import timeit
    with timeit('load 10 sentences maxlength 8'):
        assert len(list(ptb('train', n=10, maxlength=8))) == 10
    d = PTB.standard_split(PTB_ROOT)
    # skip training because it's slow to load all of it.
    #with timeit('loading train'):
    #    train = list(d.load_fold(d.train))
    with timeit('load dev'):
        dev = list(d.load_fold(d.dev))
    with timeit('load test'):
        test = list(d.load_fold(d.test))

    print 'n_sentences: dev %s' % len(dev)
    print 'n_sentences: test %s' % len(test)
    assert len(dev) == 1700
    assert len(test) == 2416
    with timeit('load dev w/ preprocessing'):
        dev = list(ptb('dev'))
    assert len(dev) == 1700
    with timeit('load test w/ preprocessing'):
        test = list(ptb('test'))
    assert len(test) == 2416

    from ldp.prune.example import Setup
    s = Setup(grammar='medium', train=0, dev=3000, minlength=0, maxlength=1000000)
    assert len(s.dev) == 1700

    #s = Setup(grammar='medium', train=0, dev=0, minlength=0, maxlength=1000000)
    assert len(list(s.load('test'))) == 2416



def examples_from_strings(grammar, corpus):
    " Manually specified training example."
    from ldp.prune.example import Example
    for s in corpus:
        a = Tree.fromstring(s)
        b = preprocess(a)
        yield Example(' '.join(b.leaves()), grammar, gold=b)


if __name__ == '__main__':
    test_load()
    assert len(list(ptb('dev', maxlength=10, n=100))) == 100
