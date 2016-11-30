"""Test dynamic parser with small hand-specified grammars.

"""

from arsenal.terminal import yellow
from arsenal.math import assert_equal

from ldp.prune.example import Example
from ldp.parse.grammar import Grammar
from ldp.parse.leftchild import pruned_parser
from ldp.cp import viterbi

gr = Grammar.tiny_grammar("""
# lexicon
0 N Papa
0 V ate
0 D the
0 N caviar
0 P with
0 N spoon
0 . .

# test comment...

# rules
0 ROOT S .
0 S NP VP
0 NP N
0 NP D N
0 VP V NP
0 PP P NP

0 VP VP PP
1 NP NP PP

# another testing comment...

""")

sentence = 'Papa ate the caviar with the spoon .'; I,K = 2,7
#sentence = 'Papa ate the caviar .'; I,K = 0,2


if 0:
    gr = Grammar.tiny_grammar("""
# lexicon
0 X Papa
0 X ate
0 X food
0 . .

# rules

0 X X X
0 ROOT X .

""")

    sentence = 'Papa ate food  .'
    I,K = 0,2


def tri(i, j):
    return j*(j-1)/2 + i

def compare_charts():
    for I in range(e.N):
        for K in range(I+1, e.N+1):
            for X in range(gr.nsymbols):
                print [I,K,X], pp.chart[tri(I,K),X], p.chart[I,K,X]['score']
                assert_equal(pp.chart[tri(I,K),X], p.chart[I,K,X]['score'])



def parse(e, gr, m):
    p = viterbi.DynamicParser(e.tokens, gr, m.copy())
    p.run()
    pp = p.state()
    #pp = pruned_parser(e.tokens, gr, m)
    return pp


e = Example(sentence, gr, None)
m = e.mask
m[:,:] = 1

print yellow % 'Sentence:'
print e.sentence

pp = parse(e, gr, m.copy())

p = viterbi.DynamicParser(e.tokens, gr, m.copy())
p.run()

print yellow % 'Derivation:'
print gr.coarse_derivation(p.derivation())

assert_equal(pp.pushes, p.state().pushes, 'pushes', verbose=1, throw=0)
assert_equal(pp.pops, p.state().pops, 'pops', throw=0)

a = 0
print
print yellow % 'span(%s,%s) = %s: %s' % (I,K,a,' '.join(sentence.split()[I:K]))
p.change(I,K,a)
m[I,K] = a

pp = parse(e, gr, m)
assert_equal(pp.pushes, p.state().pushes, 'pushes', verbose=1, throw=0)
assert_equal(pp.pops, p.state().pops, 'pops', throw=0)

a = 1
print
print yellow % 'span(%s,%s) = %s: %s' % (I,K,a,' '.join(sentence.split()[I:K]))
p.change(I,K,a)
m[I,K] = a

pp = parse(e, gr, m)
assert_equal(pp.pushes, p.state().pushes, 'pushes', verbose=1, throw=0)
assert_equal(pp.pops, p.state().pops, 'pops', throw=0)
