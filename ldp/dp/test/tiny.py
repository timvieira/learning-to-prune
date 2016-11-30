"""Test dynamic parser with small hand-specified grammars.

"""
from arsenal.terminal import colors
from arsenal.math import assert_equal
from ldp.prune.example import Example
from ldp.parse.grammar import Grammar

from ldp.dp.risk import InsideOut
from ldp.cp.viterbi import DynamicParser
from ldp.parse.leftchild import pruned_parser

from collections import namedtuple
Result = namedtuple('Result', 'reward coarse fine')


grammar = Grammar.tiny_grammar("""
# lexicon
0 N Papa
0 V ate
0 D the
0 N caviar
0 P with
0 N spoon
0 . .

# rules
0 ROOT S .
0 S NP VP
0 NP N
0 NP D N
0 VP V NP
0 PP P NP

-1 VP VP PP
0 NP NP PP

""")

# Note: Viterbi parsers should be invariant to annealing.
#grammar = grammar.anneal(1)
steps = 2
sentence = 'Papa ate the caviar with the spoon .'
#sentence = 'Papa ate the caviar .'
e = Example(sentence, grammar, None)
m = e.mask
m[:,:] = 1

print colors.yellow % 'Sentence:'
print e.sentence

p = DynamicParser(e.tokens, grammar, m.copy())
p.run()

gold = grammar.coarse_derivation(p.derivation())

print colors.yellow % 'Derivation:'
print gold

e.set_gold(gold)

if 1:
    # Now, prune away something important to that parse.
    magic = list(e.gold_spans)[0]
    m[magic] = 0
    print 'magic span:', magic

    if 1:
        # change the 'gold' parse to be the lower scoring one.
        p = DynamicParser(e.tokens, grammar, m.copy())
        p.run()
        gold = grammar.coarse_derivation(p.derivation())
        print colors.yellow % 'Derivation:'
        print gold
        e.set_gold(gold)


p = DynamicParser(e.tokens, grammar, m.copy())
p.run()

f = InsideOut(e, grammar, m*1.0, steps=steps)

def changeprop(I,K):
    "Change Propagation."
    p.change(I, K, 1-m[I,K])  # change
    d = p.derivation()
    c = grammar.coarse_derivation(d)
    r = e.recall(c)
    p.change(I, K, m[I,K])    # don't forget to change back!
    return Result(r, c, d)

def bsurrogate(I,K):
    "Brute-force annealed expected recall."
    m[I,K] = 1-m[I,K]
    s = InsideOut(e, grammar, m*1.0, steps=steps)
    m[I,K] = 1-m[I,K]
    return Result(s.val, None, None)

def bruteforce(I,K):
    "Brute-force CKY."
    m1 = m.copy()
    m1[I,K] = 1-m[I,K]
    s = pruned_parser(e.tokens, grammar, m1)
    c = grammar.coarse_derivation(s.derivation)
    r = e.recall(c)
    return Result(r, c, s.derivation)

def dynprogram(I,K):
    "Dynamic programming"
    return Result(f.est[I,K], None, None)


def run():
    for I,K in e.nodes:
        print colors.green % '%s span (%s,%s): "%s"' % ('prune' if m[I,K] else 'unprune',
                                                        I, K, ' '.join(sentence.split()[I:K]))
        cp = changeprop(I,K)
        bf = bruteforce(I,K)
        dp = dynprogram(I,K)
        bs = bsurrogate(I,K)
        if 0:
            print colors.yellow % 'New derivation:'
            print 'CP tree:', cp.coarse
            print 'BF tree:', bf.coarse
        print 'BF: %g' % bf.reward
        print 'CP: %g' % cp.reward
        print 'DP: %g' % dp.reward
        print 'BS: %g' % bs.reward
        assert_equal(bs.reward, dp.reward, 'bs v. dp', verbose=1)
        assert_equal(bf.reward, cp.reward, 'bf v. cp', verbose=1)
#        assert_equal(bf.reward, dp.reward, 'bf v. dp', verbose=1)

run()
