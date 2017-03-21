"""
Parser benchmarks: Comparing parser implementations.
"""
from __future__ import division
import numpy as np
from time import time
from pandas import DataFrame

from arsenal.timer import Timer
from arsenal.terminal import red, yellow, green
from arsenal.math.pareto import show_frontier
from arsenal.viz.util import axman

from ldp.parse import leftchild_bp
from ldp.parse import leftchild
from ldp.parse import xprod
from ldp.cp import viterbi as agenda

from ldp.parse.grammar import Grammar
from ldp.prune.example import Example, cgw_f
from ldp.parsing.ptb import ptb
from ldp.parsing.evalb import evalb, evalb_unofficial
from ldp.parsing.util import unbinarize, item_tree, fix_terminals, binarize, oneline
from argparse import ArgumentParser

# adjust pandas print settings by console width
import pandas
from arsenal.terminal import console_width
pandas.set_option('display.width', console_width())


class Parser(object):

    def __init__(self, module, grammar, chomsky):
        self.module = module
        self.grammar = grammar
        self.chomsky = chomsky

    def __call__(self, e, keep):
        tokens = self.grammar.encode_sentence(e.sentence.split())
        return self.module.pruned_parser(tokens, self.grammar, keep)

    def decode(self, e, d):
        coarse = self.grammar.coarse_derivation(d)
        coarse = fix_terminals(item_tree(coarse), e.sentence.split())
        if self.chomsky:
            u = coarse.copy()
            u.un_chomsky_normal_form()
        else:
            u = unbinarize(coarse)
        return u


class AgendaParser(Parser):

    def __init__(self, grammar, chomsky):
        super(AgendaParser, self).__init__(agenda, grammar, chomsky)

    def __call__(self, e, keep):
        tokens = self.grammar.encode_sentence(e.sentence.split())
        pp = self.module.DynamicParser(tokens, self.grammar, keep)
        pp.run()
        return pp.state()


def _many_grammars():
    "Compares CKY implementation on multiple grammars."
    P = {}

    g = Grammar.load('data/bubs/eng.M2.P1')
    P['M2.P1'] = Parser(leftchild, g, chomsky=1)

    g = Grammar.load('data/bubs/eng.M2')
    P['M2'] = Parser(leftchild, g, chomsky=1)

    g = Grammar.load('data/medium')
    P['medium'] = Parser(leftchild, g, chomsky=0)

    g = Grammar.load('data/bubs/wsj_6')
    P['wsj_6'] = Parser(leftchild, g, chomsky=0)

    color = {
        'M2.P1':   'r',
        'M2':      'b',
        'medium':  'g',
        'wsj_6':   'y',
    }

    marker = {
        'M2.P1':   '*',
        'M2':      '*',
        'medium':  '*',
        'wsj_6':   '*',
    }

    return P, color, marker


def main():
    """
    Benchmark parsers
    """
    p = ArgumentParser()
    p.add_argument('--grammar', default='big')
    p.add_argument('--maxlength', type=int, default=None)
    p.add_argument('--examples', type=int, default=None)
    p.add_argument('--experiment',
                   choices=('default-parser', 'grammar-loops', 'grammars'),
                   required=True)

    p.add_argument('--fold', default='other')
    p.add_argument('--bylength', action='store_true')
    p.add_argument('--pareto', action='store_true')
    p.add_argument('--deps', action='store_true')
    p.add_argument('--profile', choices=('yep','cprofile'))
    p.add_argument('--delta', type=float, default=1.0,
                   help=('pruning probability (false-positive rate wrt oracle). '
                         '[0.0: oracle, 1.0: unpruned]'))
    p.add_argument('--policy')

    args = p.parse_args()

    from arsenal.profiling import profiler
    with profiler(args.profile):
        _main(args)


def _main(args):

    if args.deps:
        import StanfordDependencies
        dep = StanfordDependencies.get_instance(backend='subprocess')

    delta = args.delta
    assert 0 <= delta <= 1

    T = {}
    P = {}
    check_llh = 1

    color = {name: c for name, c in zip(sorted(P), 'rgbym'*len(P))}
    marker = {name: 'o' for name in P}

    if args.experiment not in ('grammars',):
        # benchmark default parser
        if args.grammar == 'medium':
            grammar_file = 'data/medium'
            g = Grammar.load('data/medium')
        elif args.grammar == 'big':
            grammar_file = 'data/bubs/wsj_6'
        chomsky = False
        g = Grammar.load(grammar_file)

    if args.experiment == 'default-parser':
        P['lchild'] = Parser(leftchild, g, chomsky=0)

    elif args.experiment == 'grammar-loops':
        # Experiment: grammar loops
        P['lcbptr'] = Parser(leftchild_bp, g, chomsky=chomsky)
        P['lchild'] = Parser(leftchild, g, chomsky=chomsky)
        #P['x-prod'] = Parser(xprod, g, chomsky=chomsky)
        #P['agenda'] = AgendaParser(g, chomsky=chomsky)

    elif args.experiment == 'grammars':
        #P, color, marker = _leftchild_v_dense_yj_on_many_grammars()
        P, color, marker = _many_grammars()
        check_llh = False
    else:
        raise ValueError('Fail to recognize experiment %r' % args.experiment)

    T = {x: Timer(x) for x in P}
    overall = []
    errors = []

    examples = ptb(args.fold, minlength=3, maxlength=args.maxlength, n=args.examples)

    if 1:
        examples = list(examples)
        np.random.shuffle(examples)

    _evalb_gold = {}
    _evalb_pred = {}
    for k, p in enumerate(P):
        _evalb_gold[p] = open('tmp/evalb-%s.gold' % k, 'wb')
        _evalb_pred[p] = open('tmp/evalb-%s.pred' % k, 'wb')

    if args.policy:
        from ldp.prune.features import Features
        theta = np.load(args.policy)['coef']

        policy_grammar = Grammar.load('data/bubs/wsj_6')    # FIXME: shouldn't be hardcoded!
        F = Features(policy_grammar, nfeatures=2**22)

    for i, (s, t) in enumerate(examples):
        print
        print green % 'Example: %s, length: %s' % (i, len(s.split()))
        print yellow % s

        e = Example(s, grammar=None, gold=t)
        sentence = e.tokens
        N = e.N

        if args.policy:
            e.tokens = policy_grammar.encode_sentence(e.sentence.split())
            keep = F.mask(e, theta)

        else:
            # don't prune anything
            keep = np.ones((N,N+1), dtype=np.int)
            for x in e.nodes:
                keep[x] = np.random.uniform(0,1) <= delta
            for x in e.gold_spans:
                keep[x] = 1

        data = []

        #ugold = Tree.fromstring(e.gold_unbinarized)

        if args.deps:
            dep_gold = dep.convert_tree(e.gold_unbinarized, universal=0)   # TODO: include function tags???
            dep_unlabel_gold = {(z.index, z.head) for z in dep_gold}
            dep_label_gold = {(z.index, z.deprel, z.head) for z in dep_gold}

        for parser in sorted(P):

            b4 = time()

            with T[parser]:
                state = P[parser](e, keep)

            wallclock = time()-b4

            s = state.likelihood
            d = state.derivation
            pops = state.pops
            pushes = state.pushes

            ucoarse = P[parser].decode(e, d)

#            print
#            print parser
#            print ucoarse

            # write gold and predicted trees to files so we can call evalb
            print >> _evalb_gold[parser], e.gold_unbinarized
            print >> _evalb_pred[parser], oneline(ucoarse)

            GW,G,W = evalb_unofficial(e.gold_unbinarized, binarize(ucoarse))
            h = cgw_f(GW, G, W)
#            h = evalb(e.gold_unbinarized, ucoarse)

            row = {'name': parser,
                   'llh': s,
                   'sentence': sentence,
                   'N': N,
                   #'tree': tree,
                   'evalb': h,
                   'GotWant': GW,
                   'Got': G,
                   'Want': W,
                   'pops': pops,
                   'pushes': pushes,
                   'wallclock': wallclock}

            if args.deps:
                # TODO: include function tags? What is the correct way to get target trees?
                dep_parse = dep.convert_tree(oneline(ucoarse), universal=0)
                dep_label = {(z.index, z.deprel, z.head) for z in dep_parse}
                dep_unlabel = {(z.index, z.head) for z in dep_parse}

                # TODO: Use the official eval.pl script from CoNLL task.
                UAS = len(dep_unlabel & dep_unlabel_gold) / e.N
                LAS = len(dep_label & dep_label_gold) / e.N
                row['LAS'] = LAS
                row['UAS'] = UAS

            data.append(row)
            overall.append(row)

        df = DataFrame(overall).groupby('name').mean()
        #df['wallclock'] = sum_df.wallclock  # use total time

        df.sort_values('wallclock', inplace=1)
        df['speedup'] = df.wallclock.max() / df.wallclock
        df['wps'] = df['N'] / df['wallclock']   # ok to use avg instead of sum

        # Determine which columns to display given command-line options.
        show_cols = ['evalb_corpus', 'wallclock', 'wps', 'speedup', 'pushes', 'pops', 'LAS', 'UAS']
        if len(P) == 1:
            show_cols.remove('speedup')
        if not args.deps:
            show_cols.remove('LAS')
            show_cols.remove('UAS')

        def foo(df):
            "Add column"
            s = DataFrame(overall).groupby('name').sum()  # create separate sum dataframe.
            P = s.GotWant/s.Got
            R = s.GotWant/s.Want
            df['evalb_corpus'] = 2*P*R/(P+R)
            df['evalb_avg'] = df.pop('evalb')  # get rid of old column.

        foo(df)

        print df[show_cols]

        if args.pareto:
            accuracy_name = 'evalb'
            with axman('speed-accuracy ($\delta= %g$)' % delta) as ax:
                df = DataFrame(overall).groupby('name').mean()
                runtime = df.wallclock / df.wallclock.max()
                for name, x, y in zip(df.index, runtime, df[accuracy_name]):
                    c = color[name]
                    ax.scatter([x], [y], alpha=0.75, lw=0, s=50, c=c, label=name, marker=marker[name])
                ax.legend(loc=4)
                ax.set_xlim(-0.1,1.1)
                ax.set_ylim(0,1)
                ax.grid(True)
                ax.set_xlabel('runtime (relative to slowest)')
                ax.set_ylabel('accuracy (%s)' % accuracy_name)
                show_frontier(runtime, df[accuracy_name], ax=ax)

        if args.bylength:
            # Breakdown runtime differences of parsers by length.
            bylength = {name: [] for name in T}
            for length, df in DataFrame(overall).groupby('N'):
                df = df.groupby('name').mean()
                for name, v in df.wallclock.iteritems():
                    bylength[name].append([length, v])
            with axman('benchmark') as ax:
                for name, d in sorted(bylength.items()):
                    d.sort()
                    xs, ys = np.array(d).T
                    ax.plot(xs, ys, alpha=0.5, c=color[name], label=name)
                    ax.scatter(xs, ys, alpha=0.5, lw=1, c=color[name])
                ax.legend(loc=2)
                ax.set_xlabel('sentence length')
                ax.set_ylabel('seconds / sentence')

        if check_llh:
            # Only run this test when it makes sense, e.g., when all parses come
            # from the same grammar.
            s0 = data[0]['llh']
            for x in data:
                s = x['llh']
                name = x['name']
                if abs(s0 - s) > 1e-10:
                    errors.append({'parser': name,
                                   'sentence': sentence})
                    print '[%s]: name: %s expect: %g got: %g' % (red % 'error', name, s0, s)

        Timer.compare_many(*T.values(), verbose=False)

        if errors:
            print red % 'errors: %s' % len(errors)

    print
    print green % '==============================='
    print green % 'DONE!'
    print

    print 'EVALB-unofficial:'
    print 2*(df.GotWant/df.Got * df.GotWant/df.Want)/(df.GotWant/df.Got + df.GotWant/df.Want)
    print
    print 'EVALB-official:'
    import os
    for k,p in enumerate(P):
        _evalb_pred[p].close()
        _evalb_gold[p].close()
        out = 'tmp/evalb-%s.out' % k
        os.system('./bin/EVALB/evalb %s %s > %s' % (_evalb_gold[p].name, _evalb_pred[p].name, out))
        with file(out) as f:
            for x in f:
                if x.startswith('Bracketing FMeasure'):
                    print p, float(x.strip().split()[-1])
                    break                  # use the first one which is for all lengths


if __name__ == '__main__':
    main()
