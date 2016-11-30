"""Grammar preprocessing module.

"""
import re
from numpy import log
from arsenal.alphabet import Alphabet
from arsenal.iterview import iterview
from pandas import DataFrame


def preprocess_bubs_format(bubs, output):
    """Convert grammar from bubs-parser into ldp-friendly csv format. The result is
    an equivalent grammar, which is much faster to load because it has been
    integerized.

    Given a gzipped grammar from bubs-parser, e.g. `eng.M2.gr.gz`, this function
    will generate four files:

    - eng.M2.gr.csv: grammar rules
    - eng.M2.lex.csv: lexical rules
    - eng.M2.lex.alphabet: mapping from terminals to integers
    - eng.M2.sym.alphabet: mapping from syms to integers

    """

    sym = Alphabet()
    lex = Alphabet()

    import gzip
    lines = gzip.open(bubs, 'rb').readlines()
    reading_lex = False

    l = []
    f = []
    for line in iterview(lines[1:]):        # drop first line

        if line.startswith('===== LEXICON'):
            reading_lex = True
            continue

        x = line.strip().split()
        if not x:
            continue

        lhs = x[0]
        rhs = tuple(b for b in x[2:-1])
        score = x[-1]
        if len(rhs) == 1:
            rhs = (rhs[0], '')
        y, z = rhs
        lhs = sym[lhs]

        y = lex[y] if reading_lex else sym[y]
        z = sym[z] if z else -1

        if reading_lex:
            l.append({'score': score,
                      'head': lhs,
                      'left': y})
        else:
            f.append({'score': score,
                      'head': lhs,
                      'left': y,
                      'right': z})

    # non-gzipped loads faster.
    #DataFrame(f).to_csv(gzip.open(output + '.gr.csv.gz', 'wb'))
    #DataFrame(l).to_csv(gzip.open(output + '.lex.csv.gz', 'wb'))

    DataFrame(f).to_csv(output + '.gr.csv')
    DataFrame(l).to_csv(output + '.lex.csv')
    sym.save(output + '.sym.alphabet')
    lex.save(output + '.lex.alphabet')


def preprocess_berkeley_format(input_prefix, output, coarsen=False):
    """
    Preprocessing: convert PTB grammar into simple tsv format.
    """

    def g(x):
        if coarsen:
            x = x.split('^')[0]
            x = x.split('_')[0]
        return x

    sym = Alphabet()
    lex = Alphabet()

    lexical_rules = []
    for x in file(input_prefix + '.lexicon'):
        [(x, y, s)] = re.findall(r'(\S+)\s+(\S+)\s*\[(.*?)\]', x)
        s = float(s)
        x = g(x)
        y = g(y)
        lexical_rules.append({'score': log(s), 'head': sym[x], 'left': lex[y]})

    rules = []
    for x in file(input_prefix + '.grammar'):
        x, y = x.split(' -> ')
        y = y.split()
        if len(y) == 2:
            y, s = y
            s = float(s)
            z = -1
        else:
            assert len(y) == 3
            y, z, s = y
            s = float(s)
        x = g(x)
        y = g(y)
        if x == y and z == -1:
            continue
        x = sym[x]
        y = sym[y]
        if z != -1:
            z = g(z)
            z = sym[z]
        rules.append({'score': log(s), 'head': x, 'left': y, 'right': z})

    DataFrame(rules).to_csv(output + '.gr.csv')
    DataFrame(lexical_rules).to_csv(output + '.lex.csv')
    sym.save(output + '.sym.alphabet')
    lex.save(output + '.lex.alphabet')


if __name__ == '__main__':
    if 0:
        preprocess_berkeley_format('data/test.gr', 'data/medium', coarsen=True)
        exit(0)

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    p = ArgumentParser(description=preprocess_bubs_format.__doc__,
                       formatter_class=RawDescriptionHelpFormatter)
    p.add_argument('bubs')
    p.add_argument('output_prefix')
    args = p.parse_args()
    preprocess_bubs_format(args.bubs, args.output_prefix)
