"""
Look at learning curves.
"""
from __future__ import division
import cPickle
import seaborn as sns
import numpy as np
import pylab as pl
import pandas as pd
from path import path
from arsenal import colors, wide_dataframe
wide_dataframe()

from argparse import ArgumentParser
p = ArgumentParser()
p.add_argument('--grammar', choices=('medium', 'big'))
p.add_argument('-i', action='store_true')
_args = p.parse_args()

data = []
for results in (path('results/*-lols10-*/').glob('dump')
#          + path('results/*-baseline9-*/').glob('dump')
          ):

    args = cPickle.load(file(results / 'args.pkl'))
    # only grab models matching the grammar specified.
    if args.grammar != _args.grammar:
        continue

    df = pd.read_csv(results / 'log.csv')
    assert df.get('dev_new_policy_evalb_corpus') is not None

    # identify iteration with best dev reward (surrogate).
    el = df.datetime.map(pd.to_datetime).tolist()
    df['elapsed'] = [(t - el[0]).total_seconds() / (24*60*60) for t in el]
    df = df[df.elapsed <= 6]            # take the best policy <= 6 days of training.

    best = df.ix[df.dev_new_policy_reward.argmax()]

    row = {'args_%s' % k: v for k,v in args.__dict__.items()}

    row.update({'row_%s' % k: v for k,v in best.iteritems()})

    # converge a lols iteration count into passes of the data.
    # iterations * (examples/iterations) * 1/examples = unit less.
    ratio = args.minibatch / 36563 #args.train    # XXX: hardcoded training set size.

    xaxis = 'elapsed'
    #xaxis = 'iteration'
    pl.figure()
    pl.plot(df[xaxis], df.dev_new_policy_reward - df.dev_new_policy_reward.tolist()[0], c='b', label='dev')
    pl.plot(df[xaxis], df.train_new_policy_reward - df.train_new_policy_reward.tolist()[0], c='k', label='train')
    pl.axvline(best[xaxis], linestyle=':')
    pl.xlim(0,6)
    pl.xlabel(xaxis)
    pl.ylabel(r'$R_t - R_1$')
    pl.legend(loc='best')
    pl.savefig(results / 'learning-curve.svg')
    pl.close()

    iterations = df.iteration.max()
    passes = df.iteration.max() * ratio
    total_time = df.ix[df.iteration.argmax()].elapsed

    row.update({
        'earlystop_iteration': best.iteration,
        'earlystop_passes': best.iteration * ratio,
        'earlystop_elapsed':   best.elapsed,
        'total_iterations': iterations,
        'total_passes': passes,
        'total_time': total_time,
        'hours_iterations': total_time * 24 / iterations,
        'hours_pass': total_time * 24 / passes,
    })

    data.append(row)

df = pd.DataFrame(data).sort_values(['args_roll_out','args_tradeoff'])

df['dev_evalb'] = df['row_dev_new_policy_evalb_corpus']


cp = df[df.args_roll_out == 'CP']
dp = df[df.args_roll_out == 'DP']


show_cols = ['args_roll_out', 'args_tradeoff', 'dev_evalb', 'earlystop_elapsed', 'earlystop_passes',
             'hours_iterations', 'hours_pass',
             'total_time', 'total_passes', 'args_results']
print cp[show_cols]
print
print dp[show_cols]


with file('tmp/convergence-and-iterations-%s.html' % _args.grammar, 'wb') as html:
    html.write('<h1>%s</h1>' % _args.grammar)
    html.write('<h2>CP</h2>')
    html.write(cp[show_cols].to_html())
    html.write('<h2>DP</h2>')
    html.write(dp[show_cols].to_html())

    print >> html, '<center><table><tr style="text-align:center; font-size: 30pt;"><th>CP</th><th>DP</th></tr>'
    for c, d in zip(cp.args_results, dp.args_results):
        C = file(c / 'learning-curve.svg').read()
        D = file(d / 'learning-curve.svg').read()
        print >> html, '<tr><td>%s<td><td>%s</td></tr>' % (C, D)
    print >> html, '</table></center>'


print
print colors.green % 'wrote %s' % html.name
print


if _args.i:
    from arsenal.debug import ip; ip()
