#!/usr/bin/env python
from __future__ import division
import sys
import numpy as np
import pylab as pl
import seaborn as sns
import pandas as pd
from arsenal import colors

c_g = '#f1a340'
c_n = '#998ec3'

def percent(x):
    return '%4.1f%% %s' % (np.mean(x)*100, colors.magenta % '(%s/%s)' % (np.sum(x), len(x)))


def breakdown():
    # Overall effect on accuracy
    print colors.cyan % '============================================'
    print colors.cyan %'OVERALL'

    # gold/nongold breakdown
    print 'gold/nongold/total: %.2f%%/%.2f%% (%s/%s/%s)' \
        % (100*len(gold)/len(df), 100*len(nongold)/len(df),
           len(gold), len(nongold), len(df))


    print colors.cyan % '# Supervision errors <equal-weight> (<class-weight>)'

    def pp(name, ns, gs):
        n = np.mean(ns)
        g = np.mean(gs)
        print ('%s: %4.1f%% (%4.1f%%)'
               % (name,
                  100.0 * (g + n) / 2,                           # equally weighted
                  100.0 * (len(gs)*g + len(ns)*n) / len(df)))    # class-weighted

    pp(colors.light_green % 'good',
       (nongold.delta_rew < 0),
       (gold.delta_rew > 0))

    pp(colors.light_yellow % 'same',
       (nongold.delta_rew == 0),
       (gold.delta_rew == 0))

    pp(colors.light_red % 'errs',
       (nongold.delta_rew > 0),
       (gold.delta_rew < 0))


    # errors
    #e = df[(~df.gold & (df.delta_rew > 0)) | (df.gold & (df.delta_rew < 0))]
    #from arsenal.debug import ip; ip()    
    
    #print colors.cyan % '# REWARD'
    #print 'decrease: %s' % percent(df.delta_rew < 0)
    #print 'noeffect: %s' % percent(df.delta_rew == 0)
    #print 'increase: %s' % percent(df.delta_rew > 0)

    #print colors.cyan % '# ACCURACY'
    #print 'decrease: %s' % percent(df.delta_acc < 0)
    #print 'noeffect: %s' % percent(df.delta_acc == 0)
    #print 'increase: %s' % percent(df.delta_acc > 0)

    #print '# RUNTIME'
    #print 'decrease: %s' % percent(df.delta_run < 0)
    #print 'noeffect: %s' % percent(df.delta_run == 0)
    #print 'increase: %s' % percent(df.delta_run > 0)

    print colors.cyan % '# error breakdown'
    print colors.cyan % '# gold'

    print '%s: %s' % (colors.light_red % 'decrease',
                      percent(gold.delta_rew < 0))
    print '%s: %s' % (colors.light_yellow % 'noeffect',
                      percent(gold.delta_rew == 0))
    print '%s: %s' % (colors.light_green % 'increase',
                      percent(gold.delta_rew > 0))

    print colors.cyan % '# non-gold'
    print '%s: %s' % (colors.light_green % 'decrease',
                      percent(nongold.delta_rew < 0))
    print '%s: %s' % (colors.light_yellow % 'noeffect',
                      percent(nongold.delta_rew == 0))
    print '%s: %s' % (colors.light_red % 'increase',
                      percent(nongold.delta_rew > 0))


pl.ion()

df = pd.read_csv(sys.argv[1])

from path import path
import cPickle
args = cPickle.load(file((path(sys.argv[1]) / '..' / 'args.pkl').abspath()))
tradeoff = args.tradeoff
bodenstab = args.initializer_penalty

print 'GRAMMAR: %s' % args.grammar
print 'TRADEOFF: %g' % tradeoff
print 'ACC: %s' % args.accuracy
print 'RUN: %s' % args.runtime


#if 0:
#    # [2016-08-01 Mon] You can easily find a lambda which makes the baseline oracle
#    # really bad.
#    def look_at_tradeoff(df):
#        df = df.copy()
#        xs = np.linspace(0, 0.02, 1000)
#        ys = []
#        for tradeoff in xs:
#            df['delta_rew'] = df.delta_acc - tradeoff*df.delta_run
#            gold = df[df.gold]
#            #nongold = df[~df.gold]
#            err = np.mean(gold.delta_rew < 0)
#            #err = np.mean(nongold.delta_rew < 0)
#            ys.append(err)
#        pl.plot(xs, ys)
#
#    look_at_tradeoff(df)


df.delta_run *= tradeoff
#df['delta_rew'] = df.delta_acc - tradeoff*df.delta_run
df['delta_rew'] = df.delta_acc - df.delta_run

#sns.violinplot(x='gold', y='delta_rew', data=df)
sns.boxplot(x='gold', y='delta_rew', data=df)

gold = df[df.gold]
nongold = df[~df.gold]

print
print 'Gold'
print gold.delta_rew.describe()

print 'Non-gold'
print nongold.delta_rew.describe()

pl.figure()
pl.scatter(gold.delta_run, gold.delta_acc, c=c_g, lw=0, label='gold', marker='.')
pl.scatter(nongold.delta_run, nongold.delta_acc, c=c_n, lw=0, label='nongold', marker='.')

#pl.figure()
#sns.kdeplot(gold.delta_run, gold.delta_acc, shade=1)
#pl.scatter(gold.delta_run, gold.delta_acc, marker='.')
#sns.jointplot(gold.delta_run, gold.delta_acc, kind='kde', shade=1)
#pl.title('gold')

#pl.figure()

#pl.figure()
#pl.hexplot(nongold.delta_run, nongold.delta_acc, cmap='Greens')
#pl.title('nongold')

bad_gold = gold[gold.delta_rew < 0]
ok_gold = gold[gold.delta_rew >= 0]
bad_nongold = nongold[nongold.delta_rew >= 0]
ok_nongold = nongold[nongold.delta_rew < 0]

if 0:
    # Highlight incorrect 'labels'
    pl.scatter(ok_gold.delta_run, ok_gold.delta_acc, c=c_g, lw=0, label='ok gold', marker='.')
    pl.scatter(ok_nongold.delta_run, ok_nongold.delta_acc, c=c_n, lw=0, label='ok nongold', marker='.')
    pl.scatter(bad_gold.delta_run, bad_gold.delta_acc, c=c_g, lw=0, label='bad gold')
    pl.scatter(bad_nongold.delta_run, bad_nongold.delta_acc, c=c_n, lw=0, label='bad nongold')

#from viz.interact.pointbrowser import PointBrowser
#br = PointBrowser(df, xcol='delta_run', ycol='delta_acc', plot_kwargs={'alpha': 0, 'label': None})

breakdown()

xs = np.linspace(df.delta_run.min(), df.delta_run.max(), 100)
pl.plot(xs, xs, c='k', lw=3)   # Note: we already rescaled x-axis by tradeoff




# TODO: Is this the right way to show Bodenstab's penalty???
#pl.plot(xs, xs * bodenstab / tradeoff, c='k', lw=3)


#p_gold = np.polyfit(gold.delta_run, gold.delta_acc, deg=1)
#pl.plot(xs, p_gold[1] + p_gold[0]*xs, c=c_g, lw=2)

#p_nongold = np.polyfit(nongold.delta_run, nongold.delta_acc, deg=1)
#pl.plot(xs, p_nongold[1] + p_nongold[0]*xs, c=c_n, lw=2)

# TODO: How much Bodenstab cares ratio of lines?
#pl.plot(xs, xs * p_gold[0] / p_nongold[0], c=c_n, lw=2)
#pl.plot(xs, xs * p_nongold[0] / p_gold[0], c='y', lw=2)

pl.ylabel('acc(keep) - acc(prune)')
pl.xlabel(r'$\lambda \times$ (run(keep) - run(prune))')
pl.xlim(0, df.delta_run.max())
pl.ylim(df.delta_acc.min(), df.delta_acc.max())
pl.tight_layout()
pl.legend(loc=4)
pl.show()


if 0:
    # Show separate plots for gold/nongold to highlight "label" errors
    pl.figure()
    pl.scatter(ok_gold.delta_run, ok_gold.delta_acc, c=c_g, lw=0, alpha=0.25)
    pl.scatter(bad_gold.delta_run, bad_gold.delta_acc, c=c_g, lw=0, alpha=0.25)
    pl.plot(xs, xs, c='k', lw=3)
    pl.title('gold')

    pl.figure()
    pl.scatter(bad_nongold.delta_run, bad_nongold.delta_acc, c=c_n, lw=0, alpha=0.25, label='bad nongold')
    pl.scatter(ok_nongold.delta_run, ok_nongold.delta_acc, c=c_n, lw=0, alpha=0.25, label='ok nongold')
    pl.plot(xs, xs, c='k', lw=3)
    pl.title('non-gold')

#sns.jointplot('delta_run', 'delta_acc', data=gold)
#pl.title('gold')

#sns.jointplot('delta_run', 'delta_acc', data=nongold)
#pl.title('nongold')

if 0:
    from arsenal.debug import ip; ip()
else:
    pl.ioff()
    pl.show()
