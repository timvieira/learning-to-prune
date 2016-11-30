#!/usr/bin/env python
"""Use a preliminary Pareto frontier to determine which values of lambda are
promising as well as good initialization points.

For example, using the asymmetric weighting method (baseline) to sweep out a
preliminary frontier. From this frontier, we can determine a "reasonable"
mapping from the asymmetric weighting penalty to a value of lambda.

  In theory, we can even skip the initialization step by loading existing
  parameters (this is a little tricky because we leave those values in the
  aggregate dataset as a "prior").

Another use case is calibrating learning rates to variants of the objective
function. For example, if we want to use raw pops instead of speedup then we can
get a sense for the range of values of lambda look promising and where on the
Pareto curve they will lie.

"""
import pylab as pl
import numpy as np
from path import path
from pandas import read_csv
from arsenal.iterextras import window
from argparse import ArgumentParser
p = ArgumentParser()
p.add_argument('filename', help='Path to csv file containing the results to base the lambda values on.')
p.add_argument('--job-file')
p.add_argument('--accuracy', required=1)
p.add_argument('--runtime', required=1)
p.add_argument('--filters', nargs='*', default=())
p.add_argument('--minibatch', type=int, default=1000)
p.add_argument('--data', choices=('train', 'dev'), default='dev')

args = p.parse_args()

# TODO: Should we use train to pick the points? Aside from the usual engineering
# compulsion to look at dev data, we might actually benefit from using
# train-only because the baseline method gets *worse* accuracy on conservative
# penalties! So bad, that the points aren't even on the Pareto frontier.  This
# is so noisy that I'll need to use a the a proper convex hull (not my adjacent
# points heuristic).

df = read_csv(args.filename)

RUNTIME = '%s_new_policy_%s' % (args.data, args.runtime)
ACCURACY = '%s_new_policy_%s' % (args.data, args.accuracy)

#df.sort('tradeoff', inplace=1)
df.sort_values(RUNTIME, inplace=1, ascending=False)

for F in args.filters:
    df = df[eval(F)]

penalty = np.array(df.tradeoff)
[grammar] = df.args_grammar.unique()

df['penalty'] = penalty

assert not df.empty

# TODO: Baselines should all use the same regularization constant as should the
# jobs we plan to run based on initialization we derive in this script.
assert len(set(df.args_C)), 'should run with same regularizer %s' % set(df.args_C)
regularizer = np.array(df.args_C)[0]


print df[[ACCURACY, RUNTIME, 'penalty']]

runtime = np.array(df[RUNTIME])
accuracy = np.array(df[ACCURACY])


print '[fit] fitting...'
from ldp.viz.parametric_fit import fit
ff,gg = fit(runtime, accuracy)
print '[fit] done.'


pts = zip(penalty, runtime, accuracy)

# Add two fake points:
# TODO: some magic numbers...
pts = [(None, runtime.max()*2.0, accuracy.max()*1.01)] + pts + [(None, runtime.min()*.98, accuracy.min()*.98)]


ddd = []
skip = []
for (((p1,x1,y1), (p,x2,y2), (p3,x3,y3))) in window(pts, 3):

    # slopes give a range for lambda values to try.
    m12 = (y2-y1)/(x2-x1)
    m23 = (y3-y2)/(x3-x2)

    #print 'penalty=%g point=%s, lambda=%s' % (p, [x2,y2], [m12, m23])

    if m12 >= m23:
        # This point is not on the convex Pareto frontier. So, we back off to
        # the slope between the neighboring points
        m31 = (y3-y1)/(x3-x1)
        #print 'penalty=%g point=%s not on convex frontier.' % (p, [x2,y2]), 'try', m31
        m12 = m23 = m31
        skip.append([x2,y2])

    ddd.append([p, m12, m23, gg(x2)])
#
#    if 0:
#        # visual debugging.
#        b12 = y2-m12*x2
#        b23 = y2-m23*x2
#        xs = np.linspace(runtime.min(), runtime.max(), 10)
#        pl.figure()
#        pl.plot(xs, xs*m12 + b12, alpha=0.5, c='b')
#        pl.plot(xs, xs*m23 + b23, alpha=0.5, c='b')
#
#        # parametric
#        pl.plot(xs, ff(x2) + gg(x2)*(xs-x2), alpha=0.5, c='r', lw=2)
#
#        pl.scatter(df[RUNTIME], df[ACCURACY], c=df.tradeoff, lw=0)
#        #pl.ylim(accuracy.min(), 1)
#        #pl.ylim(0, 1)
#        pl.show()

pl.figure()
pl.scatter(runtime, accuracy, c=penalty, lw=0)

xxx = np.linspace(runtime.min(), runtime.max(), 100)
pl.plot(xxx, [ff(x) for x in xxx], c='r')


if skip:
    skip = np.array(skip)
    pl.scatter(skip[:,0], skip[:,1], c='r', lw=2, marker='x')

if 0:
    ps, us, ls, pp = np.array(ddd).T
    pl.figure()
    #pl.plot(ps, us, alpha=0.5, c='b')
    #pl.plot(ps, ls, alpha=0.5, c='b')
    pl.fill_between(ps, ls, us, alpha=0.25, color='b')

    pl.plot(ps, (us+ls)/2, c='b', lw=2)

    pl.plot(ps, pp, c='k', lw=2)
    pl.xlabel('initializer penalty')
    pl.ylabel('tradeoff (lambda)')
    pl.title('initializer calibration for (%s,%s)' % (args.accuracy, args.runtime))



assert len(ddd) == len(accuracy)

#f = file(args.job_file, 'wb') if args.job_file else None

# infer the rollout type
RO = 'DP' if 'expected_recall' in args.accuracy else 'CP'
RO = 'BF' if 'pushes' in args.runtime  else RO
if 'expected_recall' in args.accuracy and 'pops' == args.runtime:
    RO = 'HY'

f = file('jobs-%s-%s-%s-%s' % (grammar, RO, args.accuracy, args.runtime), 'wb')

for penalty, _, _, pp in ddd:
    init = read_csv('tmp/%s-%s.csv' % ('baseline9', grammar))
    [thing] = init[init.args_initializer_penalty == penalty].log
    initweights = path(thing).dirname() / 'new_policy-001.npz'

    # TODO: Check that several options are the same as the initial curve,
    # e.g., grammar, initializer, classifier, training data.
    zargs = ['python', '-u -m ldp.lols', '--seed 0',
             '--grammar', grammar,
             '--maxlength 40', '--minlength 3', '--train 40000', '--dev 10000',
             '--minibatch', args.minibatch,
             '--classifier LOGISTIC',
             '--initializer BODENSTAB_GOLD',
             '--initializer-penalty', penalty,
             '--tradeoff', pp,
             '--roll-out', RO,
             '--accuracy', args.accuracy,
             '--runtime', args.runtime,
             '--init-weights', initweights,
             '-C', regularizer]

    cmd = ' '.join(map(str, zargs))

    print cmd

    if f is not None:
        print >> f, cmd

pl.show()
