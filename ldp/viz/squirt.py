#!/usr/bin/env python
import pandas
import pylab as pl
import numpy as np
import seaborn as sns
from argparse import ArgumentParser
from arsenal import colors
from arsenal.math.pareto import show_frontier
from ldp.viz.cone import lambda_cone, arrow


sns.set(font_scale=2.1)
pl.rc('text', usetex=True)
pl.rc('font', family="Times New Roman")


# Light colors for trajectory and cone.
c_baseline = '#a1d76a'
c_lols = '#9e4c79'

# darker color for vector and distinguished point
c_vec_lols = '#862058'
c_vec_baseline = '#558524'

# TODO: use consistent color for each method.
c_cp = '#6a6bd7'
c_dp = '#d76ad7'


def main():

    p = ArgumentParser()
    p.add_argument('filename', help='Path to csv file containing the results.')
    p.add_argument('baseline', help='Path to csv file containing the results.')
    p.add_argument('--accuracy', required=1)
    p.add_argument('--runtime', required=1)
    p.add_argument('--data', choices=('train', 'dev'), default='dev')
#    p.add_argument('--save')

    args = p.parse_args()
    df = pandas.read_csv(args.filename)

    RUNTIME = '%s_new_policy_%s' % (args.data, args.runtime)
    ACCURACY = '%s_new_policy_%s' % (args.data, args.accuracy)
    df.sort_values(RUNTIME, inplace=1, ascending=False)
    [grammar] = df.args_grammar.unique()
    print 'Grammar: %s' % grammar

    assert not df.empty
    print df[[ACCURACY, RUNTIME, 'tradeoff', 'jobid']]

    rescale = 1/df[RUNTIME].max()

    ax = pl.figure().add_subplot(111)
    #pl.axes(frameon=0)
    #pl.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    B = pandas.read_csv(args.baseline)

    df = df[(df.args_accuracy == args.accuracy)
            & (df.args_runtime == args.runtime)
            & (df.args_classifier == 'LOGISTIC')]

    # filter out points we didn't use from the baseline.
    used = []
    for _, lols in df.iterrows():
        [ix] = B[B.tradeoff == lols.args_initializer_penalty].index
        used.append(ix)
    B = B.ix[used]

    lambda_cone(B[ACCURACY], B[RUNTIME] * rescale,
                ax, c=c_baseline, conesize=0.05)

    lols_front = []

    for _, lols in df.iterrows():

        penalty = lols.args_initializer_penalty
        init = B[B.tradeoff == penalty]
        assert len(init) == 1

        x = float(init[RUNTIME]*rescale)
        y = float(init[ACCURACY])

        tradeoff = lols.tradeoff/rescale

        # baseline vector (target direction)
        arrow(x, y, tradeoff, offset=-0.05, c=c_vec_baseline, ax=ax)

        # baseline point
        ax.scatter([B[RUNTIME]*rescale], [B[ACCURACY]], c=c_vec_baseline, lw=0, s=9)

        # LOLS learning curve (squirt)
        squirt = pandas.read_csv(lols.log)
        ax.scatter(squirt[RUNTIME]*rescale, squirt[ACCURACY], lw=0, c=c_lols, s=8, alpha=.5)

        print 'acc iter1: %g init: %g' % (squirt[squirt.iteration==1][ACCURACY], init[ACCURACY])
        print 'run iter1: %g init: %g' % (squirt[squirt.iteration==1][RUNTIME], init[RUNTIME])

        assert abs(float(squirt[squirt.iteration==1][ACCURACY]) - float(init[ACCURACY])) < 1e-3
        assert abs(float(squirt[squirt.iteration==1][RUNTIME]) - float(init[RUNTIME])) < 1e-3

        # re-do early stopping
        early_stop = squirt[ACCURACY] - squirt[RUNTIME]*squirt.tradeoff
        lols = squirt.ix[early_stop.argmax()]
#        lols = squirt.ix[squirt.iteration.argmax()]

        if abs(x - lols[RUNTIME]*rescale) + abs(y - lols[ACCURACY]) > 1e-10:
            # LOLS vector
            ax.annotate("",
                        xy=(lols[RUNTIME]*rescale, lols[ACCURACY]),
                        xytext=(x,y),
                        arrowprops=dict(arrowstyle="->", lw=2, color=c_vec_lols,
                                        connectionstyle="arc3"))
        else:
            print colors.yellow % 'no lols vector for this point'
            print abs(x - lols[RUNTIME]*rescale) + abs(y - lols[ACCURACY]), abs(x - lols[RUNTIME]*rescale), abs(y - lols[ACCURACY])
            print 'early stop iteration', lols.iteration
            print early_stop


        # LOLS vector end point (early stopping
        ax.scatter([lols[RUNTIME]*rescale], [lols[ACCURACY]], c=c_vec_lols, s=13)

        # show ugly read arrow to the last point.
        if 0:
            last = squirt.ix[squirt.iteration.argmax()]
            ax.annotate("",
                        xy=(last[RUNTIME]*rescale, last[ACCURACY]),
                        xytext=(x,y),
                        arrowprops=dict(arrowstyle="->", lw=2, color='r',
                                        connectionstyle="arc3"))

        lols_front.append([lols[RUNTIME]*rescale, lols[ACCURACY]])

    # LOLS pareto frontier.
    xx,yy=zip(*lols_front)
    show_frontier(xx, yy, c=c_vec_lols, alpha=0.4, zorder=10, interpolation='linear-convex', ax=ax)

    # tick labels.
    xx=B[RUNTIME]*rescale
    #xx = np.linspace(xx.min(), xx.max(), 12)

    if args.runtime == 'pops':
        ax.set_xlabel(r'runtime (avg constituents built)')
    elif args.runtime == 'mask':
        ax.set_xlabel('runtime (avg spans allowed)')
    elif args.runtime == 'pushes':
        ax.set_xlabel('Runtime (Avg $|E|$)')

    pl.xticks(xx, ['%.f' % (x/rescale) for x in xx], rotation=45)

    # show all learning curves.
    if 0:
        pl.figure()
        for _, lols in df.iterrows():
            squirt = pandas.read_csv(lols.log)
            #pl.figure()
            #pl.plot(squirt[RUNTIME]*rescale, squirt[ACCURACY])
            R = squirt[ACCURACY] - squirt.tradeoff*squirt[RUNTIME]
            pl.plot(squirt.iteration, R)

    #xx=B[ACCURACY]
    #pl.yticks(xx, ['%.3f' % (x) for x in xx])

    pl.ion()
    pl.show()

#    # not ready for prime time because axes limits aren't set.
#    if args.save:
#        save = True
#        if path(args.save).exists():
#            save = False
#            print bold % colors.yellow % "File exists (%s)" % args.save
#            print bold % colors.yellow % "Overwrite existing file [y/N]?",
#            if raw_input().strip().lower() in ('y','yes'):
#                save = True
#        if save:
#            print bold % colors.yellow % "Saved file %s" % args.save
#            pl.savefig(args.save)

    t = ['Controlled experiments (dev)']
    [G] = df.args_grammar.unique()
    if 'medium' in G:
        t.append('small grammar')
    elif 'big' in G:
        t.append('big grammar')

    [RO] = df.args_roll_out.unique()
    if 'CP' in RO:
        t.append('$r_{\\textit{CP}}$')
    elif 'DP' in RO:
        t.append('$r_{\\textit{DP}}$')
    elif 'BF' in RO:
        t.append('$r_{\\textit{BF}}$')
    elif 'HY' in RO:
        t.append('$r_{\\textit{HY}}$')

    if args.accuracy == 'expected_recall_avg':
        ax.set_ylabel('accuracy (expected binarized recall)')
    elif args.accuracy == 'evalb_avg':
        ax.set_ylabel('accuracy (avg single-sentence F1)')

    #pl.title(', '.join(t))

    print B[[ACCURACY, RUNTIME]]

    ax.figure.tight_layout()
    pl.ioff()
    pl.show()


if __name__ == '__main__':
    main()
