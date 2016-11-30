"""Create final Pareto frontier plots, run significance tests (and
significance.csv files for results tables).

"""
from __future__ import division
import cPickle
import numpy as np
import pylab as pl
import pandas as pd
import seaborn as sns

from path import path
from argparse import ArgumentParser

sns.set(font_scale=4)
pl.rc('text', usetex=True)
pl.rc('font', family="Times New Roman")

from ldp.prune.example import cgw_f
from ldp.viz.cone import lambda_cone, arrow
from arsenal import colors, iterview
from arsenal.math import compare
from perm import paired_permutation_test


c_baseline = '#a1d76a'
c_cp = '#6a6bd7'
c_dp = '#d76ad7'
c_hy = 'k'

# darker color for vector and distinguished point
c_vec_baseline = '#558524'

C = {
    'HY': c_hy,
    'CP': c_cp,
    'DP': c_dp,
    'baseline': c_baseline,
}


def load(d):
    D = None
    for f in d.glob('*.csv.gz'):
        dd = pd.read_csv(f)
        if D is None:
            D = dd
        else:
            D = D.append(dd)
    return D


def sanity_check(Ds):
    names, Ds = zip(*Ds)

    D0 = Ds[0]
    Ps = set(D0.policy.unique())
    Es = set(D0.example.unique())

    # Sanity check.
    for name, dd in zip(names, Ds):
        # same policy and examples
        if (set(dd.policy.unique()) != Ps
            or set(dd.example.unique()) != Es):
            print colors.bold % colors.red % '======================================'
            print colors.bold % colors.red % 'WARNING: some policies arent finished.'
            print colors.bold % colors.red % '======================================'
            print name, 'want/got sizes %s/%s' % (len(Ps), len(set(dd.policy.unique())))

        #assert set(dd.policy.unique()) == Ps  # same policies
        #assert set(dd.example.unique()) == Es  # same examples

    bestof = aggregate_multiple_runtime_trials(Ds, Ps)

    return D0, Ds, bestof


def main():

    pl.ion()

    p = ArgumentParser()
    p.add_argument('root', type=path)
    p.add_argument('--quick', action='store_true',
                   help="Load a single evaluation log (for quick tests). Won't run bestof-k runtime.")
    p.add_argument('-i', action='store_true',
                   help='Interactive mode => open an IPython shell after execution.')
    args = p.parse_args()

    runs = [r for r in sorted(args.root.glob('*')) if r.isdir()]

    if args.quick:
        print colors.bold % colors.red % 'Warning! only using some of the runs for timing information.'
        runs = runs[:1]

    Ds = [(r, load(r)) for r in iterview(runs)]

    D0, Ds, bestof = sanity_check(Ds)

#    if 0:
#        pl.figure()
#        for name, df in D0.groupby('type'):
#            pl.scatter(df.avg_bestof_time, df.evalb, c=C[name], lw=0)
#            show_frontier(df.avg_bestof_time, df.evalb, c=C[name], interpolation='linear-convex', label=name)
#            #[w,b] = np.polyfit(df.pushes, df.avg_bestof_time, deg=1)
#            #show_frontier(df.pushes*w + b, df.evalb, interpolation='linear', c=C[name])
#        pl.xlabel('sec/sentence (best of %s)' % len(Ds))
#        pl.ylabel('Corpus EVALB-F1')
#        pl.legend(loc=4)
#        pl.show()

    rescale = 1/bestof.pushes.max()
    bestof['pushes_r'] = bestof.pushes*rescale

    B = bestof[bestof.type=='baseline'].copy()
    lols = bestof[bestof.type!='baseline']

    RO_types = lols.args_roll_out.unique()

    ax = pl.figure().add_subplot(111)
    for name, df in reversed(sorted(bestof.groupby('type'))):
        pl.scatter(df.pushes_r, df.evalb, c=C[name], lw=0, zorder=10, label='', s=50)
        pts = show_frontier(df.pushes_r, df.evalb, interpolation='linear-convex', lw=2, c=C[name], label=name)
        ax.plot(pts[:,0], pts[:,1], label=name, c=C[name])

    pl.ylabel('Corpus $F_1$')
    pl.legend(loc=4)
    pl.tight_layout()

    ax = pl.gca()
    conesize = .06
    lambda_cone(np.array(B.evalb), np.array(B.pushes_r), ax=ax, c=c_baseline, conesize=conesize, lines=0)

    # --------------------------------------------------------------------------
    # Fit parametric curve to dev points show arrows on test points.
    from ldp.viz.parametric_fit import fit

    df = join_with_dev(B)

    ff, gg = fit(df.dev_pushes, df.dev_evalb)
    if 0:
        # enable to show the parametric curve.
        xs = pl.linspace(0, df.dev_pushes.max()+.1*df.dev_pushes.ptp(), 100)
        ax.plot(xs*rescale, ff(xs), c='k')

    ax = pl.gca()
    for _, z in df.iterrows():
        x, y = z.test_pushes*rescale, z.test_evalb
        arrow(x, y, gg(z.dev_pushes)/rescale, offset=-conesize, c=c_vec_baseline, ax=ax)

    # --------------------------------------------------------------------------
    B.loc[:,'tradeoff'] = np.nan

    data = []

    # Loop over all rollout types joined on initial policy (i.e., the baseline).
    for i, bl in B.iterrows():
        spawn = lols[lols.args_initializer_penalty == bl.args_tradeoff]
        assert len(spawn) == len(RO_types)

        models = {}
        for ro in RO_types:
            [ix] = spawn[spawn.args_roll_out == ro].index
            models[ro] = lols.ix[ix]

        [dev_pushes] = df[df.policy == bl.policy].dev_pushes
        tradeoff = gg(dev_pushes)
        B.loc[i,'tradeoff'] = tradeoff

        if 1:
            # uncomment to run hypothesis tests.

            print colors.bold % colors.green % '============================================================='
            print 'tradeoff: %g' % tradeoff
            print

            baseline_acc, baseline_run = get_acc_run(D0[D0.policy == bl.policy])

            row = {
                'baseline': bl.policy,
                'baseline_accuracy': baseline_acc,
                'baseline_runtime': baseline_run,
                'baseline_reward': baseline_acc - tradeoff*baseline_run,
                'wps_baseline': bl.wps,
                'wallclock_baseline': bl.avg_bestof_time,
                'tradeoff': tradeoff,
            }

            star_sty = dict(alpha=1, lw=0, marker='*', s=700, zorder=100)

            for ro, model in sorted(models.items()):
                print colors.bold % '# %s' % ro
                sig, win = paired_permutation_test(D0,
                                                   a=bl.policy,
                                                   b=model.policy,
                                                   tradeoff=tradeoff,
                                                   R=5000)
                acc, run = get_acc_run(D0[D0.policy == model.policy])

                row[ro] = model.policy
                row['%s_accuracy'  % ro] = acc
                row['%s_runtime'   % ro] = run
                row['%s_reward'    % ro] = acc - tradeoff*run
                row['wps_%s'       % ro] = model.wps
                row['wallclock_%s' % ro] = model.avg_bestof_time
                row['winner_%s'    % ro] = win
                row['sig_%s'       % ro] = sig

                if win == +1:
                    pl.scatter([model.pushes_r],
                               [model.evalb],
                               c=C[ro],
                               **star_sty)

                elif win == -1:
                    pl.scatter([bl.pushes_r],
                               [bl.evalb],
                               c=C['baseline'],
                               **star_sty)

                # draw a dotten line to the baseline point.
                pl.plot([bl.pushes_r, model.pushes_r],
                        [bl.evalb, model.evalb],
                        c=C[ro],
                        lw=1,
                        alpha=0.75,
                        label=None,
                        linestyle='--')

            data.append(row)

    #[w,b] = np.polyfit(B.pushes, B.avg_bestof_time, deg=1)

    xx = lols.pushes_r
    xx = np.linspace(xx.min(), xx.max(), 12)

    # put ticks on the top of the plot.
    #ax.xaxis.tick_top()

    #pl.xticks(xx, ['%.2g\m(%.2g)' % (x/rescale / 1e6, (x/rescale*w+b)*100) for x in xx], rotation=0)
    #pl.text(0.4, 0.401, re.sub('(\d)e([\-+]\d+)', r'\1e^{\2}', r'$\textit{seconds} \approx %.2g \cdot \textit{pushes} + %.2g$' % (w,b)))
    #pl.xlabel('average megapushes ($\\approx$ milliseconds)')

    pl.xticks(xx, [r'$%.2g$' % (x/rescale / 1e6) for x in xx])
#    pl.xticks(xx, [r'%.2g' % (x/rescale / 1e6) for x in xx], rotation=45)

    if 'medium' not in args.root:
        pl.xlabel('millions of hyperedges built per sentence')

    pl.ylim(bestof.evalb.min()-0.02, bestof.evalb.max()+0.015)
    pl.xlim(bestof.pushes_r.min()-.01, bestof.pushes_r.max()+0.01)


    zf = pd.DataFrame(data).sort_values('tradeoff')
#    print zf[['tradeoff', 'baseline_reward', 'cp_reward', 'dp_reward', 'winner_cp', 'winner_dp']].sort_values('tradeoff').to_string(float_format='%.4g'.__mod__, index=0)
#    print zf[['tradeoff',
#              'baseline_accuracy', 'baseline_runtime',
#              'cp_accuracy', 'cp_runtime',
#              'dp_accuracy', 'dp_runtime',
#              'winner_cp', 'winner_dp']].sort_values('tradeoff').to_string(float_format='%.4g'.__mod__, index=0)

    if not args.quick:
        sig_file = args.root / 'significance.csv'
        print
        print colors.green % 'wrote %s' % sig_file
        print
        zf.to_csv(sig_file)

    if args.i:
        pl.ion(); pl.show()
        from arsenal.debug import ip; ip()
    else:
        pl.ioff(); pl.show()


def join_with_dev(B):
    data = []
    for policy in B.policy.unique():
        dump = path(policy).dirname()
        log = pd.read_csv(dump / 'log.csv')
        [dev_evalb] = log.dev_new_policy_evalb_corpus
        [dev_pushes] = log.dev_new_policy_pushes
        [test_evalb] = B[B.policy == policy].evalb
        [test_pushes] = B[B.policy == policy].pushes
        data.append({'policy': policy,
                     'test_evalb': test_evalb, 'test_pushes': test_pushes,
                     'dev_evalb': dev_evalb, 'dev_pushes': dev_pushes})
    return pd.DataFrame(data)


def get_acc_run(X):
    acc = cgw_f(X.want_and_got.sum(), X.got.sum(), X.want.sum())
    run = X.pushes.mean()
    return acc, run


def aggregate_multiple_runtime_trials(Ds, Ps):
    """Collapse multiple dataframes `Ds` from different timing runes into a single
    one, by taking the min over runtimes (i.e., new runtime will be "best-of k"
    where k=|Ds|).

    Actually, this function does more than that. It appears to collapse over
    sentence too, e.g., computing corpus-EVALB and avg[best-of-k runtimes].

    """
    D0 = Ds[0]

    # Append trials together
    foo = Ds[0]
    for dd in Ds[1:]:
        foo = foo.append(dd)

    # Take min over time_total for this policy-example pair.
    minz = foo[['policy','example','time_total']].groupby(['policy','example']).min()

    data = []
    for policy in iterview(Ps):
        dump = path(policy).dirname()
        args = cPickle.load(file(dump / 'args.pkl'))
        log = pd.read_csv(dump / 'log.csv')

        # TODO: will need to add extra cases.
        if 'DP' in args.roll_out:
            type_ = 'DP'
        elif 'CP' in args.roll_out:
            type_ = 'CP'
        elif 'HY' in args.roll_out:
            type_ = 'HY'
        elif 'BODEN' in args.roll_out:
            type_ = 'baseline'
        else:
            raise ValueError(args.roll_out)

        min_times = minz.ix[policy]['time_total']

        P = D0[D0.policy == policy]
        f = cgw_f(P.want_and_got.sum(), P.got.sum(), P.want.sum())

        #pl.scatter(df.avg_bestof_time, df.evalb, c=C[name], lw=0)
        #show_frontier(df.avg_bestof_time, df.evalb, c=C[name], interpolation='linear', label=name)
        #[w,b] = np.polyfit(df.pushes, df.avg_bestof_time, deg=1)
        #show_frontier(df.pushes*w + b, df.evalb, interpolation='linear', c=C[name])

        if 0:
            # log-log plot of pushes v. seconds. Really great correlation!
            PP = P[['example','pushes']].join(min_times, on='example')
            PP['log(pushes)'] = np.log(PP.pushes)
            PP['log(seconds)'] = np.log(PP.time_total)
            compare('log(pushes)', 'log(seconds)', data=PP, scatter=1, show_regression=1)
            #pl.figure()
            # pushes v. seconds. Really great correlation!
            #PP = P[['example','pushes']].join(min_times, on='example')
            #compare('pushes', 'time_total', data=PP, scatter=1, show_regression=1)
            pl.ioff(); pl.show()

        if 0:
            # empirical runtime estimates

            # scatter plot sentence length against runtime.
            n_by_time = P[['example','N']].join(min_times, on='example')
            pl.scatter(n_by_time.N, n_by_time.time_total, alpha=0.5, lw=0)

            # highlight median runtime per sentence length.
            n_by_median_time = n_by_time.groupby('N').median()
            pl.plot(n_by_median_time.index, n_by_median_time.time_total, c='k', lw=2)

            # empirical exponent and constant factor
            compare(np.log(n_by_time.time_total), np.log(n_by_time.N), scatter=1, show_regression=1)
            pl.ioff(); pl.show()

        # use early stopping on dev to pick the policy.
        dev = log.ix[log['dev_new_policy_reward'].argmax()]

        row = {'avg_bestof_time': np.mean(min_times),
               'wps': np.mean(P.N) / np.mean(min_times),
               'pushes': np.mean(P.pushes),
               'pops': np.mean(P.pops),
               'policy': policy,
               'dev_pushes': dev.dev_new_policy_pushes,
               'dev_evalb': dev.dev_new_policy_evalb_corpus,
               'type': type_,
               'evalb': f}

        row.update({'args_'+k: v for k,v in args.__dict__.items()})

        data.append(row)

    # remove unused baselines (sorry this is a bit ugly).
    ddd = pd.DataFrame(data)
    others = ddd[ddd.type != 'baseline']
    B = ddd[ddd.type == 'baseline']
    used = set()
    for _, z in others.iterrows():
        [ix] = B[B.policy == z.args_init_weights].index
        used.add(ix)
    B = B.ix[list(used)]
    ddd = others.append(B)

    return ddd



from arsenal.iterextras import window
from arsenal.math.pareto import pareto_frontier
# TODO: this belongs in arsenal.math.pareto
def show_frontier(X, Y, maxX=False, maxY=True, dots=False,
                  XMAX=None, YMIN=None, ax=None, label=None,
                  interpolation='pessimistic', c='b', **style):
    """Plot Pareto frontier.

    Args:

      X, Y: data.

      maxX, maxY: (bool) whether to maximize or minimize along respective
        coordinate.

      dots: (bool) highlight points on the frontier (will use same color as
        `style`).

      ax: use an existing axis if non-null.

      style: keyword arguments, which will be passed to lines connecting the
        points on the Pareto frontier.

      XMAX: max value along x-axis
      YMIN: min value along y-axis

    """
    if ax is None:
        ax = pl.gca()
    sty = {'c': 'b', 'alpha': 0.3, 'zorder': 0}
    sty.update(style)

    if interpolation == 'linear-convex':
        # Convex hull by itself doesn't work, but what we have is ok because its
        # the intersection of the convex hull with the pareto frontier, which is
        # handled below.
        from scipy.spatial import ConvexHull
        X = np.array(X)
        Y = np.array(Y)
        hull = ConvexHull(np.array([X,Y]).T)
        X = X[hull.vertices]
        Y = Y[hull.vertices]

    assert not maxX and maxY, 'need to update some hardcoded logic'

    f = pareto_frontier(X, Y, maxX=maxX, maxY=maxY)
    if not f:
        print colors.yellow % '[warn] Empty frontier'
        return
    if dots:
        xs, ys = zip(*f)
        ax.scatter(xs, ys, lw=0, alpha=0.5, c=sty['c'])

    XMAX = XMAX if XMAX is not None else max(X)
    YMIN = YMIN if YMIN is not None else min(Y)
    if max(X) > XMAX:
        print '[pareto] WARNING: max(X) > XMAX. Plot will not show these points.'
    if min(Y) < YMIN:
        print '[pareto] WARNING: min(X) < XIN. Plot will not show these points.'

    # Connect corners of frontier. The first and last points on frontier have
    # lines which surround the point cloud.
    f = [(min(X), YMIN)] + f + [(XMAX, max(Y))]

    if interpolation == 'pessimistic':
        # Make line segments from adjacent points
        pts = np.array([x for ((a,b), (c,d)) in window(f, 2) for x in [[a,b], [c,b], [c,b], [c,d]]])
    elif interpolation in {'linear','linear-convex'}:
        # Make line segments from adjacent points
        pts = np.array([x for ((a,b), (c,d)) in window(f, 2) for x in [[a,b], [c,d]]])

    # Plot
    del sty['c']
#    ax.plot(pts[:,0], pts[:,1], label=label, c=color, **sty)
    return pts


if __name__ == '__main__':
    main()
