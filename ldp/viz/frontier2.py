#!/usr/bin/env python
"""
Aggregate results into a Pareto frontier plot.
"""

from __future__ import division

import numpy as np
import pylab as pl
from path import path
from itertools import cycle
from arsenal.math.pareto import show_frontier as _show_frontier
from arsenal.terminal import yellow, red
from arsenal.viz.util import AX
from arsenal.cache.pkl import load
from arsenal.humanreadable import htime
from arsenal.iterview import iterview
from viz.interact import PointBrowser

from pandas import DataFrame, to_datetime
from pandas.io.parsers import read_csv

show_frontier = _show_frontier

#pl.xkcd()

#import seaborn as sns
#sns.set_style("whitegrid")
#sns.set_context("paper")
#sns.set_context("talk")
#sns.set_palette("husl")
#sns.set_palette("deep", desat=.6)

# adjust pandas print settings by console width
import pandas
from arsenal.terminal import console_width
pandas.set_option('display.width', console_width())


LW = 2
KILL = []


class Frontier(object):

    def __init__(self, name, df, ACCURACY, RUNTIME):
        self.name = name
        self.data = []
        self.train_br = None
        self.train_ax = None
        self.dev_br = None
        self.dev_ax = None
        self.df = df
        self.YMIN = None
        self.XMAX = None
        self.ACCURACY = ACCURACY
        self.RUNTIME = RUNTIME

    def callback(self, br, x):
        print
        print x.to_string()
        print

        # Note: We can't just look at `br.X` to inspect the selected point's
        # learning curve because the dataframe in `br` has been filtered (e.g.,
        # by --last).
        log = get_log(x.log, self.ACCURACY, self.RUNTIME)

        # acc and run use to train this job.
        acc = x.args_accuracy
        run = x.args_runtime

        # show train/dev learning curves
        pl.ion()
        ax = pl.figure().add_subplot(111)
        [tradeoff] = set(log.tradeoff)

        ax.set_title(r'Reward/iteration (lam=%g, acc=%s, run=%s)' \
                     % (tradeoff, acc, run))
        #ax.plot(log.iteration, log.train_reward, c='b', label='train')
        #ax.plot(log.iteration, log.dev_reward, c='r', label='dev')

        ax.plot(log.iteration, log.train_new_policy_reward, c='b', label='train')
        ax.plot(log.iteration, log.dev_new_policy_reward, c='r', label='dev')
        maxes = running_max(list(log.iteration), list(log.dev_new_policy_reward))
        ax.scatter(maxes[:,0], maxes[:,1], lw=0)

        # indicate where selected point is on learning curve
        ax.axvline(x.iteration, lw=1, linestyle='-.', c='k')
        ax.set_xlim(0, log.iteration.max()+1)
        ax.set_ylabel('Reward')
        ax.set_xlabel('Iteration')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        if 0:
            ax = pl.figure().add_subplot(111)
            ax.set_title('Accuracy per iteration')
            ax.plot(log.iteration, log.train_accuracy, c='b', label='train')
            ax.plot(log.iteration, log.dev_accuracy, c='r', label='dev')
            ax.axvline(x.iteration, lw=1, linestyle='-.', c='k')
            ax.set_ylabel('Accuracy (%s)' % self.ACCURACY)
            ax.set_xlabel('Iteration')
            ax.set_xlim(0, log.iteration.max()+1)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

            ax = pl.figure().add_subplot(111)
            ax.set_title('Runtime per iteration')
            ax.plot(log.iteration, log.train_runtime, c='b', label='train')
            ax.plot(log.iteration, log.dev_runtime, c='r', label='dev')
            ax.axvline(x.iteration, lw=1, linestyle='-.', c='k')
            ax.set_ylabel('Runtime (%s)' % self.RUNTIME)
            ax.set_xlabel('Iteration')
            ax.set_xlim(0, log.iteration.max()+1)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

        # show trajectory of this point on Pareto frontier
        #br.ax.plot(log.dev_runtime, log.dev_accuracy, c='k', alpha=0.5, lw=2)

        if 0:
            import os
            # let's just hope there is one file in there
            [rollouts_file] = x.directory.glob('dump/init.npz.inspect_rollouts.csv.gz')
            os.system('analysis/inspect-rollouts/analyze.py %s &' % rollouts_file)

        pl.show()

    def plot(self):
        df = self.df
        if df.empty:
            return

        self.XMAX = max(df.dev_runtime.max(), df.train_runtime.max())
        self.YMIN = 0

        self.train_ax = pl.figure().add_subplot(111)
        self.dev_ax = pl.figure().add_subplot(111)

        # TODO: It doesn't make sense to use the same lambda for diferent
        # acc/run than those used to train. One option is to find a reasonable
        # value or use an upper/lower interval.

        self.dev_br = PointBrowser(df, xcol='dev_runtime', ycol='dev_accuracy',
                                   ax=self.dev_ax, callback=self.callback,
                                   plot_kwargs=dict(c=df.tradeoff, alpha=1, zorder=2))

        self.train_br = PointBrowser(df, xcol='train_runtime', ycol='train_accuracy',
                                     ax=self.train_ax, callback=self.callback,
                                     plot_kwargs=dict(c=df.tradeoff, alpha=1, zorder=2))

        self.train_ax.set_title(self.name + ' (train)')
        self.dev_ax.set_title(self.name + ' (dev)')

        # draw frontiers
        show_frontier(df.dev_runtime, df.dev_accuracy, ax=self.dev_ax,
                      XMAX=self.XMAX, YMIN=self.YMIN, lw=2, alpha=1, zorder=1)
        show_frontier(df.train_runtime, df.train_accuracy, ax=self.train_ax,
                      XMAX=self.XMAX, YMIN=self.YMIN, lw=2, alpha=1, zorder=1)

        # plot the opposite curve so see overfitting (and possibly different in data).
        #show_frontier(df.dev_runtime, df.dev_accuracy, ax=train_ax, c='g')
        #show_frontier(df.train_runtime, df.train_accuracy, ax=dev_ax, c='g')

    def show_baseline(self, d):
        "Pass in baseline dataframe"
        self.XMAX = max([self.XMAX, d.train_runtime.max(), d.dev_runtime.max()])
        self.YMIN = min([self.YMIN, d.train_accuracy.min(), d.dev_accuracy.min()])
        if not d.train_runtime.isnull().all():
            show_frontier(d.train_runtime, d.train_accuracy, ax=self.train_ax,
                          c='r', XMAX=self.XMAX, YMIN=self.YMIN, lw=LW, label='baseline')
        if not d.dev_runtime.isnull().all():
            show_frontier(d.dev_runtime, d.dev_accuracy, ax=self.dev_ax,
                          c='r', XMAX=self.XMAX, YMIN=self.YMIN, lw=LW, label='baseline')



# - [2015-12-29 Tue] The strange option to pass in `p` (policy) is to choose
#   between version of the logged performance to use for the plots. Back when we
#   were using SEARN training, there were a few options (stochastic or
#   deterministic average, new_policy). Back then, I found that deterministic
#   averaging (the column named "new_avg") worked best (for a number of
#   reasons), so I'd use that policy instead of new_policy. In the case of
#   LOLS/DAgger new_policy is very stable and the averaging policies doesn't
#   seem necessary.
def get_log(f, ACCURACY, RUNTIME, p='new_avg'):
    log = read_csv(f)

    if p not in log:
        p = 'new_policy'  # this happens with baseline


    if 'train_%s_%s' % (p, ACCURACY) not in log:
        print 'ERROR', f
        return

    log['train_accuracy'] = log['train_%s_%s' % (p, ACCURACY)]
    log['train_runtime'] = log['train_%s_%s' % (p, RUNTIME)]
    log['train_reward'] = log.train_accuracy - log.tradeoff*log.train_runtime

    log['dev_accuracy'] = log['dev_%s_%s'% (p, ACCURACY)]
    log['dev_runtime'] = log['dev_%s_%s' % (p, RUNTIME)]
    log['dev_reward'] = log.dev_accuracy - log.tradeoff*log.dev_runtime

    return log


def load_results(results, _args, filters=()):
    jobs_running = [y.split()[0] for y in list(file('tmp/jobs'))[2:]]

    data = []
    jobs = []
    msgs = []
    for x in iterview(results.glob('*')):

        # Extract name of the experiment, {YYYY-MM-DD}-{NAME}-{argument hash}
        name = x.basename()[11:].split('-')[0]

        if _args.jobids:
            if (x / 'sge-jobid.txt').exists():
                jobid = (x / 'sge-jobid.txt').text().strip()
                if jobid not in _args.jobids:
                    continue
            else:
                continue

        else:
            if not any(p == name for p in filters):   # substring match
                continue

        # TODO: the finish file doesn't get written if we call qdel, but the log
        # files now contain some timestamps, so we should grab the last
        # timestamp logged as the "finish time."
        done = (x / 'finish').exists()

        if (x / 'sge-jobid.txt').exists():
            jobid = (x / 'sge-jobid.txt').text().strip()
        else:
            jobid = None

        dump_exists = False
        log_exists = False
        args_exists = False

        args = {}

        d = x / 'dump'
        if d.exists():                 # job hasn't started (might've died)
            dump_exists = True
        else:
            # Dump doesn't exist for reasons other than it failed to get
            # scheduled. This might have to do with lack of permissions or a
            # failure in the python code (e.g., ImportError).
            assert jobid is None, x
            # try again, but this time it's not nested.
            d = x
            if d.exists():
                dump_exists = True

        if dump_exists:
            if (d / 'log.csv').exists():   # job hasn't produced first data point
                log_exists = True

            if (d / 'args.pkl').exists():
                # load command-line arguments fro pickle and add prefix (`'args_'`)
                # to avoid collisions with other column names.
                args = load(d / 'args.pkl')
                args = {'args_' + k: v for k, v in args.__dict__.items()}
                args_exists = True

        if not args_exists or not log_exists or not dump_exists:
            miss = []
            if jobid is None:
                miss.append('jobid')
            if not args_exists:
                miss.append('args')
            if not log_exists:
                miss.append('log')
            if not dump_exists:
                miss.append('dump')
            msgs.append('%s %s' % (x, yellow % '(missing: %s)' % ' '.join(miss)))

        # Note: two jobs /might/ get assigned the same job id, which might cause
        # 'hash collision' style problems.
        running = (jobid in jobs_running)

        if args_exists:
            start = to_datetime((x/'start').text())
            elapsed = (start.now() - start).total_seconds()
        else:
            continue

        J = dict(directory=x,
                 jobid=jobid,
                 start=start,
                 elapsed=elapsed,
                 done=done,
                 log_exists=log_exists,
                 dump_exists=dump_exists,
                 args_exists=args_exists,
                 running=running)

        # TODO: use dedicated CLI options for filtering jobs. The eval trick
        # (below) is unnecessary and provides a clunky filter notation.
        class __args:
            pass
        for k,v in args.items():
            if k.startswith('args_'):
                setattr(__args, k[5:], v)   # drop 'args_' prefix
        skip = 0
        for ff in _args.filter:
            fff = ff.replace('df.args_', '__args.')
            if not eval(fff):
                skip = 1
                break
        if skip:
            continue

        J.update(args)
        jobs.append(J)

        if not log_exists:
            continue

        log = get_log(d / 'log.csv', _args.accuracy, _args.runtime)

        if log is None:
            continue

        for (_, e) in log.iterrows():
            b = e.to_dict()
            status = dict(name = name,
                          jobid = jobid,
                          running = running,
                          done = done,
                          log = d / 'log.csv')
            b.update(J)
            b.update(status)
            b.update(args)
            data.append(b)

        # add column for time-elapsed (in days) since initial iteration.
        el = log.datetime.map(to_datetime).tolist()
        log['elapsed'] = [(x - el[0]).total_seconds() / (24*60*60) for x in el]

        # TODO: move to after filters are applied
        if _args.lc:        # enable to view learning curves super-imposed on one another.
            learning_curve_handler(_args, args, log, jobid)

    for msg in msgs:
        print msg

    return data, jobs


def learning_curve_handler(_args, args, log, jobid):
    # Note: iterations appear to start at 1.

    show_each = _args.show_each
    if show_each:
        ax1 = ax2 = pl.figure().add_subplot(111)

        [[ro],[acc],[run]] = [np.unique(args['args_roll_out']),
                              np.unique(args['args_accuracy']),
                              np.unique(args['args_runtime'])]

        ax1.set_title('jobid: %s. ro/acc/run %s/%s/%s'
                      % (jobid, ro, acc, run))
    else:
        if 0:
            ax1 = AX['trainlc']
            ax2 = AX['devlc']
            ax1.set_title('lc train')
            ax2.set_title('lc dev')
        else:
            # group learning curves by regularizer
            col = 'args_C'
            ax1 = AX['trainlc-%s' % args.get(col)]
            ax2 = AX['devlc-%s' % args.get(col)]
            ax1.set_title('lc train %s' % args.get(col))
            ax2.set_title('lc dev %s' % args.get(col))

    # Pick x-axis time or iterations.
    #X = log.iteration
    X = log['elapsed']
    ax1.set_xlabel('days')
    ax2.set_xlabel('days')

    if log.get('train_accuracy') is not None:
        #ax1.plot(log.iteration, log.train_accuracy - log.tradeoff * log.train_runtime, alpha=1, c='b')
        ax1.plot(X, log.train_new_policy_reward, alpha=1, c='b')
        maxes = running_max(list(X), list(log.train_new_policy_reward))
        ax1.scatter(maxes[:,0], maxes[:,1], lw=0)
    if log.get('dev_accuracy') is not None:
        #ax2.plot(X, log.dev_accuracy - log.tradeoff * log.dev_runtime, alpha=1, c='r')
        #patience(log, ax2)
        ax2.plot(X, log.dev_new_policy_reward, alpha=1, c='r')
        maxes = running_max(list(X), list(log.dev_new_policy_reward))
        ax2.scatter(maxes[:,0], maxes[:,1], lw=0)

    if show_each:
        pl.ioff()
        pl.show()

    if _args.kill_mode:
        if raw_input('kill?').startswith('y'):
            KILL.append(jobid)
            print 'KILL', ' '.join(KILL)


def patience(log, ax=None):
    ax = ax or pl.gca()
    maxes = running_max(list(log.iteration),
                        list(log.dev_accuracy - log.tradeoff * log.dev_runtime))
    ax.scatter(maxes[:,0], maxes[:,1], lw=0)


def running_max(iterations, rewards):
    v = float('-inf')
    m = []
    for i, r in sorted(zip(iterations, rewards)):
        if r > v:
            v = r
            m.append([i, r])
    return np.array(m)


def main():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--save', default='tmp/results.csv')
    p.add_argument('--interpolation',
                   choices=['linear', 'pessimistic', 'parametric', 'linear-convex'],
                   default='pessimistic')
    # reward definition
    p.add_argument('--accuracy',
                   #choices=ACC.OPTS,
                   #default='evalb',
                   required=True,
                   help='Measurement used for plotting.')
    p.add_argument('--runtime',
                   #choices=RUN.OPTS,
                   #default='pushes',
                   required=True,
                   help='Measurement used for plotting.')
    # what jobs to show
    p.add_argument('--target', required=True)
    p.add_argument('--baseline', required=False)
    p.add_argument('--others', nargs='*', default=[])
    p.add_argument('--filter', nargs='*', default=[],
                   help="e.g., --filter 'df.args_C==-12'")

    # TODO: added nicer filters for things I've been doing with --filter.
#    p.add_argment('--grammar')
#    p.add_argment('--surrogate-accuracy', help='measure to filter jobs by.')
#    p.add_argment('--surrogate-runtime', help='measure to filter jobs by.')

    # finalization
    p.add_argument('--last', action='store_true')
    p.add_argument('--early-stop', action='store_true')
    p.add_argument('--early-stop-dev-cheat', action='store_true')
    p.add_argument('--baseline-is-init', action='store_true')
    # extra plots
    p.add_argument('--tradeoff-plot', action='store_true')
    p.add_argument('--lc', action='store_true')
    p.add_argument('--show-train', action='store_true')
    p.add_argument('--show-each', action='store_true',
                   help='flip thru learning curves')
    # misc
    p.add_argument('--kill-mode', action='store_true',
                   help='flip thru learning curves asking "kill? [y/n]"')
    p.add_argument('--jobids', nargs='*', default=[])
    p.add_argument('-i', action='store_true')

    p.add_argument('--other-files', nargs='*', default=[])

    args = p.parse_args()

    # use linear interpolation for plotting Pareto frontier
    global show_frontier
    if args.interpolation in {'linear', 'linear-convex'}:
        def _show_frontier_linear(*a,**k):
            k['interpolation'] = args.interpolation #'linear'
            _show_frontier(*a, **k)
        show_frontier = _show_frontier_linear

    if args.kill_mode:
        args.show_each = True
    if args.show_each:
        args.lc = True

    results = path('results')

    filters = set(args.others + [args.target, args.baseline])

    ACCURACY = args.accuracy
    RUNTIME = args.runtime

#    if args.load:
#        D = read_csv('tmp/results.csv', index_col=0)
#        jobs = read_csv('tmp/jobs.csv', index_col=0)
#    else:

    data, jobs = load_results(results, args, filters)
    D = DataFrame(data)
    #D.to_csv('tmp/results.csv')

    jobs = DataFrame(jobs)
    #jobs.to_csv('tmp/jobs.csv')

    target = args.target
    df = D[D.name == target]

    # apply CLI filter options
    for f in args.filter:
        df = df[eval(f)]    # this is pretty ghetto.

    if args.baseline_is_init:
        # Note: this will use the same filters as the main experiment (e.g.,
        # regularization parameters).
        baseline = df[(df.iteration==1)]
    else:
        baseline = D[D.name == args.baseline]

    # NOTE: do before iteration filters (e.g., early-stop/last)
    def PPPP(df):
        "Patience: Find out how long it's been since the last improvement."
        P = []
        for jobid, D in df.groupby('jobid'):
            #p = running_max(list(D.iteration), list(D.dev_reward))
            p = running_max(list(D.iteration), list(D.dev_new_policy_reward))
            P.append({'jobid': jobid, 'patience': D.iteration.max() - p[-1,0]})
        return DataFrame(P).set_index('jobid')
    P = PPPP(df)


    # TODO: In most experiments, train rewards are based on a sample which
    # varies across iterations -- maybe I should use the same examples
    # throughout. Thus, we probably don't really want to do early stopping based
    # on this value without smoothing or something.
    if args.last or args.early_stop or args.early_stop_dev_cheat:
        assert not (args.early_stop and args.last), "Can't have both."
        ddd = []
        for _, dd in df.groupby('jobid'):

            if args.last:
                best = dd.ix[dd.iteration == dd.iteration.max()]  # last iteration
            elif args.early_stop:
                best = dd.ix[dd.train_new_policy_reward == dd.train_new_policy_reward.max()]  # best train iteration
            elif args.early_stop_dev_cheat:
                best = dd.ix[dd.dev_new_policy_reward == dd.dev_new_policy_reward.max()]  # best dev iteration

                # Break ties in dev reward in favor of the training reward
                #_best = dd.ix[dd[dd.dev_new_policy_reward == dd.dev_new_policy_reward.max()].train_new_policy_reward.argmax()]
                #_best = dd.ix[dd.dev_new_policy_reward.argmax()]

                #print _best.iteration, 'out of', dd.iteration.max(), 'iterations'
                #print _best.log
                #print >> get_params, _best.log.dirname() / 'new_policy-%03d.npz' % _best.iteration

            else:
                raise ValueError('Unrecognized option.')

            # We require the following yucky code because `best.to_dict()`
            # returns a dict with values that are each
            # <strike>single-entry</strike> dicts.
            #
            #  ^^^ I think this happens because best might contain more than one
            #  value. so clearly when you convert it to a dict you should a
            #  collection of potential values -- that's why theres a dict.
            row = {}
            for k,v in best.to_dict().items():
                v = list(v.values())[-1]          # take the last one if there are ties.
                row[k] = v
            ddd.append(row)

        df = DataFrame(ddd)

    assert not df.empty, 'DataFrame is empty.'

    if baseline is not None and not baseline.empty:
        args_check(baseline, 'baseline')
    args_check(df, 'df')

    if args.tradeoff_plot:
        # TODO: [2015-02-27 Fri] maybe we should sample tradeoff on a nonlinear
        #   scale (e.g., log-scale). We seem to get a much more linear response
        #   from training. This would help prevent the over-sampling of values
        #   with low-accuracy and low-runtime.
        pl.figure()
        pl.scatter(df.tradeoff, df.dev_accuracy, lw=0)
        pl.title(r'accuracy (%s) by $\lambda$' % ACCURACY)
        pl.xlabel(r'tradeoff ($\lambda$)')
        pl.ylabel(r'accuracy (%s)' % ACCURACY)
        pl.figure()
        pl.scatter(df.tradeoff, df.dev_runtime, lw=0)
        pl.title(r'runtime (%s) by $\lambda$' % RUNTIME)
        pl.xlabel(r'tradeoff ($\lambda$)')
        pl.ylabel(r'runtime (%s)' % RUNTIME)

        # TODO: It would be interesting to compare the baseline's tradeoff
        # parameter to ours, but they are sort of incomparable.
        #
        #pl.figure()
        #pl.scatter(baseline.tradeoff, baseline.dev_accuracy, lw=0)
        #pl.title(r'BASELINE accuracy (%s) by $\lambda$' % ACCURACY)
        #pl.xlabel(r'tradeoff ($\lambda$)')
        #pl.ylabel(r'accuracy (%s)' % ACCURACY)
        #pl.figure()
        #pl.scatter(baseline.tradeoff, baseline.dev_runtime, lw=0)
        #pl.title(r'BASELINE runtime (%s) by $\lambda$' % RUNTIME)
        #pl.xlabel(r'tradeoff ($\lambda$)')
        #pl.ylabel(r'runtime (%s)' % RUNTIME)


    frontier = Frontier(args.target, df, args.accuracy, args.runtime)
    frontier.plot()

#    # Plot reference policies (oracle1, unpruned and fast-mle).
#    #if args.target in {'searn4', 'searn5'}:
#    if ACCURACY != 'no-fail':
#        # Note: this is sort of silly. Jobs each have a copy of the
#        # baseline.csv file. Here we have the first one that comes
#        # up. (Warning: we might mix baselines, so be careful). This "guess'
#        # let's us avoid passing the file in at the command-line. It's
#        # conceivable that we might want to show multiple reference policies
#        # (e.g., different grammars on the same plot), in which case we
#        # should probably have a CLI option to specify these files.
#
#        # TODO: report baseline parser's accuracy at most-acc's runtime
#        #
#        #  - Create a class for representing a Pareto frontier, which supports the
#        #    relevant query types: accuracy @ runtime and runtime @ accuracy.
#
#        show_reference_policies = 0
#        if show_reference_policies:
#
#            baseline_csv = path('.').glob('results/*-%s-*/dump/baseline.csv' % args.target)[0]
#            B = read_csv(baseline_csv)
#
#            # Show reference policies (e.g., unpruned, oracle)
#            marker = {'oracle1': '*', 'fastmle': '^', 'unpruned': 'x'}
#            for policy in ['oracle1', 'unpruned']:
#                for name in ['train', 'dev']:
#                    if policy == 'unpruned':   # XXX: skip unpruned because it makes the plot ugly
#                        continue
#                    getattr(frontier, '%s_ax' % name) \
#                        .scatter([B['%s_%s_%s' % (name, policy, RUNTIME)]],
#                                 [B['%s_%s_%s' % (name, policy, ACCURACY)]],
#                                 c='r', s=40, marker=marker[policy])
#
#        if args.show_init:
#            if len(init_run) == 0:
#                print '[%s]' % red % 'error', 'Failed to find initializer.'
#            else:
#                frontier.dev_ax.scatter(init_run, init_acc, s=75, c='k', marker='^')
#
#        if show_reference_policies:
#            [[unpruned_acc, unpruned_run]] = B[['dev_unpruned_%s' % ACCURACY, 'dev_unpruned_%s' % RUNTIME]].get_values()
#            most_acc_acc, most_acc_run = df.ix[df['dev_accuracy'].argmax()][['dev_accuracy', 'dev_runtime']]
#            print 'unpruned: %.4f %g' % (unpruned_acc*100, unpruned_run)
#            print 'most_acc: %.4f %g' % (most_acc_acc*100, most_acc_run)
#            print 'MOSTACC:  %.2f points more accurate and %.2fx faster than unpruned.' % (100*(most_acc_acc - unpruned_acc), unpruned_run / most_acc_run)
#
#        if show_reference_policies:
#            # [2015-06-08 Mon] hack together fast-mle by piecing together
#            #   unpruned with oracle runtime (which isn't really exact if more
#            #   grammar rules fire on the unpruned mask... it's not unreasonble
#            #   more rules fire in on unpruned since the gold mask might be
#            #   unsupported by the parser).
#            [acc] = B['dev_unpruned_%s' % ACCURACY].get_values()
#            [run] = B['dev_oracle1_%s' % RUNTIME].get_values()
#            frontier.dev_ax.scatter([run], [acc], c='r', s=40, marker=marker['fastmle'])
#            [acc] = B['train_unpruned_%s' % ACCURACY].get_values()
#            [run] = B['train_oracle1_%s' % RUNTIME].get_values()
#            frontier.train_ax.scatter([run], [acc], c='r', s=40, marker=marker['fastmle'])
#
#    else:
#        print '[%s] %s' % (red % 'ERROR', 'no baseline.csv file found')

    if 1:
        frontier.show_baseline(baseline)

    # Show frontiers for 'other' things. Not the baseline (because the baseline
    # gets special handling), but things like older experiments.
    others = set(D.name.unique()) - {args.target, args.baseline}
    if others:
        print
        print yellow % 'Other curves'
        print yellow % '============'
        for other, color in zip(sorted(others), cycle(['m','g','c','k','b'])):
            print '%-9s' % other, color
            alpha = 1.0
            d = D[D.name == other]
            args_check(d, other)
            if not d.train_runtime.isnull().all():
                show_frontier(d.train_runtime, d.train_accuracy,
                              ax=frontier.train_ax,
                              c=color, alpha=alpha,
                              XMAX=frontier.XMAX, YMIN=frontier.YMIN, lw=LW)
            if not d.dev_runtime.isnull().all():

                # XXX: this is just some cruft from debugging set of jobs. Can probably delete.
                #print d[['dev_runtime','dev_accuracy']].sort('dev_accuracy')
                #print df[['dev_runtime','dev_accuracy']].sort('dev_accuracy')
                #assert (np.abs(np.array(df.dev_accuracy.sort_values()) - np.array(d.dev_accuracy.sort_values())) < 1e-5).all()
                #assert (np.abs(np.array(df.dev_runtime.sort_values()) - np.array(d.dev_runtime.sort_values())) < 1e-5).all()

                show_frontier(d.dev_runtime, d.dev_accuracy,
                              ax=frontier.dev_ax,
                              c=color, alpha=alpha, lw=LW,
                              XMAX=frontier.XMAX, YMIN=frontier.YMIN, label=other)

    for other in args.other_files:
        dd = read_csv(other, index_col=0)

        dd['dev_accuracy'] = dd['dev_new_policy_%s' % ACCURACY]
        dd['dev_runtime'] = dd['dev_new_policy_%s' % RUNTIME]
        dd['dev_reward'] = dd.dev_accuracy - dd.tradeoff*dd.dev_runtime

        show_frontier(dd.dev_runtime, dd.dev_accuracy,
                      ax=frontier.dev_ax,
                      lw=LW, label=other)

    frontier.dev_ax.legend(loc=4)

    print

    if len(df.args_C.unique()) > 1:
        show_groupby_frontiers(df, 'args_C', frontier.XMAX, frontier.YMIN, baseline=baseline)

#    if len(df.args_accuracy.unique()) > 1:
#        show_groupby_frontiers(df, 'args_accuracy', frontier.XMAX, frontier.YMIN)

    if len(df.args_accuracy.unique()) > 1:
        show_groupby_frontiers(df, 'args_roll_out', frontier.XMAX, frontier.YMIN)


#    if len(df.args_classifier.unique()) > 1:
#        show_groupby_frontiers(df, 'args_classifier', frontier.XMAX, frontier.YMIN)

    #show_groupby_frontiers(df, 'iteration', baseline=baseline)
    #asymmetry_plots(baseline)

    job_summary(jobs)

    #pl.ion()
    #pl.show()

    # Summary of jobs that are currently running, e.g., How many iterations have
    # they run for? How long has it been since they improved (patience)?
    J = df.join(jobs, 'jobid', rsuffix='_xxx')  # needs a suffix because columns overlap.
    J = J.groupby('jobid').max()
    J = J.join(P)
    J['elapsed'] = map(htime, J.elapsed)
    J['startdate'] = J.start.map(lambda x: x.date())
    J = J.sort_values('start')

    show_cols = [
        'iteration', 'running', 'patience', 'tradeoff', 'elapsed', 'startdate',
        'dev_accuracy', 'dev_runtime',
        'log',
    ]
    running = J[J['running']][show_cols]
    if running.empty:
        print red % 'No jobs running.'
    else:
        print running

    #highlight_region(df, baseline, B, frontier.dev_ax, ACCURACY, RUNTIME)
    frontier.dev_ax.set_title('Pareto frontier *DEV*')
    frontier.dev_ax.set_xlabel('runtime (%s)' % RUNTIME)
    frontier.dev_ax.set_ylabel('accuracy (%s)' % ACCURACY)
    frontier.dev_ax.set_xlim(0, None)
    frontier.dev_ax.set_ylim(0, 1)
    frontier.dev_ax.figure.canvas.draw()
    frontier.dev_ax.figure.savefig('tmp/pareto.png')

    if args.save:
        df.to_csv(args.save)
    #baseline.to_csv('tmp/baseline.csv')

    # hide the train plot.
    if not args.show_train:
        pl.close(frontier.train_ax.figure)

    if args.i:
        from arsenal.debug import ip; ip()
    else:
        pl.ioff()
        pl.show()


def job_summary(jobs):

    def cnt(name, column):
        n = len(jobs[jobs[column]])
        pct(name, n, len(jobs))

    def pct(name, n, m):
        print '%-7s %.2f (%s/%s)' % (name, n / m, n, m)

    print
    print yellow % 'Job status'
    print yellow % '=========='
    cnt('running', 'running')
    cnt('done',    'done')
    cnt('log',     'log_exists')

    qstat = load_qstat('tmp/jobs')

    print
    print yellow % 'SGE status'
    print yellow % '=========='
    if qstat.empty:
        print red % 'Failed to find anything in qstat.'
        return
    for k, d in qstat.groupby('status'):
        pct(k, len(d), len(qstat))

    if 0:
        # use kl-divergence to show what are relevant features of jobs still running.
        from arsenal.math.featureselection import kl_filter
        from arsenal.iterextras import take
        list(take(50, kl_filter([(x.done, ['%s=%s' % (k,v) for (k,v) in x.iteritems() if k.startswith('args_')])
                                 for _, x in jobs.iterrows()],
                                feature_count_cuttoff=2)))

    return qstat


def load_qstat(filename):
    "Load qstat file."
    with file(filename) as f:
        d = [np.array(x.strip().split())[:9] for i, x in enumerate(f) if i >= 2]
    names = ['jobid', 'priority', 'name', 'user', 'status',
             'submit', 'start', 'queue', 'slots']
    if len(d) == 0:
        return DataFrame(columns=names)
    return DataFrame(d, columns=names)


def show_groupby_frontiers(df, groupby, XMAX, YMIN, baseline=None):
    show_group_frontiers(df.groupby(groupby), groupby, XMAX, YMIN, baseline=baseline)


def show_group_frontiers(groups, name, XMAX, YMIN, baseline=None):
    colors = cycle(['m','g','b','y','c','k'])
    ax = pl.figure().add_subplot(111)
    groups = list(groups)

    print
    print yellow % name
    print yellow % '============'
    for color, (group_name, r) in zip(colors, groups):
        print '%s; color: %s; len %s' % (group_name, color, len(r))

    for color, (group_name, r) in zip(colors, groups):
        show_frontier(r.dev_runtime, r.dev_accuracy, label='%s (dev)' % group_name,
                      c=color, alpha=0.5, ax=ax, XMAX=XMAX, YMIN=YMIN, lw=LW)
#        show_frontier(r.train_runtime, r.train_accuracy, label='%s (train)' % group_name,
#                      c=color, linestyle=':', alpha=0.5, ax=ax, XMAX=XMAX, YMIN=YMIN, lw=LW)

    ax.set_xlabel('runtime')
    ax.set_ylabel('accuracy')

    if baseline is not None:
#        show_frontier(baseline.train_runtime, baseline.train_accuracy, ax=pl.gca(),
#                      c='r', linestyle=':', alpha=0.25, label='baseline (train)', XMAX=XMAX, YMIN=YMIN, lw=LW)
        show_frontier(baseline.dev_runtime, baseline.dev_accuracy, ax=pl.gca(),
                      c='r', alpha=0.25, label='baseline (dev)', XMAX=XMAX, YMIN=YMIN, lw=LW)

    ax.set_title('Frontiers grouped-by %s' % name)
    #ax.set_xlim(0.35, .85)
    #ax.set_ylim(.74, .83)

    if 0:
        ax.legend(loc='best')
        pl.tight_layout()
    else:
        # Shink current axis by 20%
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

    ax.figure.canvas.draw()


def args_check(d, name):
    "Check that command-line arguments are the same except the tradeoff parameter."
    [m] = d.jobid.unique().shape
    for c in d.columns:
        if c != 'iteration':
            if not c.startswith('args_') or c == 'args_results' or c == 'args_tradeoff':
                continue
        vals = d[c].unique()
        [n] = vals.shape
        if n != 1:
            print '[%s] %r mismatch got %s unique values for %s (jobs=%s)' % (yellow % 'WARN', name, n, c, m)


if __name__ == '__main__':
    main()
