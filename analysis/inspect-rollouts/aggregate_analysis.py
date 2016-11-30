from __future__ import division

import numpy as np
import pylab as pl
import seaborn as sns
sns.set(font_scale=1.75)
#sns.set(font_scale=4)
#pl.rc('text', usetex=True)
pl.rc('font', family="Times New Roman")

import pandas as pd
from arsenal import colors, wide_dataframe

wide_dataframe()


def rename_grammar(g):
    return {'big': 'fine grammar', 'medium': 'coarse grammar'}[g]


c_g = '#f1a340'
c_n = '#998ec3'


def plot2(GRAMMAR, ACC, RUN, ax, col):

    csv_init = 'tmp/inspect-rollouts-%s-%s-%s-init.csv' % (GRAMMAR, ACC, RUN)
    csv_best = 'tmp/inspect-rollouts-%s-%s-%s-policy.csv' % (GRAMMAR, ACC, RUN)

    dfs = {}
    dfs['init'] = pd.read_csv(csv_init).sort_values('tradeoff')
    dfs['best'] = pd.read_csv(csv_best).sort_values('tradeoff')

    if 0:
        for name, df in dfs.iteritems():
            kw = dict(linestyle = '-' if name == 'init' else ':')
            ax.plot(df.tradeoff, df.frac_prune_gold_improves_acc, c='r', label='acc', **kw)
            ax.plot(df.tradeoff, df.frac_prune_gold_improves_rew, c='b', label='rew', **kw)

    else:
        # The effect pruning a gold span on accuracy (just the sign), given that
        # there is difference in reward.
        for name, df in dfs.iteritems():
            kw = dict(linestyle = '-' if name == 'init' else ':')
            ax.plot(df.tradeoff, df.diff_g_acc_incr, c='g', **kw)
            ax.plot(df.tradeoff, df.diff_g_acc_same, c='y', **kw)
            ax.plot(df.tradeoff, df.diff_g_acc_decr, c='r', **kw)

    ax.set_xlim(dfs['best'].tradeoff.min(),
                dfs['best'].tradeoff.max())   # use this dataframe for x-axis, because somethimes init includes more points.

    #pl.legend(loc='best', fontsize=8)
    ax.set_title(rename_grammar(GRAMMAR))
    ax.set_xlabel(r'$\lambda$ (log scale)')
    ax.set_xscale('log')
    #ax2.legend(loc='best')

    ax.grid(False)
    if col == 1:
        ax.get_yaxis().set_visible(False)


def plot(GRAMMAR, ACC, RUN, ax, col):
    """
    Regret breakdown plot.
    """

    csv_init = 'tmp/inspect-rollouts-%s-%s-%s-init.csv' % (GRAMMAR, ACC, RUN)
    csv_best = 'tmp/inspect-rollouts-%s-%s-%s-policy.csv' % (GRAMMAR, ACC, RUN)

    dfs = {}
    dfs['init'] = pd.read_csv(csv_init).sort_values('tradeoff')
    dfs['best'] = pd.read_csv(csv_best).sort_values('tradeoff')

    for name, df in dfs.iteritems():
        kw = dict(linestyle = '-' if name == 'init' else '--', lw=3)

        # Graph #1: vertical scale is probability
        k_label = g_label = n_label = ''
        if name != 'best':
            k_label = r'$p({\sf diff} )$'
            g_label = r'$p({\sf diff} | {\sf gold})$'
            n_label = r'$p({\sf diff} | \neg\,{\sf gold})$'
        ax[0].plot(df.tradeoff, df.p_D,   c='k', label=k_label, **kw)
        ax[0].plot(df.tradeoff, df.p_D_G, c=c_g, label=g_label, **kw)
        ax[0].plot(df.tradeoff, df.p_D_N, c=c_n, label=n_label, **kw)

        # Graph #2: vertical scale is Bodenstab regret (in reward space)
        k_label = g_label = n_label = ''
        if name != 'best':
            k_label = r'$\mathbb{E}\left[{\sf regret} | {\sf diff} \right]$'
            g_label = r'$\mathbb{E}\left[{\sf regret} | {\sf diff}, {\sf gold} \right]$'
            n_label = r'$\mathbb{E}\left[{\sf regret} | {\sf diff}, \neg\,{\sf gold} \right]$'

        ax[1].plot(df.tradeoff,  df.regret_D, c='k', label=k_label, **kw)
        ax[1].plot(df.tradeoff, df.regret_GD, c=c_g, label=g_label, **kw)
        ax[1].plot(df.tradeoff, df.regret_ND, c=c_n, label=n_label, **kw)

        # Graph #3: product of graphs #1 and #2, and on a much smaller vertical scale
        k_label = g_label = n_label = ''
        if name != 'best':
            k_label = r'$\mathbb{E}\left[{\sf regret} \right]$'
            g_label = r'$\mathbb{E}\left[{\sf regret} | {\sf gold} \right]$'
            n_label = r'$\mathbb{E}\left[{\sf regret} | \neg\,{\sf gold} \right]$'

        ax[2].plot(df.tradeoff, df.regret_D  * df.p_D  , c='k', label=k_label, **kw)
        ax[2].plot(df.tradeoff, df.regret_GD * df.p_D_G, c=c_g, label=g_label, **kw)
        ax[2].plot(df.tradeoff, df.regret_ND * df.p_D_N, c=c_n, label=n_label, **kw)

        # Manually adjust the y-limits
        ax[0].set_ylim(0, 0.41)
        ax[1].set_ylim(0, 0.5)
        ax[2].set_ylim(0, 0.02)

        # Manually set x-axis limits and scale
        for i in range(3):
            ax[i].set_xlim(dfs['best'].tradeoff.min(), dfs['best'].tradeoff.max())
            ax[i].set_xscale('log')

        # set y-axis labels (and scale)
        use_log_y = 0
        if use_log_y:
            ax[0].set_ylabel('diff (log scale)')
            ax[1].set_ylabel(r'regret | diff (log scale)')
            ax[2].set_ylabel(r'regret (log scale)')
            for i in range(3):
                ax[i].set_yscale('log')
        else:
            ax[0].set_ylabel('diff')
            ax[1].set_ylabel(r'regret | diff')
            ax[2].set_ylabel(r'regret')

        # Only the bottom row get an x-label, others get there x-axis erased.
        ax[2].set_xlabel(r'$\lambda$ (log scale)')
        ax[0].get_xaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False)

        if col == 1:
            # right column has invisible y-axes.
            ax[0].get_yaxis().set_visible(False)
            ax[1].get_yaxis().set_visible(False)
            ax[2].get_yaxis().set_visible(False)

            # right column shows the legend.
            ax[0].legend(loc='best')
            ax[1].legend(loc='best')
            ax[2].legend(loc='best')

    #pl.legend(loc='best', fontsize=8)
    ax[0].set_title(rename_grammar(GRAMMAR))

    pl.tight_layout()


def main():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--grammar', choices=('both', 'medium', 'big'))
    p.add_argument('--rollout', choices=('CP', 'DP'))

    args = p.parse_args()

    CP = ('evalb_avg', 'pops')
    DP = ('expected_recall_avg', 'mask')

    GRAMMARS = ['medium', 'big'] if args.grammar == 'both' else [args.grammar]
    ACC, RUN = DP if args.rollout == 'DP' else CP
    pl.ion()

    fig1, ax1 = pl.subplots(nrows=3, #sharex=True,
                            ncols=2, figsize=(10,10))

    for i in range(3):
        for j in range(2):
            ax1[i,j].grid(False)

    fig2, ax2 = pl.subplots(nrows=1, #sharex=True,
                            ncols=2, figsize=(10,5))

    for i, GRAMMAR in enumerate(GRAMMARS):
        plot(GRAMMAR, ACC, RUN, ax=ax1[:,i], col=i)
        plot2(GRAMMAR, ACC, RUN, ax=ax2[i], col=i)

    fig1.tight_layout()
    fig2.tight_layout()

    pl.ioff()
    pl.show()


if __name__ == '__main__':
    main()
