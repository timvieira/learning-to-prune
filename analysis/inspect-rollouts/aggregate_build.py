"""This script munches on the inspect_rollouts logs to creat the
`inspect-rollouts-{GRAMMAR}-{ACC}-{RUN}-{POLICY}.csv` files that the
aggregate_plot.py script takes as input.

"""
import cPickle
import numpy as np
import pandas as pd
from path import path
from arsenal import colors, iterview, wide_dataframe

wide_dataframe()


def breakdown(data, args, df):
    # the lols one-step deviation oracle
    df['lols'] = (df.delta_rew > 0)
    df['Diff'] = (df.lols != df.gold) & (df.delta_rew != 0)

    frac_tie = np.mean(df.delta_rew == 0)

    G = df[df.gold]           # data |  gold
    N = df[~df.gold]          # data | ~gold
    D = df[df.Diff]           # data | diff
    GD = D[D.gold]            # data | diff, gold
    ND = D[~D.gold]           # data | diff, ~gold

    p_G = np.mean(df.gold)    # p( gold)
    p_N = 1-p_G               # p(~gold)

    p_D   = np.mean(df.Diff)  # p(diff)
    p_D_G = np.mean(G.Diff)   # p(diff| gold)
    p_D_N = np.mean(N.Diff)   # p(diff|~gold)

    p_G_D = np.mean(D.gold)   # p( gold | diff)
    p_N_D = 1-p_G_D           # p(~gold | diff)

    regret_GD = -np.mean(GD.delta_rew)   # E[R| G,D]
    regret_ND = np.mean(ND.delta_rew)    # E[R|~G,D]

    # E[R|D] = E[R|D,G]*p(G|D) + E[R|D,~G]*p(~G|D)
    regret_D = regret_GD * p_G_D + regret_ND * p_N_D

    regret = p_D * regret_D       # E[regret]    = p(D) * E[R|D]
    regret_G = regret_GD * p_D_G  # E[regret|G]  = E[R| G,D] * p(D| G)
    regret_N = regret_ND * p_D_N  # E[regret|~G] = E[R|~G,D] * p(D|~G)

    diff_g = G[G.Diff]
    diff_g_acc_incr = np.mean(-diff_g.delta_acc > 0)    # pruning gold, increases accuracy if -delta_acc > 0
    diff_g_acc_same = np.mean(diff_g.delta_acc == 0)
    diff_g_acc_decr = np.mean(-diff_g.delta_acc < 0)

    data.append(dict(
        accuracy = args.accuracy,
        runtime = args.runtime,
        tradeoff = args.tradeoff,
        grammar = args.grammar,
        g_size = len(G),
        n_size = len(N),

        g_decr = np.mean(G.delta_rew < 0),
        n_decr = np.mean(N.delta_rew < 0),
        g_same = np.mean(G.delta_rew == 0),
        n_same = np.mean(N.delta_rew == 0),
        g_incr = np.mean(G.delta_rew > 0),
        n_incr = np.mean(N.delta_rew > 0),

        # accuracy of policy on gold and nongold prediction
        P_acc_g = np.mean(G.policy == G.gold),
        P_acc_n = np.mean(N.policy == N.gold),

        # accuracy of policy on classification by rollouts
        #P_roll = np.mean(df.policy == (df.delta_rew > 0)),
        #P_matter = np.mean(matter.policy == (matter.delta_rew > 0)),

        # What do we generally do when it doesn't matter?
        # (look at gold and nongold prediction accuracy)
        #P_doesnt = np.mean(doesnt.policy),
        #P_doesnt_acc_g = np.mean(doesnt[doesnt.gold].policy),
        #P_doesnt_acc_n = 1-np.mean(doesnt[~doesnt.gold].policy),

        frac_tie = frac_tie,

        # Reward/regret of different (LOLS oracle and baseline oracle)
        p_G = p_G,
        p_N = p_N,
        p_D = p_D,
        p_D_G = p_D_G,
        p_D_N = p_D_N,
        p_G_D = p_G_D,
        p_N_D = p_N_D,
        regret_GD = regret_GD,
        regret_ND = regret_ND,
        regret_D = regret_D,
        regret_G = regret_G,
        regret_N = regret_N,
        regret = regret,

        diff_g_acc_incr = diff_g_acc_incr,
        diff_g_acc_same = diff_g_acc_same,
        diff_g_acc_decr = diff_g_acc_decr,

        frac_prune_gold_improves_acc = np.mean(-G.delta_acc > 0),
        frac_prune_gold_improves_rew = np.mean(-G.delta_rew > 0),

    ))


def build(GRAMMAR, ACC, RUN, INIT):

    data = []

    if INIT == 'policy':
        csvs = path('results').glob('*/dump/new_policy*.npz.inspect_rollouts.csv.gz')
    elif INIT == 'init':
        csvs = path('results').glob('*/dump/init.npz.inspect_rollouts.csv.gz')
    else:
        raise ValueError('dont understand INIT=%s' % INIT)

    for f in iterview(csvs):
        args = cPickle.load(file((f / '..' / 'args.pkl').abspath()))
        if args.grammar != GRAMMAR or args.accuracy != ACC or args.runtime != RUN:
            continue
        try:
            breakdown(data,
                      args,
                      pd.read_csv(f))
        except pd.io.common.EmptyDataError:
            print colors.red % '*** skipping empty file %s' % f
            print

    df = pd.DataFrame(data)
    df = df.sort_values('tradeoff')
    print df
    return df


def main():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--test', action='store_true')
    p.add_argument('--rollouts', choices=['DP','CP','both'], required=True)

    args = p.parse_args()

    CP = ('evalb_avg', 'pops')
    DP = ('expected_recall_avg', 'mask')

    if args.rollouts == 'both':
        RO = [DP, CP]
    elif args.rollouts == 'CP':
        RO = [CP]
    elif args.rollouts == 'DP':
        RO = [DP]

    for GRAMMAR in ['medium', 'big']:
        for ACC, RUN in RO:
            for INIT in ['policy', 'init']:

                print colors.yellow % '%s' % ((GRAMMAR, ACC, RUN, INIT),)

                filename = ('tmp/inspect-rollouts-%s-%s-%s-%s.csv' % (GRAMMAR, ACC, RUN, INIT)
                            if not args.test else None)

                if filename is not None and path(filename).exists():
                    print 'cached', filename
                    print pd.read_csv(filename)
                    continue

                df = build(GRAMMAR, ACC, RUN, INIT)

                if filename is not None:
                    df.to_csv(filename)
                    print 'wrote', filename


if __name__ == '__main__':
    main()
