"""Locally optimal learning to search."""

from __future__ import division

# Note: This import has to happen before pylab.
from ldp.prune.example import Reward, AvgReward, Setup, cgw_prf, cgw_f

import numpy as np
import pylab as pl
import cPickle
import random
from path import path
from time import time
from datetime import datetime
from pandas import DataFrame

from numpy.random import choice

from ldp.dp.risk import InsideOut
from ldp.cp.viterbi import DynamicParser
from ldp.cp.boolean import DynamicParser as BoolCP
from ldp.parse.leftchild import pruned_parser
from ldp.lols.classifier import SVM, GLM, Perceptron, Adagrad

from arsenal import ddict, colors, timeit, iterview, viz
from arsenal.fsutils import atomicwrite


class ROLLOUT:
    OPTS = [BODENSTAB_MLE, BODENSTAB_GOLD, CP, DP, BF, HY] = \
           'BODENSTAB_MLE, BODENSTAB_GOLD, CP, DP, BF, HY'.split(', ')

class INIT:
    OPTS = [BODENSTAB_MLE, BODENSTAB_GOLD, NONE] = \
           'BODENSTAB_MLE, BODENSTAB_GOLD, NONE'.split(', ')

class CLASSIFIER:
    OPTS = [SVM, LOGISTIC, PERCEPTRON, ADAGRAD, LINEAR, ADAGRAD_LINEAR, ADAGRAD_HINGE] = \
           'SVM, LOGISTIC, PERCEPTRON, ADAGRAD, LINEAR, ADAGRAD_LINEAR, ADAGRAD_HINGE'.split(', ')

class RUN:
    OPTS = [POPS, PUSHES, WALLCLOCK, MASK] = \
           'pops, pushes, wallclock, mask'.split(', ')

# accuracy options {NOFAIL, POSACC} currently disabled.
class ACC:
    OPTS = [EVALB_avg, EXPECTED_RECALL_avg, EVALB_corpus, EXPECTED_RECALL_corpus] = \
           'evalb_avg, expected_recall_avg, evalb_corpus, expected_recall_corpus'.split(', ')


class Evaluate(object):
    """NOTE: This class will run some of the more expensive evaluation measures,
    such as, evalb.

    """

    def __init__(self, accuracy, runtime):
        self.ACCURACY = accuracy
        self.RUNTIME = runtime

    def parse(self, e, grammar, policy=None, mask=None, with_derivations=0):
        """Evaluate test-time parser on a single sentence.

        For logging purposes, computes most evaluation measures.

        """

        if policy is not None:
            m = e.mask
            for x in e.nodes:
                m[x] = policy(e, m, x, e.features[x])
        elif mask is not None:
            m = mask
        else:
            assert False

        b4 = time()
        state = pruned_parser(e.tokens, grammar, m)
        wallclock = time()-b4

        # other accuracy measures
        d = state.derivation
        coarse = grammar.coarse_derivation(d)
        nofail = e.nofail(coarse)

        C,G,W = e.evalb_unofficial(coarse)
        #C_b, G_b, W_b = e.recall(coarse)
        W_b = len(e.gold_items)

        posacc = e.posacc(coarse)
        expected_C =  W_b * InsideOut(e, grammar, m*1.0, with_gradient=0).val

        #if self.ACCURACY == ACC.NOFAIL:
        #    accuracy = nofail
        #elif self.ACCURACY == ACC.POSACC:
        #    accuracy = posacc
        #else:
        accuracy = np.nan

        # other runtime measures
        pops = state.pops
        pushes = state.pushes
        masksize = 1.0*sum(m[x] for x in e.nodes)

        if self.RUNTIME == RUN.POPS:
            runtime = pops
        elif self.RUNTIME == RUN.PUSHES:
            runtime = pushes
        elif self.RUNTIME == RUN.MASK:
            runtime = masksize
        else:
            raise ValueError('Evaluate does not support runtime=%s' % self.RUNTIME)

        extra = {}
        if with_derivations:
            # derivations
            extra['derivation'] = state.derivation
            extra['coarse'] = grammar.coarse_derivation(d)

        return Reward(accuracy = accuracy,
                      runtime = runtime,
                      # simple accuracy measures
                      nofail = nofail,
                      posacc = posacc,
                      # corpus-level measures
                      ## - expected binarize recall
                      expected_C = expected_C,
                      ## - evalb
                      C=C, W=W, G=G,
                      ## - binarized evalb
                      W_b=W_b,
                      # runtime measures
                      pops = pops,
                      pushes = pushes,
                      mask = masksize,
                      wallclock = wallclock, **extra)

    def __call__(self, policy, examples, grammar, msg='eval'):
        "Evaluate test-time pruning policy ``c`` on ``examples``."
        rs = []
        for e in iterview(examples, msg=msg):
            rs.append(self.parse(e, grammar, policy))
        a = AvgReward(rs)
        if hasattr(a, self.ACCURACY):
            a.accuracy = getattr(a, self.ACCURACY)
        else:
            a.accuracy = a.attrs[self.ACCURACY]
        a.attrs['accuracy'] = a.accuracy
        return a


class DP(object):

    def __init__(self, grammar, example, policy, accuracy, runtime, tradeoff):
        self.grammar = grammar
        self.example = example
        m = example.mask
        m[:,:] = 1
        for x in example.nodes:
            m[x] = policy(example, m, x, example.features[x])
        self.m = m
        self.ACCURACY = accuracy
        self.RUNTIME = runtime
        self.tradeoff = tradeoff
        self.io = InsideOut(example, grammar, m*1.0, with_gradient=True)

    def roll_outs(self, tmp):
        e = self.example
        want = len(e.gold_items)

        r0 = Tmp(C=self.io.val * want, G=np.nan, W=want,
                 runtime=sum(self.m[x] for x in self.example.nodes))

        for (I,K) in e.nodes:
            a = self.m[I,K]

            C = self.io.est[I,K] * want

            if self.ACCURACY not in (ACC.EXPECTED_RECALL_avg, ACC.EXPECTED_RECALL_corpus):
                raise ValueError('DP does not support accuracy=%s' % self.ACCURACY)

            if self.RUNTIME == RUN.MASK:
                if self.m[I,K]:  # prune action
                    runtime = r0.runtime - 1
                else:            # unprune action
                    runtime = r0.runtime + 1
            else:
                raise ValueError('DP does not support runtime=%s' % self.RUNTIME)

            r = Tmp(C=C, G=np.nan, W=want, runtime=runtime)

            w = 1.0
            tmp.append([e, (I,K), w, (a, r0), (1-a, r)])



class HY(object):
    "Hybrid DP-risk + CP-pops rollouts."

    def __init__(self, grammar, example, policy, accuracy, runtime, tradeoff):
        self.grammar = grammar
        self.example = example
        m = example.mask
        m[:,:] = 1
        for x in example.nodes:
            m[x] = policy(example, m, x, example.features[x])
        self.m = m
        self.ACCURACY = accuracy
        self.RUNTIME = runtime
        self.tradeoff = tradeoff
        # DP-risk
        self.io = InsideOut(example, grammar, m*1.0, with_gradient=True)
        # CP-bool
        self.cp = BoolCP(example.tokens, grammar, m.copy())
        self.cp.run()

        if self.ACCURACY not in (ACC.EXPECTED_RECALL_avg, ACC.EXPECTED_RECALL_corpus):
            raise ValueError('DP does not support accuracy=%s' % self.ACCURACY)
        assert self.RUNTIME == 'pops'

    def _reward(self):
        assert self.RUNTIME == RUN.POPS
        return

    def roll_outs(self, tmp):
        m = self.m
        e = self.example

        e = self.example
        want = len(e.gold_items)

        r0 = Tmp(C=self.io.val * want, G=np.nan, W=want,
                 runtime=self.cp._pops)

        states = self.example.nodes
        np.random.shuffle(states)
        T = len(states)
        S = min(e.N*2, T)     # 2*N rollouts for example of length N.
        states = states[:S]

        w = T / S   # importance weight for not doing all of the rollouts
        for (I,K) in states:
            self.cp.start_undo()
            a = m[I,K]
            self.cp.change(I, K, 1-a)
            r = Tmp(C=self.io.est[I,K] * want,
                    G=np.nan,
                    W=want,
                    runtime=self.cp._pops)
            tmp.append([e, (I,K), w, (a, r0), (1-a, r)])
            self.cp.rewind()



class CP(object):

    def __init__(self, grammar, example, policy, accuracy, runtime, tradeoff):
        self.example = example
        m = example.mask
        for x in example.nodes:
            m[x] = policy(example, m, x, example.features[x])
        self.p = DynamicParser(example.tokens, grammar, m.copy())
        self.p.run()
        self.m = m
        self.grammar = grammar
        self.ACCURACY = accuracy
        self.RUNTIME = runtime
        self.tradeoff = tradeoff

    def _reward(self):
        e = self.example
        p = self.p
        d = p.derivation()
        coarse = self.grammar.coarse_derivation(d)

        if self.RUNTIME == RUN.POPS:
            runtime = p._pops
        elif self.RUNTIME == RUN.MASK:
            runtime = 1.0*sum(p.keep[x] for x in e.nodes)
        else:
            raise ValueError('CP does not support runtime=%s' % self.RUNTIME)

        C,G,W = e.evalb_unofficial(coarse)
        return Tmp(C,G,W,runtime)

    def roll_outs(self, tmp):
        m = self.m
        e = self.example
        p = self.p
        r0 = self._reward()

        DEBUG = 0
        if DEBUG:
            #from ldp.prune.example import oneline
            oneline = lambda x: x
            from ldp.parsing.util import unbinarize
            orig_coarse = unbinarize(self.grammar.coarse_derivation(self.p.derivation()))

        states = self.example.nodes
        np.random.shuffle(states)
        T = len(states)
        S = min(e.N*2, T)     # 2*N rollouts for example of length N.
        states = states[:S]

        w = T / S   # importance weight for not doing all of the rollouts
        for (I,K) in states:
            p.start_undo()
            a = m[I,K]
            p.change(I, K, 1-a)
            r = self._reward()

            if DEBUG:
                if (I,K) in e.gold_spans:
                    if (   (a == 1 and r0.f1() < r.f1())   # was keep and acc was lower
                        or (a == 0 and r0.f1() > r.f1())): # was prune and acc was higher
                        print
                        print
                        print 'gold span (%s,%s)' % (I,K)
                        print
                        print 'keep' if a else 'prune', r0.f1()
                        print oneline(orig_coarse)
                        print
                        print 'keep' if 1-a else 'prune', r.f1()
                        print oneline(unbinarize(self.grammar.coarse_derivation(self.p.derivation())))

            tmp.append([e, (I,K), w, (a, r0), (1-a, r)])
            p.rewind()


class BF(object):

    def __init__(self, grammar, example, policy, accuracy, runtime, tradeoff):
        self.example = example
        m = example.mask
        for x in example.nodes:
            m[x] = policy(example, m, x, example.features[x])
        self.p = DynamicParser(example.tokens, grammar, m.copy())
        self.p.run()
        self.m = m
        self.grammar = grammar
        self.ACCURACY = accuracy
        self.RUNTIME = runtime
        self.tradeoff = tradeoff

    def reward(self, m):
        s = pruned_parser(self.example.tokens, self.grammar, m)
        coarse = self.grammar.coarse_derivation(s.derivation)

        if self.RUNTIME == RUN.POPS:
            runtime = s.pops
        elif self.RUNTIME == RUN.PUSHES:
            runtime = s.pushes
        elif self.RUNTIME == RUN.MASK:
            runtime = 1.0*sum(m[x] for x in self.example.nodes)
        else:
            raise ValueError('BF does not support runtime=%s' % self.RUNTIME)

        C,G,W = self.example.evalb_unofficial(coarse)
        return Tmp(C,G,W,runtime)

    def roll_outs(self, tmp):
        m = self.m
        e = self.example

        r0 = self.reward(m)

        states = self.example.nodes
        np.random.shuffle(states)
        T = len(states)
        S = min(e.N*2, T)     # 2*N rollouts for example of length N.
        states = states[:S]

        w = T / S   # importance weight for not doing all of the rollouts
        for (I,K) in states:
            a = m[I,K]
            m[I,K] = 1-a
            r = self.reward(m)
            m[I,K] = a
            tmp.append([e, (I,K), w, (a, r0), (1-a, r)])


class Tmp(object):
    def __init__(self, C, G, W, runtime):
        self.C = C
        self.G = G
        self.W = W
        self.runtime = runtime

    def f1(self):
        return cgw_f(self.C, self.G, self.W)

    def recall(self):
        return cgw_prf(self.C, self.G, self.W)[1]


class BodenstabParser(object):

    def __init__(self, grammar, example, target, tradeoff):
        assert target in {ROLLOUT.BODENSTAB_MLE, ROLLOUT.BODENSTAB_GOLD}
        self.example = example
        self.grammar = grammar
        self.target = target
        self.tradeoff = tradeoff

    def roll_outs(self):
        e = self.example
        if self.target == ROLLOUT.BODENSTAB_MLE:
            target = e.mle_spans
        elif self.target == ROLLOUT.BODENSTAB_GOLD:
            target = e.gold_spans
        else:
            assert False, 'unrecognized target %s' % (self.target,)

        for (I,K) in e.nodes:
            if (I,K) in target:
                # importance weight is |(0-lambda*0) - (1-lambda*0)| = 1
                e.Q[I,K,0] += Reward(accuracy=0, runtime=0)(self.tradeoff)
                e.Q[I,K,1] += Reward(accuracy=1, runtime=0)(self.tradeoff)
            else:
                # importance weight is |(1-lambda*0) - (1-lambda*1)| = lambda
                e.Q[I,K,0] += Reward(accuracy=1, runtime=0)(self.tradeoff)
                e.Q[I,K,1] += Reward(accuracy=1, runtime=1)(self.tradeoff)


#_______________________________________________________________________________
#

class learn(object):

    def __init__(self, args, setup, tradeoff, iterations, minibatch,
                 results=None, C=None, roll_out=ROLLOUT.CP,
                 initializer=INIT.BODENSTAB_GOLD, initializer_penalty=0.01,
                 show_reference=0):

        self.evals = []
        self.tradeoff = tradeoff
        self.grammar = grammar = setup.grammar
        self.nfeatures = nfeatures = setup.nfeatures

        self.ACCURACY = args.accuracy
        self.RUNTIME = args.runtime

        self.C = C
        if args.classifier == CLASSIFIER.LOGISTIC:
            self.policy = GLM(nfeatures, C=self.C, loss=0)
        elif args.classifier == CLASSIFIER.LINEAR:
            self.policy = GLM(nfeatures, C=self.C, loss=1)
        elif args.classifier == CLASSIFIER.ADAGRAD:
            self.policy = Adagrad(self.nfeatures, C=self.C, loss=0, eta=args.learning_rate)
        elif args.classifier == CLASSIFIER.ADAGRAD_LINEAR:
            self.policy = Adagrad(self.nfeatures, C=self.C, loss=1, eta=args.learning_rate)
        elif args.classifier == CLASSIFIER.ADAGRAD_HINGE:
            self.policy = Adagrad(self.nfeatures, C=self.C, loss=2, eta=args.learning_rate)
        elif args.classifier == CLASSIFIER.SVM:
            self.policy = SVM(self.nfeatures, C=self.C)
        elif args.classifier == CLASSIFIER.PERCEPTRON:
            self.policy = Perceptron(self.nfeatures)
        else:
            raise AssertionError('Unrecognized classifier option %r' % args.classifier)

        if args.init_weights is not None:
            # XXX: Hack to warm start weights
            print '[init weights]', args.init_weights
            assert args.init_weights.exists()
            self.policy._coef = np.load(args.init_weights)['coef']

        self.evaluate = Evaluate(args.accuracy, args.runtime)

        sty = {
            'oracle1': dict(c='k', alpha=0.5, linestyle=':'),
            'fastmle': dict(c='g', alpha=0.5, linestyle=':'),
            'unpruned': dict(c='k', alpha=0.5, linestyle='--'),
            'new_policy': dict(c='b', lw=2),
        }
        self.lc = ddict(lambda name: viz.LearningCurve(name, sty=sty))

        train = list(setup.train)
        random.shuffle(train)
        dev = list(setup.dev)
        dataset = [('train', train)]
        if dev:
            dataset.append(('dev', dev))
        self.train = train
        self.dev = dev
        self.results = results

        # Do we need to run the unpruned parser? (This can be very slow, so we
        # should only do it when necessary.)
        if (show_reference
            or roll_out == ROLLOUT.BODENSTAB_MLE
            or initializer == INIT.BODENSTAB_MLE):

            from ldp.parsing.util import item_tree, item_tree_get_items
            for e in iterview(train+dev, msg='unpruned'):
                # unpruned
                r = self.evaluate.parse(e, grammar, mask=e.mask, with_derivations=1)
                e.mle_spans = frozenset({(I,K) for (_,I,K) in item_tree_get_items(item_tree(r.coarse)) if K-I > 1 and K-I != e.N})
                del r.coarse, r.derivation   # delete to save memory
                e.baseline = r
        # Do we need to run the oracle parser?
        if show_reference:
            for e in iterview(train+dev, msg='oracle'):
                # oracle
                m = e.mask
                for x in e.nodes:
                    m[x] = (x in e.gold_spans)
                r = self.evaluate.parse(e, grammar, mask=m, with_derivations=1)
                del r.coarse, r.derivation   # delete to save memory
                e.oracle = r
            # plot/log baselines
            self.baselines()

        # ----------------------------------
        # Baseline --
        # Assumes no dynamic features.
        if roll_out in {ROLLOUT.BODENSTAB_GOLD, ROLLOUT.BODENSTAB_MLE}:
            for e in iterview(train, msg='rollouts'):
                p = BodenstabParser(grammar, e, target=roll_out, tradeoff=tradeoff)
                p.roll_outs()
            print colors.yellow % 'Training...'
            with timeit('train'):
                self.policy.train(train)
            ps = dict(new_policy = self.policy)
            x = dict(iteration=1, tradeoff=tradeoff)
            self.iteration = 1
            self.performance(dataset, ps, x)
            return
        # ----------------------------------
        if roll_out == ROLLOUT.CP:
            Rollouts = CP
        elif roll_out == ROLLOUT.BF:
            Rollouts = BF
        elif roll_out == ROLLOUT.HY:
            Rollouts = HY
        elif roll_out == ROLLOUT.DP:
            Rollouts = DP
        else:
            raise ValueError('Unrecognized rollout option %s' % roll_out)
        for iteration in xrange(1, iterations+1):
            self.iteration = iteration
            print
            print colors.green % 'Iter %s' % iteration
            if iteration == 1 and initializer in [ROLLOUT.BODENSTAB_MLE, ROLLOUT.BODENSTAB_GOLD]:
                # first iteration uses asymmetric classification to initialize
                for e in iterview(train, msg='rollouts'):
                    p = BodenstabParser(grammar, e, target=initializer, tradeoff=initializer_penalty)
                    p.roll_outs()
            else:
                M = choice(train, min(len(train), minibatch), replace=0)

                tmp = []
                for e in iterview(M, msg='rollouts'):
                    p = Rollouts(grammar, e, self.policy,
                                 accuracy=args.accuracy,
                                 runtime=args.runtime,
                                 tradeoff=tradeoff)
                    p.roll_outs(tmp)

                # corpus-level accuracy
                if args.accuracy in (ACC.EVALB_corpus, ACC.EXPECTED_RECALL_corpus):
                    self.postprocess_rollouts_corpus(M, tmp)

                else:

                    # Compare baseline labels to LOLS's "labels" via rollouts.
                    self.asym_v_lols(tmp)

                    # propagate back to CSC datasets
                    for [e, (I,K), w, (action0, r0), (action1, r1)] in tmp:

                        if args.accuracy == ACC.EVALB_avg:
                            acc1 = r1.f1()
                            acc0 = r0.f1()
                        elif args.accuracy == ACC.EXPECTED_RECALL_avg:
                            acc1 = r1.recall()
                            acc0 = r0.recall()
                        else:
                            acc1 = r1.accuracy
                            acc0 = r0.accuracy

                        e.Q[I,K,action1] += w * (acc1 - tradeoff * r1.runtime)
                        e.Q[I,K,action0] += w * (acc0 - tradeoff * r0.runtime)

            print colors.yellow % 'Training...'
            with timeit('train'):
                self.policy.train(train)
            # Specify additional policies to evaluate on training data.
            ps = dict(new_policy = self.policy)
            # metadata to log
            x = dict(iteration = iteration, tradeoff = tradeoff)
            self.performance(dataset, ps, x)

    def postprocess_rollouts_corpus(self, M, tmp):
        """Special handling for corpus-level rollouts (e.g., Corpus-F1)

        This is yucky because rewards are no longer an average over
        sentences. Handling these measures requires a little trick for
        subtracting the old and add the new contribution of the sentence to the
        reward.

        """

        # compute counts for f-measure on roll-in
        C0 = 0
        G0 = 0
        W0 = 0
        run0 = 0.0
        check = set()
        for [e, _, _, (_, r0), _] in tmp:
            if e not in check:
                C0 += r0.C
                G0 += r0.G
                W0 += r0.W
                run0 += r0.runtime
                check.add(e)

        # compute initial precision, recall, f-measure *for the whole minibatch.*
        _,R0,F0 = cgw_prf(C0,G0,W0)

        # propagate back to CSC datasets
        for [e, (I,K), w, (action0, r0), (action1, r1)] in tmp:

            # remove old counts for this sentence and add new counts
            _,R1,F1 = cgw_prf((C0-r0.C)+r1.C,
                              (G0-r0.G)+r1.G,
                              (W0-r0.W)+r1.W)

            if self.ACCURACY == ACC.EVALB_corpus:
                acc1 = F1
                acc0 = F0
            elif self.ACCURACY == ACC.EXPECTED_RECALL_corpus:
                acc1 = R1
                acc0 = R0
            else:
                assert False

            run1 = ((run0 - r0.runtime) + r1.runtime)

            e.Q[I,K,action1] += w * (acc1 - self.tradeoff * run1/len(M))
            e.Q[I,K,action0] += w * (acc0 - self.tradeoff * run0/len(M))

    def baselines(self):
        self._baselines(self.lc['train'], self.train)
        if self.dev:
            self._baselines(self.lc['dev'], self.dev)
        # log reference policies
        if self.results is not None:
            flat = {}
            for name in self.lc:
                lcurve = self.lc[name]
                for policy, r in lcurve.baselines_reward.items():
                    for attr, val in r.attrs.items():
                        flat['%s_%s_%s' % (name, policy, attr)] = val
            DataFrame([flat]).to_csv(self.results / 'baseline.csv')

    def _baselines(self, lc, examples):
        if 'oracle1' not in lc.baselines:
            r = AvgReward([e.oracle for e in examples])
            lc.baselines_reward['oracle1'] = r
            lc.baselines['oracle1'] = r(self.tradeoff)
        if 'unpruned' not in lc.baselines:
            r = AvgReward([e.baseline for e in examples])
            lc.baselines_reward['unpruned'] = r
            lc.baselines['unpruned'] = r(self.tradeoff)
        if 'fastmle' not in lc.baselines:
            r = AvgReward([Reward(e.baseline.accuracy, 0, wallclock=0) for e in examples])
            lc.baselines_reward['fastmle'] = r
            lc.baselines['fastmle'] = r(self.tradeoff)

    def performance(self, dataset, ps, x):
        tradeoff = self.tradeoff
        # policy evaluation loops: here we evaluate a few different policies.
        for d, data in dataset:
            if d == 'train' and len(data) > 1000:
                # use a fixed random sample to evaluate learning training set.
                data = data[:1000]
            for name, policy in ps.items():
                # run evaluation
                r = self.evaluate(policy, data, self.grammar, msg='eval (%s/%s)' % (d, name))
                # Is this the current best?
                prev = self.lc[d].data[name]
                newmax = r(tradeoff) > max(s for [_,s] in prev) if prev else True
                # update learning curve
                self.lc[d].update(self.iteration, **{name: r(tradeoff)})
                # print message to terminal
                print colors.magenta % '%s/%s' % (d, name)
                print colors.yellow % 'reward:   ', '%.5f' % r(tradeoff), \
                    (colors.cyan % '*best so far*') if newmax else ''
                print colors.yellow % 'accuracy: ', '%.5f' % r.accuracy
                print colors.yellow % 'runtime:  ', '%.5f' % r.runtime
                # log reward, accuracy and runtime
                x['%s_%s_reward' % (d, name)] = r(tradeoff)
                x['%s_%s_accuracy' % (d, name)] = r.accuracy
                x['%s_%s_runtime' % (d, name)] = r.runtime
                # log all other attributes
                for k, v in r.attrs.items():
                    x['%s_%s_%s' % (d, name, k)] = v
                # log current time
                x['datetime'] = datetime.now()
        self.evals.append(x)
        if self.results is not None:
            # print a big message to terminal
            for k in sorted(x):
                print '%35s: %s' % (k, x[k])
            # write to log file
            df = DataFrame(self.evals)
            with atomicwrite(self.results / 'log.csv') as f:
                df.to_csv(f)
            # save model
            for name in ['new_policy']:
                if name not in ps:
                    continue
                with timeit('save model %r' % name):
                    ps[name].save(self.results / '%s-%03d' % (name, self.iteration))


    def asym_v_lols(self, tmp):
        """
        How locally optimal is the current policy?

        Inspect rollout information.
        This version takes only this iteration's rollouts.
        """

        if self.results is None:
            return

        data = []
        for [e, (I,K), w, (a0, r0), (a1, r1)] in tmp:

            action = a0
            if a0 == 1:  # keep
                r0, r1 = r1, r0   # swap
            del a0, a1, w

            # Note: We're ignoring the importance weight for where we chose to
            # rollout.

            if self.ACCURACY == ACC.EVALB_avg:
                _,_,acc1 = cgw_prf(r1.C, r1.G, r1.W)
                _,_,acc0 = cgw_prf(r0.C, r0.G, r0.W)
            elif self.ACCURACY == ACC.EXPECTED_RECALL_avg:
                _,acc1,_ = cgw_prf(r1.C, r1.G, r1.W)
                _,acc0,_ = cgw_prf(r0.C, r0.G, r0.W)
            else:
                acc1 = r1.accuracy
                acc0 = r0.accuracy

            rew0 = acc0 - self.tradeoff*r0.runtime
            rew1 = acc1 - self.tradeoff*r1.runtime

            assert r0.runtime <= r1.runtime, [r0.runtime, r1.runtime]

            data.append({
                'example': e.name,
                'span_begin': I,
                'span_end': K,
                'policy': action,
                'gold': (I,K) in e.gold_spans,
                'delta_rew': rew1 - rew0,
                'delta_acc': acc1 - acc0,
                'delta_run': r1.runtime - r0.runtime,
            })

        df = DataFrame(data)

        df.to_csv(self.results / ('asym_v_lols_iteration_%s.csv' % self.iteration))


def main():
    from argparse import ArgumentParser
    from arsenal.profiling.utils import profiler

    p = ArgumentParser()
    # output
    p.add_argument('--results', default=None, type=path)
    # data
    p.add_argument('--grammar', choices=('medium', 'big'))
    p.add_argument('--maxlength', type=int, default=40)
    p.add_argument('--minlength', type=int, default=3)
    p.add_argument('--train', type=int, default=1)
    p.add_argument('--dev', type=int, default=0)
    # algorithm
    p.add_argument('--iterations', type=int, default=1000000)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--tradeoff', type=float, required=1)
    p.add_argument('--classifier', choices=CLASSIFIER.OPTS, required=1)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--accuracy', choices=ACC.OPTS, required=True)
    p.add_argument('--runtime', choices=RUN.OPTS, required=True)
    p.add_argument('--roll-out', choices=ROLLOUT.OPTS, required=1)
    p.add_argument('--initializer', choices=INIT.OPTS, required=1)
    p.add_argument('--initializer-penalty', type=float, default=0.01)

    p.add_argument('--init-weights', type=path)

    p.add_argument('--minibatch', type=int, required=1)
    p.add_argument('-C', type=float, default=0,
                   help='log-regularization constant or log-slack penalty.')
    # misc
    p.add_argument('--profiler', choices=('yep', 'cprofile'))
    p.add_argument('--show-reference', action='store_true')

    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.results is not None:
        # [2016-01-05 Tue] I've added this check so that when SGE reboots a job,
        # the restarted job will die instead of clobber existing results.
        assert not args.results.exists(), 'path already exists %s' % args.results
        args.results.makedirs_p()    # create if it doesn't exist
        with file(args.results / 'args.pkl', 'wb') as pkl:
            cPickle.dump(args, pkl)

    setup = Setup(grammar = args.grammar,
                  train = args.train,
                  dev = args.dev,
                  minlength = args.minlength,
                  maxlength = args.maxlength)

    with profiler(use=args.profiler):
        learn(args, setup,
              tradeoff=args.tradeoff,
              iterations=args.iterations,
              minibatch=args.minibatch,
              results=args.results,
              C=args.C,
              initializer=args.initializer,
              initializer_penalty=args.initializer_penalty,
              show_reference=args.show_reference,
              roll_out=args.roll_out)

    print
    print colors.green % 'DONE!'


if __name__ == '__main__':
    main()
