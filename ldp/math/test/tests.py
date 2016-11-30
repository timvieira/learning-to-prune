import numpy as np

from ldp.math.util import sigmoid, dsigmoid, logadd
from ldp.lols.classifier import glm
from arsenal.math import spherical
from numpy.random import choice
from ldp.prune.example import Setup, Reward
from arsenal.math.checkgrad import check_gradient


def test_dsigmoid():
    for x in np.arange(-10, 10, 0.01):
        check_gradient(sigmoid, [dsigmoid(x)], np.array([x]), verbose=False, progress=False)
    print '[dsigmoid test] pass'


def test_glm_gradient():

    s = Setup(maxlength=10, train=1, dev=0)

    for e in s.train:
        for (I,K) in e.nodes:
            gold = (I,K) in e.gold_spans
            e.Q[I,K,0] = 0.1*gold
            e.Q[I,K,1] = gold

    # too many dimensions to test all of them, let's pick an interesting subset.
    active_features = set(k for e in s.train for x in e.nodes for k in e.features[x].get_keys())

    # add a few inactive features.
    active_features.update(choice(range(s.nfeatures),
                                  min(30, s.nfeatures), replace=0))

    print len(active_features), 'active features'

    regularizer = 0.1

    for name, loss in [('logistic',0), ('linear',1)]:
        print
        print '# testing %s' % name

        def f(x):
            return glm(x, s.train, regularizer, with_gradient=0, loss=loss)

        for _ in range(5):
            x0 = spherical(s.nfeatures)
            _, g = glm(x0, s.train, regularizer, with_gradient=1, loss=loss)
            check_gradient(f, g, x0, random_subset=active_features, verbose=True, progress=0)


if __name__ == '__main__':
    import numpy.random
    import random
    random.seed(0)
    numpy.random.seed(0)

    test_dsigmoid()
    test_glm_gradient()

    from ldp.math import adagrad
    adagrad.test()
