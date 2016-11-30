#!/usr/bin/env python
"""
Exploring the shape of the asym-runtime curve so that I can sample runtimes
evenly.
"""

from __future__ import division
import pandas
import pylab as pl
import numpy as np
from numpy import exp, log, polyfit
from scipy.special import logit, expit as sigmoid

#def logit(x):
#    return log(x/(1-x))

#def sigmoid(x):
#    "sigmoid function aka inverse logit, logistic function."
#    return 1./(1+exp(-x))

def plotF(f, xmin, xmax, pts=100, **kw):
    X = np.linspace(xmin, xmax, pts)
    pl.plot(X, [f(x) for x in X], c='r', alpha=0.5, lw=2, **kw)


def main():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('filename')
    args = p.parse_args()
    df = pandas.read_csv(args.filename)

    grammar = set(df.args_grammar)
    print 'grammar:', grammar

    #yl = 'dev_new_policy_pushes'
    yl = 'dev_new_policy_pops'
    #yl = 'dev_new_policy_mask'
    xl = 'args_tradeoff'

    # Transform parameters so that the target relationship is linear.

    # log(y) = a*logit(x) + b
    # y = exp(a*logit(x) + b)

    # Plot function in the original (i.e., nonlinear) space.
    Y = np.array(df[yl])
    X = np.array(df[xl])


    TY = np.log(Y)
    TX = logit(X)

    [m,b] = polyfit(TX,TY,deg=1)

    if 0:
        pl.figure()
        pl.plot(TX, TY, c='b', alpha=0.5, lw=2)
        pl.scatter(TX, TY, c='b', alpha=0.5, lw=2)
        pl.xlabel('logit %s' % xl)
        pl.ylabel('log %s' % yl)
        plotF(lambda x: m*x+b, TX.min(), TX.max())

    if 1:
        pl.figure()
        pl.plot(X, Y, c='b', alpha=0.5, lw=2)
        pl.scatter(X, Y, c='b', alpha=0.5, lw=2)
        pl.xlabel(xl)
        pl.ylabel(yl)
        plotF(lambda x: exp(m*logit(x)+b), X.min(), X.max())

    # Examine the inverse:
    #
    # Inverse: solve for x given y.
    #   a*logit(x) + b = log(y)
    #   logit(x) = (log(y)-b)/a
    #   x = sigmoid((log(y)-b)/a)

    if 0:
        # Swap
        pl.figure()
        pl.plot(Y, X, c='b', alpha=0.5, lw=2)
        pl.scatter(Y, X, c='b', alpha=0.5, lw=2)
        pl.xlabel(yl)
        pl.ylabel(xl)
        pl.title('inverse')
        plotF(lambda y: sigmoid((log(y)-b)/m), Y.min(), Y.max())

    # So, if I want K evenly space runtimes, what lambdas should I use?
    yy = np.linspace(Y.min(), Y.max(), 12)
    xx = sigmoid((log(yy)-b)/m)

    print 'tradeoff = sigmoid((log(runtime) - %g) / %g)' % (b, m)

    # massage function into something simpler.
    # sigmoid((log(yy)-b)/m)
    #  = sigmoid((log((yy/exp(b))**(1/m))))
    #  = (1 + exp(-(log((yy/exp(b))**(1/m)))))**-1
    #  = (1 + (yy/exp(b))**(-1/m))**-1
    #  = (1 + yy**(-1/m) * exp(b/m))**-1
    #  = (1 + yy**a * c)**-1

    c, a = exp(b/m), -1/m
    xxx = (1 + c * yy ** a)**-1
    print 'tradeoff = (1 + %g * runtime ** %g)**-1' % (c, a)
    assert np.abs(xx - xxx).max() < 1e-8

    print 'runtimes: ', yy
    print 'tradeoffs:', xx

    pl.figure()
    pl.scatter(yy, xx)
    plotF(lambda y: (1 + c * y ** a)**-1, yy.min(), yy.max())

    pl.title('which parameters to use for evenly spaced runtime...')
    pl.ylabel('tradeoff')
    pl.xlabel('runtime')

    pl.show()


if __name__ == '__main__':
    main()
