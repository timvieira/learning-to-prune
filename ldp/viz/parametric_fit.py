from __future__ import division
import pandas
import pylab as pl
import numpy as np
import seaborn as sns
from scipy.optimize import basinhopping
from numpy import exp
from scipy.special import expit  # expit=sigmoid
from ldp.viz.cone import lambda_cone, arrow


pl.rc('text', usetex=True)


def F(w, x):
    """New model is a

    c = b + a*log(x - x0)
    y = ymax*sigmoid(-c)

    which says that if we shift x by x0, then we have a linear relationshipt
    between shifted-x and y in log-logit space,

    log(x - x0) = a*log((y-ymin)/(ymax-y)) + b,

    for some x0, a, b, ymin, ymax. We hardcode ymin = 0.

    There is a similar story for the inverse function, which is what we actually
    implement. (Described above and simplified.)

    """
    x0, ymax, a, b = w
    x0 = exp(x0)                              # x0 is positive
    return ymax*expit(b + a*np.log(x + x0))


def F_inverse(w, x):
    x0, ymax, a, b = w
    x0 = exp(x0)
    return (exp(-b) * (-1 + ymax/x))**(1/a) - x0


def dFdx(w, x):
    x0, ymax, a, b = w
    x0 = exp(x0)
    s = expit(b + a*np.log(x + x0))
    return s*(1-s)*a*ymax/(x+x0)


def fit(X, Y):
    """
    Fit a parametric curve to Pareto frontier.
    """

    if 0:
        # plot random cross sections for fit objective, which is nonconvex.
        # the cross sections helped me determine the positivity constraints
        # and that L2 regression is better than L1 regression.
        from arsenal.math import spherical
        fff = lambda w: np.sum([(F(w, x) - y)**2 for x,y in zip(X,Y)])
        x0 = np.array([-1, -1, 0, -1])
        for _ in range(10):
            d = spherical(4)
            xx = np.linspace(-10,10,100)
            yy = [fff(x0 + a*d) for a in xx]
            pl.figure()
            pl.plot(xx,yy)
            pl.ylim(0, min(100, max(yy)))
            pl.show()

    # Minimize mean squared error with Basin hopping to avoid local minima.
    w = basinhopping(lambda w: np.sum([(F(w, x) - y)**2 for x,y in zip(X,Y)]),
                     np.array([0, 0, 0, 0]), niter=100).x

    # Return closure over best parameters.
    return (lambda x: F(w, x),
            lambda x: dFdx(w, x))


def main():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--accuracy', default='recall')
    p.add_argument('--runtime', default='pops')
    p.add_argument('--data', default='dev')
    p.add_argument('--log', default='dev')
    args = p.parse_args()

    df = pandas.read_csv(args.log)

    RUNTIME = '%s_new_policy_%s' % (args.data, args.runtime)
    ACCURACY = '%s_new_policy_%s' % (args.data, args.accuracy)

    df.sort_values(RUNTIME, inplace=1)

    xs = np.array(df[RUNTIME])
    ys = np.array(df[ACCURACY])

    # Notes:
    #
    # 1) I've doubled-check that fitting the curve with and without rescaling
    #    gives the same lambda estimates.
    #
    # 2) Aspect ratio and rescaling are important for inspecting the plots.
    #
    # 3) Using log10 scale is useful for making numbers user friendly.
    #
    log10_scale = int(round(np.log10(xs.ptp())))
    rescale = 10**-log10_scale
    xs = xs*rescale   # normalize runtime axis since it usually on some crazy scale relative to accuracy

    ax = pl.figure().add_subplot(111)
    ax.scatter(xs, ys, c='#558524', lw=0, s=30)

    f, g = fit(xs, ys)

    # show curve
    xx = np.linspace(xs.min()*.9, xs.max()*1.1, 100)
    ax.plot(xx, [f(x) for x in xx], c='r', alpha=0.5, lw=2, label='parametric fit')

    dx = xs.ptp()/10

    conesize = 0.2
    lambda_cone(ys, xs, ax, c='#a1d76a', conesize=conesize, lines=0)

    for x,y in zip(xs,ys):
        tradeoff = g(x)

        #print '%g => %g' % (x, g(x))
        print '%g => %g' % (x/rescale, g(x)*rescale)

        if 0:
            # tangent line along the (x, f(x))
            xx = np.linspace(x-dx, x+dx, 10)
            tt = g(x)*(xx - x) + f(x)
            pl.scatter([x], [f(x)], c='k')
            pl.plot(xx, tt, c='k', alpha=.5)

        if 0:
            # tangent line running thru (x,y)
            xx = np.linspace(x-dx, x+dx, 10)
            tt = g(x)*(xx - x) + y
            pl.plot(xx, tt, c='k', alpha=.5)

        if 1:
            # arrow along lambda direction.
            arrow(x, y, angle=tradeoff, offset=-conesize, c='#558524')

    pl.gca().set_aspect('equal')

    pl.xlabel(r'%s $(10^{%s})$' % (args.runtime.replace('_', r'\_'), log10_scale))
    pl.ylabel(r'%s' % args.accuracy.replace('_', r'\_'))
    pl.title('Parametric fit')
    pl.legend(loc=4)
    pl.show()


if __name__ == '__main__':
    main()
